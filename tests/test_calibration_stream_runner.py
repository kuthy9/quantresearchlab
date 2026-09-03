from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_continuous_replay as runner_module
from scripts.run_continuous_replay import (
    BRAIN_RUNTIME_STATE_SCHEMA_VERSION,
    BRAIN_CALIBRATION_FIELD_TYPES,
    DECISION_FIELD_TYPES,
    MARKET_CASE_INPUT_RUNTIME_STATE_SCHEMA_VERSION,
    MARKET_CASE_INPUT_DATA_CONTINUITY_POLICY,
    NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION,
    NATURAL_FUNNEL_SCHEMA_FINGERPRINT,
    OPEN_THESIS_BINDING_STAGES,
    _calendar_warmup_start,
    _compact_disposition_case_insert,
    _finalize_shadow_outputs,
    _load_market_case_input_profile,
    _natural_episode_funnel_summary,
    _open_thesis_binding_funnel_summary,
    _unexplained_episode_details_payload,
    _unexplained_episode_aggregate,
    _unexplained_episode_summary,
    _update_natural_episode_funnel,
    _update_open_thesis_binding_funnel,
    _update_unexplained_episode_summaries,
    _visualization_clocks,
)
from smc_trader.artifact_stream import atomic_parquet, sha256_file
from smc_trader.brain_calibration import (
    BrainCalibrationRecorder,
    BrainCalibrationRecord,
    RECORDER_SCHEMA_VERSION,
)
from smc_trader.causal_cases import (
    CAUSAL_CASE_INPUT_FIELD_TYPES,
    CAUSAL_CASE_OUTCOME_FIELD_TYPES,
    CAUSAL_CASE_RECORDER_SCHEMA_VERSION,
    CausalCaseRecorder,
)
from smc_trader.engine import ContinuousSMCEngine
from smc_trader.io import iter_completed_bars
from smc_trader.market_cases import (
    MARKET_CASE_INPUT_FIELD_TYPES,
    MARKET_CASE_RECORDER_SCHEMA_VERSION,
    MarketEpisodeCaseRecorder,
    expected_market_case_run_identity,
)
from smc_trader.model import (
    Action,
    Direction,
    MarketEpisodeState,
    NeutralEngineSnapshot,
    Playbook,
    PlaybookPhase,
    Timeframe,
    to_primitive,
)
from smc_trader.execution import ExecutionRealityInput
from smc_trader.observation import (
    EventMemory,
)
from smc_trader.scene_graph import (
    TemporalMarketSceneGraph,
    _is_terminal,
    market_episode_id,
)
from tests.helpers import session_bars
from smc_trader.shadow_outcome import (
    RECORDER_SCHEMA_VERSION as SHADOW_OUTCOME_RECORDER_SCHEMA_VERSION,
    SHADOW_DERIVED_SCHEMA_VERSION,
    SHADOW_EPISODE_OUTCOME_FIELD_TYPES,
    SHADOW_MECHANISM_CHALLENGE_FIELD_TYPES,
    SHADOW_MECHANISM_MOTIF_FIELD_TYPES,
    SHADOW_MOTIF_ROOT_SAMPLE_LIMIT,
    SHADOW_OUTCOME_FIELD_TYPES,
    SHADOW_OUTCOME_PROTOCOL,
    SHADOW_ROOT_EPISODE_FIELD_TYPES,
    SHADOW_ROOT_SEQUENCE_FIELD_TYPES,
    ShadowCandidateOutcomeRecord,
    ShadowCandidateOutcomeRecorder,
    ShadowEpisodeOutcomeRecord,
    ShadowMechanismChallengeRecord,
    ShadowMechanismMotifRecord,
    ShadowRootEpisodeRecord,
    ShadowRootSequenceRecord,
)


ROOT = Path(__file__).resolve().parents[1]


def _forbidden_market_action_layer(*_args, **_kwargs):
    raise AssertionError("market-case input invoked an action layer")


@pytest.mark.parametrize(
    ("start", "expected"),
    (
        (
            "2023-11-05T18:00:00-05:00",
            pd.Timestamp("2023-10-29 18:00", tz="America/New_York"),
        ),
        (
            "2023-03-12T18:00:00-04:00",
            pd.Timestamp("2023-03-05 18:00", tz="America/New_York"),
        ),
    ),
)
def test_calendar_warmup_preserves_market_wall_clock_across_dst(
    start: str,
    expected: pd.Timestamp,
) -> None:
    assert _calendar_warmup_start(
        pd.Timestamp(start),
        days=7,
    ) == expected


def test_stream_schemas_match_the_lightweight_contract() -> None:
    assert tuple(BRAIN_CALIBRATION_FIELD_TYPES) == tuple(
        field.name for field in fields(BrainCalibrationRecord)
    )
    assert tuple(SHADOW_OUTCOME_FIELD_TYPES) == tuple(
        field.name for field in fields(ShadowCandidateOutcomeRecord)
    )
    assert tuple(SHADOW_EPISODE_OUTCOME_FIELD_TYPES) == tuple(
        field.name for field in fields(ShadowEpisodeOutcomeRecord)
    )
    assert tuple(SHADOW_MECHANISM_CHALLENGE_FIELD_TYPES) == tuple(
        field.name for field in fields(ShadowMechanismChallengeRecord)
    )
    assert tuple(SHADOW_ROOT_EPISODE_FIELD_TYPES) == tuple(
        field.name for field in fields(ShadowRootEpisodeRecord)
    )
    assert tuple(SHADOW_ROOT_SEQUENCE_FIELD_TYPES) == tuple(
        field.name for field in fields(ShadowRootSequenceRecord)
    )
    assert tuple(SHADOW_MECHANISM_MOTIF_FIELD_TYPES) == tuple(
        field.name for field in fields(ShadowMechanismMotifRecord)
    )
    assert DECISION_FIELD_TYPES["asof"] == "timestamp_ny"
    assert DECISION_FIELD_TYPES["model_action"] == "large_string"
    assert DECISION_FIELD_TYPES["invalidation_source_id"] == "large_string"
    assert DECISION_FIELD_TYPES["target_ids"] == "large_string"
    assert DECISION_FIELD_TYPES["top_episode_id"] == "large_string"
    assert DECISION_FIELD_TYPES["global_market_mode"] == "large_string"
    assert DECISION_FIELD_TYPES["global_dislocated"] == "bool"
    assert DECISION_FIELD_TYPES["global_path_blocker_count"] == "int64"
    assert DECISION_FIELD_TYPES["global_material_conflict_count"] == "int64"
    assert DECISION_FIELD_TYPES["market_thesis_id"] == "large_string"
    assert DECISION_FIELD_TYPES["market_thesis_action_bound"] == "bool"
    assert (
        DECISION_FIELD_TYPES["playbook_plan_delivery_valid"] == "bool"
    )
    assert (
        DECISION_FIELD_TYPES["global_key_material_conflict_ids"]
        == "large_string"
    )
    assert "global_path_blocker_ids" not in DECISION_FIELD_TYPES
    assert "global_material_conflicts" not in DECISION_FIELD_TYPES
    assert "global_unexplained_episode_ids" not in DECISION_FIELD_TYPES
    assert DECISION_FIELD_TYPES["top_context_metadata"] == "large_string"
    assert "snapshot_hash" not in DECISION_FIELD_TYPES
    assert "group3_boundary_transitions" not in DECISION_FIELD_TYPES
    assert "group4_state" not in DECISION_FIELD_TYPES
    for repeated_identity in (
        "registry_hash",
        "model_code_hash",
        "config_hash",
        "primitive_protocol_hashes",
        "brain_input_contract_hash",
        "scene_hypothesis_id",
        "competing_scene_hypothesis_ids",
        "context_root_ids",
        "scene_revision_id",
        "protocol_version",
        "protocol_hash",
    ):
        assert repeated_identity not in BRAIN_CALIBRATION_FIELD_TYPES
    assert "evidence_revision_id" in BRAIN_CALIBRATION_FIELD_TYPES
    for context_feature in (
        "global_market_mode",
        "authority_relation",
        "authority_rank_gap",
        "conflict_role",
        "conflict_scope",
        "acceptance_state",
        "obstruction_distance_R",
        "free_path_R",
        "soft_obstruction_count",
        "hard_barrier_before_target",
        "ambiguity_count",
    ):
        assert context_feature in BRAIN_CALIBRATION_FIELD_TYPES


def test_open_thesis_binding_funnel_exposes_root_identity_bottleneck() -> None:
    thesis = SimpleNamespace(
        thesis_id="market-thesis:1",
        root_id="displacement:1",
        mechanism="directional_displacement",
        direction=Direction.LONG,
        authority_relation="unrelated",
        draw_candidate_ids=("draw:above", "draw:far-above"),
    )
    hypothesis = SimpleNamespace(
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        direction=Direction.LONG,
        market_thesis_ids=(thesis.thesis_id,),
        market_thesis_id=thesis.thesis_id,
        bound_market_thesis_id=None,
        playbook_match_strength=0.4,
        market_thesis_match_status="root_identity_unbound",
        hard_gate_results={"causal_order": True},
        context_metadata={"playbook_plan_delivery_valid": "false"},
        plan=None,
        phase=PlaybookPhase.WEAKENING,
    )
    snapshot = SimpleNamespace(
        belief=SimpleNamespace(
            global_context=SimpleNamespace(
                market_epoch_id="epoch:1",
                updated_at=pd.Timestamp("2024-01-02T09:31:00-05:00"),
                open_market_theses=(thesis,),
            ),
            hypotheses={
                f"{Playbook.DISPLACEMENT_FIRST_PULLBACK.value}:long": hypothesis,
            },
        )
    )
    state: dict[str, object] = {}
    _update_open_thesis_binding_funnel(state, snapshot)
    unbound = _open_thesis_binding_funnel_summary(state)
    assert unbound["root_stage_counts"] == {
        "open_thesis_created": 1,
        "mechanism_direction_matched": 1,
        "exact_root_bound": 0,
        "causal_gates_complete": 0,
        "plan_delivery_valid": 0,
        "executable": 0,
    }
    assert unbound["highest_stage_observed_dispositions"] == {
        "matched_root_unbound": 1,
    }
    assert unbound["compact_case_index"]["cases"][0]["root_id"] == (
        thesis.root_id
    )
    assert unbound["compact_case_index"]["cases"][0][
        "draw_candidate_count"
    ] == 2
    assert unbound["compact_case_index"]["cases"][0][
        "sample_draw_candidate_ids"
    ] == ["draw:above", "draw:far-above"]

    resumed = pickle.loads(pickle.dumps(state))
    thesis.authority_relation = "aligned"
    snapshot.belief.global_context.updated_at = pd.Timestamp(
        "2024-01-02T09:32:00-05:00"
    )
    hypothesis.bound_market_thesis_id = thesis.thesis_id
    hypothesis.market_thesis_match_status = "exact_root_bound"
    hypothesis.hard_gate_results = {"causal_order": False}
    _update_open_thesis_binding_funnel(resumed, snapshot)
    hypothesis.hard_gate_results = {"causal_order": True}
    hypothesis.context_metadata = {
        "playbook_plan_delivery_valid": "true"
    }
    hypothesis.plan = SimpleNamespace()
    hypothesis.phase = PlaybookPhase.EXECUTABLE
    _update_open_thesis_binding_funnel(resumed, snapshot)
    _update_open_thesis_binding_funnel(resumed, snapshot)

    complete = _open_thesis_binding_funnel_summary(resumed)
    assert all(value == 1 for value in complete["root_stage_counts"].values())
    assert complete["highest_stage_observed_dispositions"] == {
        "executable": 1
    }
    assert complete["last_observed_failed_hard_gate_counts"] == {}
    assert all(
        row["opportunities_first_reached"] == 1
        for row in complete["match_rows"]
    )
    stage_relations = {
        row["stage"]: row["authority_relation"]
        for row in complete["match_rows"]
    }
    assert stage_relations["open_thesis_created"] == "unrelated"
    assert stage_relations["exact_root_bound"] == "aligned"


def test_compact_binding_cases_retain_each_disposition_when_full() -> None:
    cases: dict[str, dict[str, object]] = {}
    for index in range(40):
        identity = f"early:{index:02d}"
        _compact_disposition_case_insert(
            cases,
            identity,
            {
                "record_id": identity,
                "first_seen_at": f"2024-01-01T00:{index:02d}:00+00:00",
                "disposition": "matched_root_unbound",
            },
        )

    _compact_disposition_case_insert(
        cases,
        "late:visible-draw-missing",
        {
            "record_id": "late:visible-draw-missing",
            "first_seen_at": "2024-01-02T00:00:00+00:00",
            "disposition": "visible_draw_missing",
        },
    )

    assert len(cases) == 40
    assert "late:visible-draw-missing" in cases
    assert {
        str(case["disposition"]) for case in cases.values()
    } == {"matched_root_unbound", "visible_draw_missing"}


def test_open_thesis_binding_funnel_requires_a_nonempty_gate_contract() -> None:
    thesis = SimpleNamespace(
        thesis_id="market-thesis:empty-gates",
        root_id="displacement:empty-gates",
        mechanism="directional_displacement",
        direction=Direction.LONG,
        authority_relation="aligned",
    )
    hypothesis = SimpleNamespace(
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        direction=Direction.LONG,
        market_thesis_ids=(thesis.thesis_id,),
        market_thesis_id=thesis.thesis_id,
        bound_market_thesis_id=thesis.thesis_id,
        playbook_match_strength=1.0,
        market_thesis_match_status="exact_root_bound",
        hard_gate_results={},
        context_metadata={"playbook_plan_delivery_valid": "true"},
        plan=SimpleNamespace(),
        phase=PlaybookPhase.EXECUTABLE,
    )
    snapshot = SimpleNamespace(
        belief=SimpleNamespace(
            global_context=SimpleNamespace(
                market_epoch_id="epoch:empty-gates",
                updated_at=pd.Timestamp("2024-01-02T09:31:00-05:00"),
                open_market_theses=(thesis,),
            ),
            hypotheses={
                f"{Playbook.DISPLACEMENT_FIRST_PULLBACK.value}:long": (
                    hypothesis
                ),
            },
        )
    )
    state: dict[str, object] = {}
    _update_open_thesis_binding_funnel(state, snapshot)
    summary = _open_thesis_binding_funnel_summary(state)

    assert summary["root_stage_counts"]["exact_root_bound"] == 1
    assert summary["root_stage_counts"]["causal_gates_complete"] == 0
    assert summary["last_observed_failed_hard_gate_counts"] == {
        "missing_hard_gate_results": 1
    }


def test_open_thesis_funnel_reports_secondary_root_as_unbound() -> None:
    primary = SimpleNamespace(
        thesis_id="market-thesis:primary",
        root_id="displacement:primary",
        mechanism="directional_displacement",
        direction=Direction.LONG,
        authority_relation="aligned",
    )
    secondary = SimpleNamespace(
        thesis_id="market-thesis:secondary",
        root_id="displacement:secondary",
        mechanism="directional_displacement",
        direction=Direction.LONG,
        authority_relation="aligned",
    )
    hypothesis = SimpleNamespace(
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        direction=Direction.LONG,
        market_thesis_ids=(primary.thesis_id, secondary.thesis_id),
        market_thesis_id=primary.thesis_id,
        bound_market_thesis_id=primary.thesis_id,
        playbook_match_strength=0.8,
        market_thesis_match_status="exact_root_bound",
        hard_gate_results={"causal_order": True},
        context_metadata={"playbook_plan_delivery_valid": "true"},
        plan=SimpleNamespace(),
        phase=PlaybookPhase.EXECUTABLE,
    )
    snapshot = SimpleNamespace(
        belief=SimpleNamespace(
            global_context=SimpleNamespace(
                market_epoch_id="epoch:multiple-theses",
                updated_at=pd.Timestamp("2024-01-02T09:31:00-05:00"),
                open_market_theses=(primary, secondary),
            ),
            hypotheses={
                f"{Playbook.DISPLACEMENT_FIRST_PULLBACK.value}:long": (
                    hypothesis
                ),
            },
        )
    )
    state: dict[str, object] = {}
    _update_open_thesis_binding_funnel(state, snapshot)
    summary = _open_thesis_binding_funnel_summary(state)
    cases = {
        row["thesis_id"]: row
        for row in summary["compact_case_index"]["cases"]
    }

    assert cases[primary.thesis_id]["current_match_status"] == (
        "exact_root_bound"
    )
    assert cases[secondary.thesis_id]["current_match_status"] == (
        "root_identity_unbound"
    )
    assert cases[secondary.thesis_id]["disposition"] == (
        "matched_root_unbound"
    )
    assert cases[secondary.thesis_id]["highest_stage_observed"] == (
        "mechanism_direction_matched"
    )
    assert cases[secondary.thesis_id][
        "selected_hypothesis_match_strength"
    ] is None
    assert summary["root_stage_counts"]["mechanism_direction_matched"] == 2
    assert summary["root_stage_counts"]["exact_root_bound"] == 1


def test_open_thesis_funnel_tracks_root_candidates_independently() -> None:
    primary = SimpleNamespace(
        thesis_id="market-thesis:primary-root-specific",
        root_id="displacement:primary-root-specific",
        mechanism="directional_displacement",
        direction=Direction.LONG,
        authority_relation="aligned",
    )
    secondary = SimpleNamespace(
        thesis_id="market-thesis:secondary-root-specific",
        root_id="displacement:secondary-root-specific",
        mechanism="directional_displacement",
        direction=Direction.LONG,
        authority_relation="aligned",
    )

    def candidate(thesis: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(
            playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
            direction=Direction.LONG,
            required_root_id=thesis.root_id,
            market_thesis_root_id=thesis.root_id,
            market_thesis_ids=(thesis.thesis_id,),
            market_thesis_id=thesis.thesis_id,
            bound_market_thesis_id=thesis.thesis_id,
            playbook_match_strength=0.8,
            market_thesis_match_status="exact_root_bound",
            hard_gate_results={"causal_order": True},
            context_metadata={"playbook_plan_delivery_valid": "true"},
            plan=SimpleNamespace(),
            phase=PlaybookPhase.EXECUTABLE,
        )

    primary_candidate = candidate(primary)
    secondary_candidate = candidate(secondary)
    candidate_items = (
        ("candidate:primary", primary_candidate),
        ("candidate:secondary", secondary_candidate),
    )
    snapshot = SimpleNamespace(
        belief=SimpleNamespace(
            global_context=SimpleNamespace(
                market_epoch_id="epoch:root-specific",
                updated_at=pd.Timestamp("2024-01-02T09:31:00-05:00"),
                open_market_theses=(primary, secondary),
            ),
            # The overview slot intentionally contains only one summary.  The
            # funnel must consume the two root-specific action candidates.
            hypotheses={
                f"{Playbook.DISPLACEMENT_FIRST_PULLBACK.value}:long": (
                    primary_candidate
                )
            },
            action_candidate_items=lambda: candidate_items,
        )
    )
    state: dict[str, object] = {}

    _update_open_thesis_binding_funnel(state, snapshot)
    summary = _open_thesis_binding_funnel_summary(state)
    cases = {
        row["thesis_id"]: row
        for row in summary["compact_case_index"]["cases"]
    }

    assert summary["root_theses_observed"] == 2
    assert summary["root_stage_counts"] == {
        stage: 2 for stage in OPEN_THESIS_BINDING_STAGES
    }
    assert summary["highest_stage_observed_dispositions"] == {
        "executable": 2
    }
    assert cases[primary.thesis_id]["current_match_status"] == (
        "exact_root_bound"
    )
    assert cases[secondary.thesis_id]["current_match_status"] == (
        "exact_root_bound"
    )


def _natural_snapshot(
    playbook: Playbook,
    direction: Direction,
    phase: PlaybookPhase,
    *,
    minute: int,
    action: Action = Action.WAIT,
    risk_action: Action = Action.WAIT,
) -> SimpleNamespace:
    key = f"{playbook.value}:{direction.value}"
    hypothesis = SimpleNamespace(
        key=key,
        playbook=playbook,
        direction=direction,
        episode_id=f"episode:{playbook.value}:{direction.value}",
        setup_context_id=f"setup:{playbook.value}:{direction.value}",
        phase=phase,
        terminal_reason=(
            "test_terminal"
            if phase in {PlaybookPhase.COMPLETED, PlaybookPhase.INVALIDATED}
            else None
        ),
        context_metadata={
            "source_timeframe": "5m",
            "source_tier": "B",
            "authority_relation": "supports_challenger",
        },
        hard_gate_results={"aligned_trigger": phase is PlaybookPhase.EXECUTABLE},
    )
    asof = pd.Timestamp("2022-06-06T10:00:00-04:00") + pd.Timedelta(
        minutes=minute
    )
    context = SimpleNamespace(
        market_mode=SimpleNamespace(value="transition"),
        candidate_structured_episode_ids=(),
        unexplained_structured_episode_ids=(),
        external_draw_candidates={"above": (), "below": ()},
        path_blocker_ids=(),
    )
    candidates = {key: hypothesis}
    belief = SimpleNamespace(
        hypotheses={key: hypothesis},
        global_context=context,
    )
    belief.action_candidate_items = lambda: tuple(candidates.items())
    belief.resolve_hypothesis = lambda identity: candidates.get(identity)
    return SimpleNamespace(
        observation=SimpleNamespace(asof=asof),
        belief=belief,
        decision=SimpleNamespace(
            best_hypothesis_key=key,
            selected_action=action,
        ),
        risk=SimpleNamespace(final_action=risk_action),
    )


@pytest.mark.parametrize(
    ("playbook", "direction"),
    tuple(
        (playbook, direction)
        for playbook in Playbook
        for direction in Direction
    ),
)
def test_natural_funnel_counts_each_episode_stage_once_and_preserves_same_bar_order(
    playbook: Playbook,
    direction: Direction,
) -> None:
    state: dict[str, object] = {"entry_approvals": {}}
    executable = _natural_snapshot(
        playbook,
        direction,
        PlaybookPhase.EXECUTABLE,
        minute=0,
        action=Action.ENTER,
        risk_action=Action.ENTER,
    )
    _update_natural_episode_funnel(state, executable)
    _update_natural_episode_funnel(state, executable)

    result = _natural_episode_funnel_summary(
        state,
        end_asof=executable.observation.asof,
        include_details=True,
    )
    compact = _natural_episode_funnel_summary(
        state,
        end_asof=executable.observation.asof,
    )
    assert "episodes" not in compact
    assert "candidate_root_rows" not in compact
    assert len(compact["compact_case_index"]["cases"]) <= 40
    episode = result["episodes"][0]
    assert [item["stage"] for item in episode["first_reach_sequence"]] == [
        "candidate_root",
        "forming",
        "armed",
        "waiting_location",
        "waiting_trigger",
        "executable",
        "decision_enter",
        "risk_pass",
    ]
    assert len(result["rows"]) == 8
    assert all(row["episodes_first_reached"] == 1 for row in result["rows"])


def test_natural_funnel_preserves_exact_lsr_zone_to_pullback_identity() -> None:
    state: dict[str, object] = {"entry_approvals": {}}

    def snapshot(phase: PlaybookPhase, minute: int) -> SimpleNamespace:
        value = _natural_snapshot(
            Playbook.LIQUIDITY_SWEEP_REVERSAL,
            Direction.SHORT,
            phase,
            minute=minute,
        )
        hypothesis = next(iter(value.belief.hypotheses.values()))
        hypothesis.episode_id = "lsr-entry-episode:zone-a"
        hypothesis.setup_context_id = "lsr-entry-episode:zone-a"
        hypothesis.context_id = "lsr-context:pool-a"
        hypothesis.context_thesis_id = "context-thesis:pool-a"
        hypothesis.parent_context_thesis_id = "context-thesis:pool-a"
        hypothesis.initiating_event_id = "manipulation:pool-a"
        hypothesis.market_thesis_root_id = "root:pool-a"
        hypothesis.entry_location_id = "entry-location:zone-a"
        hypothesis.entry_path_id = "zone-return:zone-a"
        hypothesis.context_metadata = {
            **hypothesis.context_metadata,
            "lsr_manipulation_id": "manipulation:pool-a",
            "lsr_pool_path_id": "pool-reversal:pool-a",
            "lsr_displacement_id": "displacement:reverse-a",
            "lsr_entry_zone_id": "fvg:zone-a",
        }
        hypothesis.plan_feasibility = SimpleNamespace(
            valid=phase is PlaybookPhase.EXECUTABLE
        )
        hypothesis.selected_trigger = (
            None
            if phase is not PlaybookPhase.EXECUTABLE
            else SimpleNamespace(
                trigger_id="micro-bos:zone-a",
                trigger_kind="micro_bos_confirmed",
                observed_at=value.observation.asof,
            )
        )
        return value

    eligible = snapshot(PlaybookPhase.WAITING_LOCATION, 0)
    pullback = snapshot(PlaybookPhase.WAITING_TRIGGER, 1)
    executable = snapshot(PlaybookPhase.EXECUTABLE, 2)
    for value in (eligible, pullback, executable):
        _update_natural_episode_funnel(state, value)

    summary = _natural_episode_funnel_summary(
        state,
        end_asof=executable.observation.asof,
        include_details=True,
    )
    assert summary["diagnostic_schema_version"] == (
        NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION
    )
    assert len(summary["episodes"]) == 1
    episode = summary["episodes"][0]
    assert episode["context_thesis_id"] == "context-thesis:pool-a"
    assert episode["parent_context_thesis_id"] == "context-thesis:pool-a"
    assert episode["lsr_manipulation_id"] == "manipulation:pool-a"
    assert episode["lsr_pool_path_id"] == "pool-reversal:pool-a"
    assert episode["lsr_displacement_id"] == "displacement:reverse-a"
    assert episode["lsr_entry_zone_id"] == "fvg:zone-a"
    assert episode["entry_location_id"] == "entry-location:zone-a"
    assert episode["entry_path_id"] == "zone-return:zone-a"
    assert episode["eligible_entry_zone"] is True
    assert episode["eligible_zone_first_observed_at"] == (
        eligible.observation.asof.isoformat()
    )
    assert episode["first_pullback_at"] == pullback.observation.asof.isoformat()
    assert episode["first_trigger_observed_at"] == (
        executable.observation.asof.isoformat()
    )
    assert episode["selected_trigger_id"] == "micro-bos:zone-a"
    assert episode["first_plan_valid_at"] == (
        executable.observation.asof.isoformat()
    )


def test_pre_entry_weakening_does_not_manufacture_executable_reach() -> None:
    state: dict[str, object] = {"entry_approvals": {}}
    weakening = _natural_snapshot(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        PlaybookPhase.WEAKENING,
        minute=0,
    )
    _update_natural_episode_funnel(state, weakening)
    result = _natural_episode_funnel_summary(
        state,
        end_asof=weakening.observation.asof,
        include_details=True,
    )
    assert [
        item["stage"] for item in result["episodes"][0]["first_reach_sequence"]
    ] == [
        "candidate_root",
        "forming",
        "armed",
        "waiting_location",
        "waiting_trigger",
    ]
    assert "executable" not in result["episodes"][0]["first_reached_at"]


def test_candidate_root_denominator_precedes_episode_and_survives_resume() -> None:
    root_id = "manipulation:candidate-root"

    def candidate_snapshot(minute: int, *, bind: bool) -> SimpleNamespace:
        snapshot = _natural_snapshot(
            Playbook.LIQUIDITY_SWEEP_REVERSAL,
            Direction.SHORT,
            PlaybookPhase.FORMING,
            minute=minute,
        )
        hypothesis = next(iter(snapshot.belief.hypotheses.values()))
        hypothesis.episode_id = "episode:bound" if bind else None
        hypothesis.initiating_event_id = root_id if bind else None
        snapshot.belief.global_context.candidate_structured_episode_ids = (
            (root_id,) if not bind else ()
        )
        snapshot.observation.manipulations = (
            SimpleNamespace(
                manipulation_id=root_id,
                source_inventory_item_id="pool:candidate-root",
                formed_at=pd.Timestamp("2022-06-06T10:00:00-04:00"),
                resolved_at=None,
                censored_at=None,
                source_timeframe=Timeframe.M5,
                side="above",
                source_kind="formed_liquidity_pool",
                lifecycle=SimpleNamespace(value="reaccepted"),
            ),
        )
        snapshot.observation.path_sequences = ()
        snapshot.observation.liquidity_inventory = (
            SimpleNamespace(
                item_id="pool:candidate-root",
                structural_rank="intermediate",
            ),
        )
        return snapshot

    first = candidate_snapshot(0, bind=False)
    uninterrupted: dict[str, object] = {}
    _update_natural_episode_funnel(uninterrupted, first)
    resumed = pickle.loads(pickle.dumps(uninterrupted))

    linked = candidate_snapshot(1, bind=True)
    _update_natural_episode_funnel(uninterrupted, linked)
    _update_natural_episode_funnel(resumed, linked)
    _update_natural_episode_funnel(resumed, linked)

    expected = _natural_episode_funnel_summary(
        uninterrupted,
        end_asof=linked.observation.asof,
        include_details=True,
    )
    actual = _natural_episode_funnel_summary(
        resumed,
        end_asof=linked.observation.asof,
        include_details=True,
    )
    assert actual == expected
    assert actual["denominators"] == {
        "candidate_roots_observed": 1,
        "candidate_roots_linked_to_episode": 1,
        "candidate_roots_unlinked": 0,
        "formed_episodes_observed": 1,
    }
    assert actual["candidate_root_rows"][0]["linked_episode_keys"]
    assert actual["episodes"][0]["episode_id"] == "episode:bound"
    assert actual["episodes"][0]["first_reached_at"]["forming"] == (
        first.observation.asof.isoformat()
    )
    assert actual["episodes"][0]["phase_duration_seconds"]["forming"] == 60.0


def test_compact_candidate_root_binds_when_root_departs_on_episode_clock() -> None:
    root_id = "manipulation:compact-root"
    first = _natural_snapshot(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        PlaybookPhase.FORMING,
        minute=0,
    )
    first_hypothesis = next(iter(first.belief.hypotheses.values()))
    first_hypothesis.episode_id = None
    first.belief.global_context.candidate_structured_episode_ids = (root_id,)
    first.observation.manipulations = (
        SimpleNamespace(
            manipulation_id=root_id,
            source_inventory_item_id="pool:compact-root",
            formed_at=first.observation.asof,
            resolved_at=None,
            censored_at=None,
            source_timeframe=Timeframe.M5,
            side="above",
            source_kind="formed_liquidity_pool",
            lifecycle=SimpleNamespace(value="reaccepted"),
        ),
    )
    first.observation.path_sequences = ()
    first.observation.liquidity_inventory = ()
    state: dict[str, object] = {}
    _update_natural_episode_funnel(state, first, retain_details=False)

    linked = _natural_snapshot(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        PlaybookPhase.FORMING,
        minute=1,
    )
    linked_hypothesis = next(iter(linked.belief.hypotheses.values()))
    linked_hypothesis.episode_id = "episode:compact-bound"
    linked_hypothesis.initiating_event_id = root_id
    linked.belief.global_context.candidate_structured_episode_ids = ()
    linked.observation.manipulations = ()
    linked.observation.path_sequences = ()
    linked.observation.liquidity_inventory = ()
    _update_natural_episode_funnel(state, linked, retain_details=False)
    summary = _natural_episode_funnel_summary(
        state,
        end_asof=linked.observation.asof,
    )
    assert summary["denominators"] == {
        "candidate_roots_observed": 1,
        "candidate_roots_linked_to_episode": 1,
        "candidate_roots_unlinked": 0,
        "formed_episodes_observed": 1,
    }


def test_candidate_root_binding_does_not_join_distinct_same_pool_episode() -> None:
    state: dict[str, object] = {}
    first = _natural_snapshot(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        PlaybookPhase.FORMING,
        minute=0,
    )
    root_id = "manipulation:first"
    first_hypothesis = next(iter(first.belief.hypotheses.values()))
    first_hypothesis.episode_id = None
    first.belief.global_context.candidate_structured_episode_ids = (root_id,)
    first.observation.manipulations = ()
    first.observation.path_sequences = ()
    first.observation.liquidity_inventory = ()
    _update_natural_episode_funnel(state, first)

    later = _natural_snapshot(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        PlaybookPhase.FORMING,
        minute=1,
    )
    later_hypothesis = next(iter(later.belief.hypotheses.values()))
    later_hypothesis.episode_id = "manipulation:second"
    later_hypothesis.initiating_event_id = "manipulation:second"
    # A shared shorter source identity must not suffix-bind two episodes.
    later_hypothesis.competing_episode_ids = ("pool:shared",)
    later.belief.global_context.candidate_structured_episode_ids = ()
    later.observation.manipulations = ()
    later.observation.path_sequences = ()
    later.observation.liquidity_inventory = ()
    _update_natural_episode_funnel(state, later)

    result = _natural_episode_funnel_summary(
        state,
        end_asof=later.observation.asof,
        include_details=True,
    )
    assert result["denominators"]["candidate_roots_unlinked"] == 1


def test_unexplained_summary_records_first_failed_gate_without_trace() -> None:
    root_id = "manipulation:1"
    manipulation = SimpleNamespace(
        manipulation_id=root_id,
        source_inventory_item_id="pool:1",
        formed_at=pd.Timestamp("2022-06-06T10:00:00-04:00"),
        resolved_at=None,
        censored_at=None,
        source_timeframe=Timeframe.M5,
        side="above",
        source_kind="formed_liquidity_pool",
        lifecycle=SimpleNamespace(value="reaccepted"),
    )
    hypothesis = SimpleNamespace(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        direction=Direction.SHORT,
        hard_gate_results={
            "eligible_root": True,
            "opposite_displacement": False,
        },
        context_metadata={"authority_relation": "local_countertrend"},
    )
    context = SimpleNamespace(
        unexplained_structured_episode_ids=(root_id,),
        external_draw_candidates={"above": (), "below": ("draw:1",)},
        path_blocker_ids=("barrier:long", "barrier:short"),
        obstruction_views={
                "long": SimpleNamespace(
                    hard_barriers=(
                        SimpleNamespace(
                            obstruction_id="barrier:long",
                            contact_price=lambda _direction: 101.0,
                        ),
                    )
                ),
                "short": SimpleNamespace(
                    hard_barriers=(
                        SimpleNamespace(
                            obstruction_id="barrier:short",
                            contact_price=lambda _direction: 99.0,
                        ),
                    )
                ),
        },
    )
    snapshot = SimpleNamespace(
        observation=SimpleNamespace(
            asof=pd.Timestamp("2022-06-06T10:02:00-04:00"),
            price=100.0,
            manipulations=(manipulation,),
            path_sequences=(),
            liquidity_inventory=(
                SimpleNamespace(item_id="pool:1", structural_rank="external"),
            ),
        ),
        belief=SimpleNamespace(
            global_context=context,
            hypotheses={"lsr:short": hypothesis},
        ),
    )
    state: dict[str, object] = {}

    _update_unexplained_episode_summaries(state, snapshot)
    rows = _unexplained_episode_summary(
        state,
        end_asof=snapshot.observation.asof,
    )

    assert len(rows) == 1
    assert rows[0]["first_failed_gate"] == "opposite_displacement"
    assert rows[0]["unexplained_reason"] == "no_reverse_displacement"
    assert rows[0]["nearest_playbook"] == (
        Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    )
    assert rows[0]["hard_blocker_count"] == 1
    assert rows[0]["nearest_blocker_id"] == "barrier:short"
    assert rows[0]["visible_draw_count"] == 1
    assert rows[0]["nearest_draw_id"] == "draw:1"
    assert rows[0]["censored"] is True
    assert "observation" not in rows[0]
    aggregate = _unexplained_episode_aggregate(
        state,
        end_asof=snapshot.observation.asof,
    )
    assert aggregate["episodes_observed"] == 1
    assert aggregate["rows"][0]["episodes"] == 1


def test_natural_funnel_counts_same_bar_fill_and_close_and_final_duration() -> None:
    playbook = Playbook.LIQUIDITY_SWEEP_REVERSAL
    direction = Direction.SHORT
    state: dict[str, object] = {
        "entry_approvals": {
            "thesis:1": {
                "episode_id": f"episode:{playbook.value}:{direction.value}",
            }
        }
    }
    opened = _natural_snapshot(
        playbook,
        direction,
        PlaybookPhase.ENTERED,
        minute=0,
    )
    _update_natural_episode_funnel(state, opened)
    closed = _natural_snapshot(
        playbook,
        direction,
        PlaybookPhase.ENTERED,
        minute=2,
    )
    trade = SimpleNamespace(
        thesis_hash="thesis:1",
        playbook=playbook.value,
        direction=direction.value,
        exit_reason="same_bar_conservative_target_stop",
    )
    _update_natural_episode_funnel(
        state,
        closed,
        step=SimpleNamespace(position=None, closed_trades=(trade,)),
    )

    result = _natural_episode_funnel_summary(
        state,
        end_asof=closed.observation.asof,
        include_details=True,
    )
    episode = result["episodes"][0]
    assert [
        item["stage"] for item in episode["first_reach_sequence"][-2:]
    ] == ["order_filled", "position_terminal"]
    assert episode["terminal"] is True
    assert episode["phase_duration_seconds"]["entered"] == 120.0


def test_compact_natural_funnel_counts_position_after_action_root_closes() -> None:
    state: dict[str, object] = {"entry_approvals": {}}
    opened = _natural_snapshot(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        PlaybookPhase.ENTERED,
        minute=0,
    )
    hypothesis = next(iter(opened.belief.hypotheses.values()))
    _update_natural_episode_funnel(
        state,
        opened,
        retain_details=False,
    )

    closed_root = _natural_snapshot(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        PlaybookPhase.INACTIVE,
        minute=1,
    )
    closed_root.belief.action_candidate_items = lambda: ()
    closed_root.belief.resolve_hypothesis = lambda identity: None
    position = SimpleNamespace(
        playbook=hypothesis.playbook,
        direction=hypothesis.direction,
        setup_id=hypothesis.setup_context_id,
    )
    _update_natural_episode_funnel(
        state,
        closed_root,
        step=SimpleNamespace(position=position, closed_trades=()),
        retain_details=False,
    )
    summary = _natural_episode_funnel_summary(
        state,
        end_asof=closed_root.observation.asof,
    )
    assert any(
        row["stage"] == "order_filled"
        and row["episodes_first_reached"] == 1
        for row in summary["rows"]
    )


def test_compact_natural_funnel_retires_terminal_identity_after_counting_reason() -> None:
    state: dict[str, object] = {"entry_approvals": {}}
    terminal = _natural_snapshot(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        PlaybookPhase.INVALIDATED,
        minute=0,
    )
    hypothesis = next(iter(terminal.belief.hypotheses.values()))
    hypothesis.terminal_reason = "protected_structure_broken"
    _update_natural_episode_funnel(state, terminal, retain_details=False)
    at_terminal = _natural_episode_funnel_summary(
        state,
        end_asof=terminal.observation.asof,
    )
    assert at_terminal["terminal_disposition_counts"] == {
        "protected_structure_broken": 1
    }

    absent = _natural_snapshot(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        PlaybookPhase.INACTIVE,
        minute=1,
    )
    absent.belief.action_candidate_items = lambda: ()
    absent.belief.resolve_hypothesis = lambda identity: None
    _update_natural_episode_funnel(state, absent, retain_details=False)
    compact = state["natural_funnel_compact_state"]
    assert compact["episode_masks"] == {}
    assert compact["episode_metadata"] == {}
    summary = _natural_episode_funnel_summary(
        state,
        end_asof=absent.observation.asof,
    )
    assert summary["terminal_disposition_counts"] == {
        "protected_structure_broken": 1
    }


@pytest.mark.parametrize("retain_details", (True, False))
def test_natural_funnel_observes_retained_episode_terminal_without_action_authority(
    retain_details: bool,
) -> None:
    state: dict[str, object] = {"entry_approvals": {}}
    forming = _natural_snapshot(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        PlaybookPhase.FORMING,
        minute=0,
    )
    _update_natural_episode_funnel(
        state,
        forming,
        retain_details=retain_details,
    )

    terminal = _natural_snapshot(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        PlaybookPhase.INVALIDATED,
        minute=1,
    )
    terminal_hypothesis = next(iter(terminal.belief.hypotheses.values()))
    terminal.belief.action_candidate_items = lambda: ()
    terminal.belief.lifecycle_candidate_items = lambda: (
        (terminal_hypothesis.key, terminal_hypothesis),
    )
    assert terminal.belief.action_candidate_items() == ()
    _update_natural_episode_funnel(
        state,
        terminal,
        retain_details=retain_details,
    )

    absent = _natural_snapshot(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        PlaybookPhase.INACTIVE,
        minute=2,
    )
    absent.belief.action_candidate_items = lambda: ()
    absent.belief.lifecycle_candidate_items = lambda: ()
    absent.belief.resolve_hypothesis = lambda identity: None
    _update_natural_episode_funnel(
        state,
        absent,
        retain_details=retain_details,
    )

    summary = _natural_episode_funnel_summary(
        state,
        end_asof=absent.observation.asof,
        include_details=retain_details,
    )
    if retain_details:
        assert len(summary["episodes"]) == 1
        episode = summary["episodes"][0]
        assert episode["terminal"] is True
        assert episode["censored"] is False
        assert episode["exit_reason"] == "test_terminal"
        assert episode["phase_duration_seconds"]["forming"] == 60.0
    else:
        assert summary["terminal_disposition_counts"] == {
            "test_terminal": 1
        }
        compact = state["natural_funnel_compact_state"]
        assert compact["episode_masks"] == {}
        assert compact["episode_metadata"] == {}


def test_compact_natural_funnel_summary_does_not_mutate_checkpoint_cases() -> None:
    state: dict[str, object] = {"entry_approvals": {}}
    snapshot = _natural_snapshot(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        PlaybookPhase.FORMING,
        minute=0,
    )
    _update_natural_episode_funnel(
        state,
        snapshot,
        retain_details=False,
    )
    internal_cases = state["natural_funnel_compact_state"]["episode_cases"]
    assert all(not row["censored"] for row in internal_cases.values())
    _natural_episode_funnel_summary(
        state,
        end_asof=snapshot.observation.asof,
    )
    assert all(not row["censored"] for row in internal_cases.values())


@pytest.mark.parametrize("detail_mode", (True, False))
def test_unexplained_root_departure_is_a_terminal_resolution(
    detail_mode: bool,
) -> None:
    root_id = "manipulation:departing"
    context = SimpleNamespace(
        unexplained_structured_episode_ids=(root_id,),
        external_draw_candidates={"above": (), "below": ()},
        path_blocker_ids=(),
    )
    manipulation = SimpleNamespace(
        manipulation_id=root_id,
        source_inventory_item_id="pool:departing",
        formed_at=pd.Timestamp("2022-06-06T10:00:00-04:00"),
        resolved_at=None,
        censored_at=None,
        source_timeframe=Timeframe.M5,
        side="below",
        source_kind="formed_liquidity_pool",
        lifecycle=SimpleNamespace(value="reaccepted"),
    )
    hypothesis = SimpleNamespace(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        direction=Direction.LONG,
        hard_gate_results={"entry_zone": False},
        context_metadata={},
    )
    snapshot = SimpleNamespace(
        observation=SimpleNamespace(
            asof=pd.Timestamp("2022-06-06T10:01:00-04:00"),
            manipulations=(manipulation,),
            path_sequences=(),
            liquidity_inventory=(),
        ),
        belief=SimpleNamespace(
            global_context=context,
            hypotheses={"lsr:long": hypothesis},
        ),
    )
    state: dict[str, object] = {
        "unexplained_episode_detail_mode": detail_mode,
    }
    _update_unexplained_episode_summaries(state, snapshot)
    snapshot.observation.asof += pd.Timedelta(minutes=1)
    context.unexplained_structured_episode_ids = ()
    _update_unexplained_episode_summaries(state, snapshot)

    if detail_mode:
        row = _unexplained_episode_summary(state)[0]
        assert row["terminal"] is True
        assert row["censored"] is False
        assert row["resolution"] == "explained_or_root_resolved"
    else:
        assert _unexplained_episode_summary(state) == []
        aggregate = _unexplained_episode_aggregate(state)
        assert aggregate["episodes_observed"] == 1
        assert aggregate["rows"][0]["resolution"] == (
            "explained_or_root_resolved"
        )


def _write_source(tmp_path: Path, *, periods: int = 32) -> Path:
    index = pd.date_range(
        "2022-06-06 18:00",
        periods=periods,
        freq="min",
        tz="America/New_York",
        name="ts",
    )
    steps = pd.Series(range(len(index)), index=index, dtype=float)
    close = 12_500.0 + 0.25 * (steps % 9)
    frame = pd.DataFrame(
        {
            "open": close.shift(1, fill_value=close.iloc[0]),
            "close": close,
            "volume": 10.0 + (steps % 5),
            "symbol": "NQU2",
            "instrument_id": 1,
        },
        index=index,
    )
    frame["high"] = frame[["open", "close"]].max(axis=1) + 0.25
    frame["low"] = frame[["open", "close"]].min(axis=1) - 0.25
    source = tmp_path / "research_previous_session_front.parquet"
    frame.to_parquet(source)
    return source


def _command(
    source: Path,
    output: Path,
    *,
    resume: bool = False,
    stop_after: int = 0,
    brain_calibration: bool = False,
    brain_diagnostics: bool = False,
    calibration_only: bool = False,
    compact_scene_graph: bool = False,
    shadow_outcomes: bool = False,
    shadow_details: bool = False,
    causal_case_library: bool = False,
    market_case_input: bool = False,
    fail_shadow_finalize_after_batches: int = 0,
    unexplained_details: bool = False,
    natural_funnel_details: bool = False,
    end: str = "2022-06-06T18:32:00-04:00",
    visualize_at: tuple[str, ...] = (),
    validation_protocol: str | Path = "configs/data_splits.json",
    market_case_profile_registry: str | Path | None = None,
    model_config: str | Path = "configs/model.json",
    action_disabled_playbooks: tuple[str, ...] = (),
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_continuous_replay.py",
        "--source",
        str(source),
        "--output",
        str(output),
        "--config",
        str(model_config),
        "--validation-protocol",
        str(validation_protocol),
        "--start",
        "2022-06-06T18:00:00-04:00",
        "--end",
        end,
        "--warmup-days",
        "0",
        "--shard-rows",
        "7",
        "--checkpoint-bars",
        "5",
        "--acknowledge-research-roll-lineage",
    ]
    if market_case_profile_registry is not None:
        command.extend(
            [
                "--market-case-profile-registry",
                str(market_case_profile_registry),
            ]
        )
    if resume:
        command.append("--resume")
    if stop_after:
        command.extend(["--diagnostic-stop-after-bars", str(stop_after)])
    if brain_calibration:
        command.append("--brain-calibration")
    if brain_diagnostics:
        command.append("--brain-diagnostics")
    if calibration_only:
        command.append("--calibration-only")
    if compact_scene_graph:
        command.append("--compact-scene-graph")
    if shadow_outcomes:
        command.append("--shadow-outcomes")
    if shadow_details:
        command.append("--shadow-details")
    if causal_case_library:
        command.append("--causal-case-library")
    if market_case_input:
        command.append("--market-case-input")
    if fail_shadow_finalize_after_batches:
        command.extend(
            [
                "--diagnostic-fail-shadow-finalize-after-batches",
                str(fail_shadow_finalize_after_batches),
            ]
        )
    if unexplained_details:
        command.append("--include-unexplained-episode-details")
    if natural_funnel_details:
        command.append("--include-natural-funnel-details")
    for playbook in action_disabled_playbooks:
        command.extend(["--action-disabled-playbook", playbook])
    for clock in visualize_at:
        command.extend(["--visualize-at", clock])
    return command


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_short_market_input_protocol(
    tmp_path: Path,
    *,
    end_exclusive: str = "2022-06-06T18:32:00-04:00",
) -> Path:
    payload = json.loads(
        (
            ROOT / "configs/market_case_input_profiles_v2.json"
        ).read_text(encoding="utf-8")
    )
    profile = payload["market_case_input_profiles"][
        "market_episode_input_smoke_2024_01_08"
    ]
    profile.update(
        {
            "allowed_ohlcv_role": "calibration",
            "start": "2022-06-06T18:00:00-04:00",
            "end_exclusive": end_exclusive,
            "warmup_calendar_days": 0,
        }
    )
    protocol = tmp_path / "short-market-input-profile-registry.json"
    protocol.write_text(json.dumps(payload), encoding="utf-8")
    return protocol


def _write_same_contract_gap_source(tmp_path: Path) -> Path:
    source = _write_source(tmp_path, periods=40)
    frame = pd.read_parquet(source)
    frame = frame.drop(frame.index[12:20])
    frame.to_parquet(source)
    return source


def _decision_rows(output: Path) -> pd.DataFrame:
    manifest = json.loads(
        (output / "decision_shards.manifest.json").read_text()
    )
    return pd.concat(
        [pd.read_parquet(output / item["path"]) for item in manifest["shards"]],
        ignore_index=True,
    )


def _materialized_stream_rows(output: Path, name: str) -> pd.DataFrame:
    manifest = json.loads(
        (output / f"{name}.manifest.json").read_text(encoding="utf-8")
    )
    if not manifest["shards"]:
        return pd.DataFrame(columns=tuple(manifest["field_types"]))
    return pd.concat(
        [
            pd.read_parquet(output / shard["path"])
            for shard in manifest["shards"]
        ],
        ignore_index=True,
    )


def _brain_calibration_rows(output: Path) -> pd.DataFrame:
    manifest = json.loads(
        (output / "brain_calibration_shards.manifest.json").read_text()
    )
    frames = [
        pd.read_parquet(output / item["path"])
        for item in manifest["shards"]
    ]
    if not frames:
        return pd.DataFrame(columns=tuple(manifest["field_types"]))
    return pd.concat(frames, ignore_index=True)


def _shadow_raw_challenge_row(
    playbook: str,
    *,
    candidate_id: str = "candidate-a",
    observed: pd.Timestamp | None = None,
    root_id: str | None = "root-a",
    geometry_complete: bool = True,
    accepted: bool = False,
) -> dict[str, object]:
    observed = observed or pd.Timestamp("2024-01-02T10:00:00-05:00")
    row: dict[str, object] = {
        name: False if kind == "bool" else 0 if kind == "int64" else None
        for name, kind in SHADOW_OUTCOME_FIELD_TYPES.items()
    }
    row.update(
        {
            "candidate_id": candidate_id,
            "event_kind": "confirmed_bos",
            "event_id": f"bos:{candidate_id}",
            "observed_at": observed,
            "resolved_at": observed + pd.Timedelta(minutes=2),
            "symbol": "NQ",
            "instrument_id": 1,
            "direction": "long",
            "source_timeframe": "5m",
            "source_ids": "[]",
            "candidate_origin": "eye_event",
            "available_trigger_kinds": "[]",
            "decision_price": 100.0,
            "entry_rule": "next_bar",
            "entry_reference_price": 100.0,
            "entry_price": 100.0,
            "entry_at": observed + pd.Timedelta(minutes=1),
            "invalidation_price": 99.0,
            "invalidation_source_id": "stop-a",
            "draw_id": "draw-a",
            "target_price": 102.0,
            "elapsed_real_1m_bars": 2,
            "geometry_complete": geometry_complete,
            "geometry_incomplete_reason": (
                None if geometry_complete else "entry_missing"
            ),
            "filled": True,
            "resolution": "target_first",
            "censored": False,
            "target_before_invalidation": True,
            "invalidation_before_target": False,
            "same_bar_collision": False,
            "mfe_R": 2.0,
            "mae_R": 0.25,
            "hit_0_5R": True,
            "hit_1R": True,
            "hit_2R": True,
            "structural_thesis_invalidated": False,
            "decision_action": "wait",
            "risk_action": "abstain",
            "playbook_outcomes": json.dumps(
                [
                    {
                        "playbook": playbook,
                        "runtime_enabled": True,
                        "accepted": accepted,
                        "first_failed_gate": "gate-a",
                        "failed_gates": ["gate-a"],
                        "event_order_signature": ["confirmed_bos"],
                        "plan_feasibility_valid": True,
                        "plan_feasibility_failure_reason": None,
                        "graph_connected": root_id is not None,
                        "exact_root_bound": root_id is not None,
                        "market_thesis_match_status": (
                            "exact_root_bound"
                            if root_id is not None
                            else "no_open_thesis"
                        ),
                        "market_thesis_id": (
                            None if root_id is None else "thesis-a"
                        ),
                        "market_thesis_root_id": root_id,
                        "market_thesis_mechanism": (
                            None if root_id is None else "directional_delivery"
                        ),
                        "market_thesis_authority_relation": (
                            None if root_id is None else "aligned"
                        ),
                        "liquidity_route_id": f"route:{playbook}",
                        "context_draw_id": "draw:h4:terminal",
                        "intermediate_liquidity_ids": ["draw:m5:waypoint"],
                        "primary_deliverable_target_id": "draw:h1:primary",
                        "terminal_draw_id": "draw:h4:terminal",
                        "authority_barrier_id": "obstruction:h4:protected",
                        "authority_barrier_price": 104.25,
                    }
                ]
            ),
        }
    )
    return row


def _write_shadow_test_stream(
    destination: Path,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    raw_path = destination / "shadow_outcome_shards/part-00000.parquet"
    atomic_parquet(
        pd.DataFrame(rows, columns=list(SHADOW_OUTCOME_FIELD_TYPES)),
        raw_path,
        field_types=SHADOW_OUTCOME_FIELD_TYPES,
    )
    return {
        "rows": len(rows),
        "next_shard_index": 1,
        "committed_shards": [
            {
                "index": 0,
                "path": "shadow_outcome_shards/part-00000.parquet",
                "rows": len(rows),
                "first_key": str(rows[0]["candidate_id"]),
                "last_key": str(rows[-1]["candidate_id"]),
                "sha256": sha256_file(raw_path),
            }
        ],
    }


def _shadow_raw_executable_row(candidate_id: str) -> dict[str, object]:
    playbook = Playbook.DISPLACEMENT_FIRST_PULLBACK.value
    row = _shadow_raw_challenge_row(
        playbook,
        candidate_id=candidate_id,
    )
    diagnostic = json.loads(str(row["playbook_outcomes"]))[0]
    diagnostic.update(
        {
            "accepted": True,
            "episode_id": "episode-a",
            "setup_id": "setup-a",
            "bound_market_thesis_id": "thesis-a",
            "phase": "executable",
        }
    )
    row.update(
        {
            "event_kind": "playbook_executable",
            "event_id": f"{playbook}:long:episode-a",
            "source_playbook": playbook,
            "source_episode_id": "episode-a",
            "source_setup_id": "setup-a",
            "playbook_outcomes": json.dumps([diagnostic]),
        }
    )
    return row


def test_shadow_finalizer_streams_batch_fragments_and_is_restartable(
    tmp_path: Path,
) -> None:

    stream = _write_shadow_test_stream(
        tmp_path,
        [
            _shadow_raw_challenge_row(
                Playbook.DISPLACEMENT_FIRST_PULLBACK.value
            ),
            _shadow_raw_challenge_row(
                Playbook.LIQUIDITY_SWEEP_REVERSAL.value
            ),
        ],
    )
    with pytest.raises(RuntimeError, match="Shadow finalizer interruption"):
        _finalize_shadow_outputs(
            tmp_path,
            shadow_stream_state=stream,
            source_stream_manifest="shadow_outcome_shards.manifest.json",
            maximum_rows=1,
            include_details=False,
            fail_after_batches=1,
        )
    assert not (tmp_path / "shadow_derived_manifest.json").exists()
    assert not (tmp_path / ".shadow_derived.tmp").exists()

    outputs = _finalize_shadow_outputs(
        tmp_path,
        shadow_stream_state=stream,
        source_stream_manifest="shadow_outcome_shards.manifest.json",
        maximum_rows=1,
        include_details=False,
    )
    assert outputs["independent_root_episode_count"] == 1
    assert outputs["raw_challenge_row_count"] == 2
    assert outputs["representative_root_challenge_count"] == 2
    assert outputs["unbound_challenge_row_count"] == 0
    root_manifest = json.loads(
        (tmp_path / outputs["root_episode_challenges"]).read_text()
    )
    assert root_manifest["rows"] == 1
    root_row = pd.read_parquet(
        tmp_path / "shadow_derived" / root_manifest["shards"][0]["path"]
    ).iloc[0]
    assert bool(root_row["dfp_rejected"])
    assert bool(root_row["lsr_rejected"])
    routes = json.loads(root_row["playbook_liquidity_routes"])
    assert routes[Playbook.DISPLACEMENT_FIRST_PULLBACK.value][
        "intermediate_liquidity_ids"
    ] == ["draw:m5:waypoint"]
    assert routes[Playbook.LIQUIDITY_SWEEP_REVERSAL.value][
        "authority_barrier_id"
    ] == "obstruction:h4:protected"
    assert routes[Playbook.LIQUIDITY_SWEEP_REVERSAL.value][
        "authority_barrier_price"
    ] == pytest.approx(104.25)
    assert not (tmp_path / ".shadow_derived.tmp").exists()
    assert not tuple((tmp_path / "shadow_derived").glob("*.sqlite3"))
    derived_manifest = json.loads(
        (tmp_path / "shadow_derived_manifest.json").read_text()
    )
    assert derived_manifest["schema_version"] == SHADOW_DERIVED_SCHEMA_VERSION
    sequence_manifest = json.loads(
        (tmp_path / outputs["root_episode_sequences"]).read_text()
    )
    sequence_shard = (
        tmp_path
        / "shadow_derived"
        / sequence_manifest["shards"][0]["path"]
    )
    sequence_shard.write_bytes(sequence_shard.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="shard hash is invalid"):
        _finalize_shadow_outputs(
            tmp_path,
            shadow_stream_state=stream,
            source_stream_manifest="shadow_outcome_shards.manifest.json",
            maximum_rows=1,
            include_details=False,
        )
    assert not (tmp_path / ".shadow_derived.tmp").exists()


def test_shadow_neutral_candidate_with_two_playbooks_remains_unbound(
    tmp_path: Path,
) -> None:
    row = _shadow_raw_challenge_row(
        Playbook.DISPLACEMENT_FIRST_PULLBACK.value,
        candidate_id="neutral-candidate",
        root_id=None,
    )
    diagnostics = json.loads(str(row["playbook_outcomes"]))
    lsr = dict(diagnostics[0])
    lsr.update(
        {
            "playbook": Playbook.LIQUIDITY_SWEEP_REVERSAL.value,
            "liquidity_route_id": "route:lsr",
        }
    )
    row["playbook_outcomes"] = json.dumps([diagnostics[0], lsr])
    stream = _write_shadow_test_stream(tmp_path, [row])

    outputs = _finalize_shadow_outputs(
        tmp_path,
        shadow_stream_state=stream,
        source_stream_manifest="shadow_outcome_shards.manifest.json",
        maximum_rows=1,
        include_details=False,
    )

    assert outputs["raw_challenge_row_count"] == 2
    assert outputs["unbound_challenge_row_count"] == 2
    assert outputs["representative_root_challenge_count"] == 0
    assert outputs["independent_root_episode_count"] == 0
    assert not (tmp_path / ".shadow_derived.tmp").exists()


def test_shadow_root_selection_rejects_one_candidate_with_mixed_roots(
    tmp_path: Path,
) -> None:
    row = _shadow_raw_challenge_row(
        Playbook.DISPLACEMENT_FIRST_PULLBACK.value,
        candidate_id="mixed-root-candidate",
        root_id="root-a",
    )
    diagnostics = json.loads(str(row["playbook_outcomes"]))
    lsr = dict(diagnostics[0])
    lsr.update(
        {
            "playbook": Playbook.LIQUIDITY_SWEEP_REVERSAL.value,
            "market_thesis_id": "thesis-b",
            "market_thesis_root_id": "root-b",
            "liquidity_route_id": "route:lsr",
        }
    )
    row["playbook_outcomes"] = json.dumps([diagnostics[0], lsr])
    stream = _write_shadow_test_stream(tmp_path, [row])

    with pytest.raises(
        ValueError,
        match="Shadow root selection batch mixes root revisions",
    ):
        _finalize_shadow_outputs(
            tmp_path,
            shadow_stream_state=stream,
            source_stream_manifest="shadow_outcome_shards.manifest.json",
            maximum_rows=1,
            include_details=False,
        )
    assert not (tmp_path / ".shadow_derived.tmp").exists()
    assert not (tmp_path / "shadow_derived_manifest.json").exists()


def test_shadow_v5_counts_raw_representative_and_unbound_separately(
    tmp_path: Path,
) -> None:
    playbooks = (
        Playbook.DISPLACEMENT_FIRST_PULLBACK.value,
        Playbook.LIQUIDITY_SWEEP_REVERSAL.value,
    )
    first = pd.Timestamp("2024-01-02T10:00:00-05:00")
    rows = [
        _shadow_raw_challenge_row(
            playbook,
            candidate_id=candidate_id,
            observed=first + pd.Timedelta(minutes=offset),
            geometry_complete=geometry_complete,
        )
        for candidate_id, offset, geometry_complete in (
            ("revision-incomplete", 0, False),
            ("revision-selected", 1, True),
            ("revision-later", 2, True),
        )
        for playbook in playbooks
    ]
    rows.extend(
        _shadow_raw_challenge_row(
            playbook,
            candidate_id="unbound-candidate",
            observed=first + pd.Timedelta(minutes=3),
            root_id=None,
            accepted=True,
        )
        for playbook in playbooks
    )

    summaries: list[dict[str, object]] = []
    for include_details in (False, True):
        destination = tmp_path / ("details" if include_details else "compact")
        stream = _write_shadow_test_stream(destination, rows)
        outputs = _finalize_shadow_outputs(
            destination,
            shadow_stream_state=stream,
            source_stream_manifest="shadow_outcome_shards.manifest.json",
            maximum_rows=1,
            include_details=include_details,
        )
        assert outputs["raw_challenge_row_count"] == 8
        assert outputs["representative_root_challenge_count"] == 2
        assert outputs["unbound_challenge_row_count"] == 2
        assert (
            outputs["raw_challenge_row_count"]
            - outputs["unbound_challenge_row_count"]
            == 6
        )
        assert outputs["independent_root_episode_count"] == 1
        assert outputs["root_sequence_unit_count"] == 2
        assert outputs["mechanism_challenge_four_quadrants"] == {
            playbook: {"rejected_path_valid": 1}
            for playbook in playbooks
        }
        assert outputs["representative_playbook_acceptance"] == {
            playbook: {
                "accepted": 0,
                "rejected": 1,
                "total": 1,
                "acceptance_rate": 0.0,
            }
            for playbook in playbooks
        }
        manifest = json.loads(
            (destination / "shadow_derived_manifest.json").read_text()
        )
        assert manifest["schema_version"] == SHADOW_DERIVED_SCHEMA_VERSION
        root_manifest = json.loads(
            (
                destination / outputs["root_episode_challenges"]
            ).read_text()
        )
        root_row = pd.read_parquet(
            destination
            / "shadow_derived"
            / root_manifest["shards"][0]["path"]
        ).iloc[0]
        assert root_row["candidate_id"] == "revision-selected"
        sequence_manifest = json.loads(
            (
                destination / outputs["root_episode_sequences"]
            ).read_text()
        )
        assert sequence_manifest["rows"] == 2
        assert not {
            "target_price",
            "invalidation_price",
            "mfe_R",
            "mae_R",
            "path_valid",
            "outcome_evaluable",
            "resolved_at",
        } & set(sequence_manifest["field_types"])
        sequence_rows = pd.concat(
            [
                pd.read_parquet(
                    destination / "shadow_derived" / shard["path"]
                )
                for shard in sequence_manifest["shards"]
            ],
            ignore_index=True,
        )
        assert set(sequence_rows["lifecycle_terminal_status"]) == {
            "unknown_not_recorded"
        }
        assert not sequence_rows["action_authority"].any()
        assert not tuple(
            (destination / "shadow_derived").glob("*.sqlite3")
        )
        if include_details:
            assert manifest["streams"]["mechanism_challenges"]["rows"] == 8
            motif_manifest = json.loads(
                (
                    destination / outputs["mechanism_motifs"]
                ).read_text()
            )
            motif_row = pd.read_parquet(
                destination
                / "shadow_derived"
                / motif_manifest["shards"][0]["path"]
            ).iloc[0]
            assert motif_row["sample_root_episode_count"] == 1
            assert not bool(motif_row["root_episode_keys_truncated"])
        aggregate_keys = set(outputs) - {
            "manifest",
            "episode_outcomes",
            "root_episode_challenges",
            "root_episode_sequences",
            "mechanism_challenges",
            "mechanism_motifs",
        }
        summaries.append({key: outputs[key] for key in aggregate_keys})

    assert summaries[0] == summaries[1]


def test_shadow_v5_selects_one_revision_per_root_and_playbook(
    tmp_path: Path,
) -> None:
    playbooks = (
        Playbook.DISPLACEMENT_FIRST_PULLBACK.value,
        Playbook.LIQUIDITY_SWEEP_REVERSAL.value,
    )
    first = pd.Timestamp("2024-01-02T10:00:00-05:00")
    rows = [
        _shadow_raw_challenge_row(
            playbook,
            candidate_id=candidate_id,
            observed=first + pd.Timedelta(minutes=offset),
            geometry_complete=geometry_complete,
        )
        for candidate_id, offset, geometry_complete in (
            ("revision-incomplete", 0, False),
            ("revision-selected", 1, True),
            ("revision-later", 2, True),
        )
        for playbook in playbooks
    ]
    signature_by_candidate = {
        "revision-incomplete": ["market_root"],
        "revision-selected": ["market_root", "location"],
        "revision-later": ["market_root", "location", "trigger"],
    }
    for row in rows:
        diagnostic = json.loads(str(row["playbook_outcomes"]))[0]
        diagnostic["event_order_signature"] = signature_by_candidate[
            str(row["candidate_id"])
        ]
        row["playbook_outcomes"] = json.dumps([diagnostic])
    stream = _write_shadow_test_stream(tmp_path, rows)

    outputs = _finalize_shadow_outputs(
        tmp_path,
        shadow_stream_state=stream,
        source_stream_manifest="shadow_outcome_shards.manifest.json",
        maximum_rows=1,
        include_details=False,
    )

    assert outputs["raw_challenge_row_count"] == 6
    assert outputs["representative_root_challenge_count"] == 2
    assert outputs["unbound_challenge_row_count"] == 0
    assert outputs["independent_root_episode_count"] == 1
    root_manifest = json.loads(
        (tmp_path / outputs["root_episode_challenges"]).read_text()
    )
    representative = pd.read_parquet(
        tmp_path / "shadow_derived" / root_manifest["shards"][0]["path"]
    ).iloc[0]
    assert representative["candidate_id"] == "revision-selected"
    sequence_manifest = json.loads(
        (tmp_path / outputs["root_episode_sequences"]).read_text()
    )
    sequence_rows = pd.concat(
        [
            pd.read_parquet(
                tmp_path / "shadow_derived" / shard["path"]
            )
            for shard in sequence_manifest["shards"]
        ],
        ignore_index=True,
    )
    assert set(sequence_rows["selected_candidate_id"]) == {
        "revision-later"
    }
    assert set(sequence_rows["event_order_length"]) == {3}
    assert all(
        json.loads(value) == [
            ["market_root", "location"],
            ["location", "trigger"],
        ]
        for value in sequence_rows["event_bigrams"]
    )


def test_shadow_v5_finalizer_bounds_motif_roots_across_shards_and_batches(
    tmp_path: Path,
) -> None:
    playbooks = (
        Playbook.DISPLACEMENT_FIRST_PULLBACK.value,
        Playbook.LIQUIDITY_SWEEP_REVERSAL.value,
    )
    root_count = SHADOW_MOTIF_ROOT_SAMPLE_LIMIT + 5
    observed = pd.Timestamp("2024-01-02T10:00:00-05:00")
    rows = [
        _shadow_raw_challenge_row(
            playbook,
            candidate_id=f"candidate-{index:03d}",
            root_id=f"root-{index:03d}",
            observed=observed + pd.Timedelta(index, unit="min"),
        )
        for index in range(root_count)
        for playbook in playbooks
    ]
    split_at = len(rows) // 2
    input_shards = (rows[:split_at], rows[split_at:])
    committed_shards: list[dict[str, object]] = []
    for index, shard_rows in enumerate(input_shards):
        path = (
            tmp_path
            / "shadow_outcome_shards"
            / f"part-{index:05d}.parquet"
        )
        atomic_parquet(
            pd.DataFrame(
                shard_rows,
                columns=list(SHADOW_OUTCOME_FIELD_TYPES),
            ),
            path,
            field_types=SHADOW_OUTCOME_FIELD_TYPES,
        )
        committed_shards.append(
            {
                "index": index,
                "path": str(path.relative_to(tmp_path)),
                "rows": len(shard_rows),
                "first_key": str(shard_rows[0]["candidate_id"]),
                "last_key": str(shard_rows[-1]["candidate_id"]),
                "sha256": sha256_file(path),
            }
        )
    stream = {
        "rows": len(rows),
        "next_shard_index": len(committed_shards),
        "committed_shards": committed_shards,
    }

    outputs = _finalize_shadow_outputs(
        tmp_path,
        shadow_stream_state=stream,
        source_stream_manifest="shadow_outcome_shards.manifest.json",
        maximum_rows=7,
        include_details=True,
    )

    assert len(committed_shards) == 2
    assert outputs["independent_root_episode_count"] == root_count
    root_manifest = json.loads(
        (tmp_path / outputs["root_episode_challenges"]).read_text()
    )
    assert root_manifest["rows"] == root_count
    motif_manifest = json.loads(
        (tmp_path / outputs["mechanism_motifs"]).read_text()
    )
    motif_rows = pd.concat(
        [
            pd.read_parquet(
                tmp_path / "shadow_derived" / shard["path"]
            )
            for shard in motif_manifest["shards"]
        ],
        ignore_index=True,
    )
    assert len(motif_rows) == 1
    motif = motif_rows.iloc[0]
    sampled_roots = json.loads(motif["sample_root_episode_keys"])
    assert int(motif["episode_count"]) == root_count
    assert int(motif["sample_root_episode_count"]) == (
        SHADOW_MOTIF_ROOT_SAMPLE_LIMIT
    )
    assert len(sampled_roots) == SHADOW_MOTIF_ROOT_SAMPLE_LIMIT
    assert bool(motif["root_episode_keys_truncated"])
    assert not (tmp_path / ".shadow_derived.tmp").exists()
    assert not tuple(tmp_path.rglob("*.sqlite3"))


def test_shadow_v5_derived_manifest_is_not_reused_as_v6(tmp_path: Path) -> None:
    manifest_path = tmp_path / "shadow_derived_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "status": "complete",
                "source_stream_manifest": (
                    "shadow_outcome_shards.manifest.json"
                ),
                "details_enabled": False,
                "outputs": {},
            }
        ),
        encoding="utf-8",
    )
    original_manifest = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="manifest conflicts"):
        _finalize_shadow_outputs(
            tmp_path,
            shadow_stream_state={"committed_shards": ()},
            source_stream_manifest="shadow_outcome_shards.manifest.json",
            maximum_rows=1,
            include_details=False,
        )
    assert manifest_path.read_bytes() == original_manifest
    assert not (tmp_path / ".shadow_derived.tmp").exists()
    assert not (tmp_path / "shadow_derived").exists()


def test_shadow_sqlite_rejects_duplicate_episode_across_batches(
    tmp_path: Path,
) -> None:
    stream = _write_shadow_test_stream(
        tmp_path,
        [
            _shadow_raw_executable_row("executable-a"),
            _shadow_raw_executable_row("executable-b"),
        ],
    )

    with pytest.raises(
        ValueError,
        match="duplicate first-executable outcome across Shadow shards",
    ):
        _finalize_shadow_outputs(
            tmp_path,
            shadow_stream_state=stream,
            source_stream_manifest="shadow_outcome_shards.manifest.json",
            maximum_rows=1,
            include_details=False,
        )
    assert not (tmp_path / ".shadow_derived.tmp").exists()


def test_default_replay_is_lightweight_resumable_and_deterministic(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    resumed_output = tmp_path / "resumed"
    interrupted = _run(_command(source, resumed_output, stop_after=11))
    assert interrupted.returncode != 0

    progress = json.loads((resumed_output / "progress.json").read_text())
    assert progress["status"] == "failed"
    assert progress["resume_supported"] is True
    assert 0.0 < progress["completed_percent"] < 100.0

    checkpoint_manifest = json.loads(
        (resumed_output / "_checkpoint/manifest.json").read_text()
    )
    checkpoint = pickle.loads(
        (
            resumed_output / "_checkpoint" / checkpoint_manifest["state_file"]
        ).read_bytes()
    )
    assert set(checkpoint["buffers"]) == {"decision_shards"}
    assert len(checkpoint["buffers"]["decision_shards"]) < 7
    assert checkpoint["brain_diagnostics_enabled"] is False
    assert "causal_cases" not in checkpoint
    assert "causal_case_input_rows" not in checkpoint
    assert "causal_case_outcome_rows" not in checkpoint
    empty_action_policy = {
        "schema_version": 2,
        "action_pipeline_mode": "legacy_decision_risk_compat",
        "scope": "new_entry_action_candidates_only",
        "disabled_new_entry_playbooks": [],
        "decision_belief_projection": "action_filtered",
        "engine_snapshot_belief_projection": "raw",
        "position_management_projection": "raw",
        "trade_intent_projection": "disabled_in_legacy_compat",
    }
    assert checkpoint["runtime_action_policy_identity"] == empty_action_policy
    assert checkpoint["replay"].engine.runtime_action_policy_identity == (
        empty_action_policy
    )
    for diagnostic_key in (
        "natural_funnel_detail_mode",
        "natural_funnel_compact_state",
        "natural_episode_funnel_records",
        "natural_candidate_root_records",
        "open_thesis_binding_records",
        "unexplained_episode_summaries",
        "unexplained_episode_aggregate_counts",
    ):
        assert diagnostic_key not in checkpoint

    manifest_path = resumed_output / "run_manifest.json"
    current_manifest_bytes = manifest_path.read_bytes()
    old_runtime_manifest = json.loads(current_manifest_bytes)
    old_runtime_manifest["brain_runtime_identity"][
        "runtime_state_schema_version"
    ] = 14
    manifest_path.write_text(
        json.dumps(old_runtime_manifest),
        encoding="utf-8",
    )
    incompatible_runtime = _run(
        _command(source, resumed_output, resume=True)
    )
    assert incompatible_runtime.returncode != 0
    assert "run manifest differs" in incompatible_runtime.stderr
    manifest_path.write_bytes(current_manifest_bytes)

    legacy_manifest = json.loads(current_manifest_bytes)
    legacy_manifest.pop("brain_runtime_identity")
    manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
    incompatible = _run(_command(source, resumed_output, resume=True))
    assert incompatible.returncode != 0
    assert "run manifest differs" in incompatible.stderr
    manifest_path.write_bytes(current_manifest_bytes)

    incompatible_action_policy = _run(
        _command(
            source,
            resumed_output,
            resume=True,
            action_disabled_playbooks=(
                Playbook.LIQUIDITY_SWEEP_REVERSAL.value,
            ),
        )
    )
    assert incompatible_action_policy.returncode != 0
    assert "run manifest differs" in incompatible_action_policy.stderr

    resumed = _run(_command(source, resumed_output, resume=True))
    assert resumed.returncode == 0, resumed.stderr

    uninterrupted_output = tmp_path / "uninterrupted"
    uninterrupted = _run(_command(source, uninterrupted_output))
    assert uninterrupted.returncode == 0, uninterrupted.stderr

    resumed_summary = json.loads((resumed_output / "summary.json").read_text())
    uninterrupted_summary = json.loads(
        (uninterrupted_output / "summary.json").read_text()
    )
    assert resumed_summary["decision_rows"] == uninterrupted_summary["decision_rows"]
    assert "lsr_episode_phase_funnel" not in resumed_summary
    assert resumed_summary["brain_diagnostics"] is False
    assert resumed_summary["natural_episode_funnel"] is None
    assert resumed_summary["natural_funnel_details"] is None
    assert not (resumed_output / "natural_funnel_diagnostics").exists()
    assert resumed_summary["open_thesis_binding_funnel"] is None
    assert resumed_summary["unexplained_episode_aggregate"] is None
    pd.testing.assert_frame_equal(
        _decision_rows(resumed_output),
        _decision_rows(uninterrupted_output),
    )
    assert resumed_summary["resume_count"] == 1
    assert max(resumed_summary["peak_buffer_rows"].values()) <= 7
    decision_rows = _decision_rows(resumed_output)
    for field in (
        "global_market_mode",
        "global_authority_timeframe",
        "global_authority_direction",
        "global_authority_source_ids",
        "global_dislocated",
        "global_scale_relations",
        "global_path_blocker_count",
        "global_nearest_path_blocker_id",
        "global_key_path_blocker_ids",
        "global_material_conflict_count",
        "global_key_material_conflict_ids",
        "global_unexplained_episode_count",
        "top_context_metadata",
        "top_competing_episode_ids",
        "market_thesis_id",
        "bound_market_thesis_id",
        "market_thesis_root_id",
        "market_thesis_mechanism",
        "market_thesis_authority_relation",
        "playbook_match_strength",
        "market_thesis_binding_required",
        "market_thesis_action_bound",
        "market_thesis_match_status",
        "playbook_first_failed_hard_gate_id",
        "playbook_plan_delivery_valid",
    ):
        assert field in decision_rows

    run_manifest = json.loads((resumed_output / "run_manifest.json").read_text())
    assert len(run_manifest["source"]["sha256"]) == 64
    assert run_manifest["model_config"]["identity"] == hashlib.sha256(
        (ROOT / "configs/model.json").read_bytes()
    ).hexdigest()
    assert run_manifest["execution"]["mbo_source_sha256"] is None
    runtime_identity = run_manifest["brain_runtime_identity"]
    assert runtime_identity["runtime_state_schema_version"] == (
        BRAIN_RUNTIME_STATE_SCHEMA_VERSION
    )
    assert runtime_identity["runtime_state_schema_version"] == 15
    assert len(runtime_identity["registry_fingerprint"]) == 64
    assert runtime_identity["registry_schema_version"] == 1
    assert set(runtime_identity["playbook_schema_versions"]) == {
        playbook.value for playbook in Playbook
    }
    assert runtime_identity["runtime_action_policy"] == empty_action_policy
    assert run_manifest["runtime_action_policy_identity"] == empty_action_policy
    assert resumed_summary["runtime_action_policy_identity"] == (
        empty_action_policy
    )
    assert run_manifest["brain_calibration_identity"] is None
    assert run_manifest["brain_calibration_fit_admission"] is None
    assert run_manifest["output"]["brain_diagnostics"] is False
    assert "causal_case_identity" not in run_manifest
    assert "market_case_input_identity" not in run_manifest
    assert "repository" not in run_manifest
    assert "last_completed_asof" not in run_manifest["source"]
    assert {"symbol", "instrument_id"}.isdisjoint(run_manifest["source"])
    assert "causal_case_library" not in run_manifest["output"]
    completed_marker = json.loads((resumed_output / "COMPLETED.json").read_text())
    assert completed_marker["status"] == "complete"
    assert "causal_case_library" not in completed_marker
    assert all("sha" not in key for key in completed_marker)
    retired_checkpoint = json.loads(
        (resumed_output / "_checkpoint/manifest.json").read_text()
    )
    assert retired_checkpoint["status"] == "complete"
    assert retired_checkpoint["resume_supported"] is False
    assert "state_file" not in retired_checkpoint
    assert not tuple((resumed_output / "_checkpoint").glob("state-*.pkl"))

    assert (resumed_output / "decision_shards.manifest.json").is_file()
    for retired in (
        "funnel_transition_shards.manifest.json",
        "path_test_shards.manifest.json",
        "decision_trace_shards.manifest.json",
        "ai_primitive_proposals.json",
    ):
        assert not (resumed_output / retired).exists()
    assert not (resumed_output / "visualizations").exists()
    assert not (resumed_output / "unexplained_episodes.json").exists()


def test_brain_diagnostics_are_checkpointed_resumable_and_explicit(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    resumed_output = tmp_path / "resumed-diagnostics"
    interrupted = _run(
        _command(
            source,
            resumed_output,
            brain_diagnostics=True,
            stop_after=11,
        )
    )
    assert interrupted.returncode != 0

    checkpoint_manifest = json.loads(
        (resumed_output / "_checkpoint/manifest.json").read_text()
    )
    checkpoint = pickle.loads(
        (
            resumed_output
            / "_checkpoint"
            / checkpoint_manifest["state_file"]
        ).read_bytes()
    )
    assert checkpoint["brain_diagnostics_enabled"] is True
    assert checkpoint["natural_funnel_diagnostic_schema_version"] == (
        NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION
    )
    assert checkpoint["natural_funnel_detail_mode"] is False
    assert "natural_funnel_compact_state" in checkpoint
    assert "open_thesis_binding_records" in checkpoint
    assert "unexplained_episode_summaries" in checkpoint

    manifest_path = resumed_output / "run_manifest.json"
    current_manifest_bytes = manifest_path.read_bytes()
    run_manifest = json.loads(current_manifest_bytes)
    assert run_manifest["brain_diagnostics_identity"] == {
        "natural_funnel_schema_version": (
            NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION
        )
    }
    run_manifest["brain_diagnostics_identity"][
        "natural_funnel_schema_version"
    ] = NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION - 1
    manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    incompatible = _run(
        _command(
            source,
            resumed_output,
            resume=True,
            brain_diagnostics=True,
        )
    )
    assert incompatible.returncode != 0
    assert "run manifest differs" in incompatible.stderr
    manifest_path.write_bytes(current_manifest_bytes)

    changed_mode = _run(
        _command(source, resumed_output, resume=True)
    )
    assert changed_mode.returncode != 0
    assert "run manifest differs" in changed_mode.stderr

    resumed = _run(
        _command(
            source,
            resumed_output,
            resume=True,
            brain_diagnostics=True,
        )
    )
    assert resumed.returncode == 0, resumed.stderr

    uninterrupted_output = tmp_path / "uninterrupted-diagnostics"
    uninterrupted = _run(
        _command(
            source,
            uninterrupted_output,
            brain_diagnostics=True,
        )
    )
    assert uninterrupted.returncode == 0, uninterrupted.stderr

    resumed_summary = json.loads((resumed_output / "summary.json").read_text())
    uninterrupted_summary = json.loads(
        (uninterrupted_output / "summary.json").read_text()
    )
    assert resumed_summary["brain_diagnostics"] is True
    for field in (
        "natural_episode_funnel",
        "open_thesis_binding_funnel",
        "unexplained_episode_aggregate",
    ):
        assert resumed_summary[field] == uninterrupted_summary[field]
        assert resumed_summary[field] is not None
    assert (
        json.loads((resumed_output / "run_manifest.json").read_text())[
            "output"
        ]["brain_diagnostics"]
        is True
    )


def test_natural_funnel_details_require_explicit_diagnostic_output(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    rejected = _run(
        _command(
            source,
            tmp_path / "natural-funnel-without-diagnostics",
            natural_funnel_details=True,
        )
    )
    assert rejected.returncode != 0
    assert "diagnostic detail output requires" in rejected.stderr
    unexplained_rejected = _run(
        _command(
            source,
            tmp_path / "unexplained-without-diagnostics",
            unexplained_details=True,
        )
    )
    assert unexplained_rejected.returncode != 0
    assert "diagnostic detail output requires" in unexplained_rejected.stderr

    output = tmp_path / "natural-funnel-diagnostics"
    completed = _run(
        _command(
            source,
            output,
            brain_diagnostics=True,
            natural_funnel_details=True,
        )
    )
    assert completed.returncode == 0, completed.stderr

    manifest_path = output / "natural_funnel_diagnostics/manifest.json"
    assert manifest_path.is_file()
    diagnostic = json.loads(manifest_path.read_text())
    assert diagnostic["format_version"] == (
        NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION
    )
    assert diagnostic["schema_fingerprint"] == (
        NATURAL_FUNNEL_SCHEMA_FINGERPRINT
    )
    assert all(len(shard["sha256"]) == 64 for shard in diagnostic["shards"])
    assert all(
        shard["first_key"] <= shard["last_key"]
        for shard in diagnostic["shards"]
    )
    assert all(
        sha256_file(output / shard["path"]) == shard["sha256"]
        for shard in diagnostic["shards"]
    )
    assert diagnostic["rows"] == (
        diagnostic["candidate_root_rows"] + diagnostic["episode_rows"]
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["natural_funnel_details"] == (
        "natural_funnel_diagnostics/manifest.json"
    )
    assert "episodes" not in summary["natural_episode_funnel"]
    assert "candidate_root_rows" not in summary["natural_episode_funnel"]
    assert summary["natural_episode_funnel"][
        "diagnostic_schema_version"
    ] == NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    assert run_manifest["output"]["brain_diagnostics"] is True
    assert run_manifest["output"]["natural_funnel_details"] is True
    assert run_manifest["brain_diagnostics_identity"] == {
        "natural_funnel_schema_version": (
            NATURAL_FUNNEL_DIAGNOSTIC_SCHEMA_VERSION
        )
    }


def test_default_compact_natural_funnel_matches_diagnostic_aggregates(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    compact_output = tmp_path / "compact-natural-funnel"
    detailed_output = tmp_path / "detailed-natural-funnel"
    compact = _run(
        _command(source, compact_output, brain_diagnostics=True)
    )
    detailed = _run(
        _command(
            source,
            detailed_output,
            brain_diagnostics=True,
            natural_funnel_details=True,
        )
    )
    assert compact.returncode == 0, compact.stderr
    assert detailed.returncode == 0, detailed.stderr

    compact_summary = json.loads(
        (compact_output / "summary.json").read_text()
    )["natural_episode_funnel"]
    detailed_summary = json.loads(
        (detailed_output / "summary.json").read_text()
    )["natural_episode_funnel"]
    for field in (
        "diagnostic_schema_version",
        "counting_basis",
        "denominators",
        "candidate_root_counts",
        "stage_order",
        "strata",
        "episodes_observed",
        "rows",
        "terminal_disposition_counts",
    ):
        assert compact_summary[field] == detailed_summary[field]
    assert len(compact_summary["compact_case_index"]["cases"]) <= 40


def test_visualization_clocks_are_bounded_causal_and_unique() -> None:
    start = pd.Timestamp("2022-06-06T18:00:00-04:00")
    end = pd.Timestamp("2022-06-06T19:00:00-04:00")
    clocks = _visualization_clocks(
        ["2022-06-06T18:30:00-04:00"],
        start=start,
        end=end,
    )
    assert clocks == (pd.Timestamp("2022-06-06T22:30:00Z"),)

    with pytest.raises(ValueError, match="timezone-aware"):
        _visualization_clocks(
            ["2022-06-06T18:30:00"],
            start=start,
            end=end,
        )
    with pytest.raises(ValueError, match="inside the replay interval"):
        _visualization_clocks(
            ["2022-06-06T19:00:00-04:00"],
            start=start,
            end=end,
        )
    with pytest.raises(ValueError, match="unique"):
        _visualization_clocks(
            [
                "2022-06-06T18:30:00-04:00",
                "2022-06-06T22:30:00Z",
            ],
            start=start,
            end=end,
        )


def test_explicit_decision_clock_renders_one_causal_visual(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path, periods=241)
    output = tmp_path / "visualized"
    decision_clock = "2022-06-06T22:01:00-04:00"
    completed = _run(
        _command(
            source,
            output,
            end="2022-06-06T22:02:00-04:00",
            visualize_at=(decision_clock,),
        )
    )
    assert completed.returncode == 0, completed.stderr

    image = output / "visualizations/20220607T020100Z.png"
    index = output / "visualizations/index.html"
    assert image.is_file()
    assert image.stat().st_size > 0
    assert index.is_file()
    assert image.name in index.read_text(encoding="utf-8")

    summary = json.loads((output / "summary.json").read_text())
    assert summary["visualization_capture"] is True
    assert summary["visualization_decisions"] == 1
    assert summary["visualization_index"] == "visualizations/index.html"

    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["output"]["visualization"] == {
        "enabled": True,
        "decision_clocks_utc": ["2022-06-07T02:01:00+00:00"],
        "directory": "visualizations",
        "selection": "explicit_decision_clocks",
    }
    marker = json.loads((output / "COMPLETED.json").read_text())
    assert marker["visualizations_index"] == "visualizations/index.html"


def test_brain_calibration_is_the_only_optional_default_stream(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    output = tmp_path / "calibration"
    completed = _run(_command(source, output, brain_calibration=True))
    assert completed.returncode == 0, completed.stderr
    assert (output / "decision_shards.manifest.json").is_file()
    assert (output / "brain_calibration_shards.manifest.json").is_file()
    state = json.loads((output / "summary.json").read_text())
    assert state["brain_diagnostics"] is False
    assert state["natural_episode_funnel"] is None
    assert state["open_thesis_binding_funnel"] is None
    assert state["unexplained_episode_aggregate"] is None
    assert set(state["peak_buffer_rows"]) == {
        "decision_shards",
        "brain_calibration_shards",
    }
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["output"]["brain_diagnostics"] is False
    identity = manifest["brain_calibration_identity"]
    assert identity["recorder_schema_version"] == RECORDER_SCHEMA_VERSION
    assert len(identity["registry_fingerprint"]) == 64
    assert identity["registry_schema_version"] == 1
    assert set(identity["playbook_schema_versions"]) == {
        Playbook.DISPLACEMENT_FIRST_PULLBACK.value,
        Playbook.LIQUIDITY_SWEEP_REVERSAL.value,
    }
    assert manifest["brain_calibration_fit_admission"] == {
        "minimum_dimension_units": 200,
        "minimum_plan_valid_roots": 30,
        "minimum_executable_episodes": 20,
    }


def test_calibration_only_omits_decision_shards_and_retires_checkpoint(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    invalid = _run(
        _command(
            source,
            tmp_path / "invalid-calibration-only",
            calibration_only=True,
        )
    )
    assert invalid.returncode != 0
    assert "--calibration-only requires --brain-calibration" in invalid.stderr

    output = tmp_path / "calibration-only"
    interrupted = _run(
        _command(
            source,
            output,
            brain_calibration=True,
            brain_diagnostics=True,
            calibration_only=True,
            unexplained_details=True,
            stop_after=11,
        )
    )
    assert interrupted.returncode != 0
    checkpoint_manifest = json.loads(
        (output / "_checkpoint/manifest.json").read_text()
    )
    checkpoint_state = pickle.loads(
        (output / "_checkpoint" / checkpoint_manifest["state_file"]).read_bytes()
    )
    assert set(checkpoint_state["buffers"]) == {
        "brain_calibration_shards"
    }
    assert checkpoint_state["decision_rows"] > 0

    completed = _run(
        _command(
            source,
            output,
            resume=True,
            brain_calibration=True,
            brain_diagnostics=True,
            calibration_only=True,
            unexplained_details=True,
        )
    )
    assert completed.returncode == 0, completed.stderr
    assert not (output / "decision_shards.manifest.json").exists()
    assert (output / "brain_calibration_shards.manifest.json").is_file()
    assert (output / "unexplained_episodes.json").is_file()

    summary = json.loads((output / "summary.json").read_text())
    assert summary["decision_rows"] > 0
    assert summary["output_contract"] == "calibration_rows_and_aggregates"
    assert set(summary["stream_rows"]) == {"brain_calibration_shards"}
    assert set(summary["peak_buffer_rows"]) == {"brain_calibration_shards"}
    assert "unexplained_episode_summaries" not in summary
    assert summary["unexplained_episode_details"] == "unexplained_episodes.json"

    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["output"]["calibration_only"] is True
    assert manifest["output"]["decision_shards"] is False
    assert manifest["output"]["stream_families"] == [
        "brain_calibration_shards"
    ]
    checkpoint = json.loads(
        (output / "_checkpoint/manifest.json").read_text()
    )
    assert checkpoint["status"] == "complete"
    assert checkpoint["resume_supported"] is False
    assert "state_file" not in checkpoint
    assert not tuple((output / "_checkpoint").glob("state-*.pkl"))


def test_unexplained_episode_details_use_a_stable_nonempty_envelope() -> None:
    episodes = [
        {
            "episode_root_id": "root:one",
            "first_observed_at": "2022-06-06T18:01:00-04:00",
            "source_kind": "manipulation",
            "timeframe": "5m",
            "direction": "long",
            "nearest_playbook": "liquidity_sweep_reversal",
            "unexplained_reason": "no_reverse_displacement",
            "resolution": "window_right_censored",
        },
        {
            "episode_root_id": "root:two",
            "first_observed_at": "2022-06-06T18:02:00-04:00",
            "source_kind": "displacement",
            "timeframe": "5m",
            "direction": "short",
            "nearest_playbook": "displacement_first_pullback",
            "unexplained_reason": "no_frozen_entry_zone",
            "resolution": "explained_or_root_resolved",
        },
    ]

    payload = _unexplained_episode_details_payload(episodes, maximum=1)

    assert payload == {
        "schema_version": 1,
        "counting_basis": (
            "bounded_deterministic_stratified_sample_of_unexplained_episodes"
        ),
        "population_count": 2,
        "sample_count": 1,
        "maximum_sample_count": 1,
        "episodes": [episodes[1]],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert json.loads(encoded) == payload


def test_unexplained_details_finalizer_failure_exact_resume_keeps_shards(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    output = tmp_path / "unexplained-finalizer-resume"
    common = {
        "brain_calibration": True,
        "brain_diagnostics": True,
        "calibration_only": True,
        "unexplained_details": True,
    }
    interrupted = _run(
        _command(source, output, stop_after=11, **common)
    )
    assert interrupted.returncode != 0

    blocker = output / "unexplained_episodes.json"
    blocker.mkdir()
    failed_finalizer = _run(
        _command(source, output, resume=True, **common)
    )
    assert failed_finalizer.returncode != 0
    assert not (output / "COMPLETED.json").exists()

    checkpoint_manifest = json.loads(
        (output / "_checkpoint/manifest.json").read_text()
    )
    checkpoint_state = pickle.loads(
        (
            output
            / "_checkpoint"
            / checkpoint_manifest["state_file"]
        ).read_bytes()
    )
    assert checkpoint_state["finalized"] is True
    processed_bars = int(checkpoint_state["processed_bars"])
    source_rows_consumed = int(checkpoint_state["source_rows_consumed"])
    stream_manifest_path = output / "brain_calibration_shards.manifest.json"
    stream_manifest_before = stream_manifest_path.read_bytes()
    stream_manifest = json.loads(stream_manifest_before)
    shard_hashes_before = {
        item["path"]: sha256_file(output / item["path"])
        for item in stream_manifest["shards"]
    }

    temporary = output / ".unexplained_episodes.json.tmp"
    if temporary.exists():
        temporary.unlink()
    blocker.rmdir()
    resumed = _run(
        _command(source, output, resume=True, **common)
    )

    assert resumed.returncode == 0, resumed.stderr
    assert (output / "COMPLETED.json").is_file()
    assert stream_manifest_path.read_bytes() == stream_manifest_before
    assert {
        item["path"]: sha256_file(output / item["path"])
        for item in stream_manifest["shards"]
    } == shard_hashes_before
    summary = json.loads((output / "summary.json").read_text())
    assert summary["decision_rows"] == processed_bars
    assert summary["resume_count"] == 2
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    assert run_manifest["source"]["rows"] == source_rows_consumed
    details = json.loads((output / "unexplained_episodes.json").read_text())
    assert details["schema_version"] == 1
    assert details["sample_count"] == len(details["episodes"])
    assert details["sample_count"] <= details["maximum_sample_count"]


def test_scene_graph_compaction_is_calibration_only_and_preserves_rows(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path, periods=480)
    invalid = _run(
        _command(
            source,
            tmp_path / "invalid-compaction",
            compact_scene_graph=True,
        )
    )
    assert invalid.returncode != 0
    assert "requires --brain-calibration --calibration-only" in invalid.stderr

    baseline_output = tmp_path / "baseline-calibration"
    compact_output = tmp_path / "compact-calibration"
    common = {
        "brain_calibration": True,
        "calibration_only": True,
        "end": "2022-06-07T02:00:00-04:00",
    }
    baseline = _run(
        _command(source, baseline_output, **common)
    )
    assert baseline.returncode == 0, baseline.stderr
    compact = _run(
        _command(
            source,
            compact_output,
            compact_scene_graph=True,
            **common,
        )
    )
    assert compact.returncode == 0, compact.stderr

    pd.testing.assert_frame_equal(
        _brain_calibration_rows(baseline_output),
        _brain_calibration_rows(compact_output),
    )
    compact_summary = json.loads(
        (compact_output / "summary.json").read_text()
    )
    compaction = compact_summary["scene_graph_compaction"]
    assert compaction["runs"] >= 2
    assert compaction["last_result"]["after"]["nodes"] <= (
        compaction["last_result"]["before"]["nodes"]
    )
    baseline_manifest = json.loads(
        (baseline_output / "run_manifest.json").read_text()
    )
    compact_manifest = json.loads(
        (compact_output / "run_manifest.json").read_text()
    )
    assert baseline_manifest["output"]["scene_graph_compaction"][
        "enabled"
    ] is False
    assert compact_manifest["output"]["scene_graph_compaction"][
        "enabled"
    ] is True


def test_scene_graph_compaction_preserves_each_engine_snapshot_and_recorder(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path, periods=180)
    frame = pd.read_parquet(source)
    baseline = ContinuousSMCEngine.from_config(
        ROOT / "configs/model.json",
        runtime_mode="development",
    )
    compact = ContinuousSMCEngine.from_config(
        ROOT / "configs/model.json",
        runtime_mode="development",
    )
    baseline_recorder = BrainCalibrationRecorder()
    compact_recorder = BrainCalibrationRecorder()
    baseline_rows: list[dict[str, object]] = []
    compact_rows: list[dict[str, object]] = []
    compaction_counts: list[int] = []
    for index, bar in enumerate(iter_completed_bars(frame), start=1):
        execution = ExecutionRealityInput(
            deadline=bar.end + pd.Timedelta(minutes=60),
            source="ohlcv_only_execution_unavailable",
        )
        baseline_recorder.on_bar(bar)
        compact_recorder.on_bar(bar)
        baseline_snapshot = baseline.on_bar(bar, execution=execution)
        compact_snapshot = compact.on_bar(bar, execution=execution)
        assert to_primitive(compact_snapshot) == to_primitive(
            baseline_snapshot
        )
        baseline_recorder.observe(baseline_snapshot, source_bar=bar)
        compact_recorder.observe(compact_snapshot, source_bar=bar)
        baseline_rows.extend(
            to_primitive(value)
            for value in baseline_recorder.drain_rows()
        )
        compact_rows.extend(
            to_primitive(value)
            for value in compact_recorder.drain_rows()
        )
        assert compact_rows == baseline_rows
        if index % 5 == 0:
            result = compact.compact_scene_graph_runtime()
            compaction_counts.append(int(result["after"]["nodes"]))
    assert compaction_counts
    assert compact.scene_graph.revision_id == (
        baseline.scene_graph.revision_id
    )
    assert compact.brain.current is not None
    assert to_primitive(compact.brain.current) == to_primitive(
        baseline.brain.current
    )


def test_scene_graph_compaction_preserves_nonempty_market_recorder_rows() -> None:
    baseline = ContinuousSMCEngine.from_config(
        ROOT / "configs/model.json",
        runtime_mode="development",
    )
    compact = ContinuousSMCEngine.from_config(
        ROOT / "configs/model.json",
        runtime_mode="development",
    )
    bars = []
    for bar in session_bars(1)[:260]:
        open_price = round(bar.open / 0.25) * 0.25
        close = round(bar.close / 0.25) * 0.25
        bars.append(
            replace(
                bar,
                open=open_price,
                high=max(open_price, close) + 0.5,
                low=min(open_price, close) - 0.5,
                close=close,
            )
        )
    selected = None
    for index, bar in enumerate(bars, start=1):
        baseline_snapshot = baseline.on_bar_neutral_input(bar)
        compact_snapshot = compact.on_bar_neutral_input(bar)
        assert to_primitive(compact_snapshot) == to_primitive(
            baseline_snapshot
        )
        if index in {120, 240}:
            compact.compact_scene_graph_runtime()
        if index > 240 and (
            baseline_snapshot.observation.scene_added_edge_ids
            or baseline_snapshot.observation.scene_revised_edge_ids
        ):
            selected = (index, bar, baseline_snapshot, compact_snapshot)
            break
    assert selected is not None
    source_ordinal, source_bar, baseline_snapshot, compact_snapshot = selected
    neutral_state = baseline_snapshot.neutral_market_state
    asof = baseline_snapshot.observation.asof
    episode = MarketEpisodeState(
        episode_id=market_episode_id(
            neutral_state.market_epoch_id,
            "location:compaction-fixture",
            "path:compaction-fixture",
            Direction.LONG,
        ),
        market_epoch_id=neutral_state.market_epoch_id,
        symbol=baseline_snapshot.observation.symbol,
        instrument_id=baseline_snapshot.observation.instrument_id,
        direction=Direction.LONG,
        entry_location_id="location:compaction-fixture",
        entry_path_id="path:compaction-fixture",
        source_zone_id="fvg:compaction-fixture",
        source_displacement_id="displacement:compaction-fixture",
        entry_location_protocol_hash="entry-location-protocol:fixture",
        source_zone_detector_protocol_hash="group3-protocol:fixture",
        source_zone_kind="fvg",
        source_zone_protocol_hash="fvg-protocol:fixture",
        source_bos_id=None,
        lower_bound=19_999.0,
        upper_bound=20_001.0,
        midpoint=20_000.0,
        near_edge=20_001.0,
        far_edge=19_999.0,
        failure_boundary=19_999.0,
        formed_at=asof,
        updated_at=asof,
        binding_status="unbound",
        claims=(),
        active_claim_ids=(),
        claim_status="unbound",
        first_pullback_step_id=None,
        first_pullback_at=None,
        trigger_step_id=None,
        trigger_event_id=None,
        trigger_at=None,
        successful_pulse_at=None,
        successful_pulse_reason=None,
        lifecycle="registered",
        terminal_at=None,
        terminal_reason=None,
    )

    def with_episode(snapshot):
        state = replace(
            snapshot.neutral_market_state,
            market_episodes=(episode,),
            episode_transitions_this_update=(episode,),
        )
        return replace(snapshot, neutral_market_state=state)

    baseline_recorder = MarketEpisodeCaseRecorder(capture_start=bars[0].end)
    compact_recorder = MarketEpisodeCaseRecorder(capture_start=bars[0].end)
    baseline_recorder.observe(
        with_episode(baseline_snapshot),
        source_bar=source_bar,
        source_row_ordinal=source_ordinal - 1,
        replay_update_ordinal=source_ordinal - 1,
        scene_graph=baseline.scene_graph,
    )
    compact_recorder.observe(
        with_episode(compact_snapshot),
        source_bar=source_bar,
        source_row_ordinal=source_ordinal - 1,
        replay_update_ordinal=source_ordinal - 1,
        scene_graph=compact.scene_graph,
    )
    baseline_rows = tuple(
        row.to_dict() for row in baseline_recorder.drain_input_rows()
    )
    compact_rows = tuple(
        row.to_dict() for row in compact_recorder.drain_input_rows()
    )
    assert len(baseline_rows) == 1
    assert "episode_created" in json.loads(
        baseline_rows[0]["transition_kinds_json"]
    )
    scene_delta = json.loads(baseline_rows[0]["scene_graph_delta_json"])
    assert scene_delta["relation_descriptors"]
    assert scene_delta["relation_descriptors_complete"] is True
    assert compact_rows == baseline_rows


def test_hot_path_caches_match_uncached_snapshots_rows_and_pickle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path, periods=180)
    bars = tuple(iter_completed_bars(pd.read_parquet(source)))

    def uncached_kind_nodes(
        graph: TemporalMarketSceneGraph,
        kind: str,
    ) -> tuple[object, ...]:
        return tuple(
            sorted(
                (
                    graph._nodes[node_id]
                    for node_id in graph._current_epoch_node_ids_by_kind.get(
                        kind,
                        (),
                    )
                    if node_id in graph._nodes
                    and graph._nodes[node_id].market_epoch_id
                    == graph._market_epoch_id
                ),
                key=lambda value: (value.observed_at, value.node_id),
            )
        )

    def uncached_context_nodes(
        graph: TemporalMarketSceneGraph,
        kind: str,
    ) -> tuple[object, ...]:
        return tuple(
            node
            for node in uncached_kind_nodes(graph, kind)
            if node.market_epoch_id == graph._market_epoch_id
            and not _is_terminal(node.kind, node.lifecycle)
        )

    def uncached_neighbors(
        graph: TemporalMarketSceneGraph,
        node_id: str,
        *,
        asof: pd.Timestamp | None = None,
    ):
        current_view = (
            asof is None
            or (
                graph._last_asof is not None
                and pd.Timestamp(asof) >= graph._last_asof
            )
        )
        edge_by_id = (
            graph._edges
            if current_view
            else {
                edge.edge_id: edge for edge in graph.edges_asof(asof)
            }
        )
        for edge_id in graph._outgoing.get(node_id, ()):
            edge = edge_by_id.get(edge_id)
            if edge is not None and edge.lifecycle == "active":
                yield edge.target_node_id, edge
        for edge_id in graph._incoming.get(node_id, ()):
            edge = edge_by_id.get(edge_id)
            if edge is not None and edge.lifecycle == "active":
                yield edge.source_node_id, edge
        if current_view:
            for edge in graph._current_path_block_edges.values():
                if edge.source_node_id == node_id:
                    yield edge.target_node_id, edge
                elif edge.target_node_id == node_id:
                    yield edge.source_node_id, edge

    def legacy_temporal_metrics(
        memory: EventMemory,
        asof: pd.Timestamp,
    ) -> tuple[dict[str, int], dict[str, int]]:
        durations: dict[str, int] = {}
        ages: dict[str, int] = {}
        active = {
            event.event_id
            for event in memory._latest_by_entity.values()
        }
        events = {
            event.event_id: event
            for event in memory._events
        }
        for event in events.values():
            if event.event_id in memory._closed_durations:
                durations[event.event_id] = memory._closed_durations[
                    event.event_id
                ]
            elif event.event_id in active:
                durations[event.event_id] = max(
                    0,
                    memory._elapsed_minutes(
                        event.formed_at or event.observed_at,
                        asof,
                    ),
                )
            else:
                durations[event.event_id] = 0
            origin = (
                event.formed_at
                or event.confirmed_at
                or event.observed_at
            )
            ages[event.event_id] = max(
                0,
                memory._elapsed_minutes(origin, asof),
            )
        for timeline in memory._entity_timelines.values():
            for index, event in enumerate(timeline):
                if index + 1 < len(timeline):
                    durations[event.event_id] = max(
                        0,
                        memory._elapsed_minutes(
                            event.observed_at,
                            timeline[index + 1].observed_at,
                        ),
                    )
                elif event.ended_at is not None:
                    durations[event.event_id] = 0
                else:
                    durations[event.event_id] = max(
                        0,
                        memory._elapsed_minutes(event.observed_at, asof),
                    )
                origin = (
                    event.formed_at
                    or event.confirmed_at
                    or event.observed_at
                )
                ages[event.event_id] = max(
                    0,
                    memory._elapsed_minutes(origin, asof),
                )
        return durations, ages

    def run(
        engine: ContinuousSMCEngine,
        recorder: BrainCalibrationRecorder,
        *,
        pickle_at: int | None = None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        snapshots: list[dict[str, object]] = []
        rows: list[dict[str, object]] = []
        for index, bar in enumerate(bars, start=1):
            execution = ExecutionRealityInput(
                deadline=bar.end + pd.Timedelta(minutes=60),
                source="ohlcv_only_execution_unavailable",
            )
            recorder.on_bar(bar)
            snapshot = engine.on_bar(bar, execution=execution)
            recorder.observe(snapshot, source_bar=bar)
            snapshots.append(to_primitive(snapshot))
            rows.extend(
                to_primitive(value) for value in recorder.drain_rows()
            )
            if pickle_at is not None and index == pickle_at:
                engine, recorder = pickle.loads(
                    pickle.dumps((engine, recorder))
                )
        return snapshots, rows

    with monkeypatch.context() as uncached:
        uncached.setattr(
            TemporalMarketSceneGraph,
            "_current_kind_nodes",
            uncached_kind_nodes,
        )
        uncached.setattr(
            TemporalMarketSceneGraph,
            "_current_nonterminal_kind_nodes",
            uncached_context_nodes,
        )
        uncached.setattr(
            TemporalMarketSceneGraph,
            "_neighbors",
            uncached_neighbors,
        )
        uncached.setattr(
            EventMemory,
            "temporal_metrics",
            legacy_temporal_metrics,
        )
        baseline_snapshots, baseline_rows = run(
            ContinuousSMCEngine.from_config(
                ROOT / "configs/model.json",
                runtime_mode="development",
            ),
            BrainCalibrationRecorder(),
        )

    cached_snapshots, cached_rows = run(
        ContinuousSMCEngine.from_config(
            ROOT / "configs/model.json",
            runtime_mode="development",
        ),
        BrainCalibrationRecorder(),
        pickle_at=90,
    )
    assert cached_snapshots == baseline_snapshots
    assert cached_rows == baseline_rows


def test_scene_graph_compaction_checkpoint_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path, periods=480)
    resumed_output = tmp_path / "compact-resumed"
    common = {
        "brain_calibration": True,
        "calibration_only": True,
        "compact_scene_graph": True,
        "end": "2022-06-07T02:00:00-04:00",
    }
    interrupted = _run(
        _command(source, resumed_output, stop_after=31, **common)
    )
    assert interrupted.returncode != 0
    checkpoint_manifest = json.loads(
        (resumed_output / "_checkpoint/manifest.json").read_text()
    )
    checkpoint_state = pickle.loads(
        (
            resumed_output
            / "_checkpoint"
            / checkpoint_manifest["state_file"]
        ).read_bytes()
    )
    checkpoint_compaction = checkpoint_state["scene_graph_compaction"]
    assert checkpoint_compaction["runs"] >= 1
    assert checkpoint_state[
        "replay"
    ].engine.scene_graph.history_retention_floor is not None

    resumed = _run(
        _command(source, resumed_output, resume=True, **common)
    )
    assert resumed.returncode == 0, resumed.stderr
    uninterrupted_output = tmp_path / "compact-uninterrupted"
    uninterrupted = _run(
        _command(source, uninterrupted_output, **common)
    )
    assert uninterrupted.returncode == 0, uninterrupted.stderr
    pd.testing.assert_frame_equal(
        _brain_calibration_rows(resumed_output),
        _brain_calibration_rows(uninterrupted_output),
    )


def test_shadow_outcomes_are_an_independent_checkpointed_diagnostic_stream(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    payload = json.loads(
        (ROOT / "configs/data_splits.json").read_text(encoding="utf-8")
    )
    profile = payload["shadow_diagnostic_profiles"][
        "brain_playbook_reverse_validation_2024_01"
    ]
    profile.update(
        {
            "allowed_ohlcv_role": "calibration",
            "start": "2022-06-06T18:00:00-04:00",
            "end_exclusive": "2022-06-06T18:32:00-04:00",
            "warmup_calendar_days": 0,
        }
    )
    action_policy = tuple(profile["action_disabled_playbooks"])
    protocol = tmp_path / "shadow-data-splits.json"
    protocol.write_text(json.dumps(payload), encoding="utf-8")

    alternate_config = tmp_path / "model.json"
    alternate_config.write_bytes((ROOT / "configs/model.json").read_bytes())
    wrong_config = _run(
        _command(
            source,
            tmp_path / "shadow-wrong-config",
            shadow_outcomes=True,
            validation_protocol=protocol,
            model_config=alternate_config,
            action_disabled_playbooks=action_policy,
        )
    )
    assert wrong_config.returncode != 0
    assert "requires the frozen configs/model.json" in wrong_config.stderr

    output = tmp_path / "shadow"
    interrupted = _run(
        _command(
            source,
            output,
            shadow_outcomes=True,
            validation_protocol=protocol,
            action_disabled_playbooks=action_policy,
            stop_after=11,
        )
    )
    assert interrupted.returncode != 0
    checkpoint_manifest = json.loads(
        (output / "_checkpoint/manifest.json").read_text()
    )
    checkpoint_state = pickle.loads(
        (
            output
            / "_checkpoint"
            / checkpoint_manifest["state_file"]
        ).read_bytes()
    )
    assert isinstance(
        checkpoint_state["shadow_outcomes"],
        ShadowCandidateOutcomeRecorder,
    )
    assert (
        checkpoint_state["replay"]
        .engine.observer.config.typed_transition_delta_transport
        is True
    )
    assert checkpoint_state["runtime_action_policy_identity"][
        "disabled_new_entry_playbooks"
    ] == [Playbook.LIQUIDITY_SWEEP_REVERSAL.value]
    assert checkpoint_state["replay"].engine.action_disabled_playbooks == (
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
    )
    assert "shadow_outcome_shards" in checkpoint_state["buffers"]

    completed = _run(
        _command(
            source,
            output,
            resume=True,
            shadow_outcomes=True,
            validation_protocol=protocol,
            action_disabled_playbooks=action_policy,
        )
    )
    assert completed.returncode == 0, completed.stderr
    assert (output / "decision_shards.manifest.json").is_file()
    assert not (output / "brain_calibration_shards.manifest.json").exists()
    assert (output / "shadow_outcome_shards.manifest.json").is_file()
    assert (output / "shadow_derived/episode_outcomes.manifest.json").is_file()
    assert (
        output / "shadow_derived/root_episode_challenges.manifest.json"
    ).is_file()
    assert (
        output / "shadow_derived/root_episode_sequences.manifest.json"
    ).is_file()
    assert not (output / "shadow_derived/mechanism_challenges.manifest.json").exists()
    assert not (output / "shadow_derived/mechanism_motifs.manifest.json").exists()
    assert (output / "shadow_derived_manifest.json").is_file()
    assert not tuple((output / "_checkpoint").glob("state-*.pkl"))

    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["output"]["shadow_outcomes"] is True
    assert manifest["output"]["brain_diagnostics"] is False
    assert manifest["output"]["brain_calibration"] is False
    action_identity = manifest["runtime_action_policy_identity"]
    assert action_identity["disabled_new_entry_playbooks"] == [
        Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    ]
    assert manifest["brain_runtime_identity"][
        "runtime_action_policy"
    ] == action_identity
    assert manifest["shadow_outcome_identity"]["profile_name"] == (
        "brain_playbook_reverse_validation_2024_01"
    )
    assert {
        key: manifest["shadow_outcome_identity"][key]
        for key in (
            "recorder_schema_version",
            "derived_schema_version",
            "protocol_version",
        )
    } == {
        key: profile[key]
        for key in (
            "recorder_schema_version",
            "derived_schema_version",
            "protocol_version",
        )
    }
    assert set(manifest["shadow_outcome_identity"]) == {
        "profile_name",
        "recorder_schema_version",
        "derived_schema_version",
        "protocol_version",
    }
    assert set(manifest["output"]["stream_families"]) == {
        "decision_shards",
        "shadow_outcome_shards",
    }
    summary = json.loads((output / "summary.json").read_text())
    assert summary["output_contract"] == (
        "lightweight_decision_and_shadow_shards"
    )
    assert summary["shadow_outcomes"]["output_affects_model"] is False
    assert summary["runtime_action_policy_identity"] == action_identity
    assert "protocol_version" not in summary["shadow_outcomes"]
    assert summary["shadow_derived_outputs"]["manifest"] == (
        "shadow_derived_manifest.json"
    )
    assert (
        summary["shadow_outcomes"][
            "typed_delta_missing_observations"
        ]
        == 0
    )
    derived_manifest = json.loads(
        (output / "shadow_derived_manifest.json").read_text()
    )
    assert derived_manifest["schema_version"] == SHADOW_DERIVED_SCHEMA_VERSION
    assert derived_manifest["details_enabled"] is False
    assert derived_manifest["streams"]["episode_outcomes"]["action_authority"] is False
    assert set(derived_manifest["streams"]) == {
        "episode_outcomes",
        "root_episode_challenges",
        "root_episode_sequences",
    }
    assert not tuple(output.rglob("shadow_index.sqlite3"))
    assert set(summary["stream_rows"]) == {
        "decision_shards",
        "shadow_outcome_shards",
    }

    combined_output = tmp_path / "shadow-with-calibration-and-details"
    combined = _run(
        _command(
            source,
            combined_output,
            brain_calibration=True,
            shadow_outcomes=True,
            shadow_details=True,
            validation_protocol=protocol,
            action_disabled_playbooks=action_policy,
        )
    )
    assert combined.returncode == 0, combined.stderr
    combined_manifest = json.loads(
        (combined_output / "run_manifest.json").read_text()
    )
    assert combined_manifest["output"]["brain_calibration"] is True
    assert combined_manifest["output"]["shadow_outcomes"] is True
    assert combined_manifest["output"]["brain_diagnostics"] is False
    assert set(combined_manifest["output"]["stream_families"]) == {
        "decision_shards",
        "brain_calibration_shards",
        "shadow_outcome_shards",
    }
    combined_derived = json.loads(
        (combined_output / "shadow_derived_manifest.json").read_text()
    )
    assert combined_derived["schema_version"] == SHADOW_DERIVED_SCHEMA_VERSION
    assert combined_derived["details_enabled"] is True
    assert set(combined_derived["streams"]) == {
        "episode_outcomes",
        "root_episode_challenges",
        "root_episode_sequences",
        "mechanism_challenges",
        "mechanism_motifs",
    }
    assert not tuple(combined_output.rglob("shadow_index.sqlite3"))


def test_causal_case_library_is_optional_resumable_and_manifest_bound(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    rejected = _run(
        _command(
            source,
            tmp_path / "case-without-shadow",
            causal_case_library=True,
        )
    )
    assert rejected.returncode != 0
    assert "requires --shadow-outcomes" in rejected.stderr

    payload = json.loads(
        (ROOT / "configs/data_splits.json").read_text(encoding="utf-8")
    )
    profile = payload["shadow_diagnostic_profiles"][
        "brain_playbook_reverse_validation_2024_01"
    ]
    profile.update(
        {
            "allowed_ohlcv_role": "calibration",
            "start": "2022-06-06T18:00:00-04:00",
            "end_exclusive": "2022-06-06T18:32:00-04:00",
            "warmup_calendar_days": 0,
        }
    )
    protocol = tmp_path / "case-data-splits.json"
    protocol.write_text(json.dumps(payload), encoding="utf-8")
    common = {
        "shadow_outcomes": True,
        "causal_case_library": True,
        "validation_protocol": protocol,
        "action_disabled_playbooks": tuple(
            profile["action_disabled_playbooks"]
        ),
    }
    resumed_output = tmp_path / "case-resumed"
    interrupted = _run(
        _command(source, resumed_output, stop_after=11, **common)
    )
    assert interrupted.returncode != 0
    checkpoint_manifest = json.loads(
        (resumed_output / "_checkpoint/manifest.json").read_text()
    )
    checkpoint_state = pickle.loads(
        (
            resumed_output
            / "_checkpoint"
            / checkpoint_manifest["state_file"]
        ).read_bytes()
    )
    assert isinstance(checkpoint_state["causal_cases"], CausalCaseRecorder)
    assert checkpoint_state["causal_cases"].recorder_schema_version == (
        CAUSAL_CASE_RECORDER_SCHEMA_VERSION
    )
    assert set(checkpoint_state["buffers"]) == {
        "decision_shards",
        "shadow_outcome_shards",
        "causal_case_input_shards",
        "causal_case_outcome_shards",
    }
    assert all(len(buffer) < 7 for buffer in checkpoint_state["buffers"].values())
    assert checkpoint_state["causal_cases"]._outcome_rows == []

    run_manifest_path = resumed_output / "run_manifest.json"
    current_run_manifest_bytes = run_manifest_path.read_bytes()
    legacy_run_manifest = json.loads(current_run_manifest_bytes)
    legacy_run_manifest["causal_case_identity"].update(
        {
            "recorder_schema_version": CAUSAL_CASE_RECORDER_SCHEMA_VERSION - 1,
            "protocol": {
                **legacy_run_manifest["causal_case_identity"]["protocol"],
                "protocol_version": "entry-episode-causal-case-1.5.0",
            },
        }
    )
    run_manifest_path.write_text(json.dumps(legacy_run_manifest), encoding="utf-8")
    incompatible_manifest = _run(
        _command(source, resumed_output, resume=True, **common)
    )
    assert incompatible_manifest.returncode != 0
    assert "run manifest differs" in incompatible_manifest.stderr
    run_manifest_path.write_bytes(current_run_manifest_bytes)

    checkpoint_root = resumed_output / "_checkpoint"
    checkpoint_manifest_path = checkpoint_root / "manifest.json"
    current_checkpoint_manifest_bytes = checkpoint_manifest_path.read_bytes()
    current_checkpoint_state_path = (
        checkpoint_root / checkpoint_manifest["state_file"]
    )
    current_checkpoint_state_bytes = current_checkpoint_state_path.read_bytes()
    legacy_checkpoint_state = pickle.loads(current_checkpoint_state_bytes)
    legacy_checkpoint_state["causal_cases"]._recorder_schema_version = (
        CAUSAL_CASE_RECORDER_SCHEMA_VERSION - 1
    )
    legacy_checkpoint_bytes = pickle.dumps(
        legacy_checkpoint_state,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    legacy_checkpoint_sha = hashlib.sha256(legacy_checkpoint_bytes).hexdigest()
    legacy_checkpoint_file = f"state-{legacy_checkpoint_sha}.pkl"
    (checkpoint_root / legacy_checkpoint_file).write_bytes(legacy_checkpoint_bytes)
    legacy_checkpoint_manifest = json.loads(current_checkpoint_manifest_bytes)
    legacy_checkpoint_manifest.update(
        {
            "state_file": legacy_checkpoint_file,
            "state_sha256": legacy_checkpoint_sha,
        }
    )
    checkpoint_manifest_path.write_text(
        json.dumps(legacy_checkpoint_manifest),
        encoding="utf-8",
    )
    incompatible_checkpoint = _run(
        _command(source, resumed_output, resume=True, **common)
    )
    assert incompatible_checkpoint.returncode != 0
    assert (
        "checkpoint causal-case recorder schema changed"
        in incompatible_checkpoint.stderr
    )
    for state_path in checkpoint_root.glob("state-*.pkl"):
        state_path.unlink()
    current_checkpoint_state_path.write_bytes(current_checkpoint_state_bytes)
    checkpoint_manifest_path.write_bytes(current_checkpoint_manifest_bytes)

    resumed = _run(
        _command(source, resumed_output, resume=True, **common)
    )
    assert resumed.returncode == 0, resumed.stderr
    uninterrupted_output = tmp_path / "case-uninterrupted"
    uninterrupted = _run(
        _command(source, uninterrupted_output, **common)
    )
    assert uninterrupted.returncode == 0, uninterrupted.stderr

    def stream_rows(destination: Path, name: str) -> pd.DataFrame:
        manifest = json.loads(
            (destination / f"{name}.manifest.json").read_text()
        )
        if not manifest["shards"]:
            return pd.DataFrame(columns=tuple(manifest["field_types"]))
        return pd.concat(
            [
                pd.read_parquet(destination / shard["path"])
                for shard in manifest["shards"]
            ],
            ignore_index=True,
        )

    for stream_name, field_types in (
        ("causal_case_input_shards", CAUSAL_CASE_INPUT_FIELD_TYPES),
        ("causal_case_outcome_shards", CAUSAL_CASE_OUTCOME_FIELD_TYPES),
    ):
        resumed_manifest = json.loads(
            (resumed_output / f"{stream_name}.manifest.json").read_text()
        )
        assert resumed_manifest["field_types"] == dict(field_types)
        assert all(
            sha256_file(resumed_output / shard["path"])
            == shard["sha256"]
            for shard in resumed_manifest["shards"]
        )
        pd.testing.assert_frame_equal(
            stream_rows(resumed_output, stream_name),
            stream_rows(uninterrupted_output, stream_name),
        )

    pair = json.loads(
        (resumed_output / "causal_case_library.manifest.json").read_text()
    )
    assert pair["status"] == "complete"
    assert pair["recorder_schema_version"] == (
        CAUSAL_CASE_RECORDER_SCHEMA_VERSION
    )
    assert pair["leakage_contract"] == {
        "embedding_source": "input_stream_only",
        "episode_split_disjoint_required": True,
        "normalization_prefix_only": True,
        "outcome_fields_in_input_schema": False,
    }
    run_manifest = json.loads(
        (resumed_output / "run_manifest.json").read_text()
    )
    assert run_manifest["output"]["causal_case_library"] is True
    assert "market_case_input_identity" not in run_manifest
    assert "last_completed_asof" not in run_manifest["source"]
    assert {"symbol", "instrument_id"}.isdisjoint(run_manifest["source"])
    assert run_manifest["causal_case_identity"][
        "future_visible_to_input"
    ] is False
    summary = json.loads((resumed_output / "summary.json").read_text())
    assert summary["causal_case_library"]["output_affects_model"] is False
    assert summary["stream_rows"] == json.loads(
        (uninterrupted_output / "summary.json").read_text()
    )["stream_rows"]
    completed = json.loads((resumed_output / "COMPLETED.json").read_text())
    assert completed["causal_case_library"] == (
        "causal_case_library.manifest.json"
    )
    retired = json.loads(
        (resumed_output / "_checkpoint/manifest.json").read_text()
    )
    assert retired["status"] == "complete"
    assert not tuple((resumed_output / "_checkpoint").glob("state-*.pkl"))


@pytest.mark.parametrize(
    "extra_args",
    (
        ("--brain-calibration",),
        ("--shadow-outcomes",),
        ("--causal-case-library",),
        ("--brain-diagnostics",),
        ("--visualize-at", "2022-06-06T18:05:00-04:00"),
        ("--simulate-execution",),
        ("--mbo-execution", "missing.parquet"),
        ("--spread-points", "0.25"),
        ("--slippage-points", "0.25"),
        ("--action-disabled-playbook", "liquidity_sweep_reversal"),
    ),
)
def test_market_case_input_mode_rejects_every_output_or_execution_feature(
    tmp_path: Path,
    extra_args: tuple[str, ...],
) -> None:
    source = _write_source(tmp_path)
    command = _command(
        source,
        tmp_path / ("rejected-" + extra_args[0].lstrip("-").replace("-", "_")),
        market_case_input=True,
    )
    command.extend(extra_args)
    result = _run(command)
    assert result.returncode != 0
    assert "--market-case-input is exclusive" in result.stderr


def test_market_case_input_profiles_allow_additional_registered_windows(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        (
            ROOT / "configs/market_case_input_profiles_v2.json"
        ).read_text(encoding="utf-8")
    )
    profiles = payload["market_case_input_profiles"]
    assert {
        "market_episode_input_smoke_2024_01_08",
        "market_episode_input_2024_01",
    }.issubset(profiles)
    expected_false = {
        "threshold_search",
        "calibration_fit_allowed",
        "future_path_used",
        "outcome_used",
        "pnl_used",
        "mbo_used",
        "brain_output",
        "decision_output",
        "risk_output",
        "execution_output",
        "shadow_output",
        "legacy_case_output",
    }
    for profile in profiles.values():
        assert profile["runner_mode"] == "market_episode_input_only"
        assert profile["observer_transition_delta_transport"] is True
        assert profile["recorder_schema_version"] == (
            MARKET_CASE_RECORDER_SCHEMA_VERSION
        )
        assert profile["protocol_version"] == (
            expected_market_case_run_identity()["protocol"][
                "protocol_version"
            ]
        )
        assert all(profile[name] is False for name in expected_false)

    profiles["market_episode_input_future_preregistered"] = {
        **profiles["market_episode_input_2024_01"],
        "start": "2024-02-01T18:00:00-05:00",
        "end_exclusive": "2024-03-01T18:00:00-05:00",
    }
    protocol = tmp_path / "data-splits-with-additional-market-profile.json"
    protocol.write_text(json.dumps(payload), encoding="utf-8")

    name, profile = _load_market_case_input_profile(
        protocol,
        start=pd.Timestamp("2024-01-08T18:00:00-05:00"),
        end=pd.Timestamp("2024-01-09T18:00:00-05:00"),
        warmup_days=7,
    )
    assert name == "market_episode_input_smoke_2024_01_08"
    assert profile == profiles[name]


@pytest.mark.parametrize(
    "git_result",
    (
        OSError("git unavailable"),
        SimpleNamespace(returncode=1, stdout="", stderr="not a repository"),
        SimpleNamespace(returncode=0, stdout="A" * 40, stderr=""),
    ),
)
def test_repository_commit_identity_fails_closed_without_valid_git(
    monkeypatch: pytest.MonkeyPatch,
    git_result: object,
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> object:
        if isinstance(git_result, BaseException):
            raise git_result
        return git_result

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="repository HEAD commit identity"):
        runner_module._repository_commit_identity()


def test_market_case_input_rejects_multi_contract_before_output(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    frame = pd.read_parquet(source)
    frame.loc[frame.index[-2], "symbol"] = "NQZ2"
    frame.loc[frame.index[-2], "instrument_id"] = 2
    frame["instrument_id"] = frame["instrument_id"].astype("int64")
    frame.to_parquet(source)
    protocol = _write_short_market_input_protocol(tmp_path)
    output = tmp_path / "market-input-multi-contract"

    result = _run(
        _command(
            source,
            output,
            market_case_input=True,
            market_case_profile_registry=protocol,
        )
    )

    assert result.returncode != 0
    assert "requires exactly one distinct (symbol, instrument_id)" in result.stderr
    assert not output.exists()


def test_market_case_input_same_contract_gap_resets_and_resumes_exactly(
    tmp_path: Path,
) -> None:
    source = _write_same_contract_gap_source(tmp_path)
    end = "2022-06-06T18:40:00-04:00"
    protocol = _write_short_market_input_protocol(
        tmp_path,
        end_exclusive=end,
    )
    default_result = _run(
        _command(
            source,
            tmp_path / "default-gap-rejected",
            end=end,
        )
    )
    assert default_result.returncode != 0
    assert "unresolved open-market gap" in default_result.stderr
    assert "same_contract=True" in default_result.stderr
    common = {
        "market_case_input": True,
        "market_case_profile_registry": protocol,
        "end": end,
    }
    resumed_output = tmp_path / "market-input-gap-resumed"
    interrupted = _run(
        _command(source, resumed_output, stop_after=11, **common)
    )
    assert interrupted.returncode != 0
    assert "intentional diagnostic interruption" in interrupted.stderr

    checkpoint_manifest = json.loads(
        (resumed_output / "_checkpoint/manifest.json").read_text()
    )
    checkpoint_state = pickle.loads(
        (
            resumed_output
            / "_checkpoint"
            / checkpoint_manifest["state_file"]
        ).read_bytes()
    )
    assert checkpoint_state["source_rows_consumed"] == 11
    assert checkpoint_state["last_source_start"] == pd.Timestamp(
        "2022-06-06T18:10:00-04:00"
    )

    run_manifest_path = resumed_output / "run_manifest.json"
    original_manifest = run_manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    assert manifest["runtime_state_schema_version"] == 8
    assert manifest["data_continuity"] == dict(
        MARKET_CASE_INPUT_DATA_CONTINUITY_POLICY
    )
    assert manifest["data_continuity"] == {
        "maximum_no_trade_gap_minutes": 5,
        "allow_same_contract_data_gap_reset": True,
        "data_gap_reset_anomaly": "data_gap_history_reset",
        "allow_cross_contract_data_gap_reset": False,
        "synthesize_over_cap_missing_minutes": False,
    }

    schema_seven = json.loads(original_manifest)
    schema_seven["runtime_state_schema_version"] = 7
    run_manifest_path.write_text(json.dumps(schema_seven), encoding="utf-8")
    stale_resume = _run(
        _command(source, resumed_output, resume=True, **common)
    )
    assert stale_resume.returncode != 0
    assert "run manifest differs" in stale_resume.stderr
    run_manifest_path.write_bytes(original_manifest)

    tampered = json.loads(original_manifest)
    tampered["data_continuity"][
        "allow_same_contract_data_gap_reset"
    ] = False
    run_manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    tampered_resume = _run(
        _command(source, resumed_output, resume=True, **common)
    )
    assert tampered_resume.returncode != 0
    assert "run manifest differs" in tampered_resume.stderr
    run_manifest_path.write_bytes(original_manifest)

    resumed = _run(
        _command(source, resumed_output, resume=True, **common)
    )
    assert resumed.returncode == 0, resumed.stderr
    control_output = tmp_path / "market-input-gap-control"
    control = _run(_command(source, control_output, **common))
    assert control.returncode == 0, control.stderr

    assert (resumed_output / "run_manifest.json").read_bytes() == (
        control_output / "run_manifest.json"
    ).read_bytes()
    assert (
        resumed_output / "market_case_input_shards.manifest.json"
    ).read_bytes() == (
        control_output / "market_case_input_shards.manifest.json"
    ).read_bytes()
    resumed_summary = json.loads(
        (resumed_output / "summary.json").read_text()
    )
    control_summary = json.loads(
        (control_output / "summary.json").read_text()
    )
    assert resumed_summary["market_case_input"]["epoch_resets"] == 1
    assert control_summary["market_case_input"]["epoch_resets"] == 1
    assert resumed_summary["processed_bars"] == 31
    assert resumed_summary["processed_bars"] == (
        resumed_summary["source_rows_processed"]
    )
    assert resumed_summary["stream_rows"] == control_summary["stream_rows"]


def test_market_case_input_runner_never_invokes_action_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path)
    protocol = _write_short_market_input_protocol(tmp_path)
    output = tmp_path / "market-input-neutral-only"
    engine = ContinuousSMCEngine.from_config(
        ROOT / "configs/model.json",
        runtime_mode="development",
    )
    monkeypatch.setattr(
        type(engine.brain),
        "reset",
        _forbidden_market_action_layer,
    )
    monkeypatch.setattr(
        type(engine.brain),
        "update",
        _forbidden_market_action_layer,
    )
    monkeypatch.setattr(
        type(engine.decision),
        "decide",
        _forbidden_market_action_layer,
    )
    monkeypatch.setattr(
        type(engine.risk),
        "review",
        _forbidden_market_action_layer,
    )
    monkeypatch.setattr(
        runner_module.ContinuousSMCEngine,
        "from_config",
        classmethod(lambda _cls, *_args, **_kwargs: engine),
    )
    monkeypatch.setattr(
        runner_module.CalibrationSequentialReplay,
        "on_bar",
        _forbidden_market_action_layer,
    )
    command = _command(
        source,
        output,
        market_case_input=True,
        market_case_profile_registry=protocol,
    )
    monkeypatch.setattr(sys, "argv", command[1:])

    runner_module._streamed_main(runner_module.parse_args())

    assert isinstance(engine.last_snapshot, NeutralEngineSnapshot)
    assert engine.brain.current is None
    assert (output / "COMPLETED.json").is_file()


def test_market_case_input_is_minimal_resumable_and_row_exact(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    protocol = _write_short_market_input_protocol(tmp_path)
    common = {
        "market_case_input": True,
        "market_case_profile_registry": protocol,
    }
    resumed_output = tmp_path / "market-input-resumed"
    interrupted = _run(
        _command(source, resumed_output, stop_after=11, **common)
    )
    assert interrupted.returncode != 0
    assert not tuple(resumed_output.glob("*.manifest.json"))
    assert not (resumed_output / "summary.json").exists()
    assert not (resumed_output / "COMPLETED.json").exists()

    checkpoint_manifest = json.loads(
        (resumed_output / "_checkpoint/manifest.json").read_text()
    )
    checkpoint_state = pickle.loads(
        (
            resumed_output
            / "_checkpoint"
            / checkpoint_manifest["state_file"]
        ).read_bytes()
    )
    assert set(checkpoint_state) == {
        "replay",
        "market_cases",
        "streams",
        "buffers",
        "processed_bars",
        "source_rows_consumed",
        "last_checkpoint_processed_bars",
        "decision_rows",
        "market_case_input_rows",
        "last_source_start",
        "last_asof",
        "resume_count",
        "finalized",
        "peak_buffer_rows",
        "next_shard_index",
        "committed_shards",
        "scene_graph_compaction",
    }
    checkpoint_compaction = checkpoint_state["scene_graph_compaction"]
    assert checkpoint_compaction["runs"] >= 1
    assert checkpoint_compaction["last_processed_bars"] == (
        checkpoint_state["processed_bars"]
    )
    assert checkpoint_compaction["last_result"] is not None
    assert checkpoint_state[
        "replay"
    ].engine.scene_graph.history_retention_floor is not None
    assert isinstance(
        checkpoint_state["market_cases"],
        MarketEpisodeCaseRecorder,
    )
    assert checkpoint_state["market_cases"].recorder_schema_version == (
        MARKET_CASE_RECORDER_SCHEMA_VERSION
    )
    assert set(checkpoint_state["streams"]) == {
        "market_case_input_shards"
    }
    assert set(checkpoint_state["buffers"]) == {
        "market_case_input_shards"
    }
    assert len(checkpoint_state["buffers"]["market_case_input_shards"]) < 7
    for forbidden in (
        "brain_calibration",
        "shadow_outcomes",
        "causal_cases",
        "natural_episode_funnel_records",
        "open_thesis_binding_records",
        "unexplained_episode_summaries",
        "visual_artifacts",
        "entry_approvals",
    ):
        assert forbidden not in checkpoint_state

    wrong_mode = _run(
        _command(
            source,
            resumed_output,
            resume=True,
        )
    )
    assert wrong_mode.returncode != 0
    assert "run manifest differs" in wrong_mode.stderr

    run_manifest_path = resumed_output / "run_manifest.json"
    original_run_manifest = run_manifest_path.read_bytes()
    run_manifest_sha256 = hashlib.sha256(original_run_manifest).hexdigest()
    version_seven_run_manifest = json.loads(original_run_manifest)
    version_seven_run_manifest["runtime_state_schema_version"] = 7
    run_manifest_path.write_text(json.dumps(version_seven_run_manifest))
    version_seven_run_resume = _run(
        _command(source, resumed_output, resume=True, **common)
    )
    assert version_seven_run_resume.returncode != 0
    assert "run manifest differs" in version_seven_run_resume.stderr
    run_manifest_path.write_bytes(original_run_manifest)

    protocol_one_one_run_manifest = json.loads(original_run_manifest)
    protocol_one_one_run_manifest["market_case_input_identity"]["protocol"][
        "protocol_version"
    ] = "market-episode-input-only-1.1.0"
    run_manifest_path.write_text(json.dumps(protocol_one_one_run_manifest))
    protocol_one_one_resume = _run(
        _command(source, resumed_output, resume=True, **common)
    )
    assert protocol_one_one_resume.returncode != 0
    assert "run manifest differs" in protocol_one_one_resume.stderr
    run_manifest_path.write_bytes(original_run_manifest)

    checkpoint_manifest_path = resumed_output / "_checkpoint/manifest.json"
    original_checkpoint_manifest = checkpoint_manifest_path.read_bytes()
    invalid_compaction_state = pickle.loads(
        pickle.dumps(checkpoint_state, protocol=pickle.HIGHEST_PROTOCOL)
    )
    invalid_compaction_state["scene_graph_compaction"]["last_result"][
        "after"
    ]["nodes"] += 1
    invalid_compaction_raw = pickle.dumps(
        invalid_compaction_state,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    invalid_compaction_hash = hashlib.sha256(
        invalid_compaction_raw
    ).hexdigest()
    invalid_compaction_file = (
        resumed_output
        / "_checkpoint"
        / f"state-{invalid_compaction_hash}.pkl"
    )
    invalid_compaction_file.write_bytes(invalid_compaction_raw)
    invalid_compaction_manifest = json.loads(original_checkpoint_manifest)
    invalid_compaction_manifest.update(
        {
            "state_file": invalid_compaction_file.name,
            "state_sha256": invalid_compaction_hash,
        }
    )
    checkpoint_manifest_path.write_text(
        json.dumps(invalid_compaction_manifest)
    )
    invalid_compaction_manifest_bytes = checkpoint_manifest_path.read_bytes()
    invalid_compaction_resume = _run(
        _command(source, resumed_output, resume=True, **common)
    )
    assert invalid_compaction_resume.returncode != 0
    assert "Scene Graph compaction" in invalid_compaction_resume.stderr
    assert checkpoint_manifest_path.read_bytes() == (
        invalid_compaction_manifest_bytes
    )
    checkpoint_manifest_path.write_bytes(original_checkpoint_manifest)

    version_one_checkpoint_state = dict(checkpoint_state)
    version_one_checkpoint_state.pop("scene_graph_compaction")
    version_one_checkpoint_raw = pickle.dumps(
        version_one_checkpoint_state,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    version_one_checkpoint_hash = hashlib.sha256(
        version_one_checkpoint_raw
    ).hexdigest()
    version_one_checkpoint_file = (
        resumed_output
        / "_checkpoint"
        / f"state-{version_one_checkpoint_hash}.pkl"
    )
    version_one_checkpoint_file.write_bytes(version_one_checkpoint_raw)
    version_one_checkpoint_manifest = json.loads(
        original_checkpoint_manifest
    )
    version_one_checkpoint_manifest.update(
        {
            "state_file": version_one_checkpoint_file.name,
            "state_sha256": version_one_checkpoint_hash,
        }
    )
    checkpoint_manifest_path.write_text(
        json.dumps(version_one_checkpoint_manifest)
    )
    version_one_checkpoint_resume = _run(
        _command(source, resumed_output, resume=True, **common)
    )
    assert version_one_checkpoint_resume.returncode != 0
    assert "market-case input checkpoint state changed" in (
        version_one_checkpoint_resume.stderr
    )
    checkpoint_manifest_path.write_bytes(original_checkpoint_manifest)

    tampered_run_manifest = json.loads(original_run_manifest)
    tampered_run_manifest["source"]["symbol"] = "NQZ2"
    run_manifest_path.write_text(json.dumps(tampered_run_manifest))
    tampered_resume = _run(
        _command(source, resumed_output, resume=True, **common)
    )
    assert tampered_resume.returncode != 0
    assert "run manifest differs" in tampered_resume.stderr
    run_manifest_path.write_bytes(original_run_manifest)

    tampered_repository_manifest = json.loads(original_run_manifest)
    tampered_repository_manifest["repository"]["commit"] = "0" * 40
    run_manifest_path.write_text(json.dumps(tampered_repository_manifest))
    tampered_repository_resume = _run(
        _command(source, resumed_output, resume=True, **common)
    )
    assert tampered_repository_resume.returncode != 0
    assert "run manifest differs" in tampered_repository_resume.stderr
    run_manifest_path.write_bytes(original_run_manifest)

    resumed = _run(_command(source, resumed_output, resume=True, **common))
    assert resumed.returncode == 0, resumed.stderr
    control_output = tmp_path / "market-input-control"
    control = _run(_command(source, control_output, **common))
    assert control.returncode == 0, control.stderr

    runner_source = (ROOT / "scripts/run_continuous_replay.py").read_text()
    assert runner_source.count("step = replay.on_bar(") == 1
    assert runner_source.count(
        "snapshot = replay.engine.on_bar_neutral_input(bar)"
    ) == 1
    pd.testing.assert_frame_equal(
        _materialized_stream_rows(
            resumed_output,
            "market_case_input_shards",
        ),
        _materialized_stream_rows(
            control_output,
            "market_case_input_shards",
        ),
    )
    stream_manifest_path = (
        resumed_output / "market_case_input_shards.manifest.json"
    )
    stream_manifest = json.loads(stream_manifest_path.read_text())
    assert stream_manifest["field_types"] == dict(MARKET_CASE_INPUT_FIELD_TYPES)
    assert stream_manifest["bindings"] == {
        "run_manifest": "run_manifest.json",
        "run_manifest_sha256": run_manifest_sha256,
    }
    assert all(
        sha256_file(resumed_output / shard["path"]) == shard["sha256"]
        for shard in stream_manifest["shards"]
    )

    run_manifest = json.loads(
        (resumed_output / "run_manifest.json").read_text()
    )
    assert set(run_manifest) == {
        "schema_version",
        "runner",
        "mode",
        "runtime_state_schema_version",
        "data_continuity",
        "repository",
        "profile",
        "profile_registry",
        "source",
        "model_config",
        "market_case_input_identity",
        "window",
        "output",
    }
    assert run_manifest["mode"] == "market_case_input"
    assert run_manifest["runtime_state_schema_version"] == (
        MARKET_CASE_INPUT_RUNTIME_STATE_SCHEMA_VERSION
    )
    assert run_manifest["runtime_state_schema_version"] == 8
    assert run_manifest["data_continuity"] == dict(
        MARKET_CASE_INPUT_DATA_CONTINUITY_POLICY
    )
    repository_commit = run_manifest["repository"]["commit"]
    assert set(run_manifest["repository"]) == {"commit"}
    assert len(repository_commit) == 40
    assert all(
        character in "0123456789abcdef"
        for character in repository_commit
    )
    expected_commit = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert repository_commit == expected_commit
    assert original_run_manifest.count(b'"commit":') == 1
    assert run_manifest["profile_registry"] == {
        "path": str(protocol.resolve()),
        "sha256": sha256_file(protocol),
        "schema_version": 1,
    }
    assert set(run_manifest["source"]) == {
        "path",
        "sha256",
        "rows",
        "first",
        "last",
        "last_completed_asof",
        "role",
        "symbol",
        "instrument_id",
    }
    assert run_manifest["source"]["symbol"] == "NQU2"
    assert run_manifest["source"]["instrument_id"] == 1
    assert run_manifest["market_case_input_identity"] == (
        expected_market_case_run_identity()
    )
    assert run_manifest["output"]["stream_families"] == [
        "market_case_input_shards"
    ]
    assert pd.Timestamp(run_manifest["source"]["last"]) + pd.Timedelta(
        minutes=1
    ) == pd.Timestamp(run_manifest["source"]["last_completed_asof"])
    assert run_manifest["model_config"]["tick_size"] == 0.25
    assert run_manifest["model_config"]["sha256"] == sha256_file(
        ROOT / "configs/model.json"
    )

    input_rows = _materialized_stream_rows(
        resumed_output,
        "market_case_input_shards",
    )
    assert {
        "source_path",
        "source_sha256",
        "source_role",
        "split_role",
        "model_versions_json",
        "symbol",
        "instrument_id",
        "repository",
        "commit",
    }.isdisjoint(input_rows.columns)
    summary = json.loads((resumed_output / "summary.json").read_text())
    assert summary["mode"] == "market_case_input"
    assert summary["run_manifest_sha256"] == run_manifest_sha256
    assert summary["market_case_input"]["outcome_joined"] is False
    compaction = summary["scene_graph_compaction"]
    assert compaction["runs"] >= 1
    assert compaction["last_processed_bars"] == summary["processed_bars"]
    assert compaction["last_result"]["history_retention_floor"] is not None
    control_summary = json.loads(
        (control_output / "summary.json").read_text()
    )
    assert compaction == control_summary["scene_graph_compaction"]
    assert summary["stream_rows"] == {
        "market_case_input_shards": int(stream_manifest["rows"])
    }
    completed_payload = json.loads(
        (resumed_output / "COMPLETED.json").read_text()
    )
    assert completed_payload == {
        "schema_version": 1,
        "status": "complete",
        "run_manifest": "run_manifest.json",
        "run_manifest_sha256": run_manifest_sha256,
        "summary": "summary.json",
        "progress": "progress.json",
        "market_case_input_shards": (
            "market_case_input_shards.manifest.json"
        ),
    }
    allowed_names = {
        "_checkpoint",
        "run_manifest.json",
        "progress.json",
        "summary.json",
        "COMPLETED.json",
        "market_case_input_shards.manifest.json",
        *{
            shard["path"]
            for shard in stream_manifest["shards"]
        },
    }
    assert {path.name for path in resumed_output.iterdir()} == allowed_names
    retired = json.loads(
        (resumed_output / "_checkpoint/manifest.json").read_text()
    )
    assert retired["status"] == "complete"
    assert retired["resume_supported"] is False
    assert not tuple((resumed_output / "_checkpoint").glob("state-*.pkl"))


def test_shadow_finalizer_failure_resumes_without_market_replay(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path, periods=241)
    payload = json.loads(
        (ROOT / "configs/data_splits.json").read_text(encoding="utf-8")
    )
    profile = payload["shadow_diagnostic_profiles"][
        "brain_playbook_reverse_validation_2024_01"
    ]
    profile.update(
        {
            "allowed_ohlcv_role": "calibration",
            "start": "2022-06-06T18:00:00-04:00",
            "end_exclusive": "2022-06-06T22:02:00-04:00",
            "warmup_calendar_days": 0,
        }
    )
    action_policy = tuple(profile["action_disabled_playbooks"])
    protocol = tmp_path / "shadow-resume-data-splits.json"
    protocol.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "shadow-finalizer-resume"

    failed = _run(
        _command(
            source,
            output,
            shadow_outcomes=True,
            validation_protocol=protocol,
            end="2022-06-06T22:02:00-04:00",
            fail_shadow_finalize_after_batches=1,
            action_disabled_playbooks=action_policy,
        )
    )
    assert failed.returncode != 0
    assert "intentional Shadow finalizer interruption" in failed.stderr
    assert not (output / "COMPLETED.json").exists()
    checkpoint_manifest = json.loads(
        (output / "_checkpoint/manifest.json").read_text()
    )
    checkpoint_state = pickle.loads(
        (
            output
            / "_checkpoint"
            / checkpoint_manifest["state_file"]
        ).read_bytes()
    )
    assert checkpoint_state["finalized"] is True
    processed_bars = int(checkpoint_state["processed_bars"])
    source_rows_consumed = int(checkpoint_state["source_rows_consumed"])
    decision_rows = int(checkpoint_state["decision_rows"])
    source_stream_manifests = {
        name: (output / name).read_bytes()
        for name in (
            "decision_shards.manifest.json",
            "shadow_outcome_shards.manifest.json",
        )
    }

    run_manifest_path = output / "run_manifest.json"
    current_run_manifest_bytes = run_manifest_path.read_bytes()
    legacy_shadow_manifest = json.loads(current_run_manifest_bytes)
    assert legacy_shadow_manifest["shadow_outcome_identity"] == {
        "profile_name": "brain_playbook_reverse_validation_2024_01",
        "recorder_schema_version": SHADOW_OUTCOME_RECORDER_SCHEMA_VERSION,
        "derived_schema_version": SHADOW_DERIVED_SCHEMA_VERSION,
        "protocol_version": SHADOW_OUTCOME_PROTOCOL["protocol_version"],
    }
    legacy_shadow_manifest["shadow_outcome_identity"][
        "derived_schema_version"
    ] = SHADOW_DERIVED_SCHEMA_VERSION - 1
    run_manifest_path.write_text(
        json.dumps(legacy_shadow_manifest),
        encoding="utf-8",
    )
    incompatible_derived = _run(
        _command(
            source,
            output,
            resume=True,
            shadow_outcomes=True,
            validation_protocol=protocol,
            end="2022-06-06T22:02:00-04:00",
            action_disabled_playbooks=action_policy,
        )
    )
    assert incompatible_derived.returncode != 0
    assert "run manifest differs" in incompatible_derived.stderr
    legacy_shadow_manifest = json.loads(current_run_manifest_bytes)
    legacy_shadow_manifest["shadow_outcome_identity"].update(
        {
            "recorder_schema_version": (
                SHADOW_OUTCOME_RECORDER_SCHEMA_VERSION - 1
            ),
            "derived_schema_version": SHADOW_DERIVED_SCHEMA_VERSION - 1,
            "protocol_version": "shadow-candidate-outcome-1.4.0",
        }
    )
    run_manifest_path.write_text(
        json.dumps(legacy_shadow_manifest),
        encoding="utf-8",
    )
    incompatible = _run(
        _command(
            source,
            output,
            resume=True,
            shadow_outcomes=True,
            validation_protocol=protocol,
            end="2022-06-06T22:02:00-04:00",
            action_disabled_playbooks=action_policy,
        )
    )
    assert incompatible.returncode != 0
    assert "run manifest differs" in incompatible.stderr
    run_manifest_path.write_bytes(current_run_manifest_bytes)

    resumed = _run(
        _command(
            source,
            output,
            resume=True,
            shadow_outcomes=True,
            validation_protocol=protocol,
            end="2022-06-06T22:02:00-04:00",
            action_disabled_playbooks=action_policy,
        )
    )
    assert resumed.returncode == 0, resumed.stderr
    assert (output / "COMPLETED.json").is_file()
    assert not (output / ".shadow_derived.tmp").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["resume_count"] == 1
    assert summary["decision_rows"] == decision_rows
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    assert run_manifest["source"]["rows"] == source_rows_consumed
    assert processed_bars == source_rows_consumed
    for name, before in source_stream_manifests.items():
        assert (output / name).read_bytes() == before
