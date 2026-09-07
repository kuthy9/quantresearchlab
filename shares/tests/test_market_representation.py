from __future__ import annotations

import builtins
import copy
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import shares.core.market_cases as market_cases_module
from shares.core.case_retrieval import MarketEpisodeCaseIndex
from eyes.core.interaction import interaction_artifact_collections
from shares.core.market_cases import expected_market_case_run_identity
from shares.core.market_representation import (
    BAR_END_INDEX_BINDING,
    CAUSAL_CANDLE_FEATURES,
    EMBEDDING_DIM,
    EmbeddingEvaluationSample,
    FEATURE_SCHEMA_VERSION,
    NEXT_EVENT_TYPE_VOCAB,
    NEXT_LIFECYCLE_VOCAB,
    NEUTRAL_MARKET_LIFECYCLE_TARGETS,
    NEUTRAL_MARKET_TRANSITION_KINDS,
    NEUTRAL_SPARSE_ACTIVE_TARGETS,
    NEUTRAL_SPARSE_DISABLED_TARGETS,
    PARAMETER_BUDGET,
    TIMEFRAMES,
    TORCH_AVAILABLE,
    CanonicalOHLCVStore,
    CanonicalSourceKey,
    DecisionTimeEmbeddingRecord,
    EventGraphObservation,
    MarketRepresentationModel,
    PrefixIndexRange,
    RepresentationCase,
    RepresentationDataError,
    SelfSupervisedTarget,
    TorchUnavailableError,
    assign_leakage_safe_splits,
    build_neutral_market_revision_targets,
    build_observable_revision_targets,
    causal_input_fingerprint,
    collate_representation_cases,
    compare_reconstruction_to_baselines,
    compare_validation_to_baseline,
    deduplicate_causal_inputs,
    encode_decision_time_records,
    encode_decision_time_head_records,
    evaluate_outcome_blind_embedding_space,
    encode_market_episode_active_head_records,
    encode_market_episode_records,
    mask_direct_label_source_tokens,
    market_case_input_to_representation_mapping,
    prepare_representation_case,
    prepare_neutral_representation_case,
    load_representation_checkpoint,
    load_neutral_representation_checkpoint,
    neutral_b1_vicreg_loss,
    representation_case_from_case_input_row,
    representation_case_from_market_case_input_row,
    representation_multitask_loss,
    representation_checkpoint_id,
    save_representation_checkpoint,
    save_neutral_representation_checkpoint,
    select_first_causal_stage_revisions,
    supervised_causal_contrastive_loss,
    validate_split_integrity,
    validate_single_embedding_checkpoint,
)
from shares.core.model import (
    AuthorityLayer,
    Direction,
    DirectionalObstructionView,
    FairValueGapLifecycle,
    GlobalMarketContext,
    LiquidityInventoryLifecycle,
    MarketMode,
    OpenMarketThesis,
    ScaleRelation,
    ScaleRelationState,
    Timeframe,
    to_primitive,
)
from shares.core.scene_graph import (
    SceneEdgeKind,
    StructuralScale,
    market_episode_id,
)

from eyes.tests.test_interaction_eye_brain_boundary import _strict_interaction
from eyes.tests.test_v3_group5_primitives import _draw, _fvg


def _frame(timeframe: str, *, scale: float = 1.0) -> pd.DataFrame:
    frequency = {
        "4h": "4h",
        "1h": "1h",
        "15m": "15min",
        "5m": "5min",
        "1m": "1min",
    }[timeframe]
    index = pd.date_range("2022-01-03 09:31", periods=12, freq=frequency, tz="UTC")
    base = (100.0 + np.arange(12) * 0.25) * scale
    close = base + np.where(np.arange(12) % 2 == 0, 0.10, -0.05) * scale
    return pd.DataFrame(
        {
            "open": base,
            "high": np.maximum(base, close) + 0.20 * scale,
            "low": np.minimum(base, close) - 0.15 * scale,
            "close": close,
            "volume": 100 + np.arange(12) * 3,
        },
        index=index,
    )


def _case(
    *,
    case_id: str = "case-a",
    revision_id: str = "revision-a",
    episode_id: str = "episode-a",
    context_id: str = "context-a",
    epoch_id: str = "epoch-a",
    direction: int = 1,
    regime: str = "continuation",
) -> RepresentationCase:
    frames = {timeframe: _frame(timeframe) for timeframe in TIMEFRAMES}
    asof = pd.Timestamp("2022-01-03 09:38", tz="UTC")
    prefixes = {
        timeframe: PrefixIndexRange(
            market_epoch_id=epoch_id,
            timeframe=timeframe,
            canonical_source_id="canonical-a",
            row_start=0,
            row_end_exclusive=int(frame.index.searchsorted(asof, side="right")),
        )
        for timeframe, frame in frames.items()
    }
    return RepresentationCase(
        case_id=case_id,
        revision_id=revision_id,
        market_epoch_id=epoch_id,
        context_thesis_id=context_id,
        entry_episode_id=episode_id,
        asof=asof,
        direction=direction,
        regime=regime,
        prefixes=prefixes,
        events=(
            EventGraphObservation(
                event_id=f"event-{case_id}",
                event_type="reverse_displacement",
                lifecycle="active",
                observed_at=asof - timedelta(minutes=2),
                active_since=asof - timedelta(minutes=3),
                duration_seconds=60.0,
                relation_types=("caused_by", "connected_to"),
                direction=direction,
                scale="1m",
                market_epoch_id=epoch_id,
            ),
        ),
    )


def _store(*, scale: float = 1.0, mutate_future: bool = False) -> CanonicalOHLCVStore:
    frames = {}
    ticks = {}
    availability = {}
    for timeframe in TIMEFRAMES:
        key = CanonicalSourceKey("epoch-a", timeframe, "canonical-a")
        frame = _frame(timeframe, scale=scale)
        if mutate_future:
            frame.iloc[-1, frame.columns.get_loc("close")] += 10_000 * scale
            frame.iloc[-1, frame.columns.get_loc("high")] += 10_000 * scale
        frames[key] = frame
        ticks[key] = 0.25 * scale
        availability[key] = BAR_END_INDEX_BINDING
    return CanonicalOHLCVStore(
        frames,
        tick_sizes=ticks,
        availability_bindings=availability,
        normalization_window=3,
    )


def _case_library_row(
    *,
    asof: pd.Timestamp = pd.Timestamp("2022-01-03 09:38", tz="UTC"),
    revision_id: str = "revision-library-a",
    revision_index: int = 0,
    revision_stage: str = "episode_created",
) -> dict[str, object]:
    prefix_start = pd.Timestamp("2022-01-03 09:30", tz="UTC")
    prefixes = [
        {
            "timeframe": timeframe,
            "frame_row_start": 0,
            # Deliberately snapshot-local and larger than the external view.
            "frame_row_end_exclusive": 999,
            "replay_view_1m_row_start": 100,
            "replay_view_1m_row_end_exclusive": 109,
            "start_at": prefix_start.isoformat(),
            "end_at": asof.isoformat(),
            "asof": asof.isoformat(),
            "boundary_semantics": "[start_at,end_at]_completed_prefix",
            "reload_rule": (
                "filter_canonical_source_by_timezone_aware_time_bounds;"
                "aggregate_to_timeframe;never_use_frame_row_as_global_index"
            ),
        }
        for timeframe in ("4H", "1H", "15m", "5m", "1m")
    ]
    physical_event = to_primitive(
        replace(
            _fvg(asof, identity="zone-a"),
            symbol="NQ",
            instrument_id=1,
        )
    )
    update_collections = {
        name: [] for name in _NEUTRAL_TRANSITION_COLLECTIONS
    }
    update_collections[
        "group3_fvg_transitions_this_update"
    ] = [physical_event]
    aggregate_collections = {
        name: list(update_collections[name])
        for name in _CAUSAL_AGGREGATE_TRANSITION_COLLECTIONS
    }
    transition = {
        "typed_transition_delta_available": True,
        "coverage": {
            "coverage_start_at": pd.Timestamp(
                "2022-01-03 09:31", tz="UTC"
            ).isoformat(),
            "coverage_start_exclusive": False,
            "coverage_end_at": asof.isoformat(),
            "coverage_start_replay_update_ordinal": 0,
            "coverage_end_replay_update_ordinal": 0,
            "complete": True,
            "observed_update_count": 1,
            "eventful_update_count": 1,
            "last_observed_update_at": asof.isoformat(),
            "last_observed_replay_update_ordinal": 0,
            "all_typed_deltas_available": True,
            "gap_free": True,
        },
        "collections": aggregate_collections,
        "updates": [
            {
                "asof": asof.isoformat(),
                "replay_update_ordinal": 0,
                "typed_transition_delta_available": True,
                "collections": update_collections,
            }
        ],
    }
    coverage = transition["coverage"]
    relation_descriptor = {
        "edge_id": "edge-a",
        "relation": SceneEdgeKind.CREATES.value,
        "lifecycle": "active",
        "observed_at": asof.isoformat(),
        "source": {
            "node_id": "node-source-a",
            "kind": "manipulation",
            "role": "manipulation",
            "timeframe": "1m",
            "structural_scale": StructuralScale.INTERNAL.value,
            "lifecycle": "active",
        },
        "target": {
            "node_id": "node-target-a",
            "kind": "entry_location",
            "role": "entry_location",
            "timeframe": "1m",
            "structural_scale": StructuralScale.INTERNAL.value,
            "lifecycle": "active",
        },
    }
    graph = {
        "asof": asof.isoformat(),
        "revision_id": "scene-a",
        "coverage": coverage,
        "added_node_ids": ["zone-a"],
        "revised_node_ids": [],
        "added_edge_ids": ["edge-a"],
        "revised_edge_ids": [],
        "resolution_event_ids": [],
        "relation_descriptors": [relation_descriptor],
        "relation_descriptors_complete": True,
        "updates": [
            {
                "asof": asof.isoformat(),
                "replay_update_ordinal": 0,
                "revision_id": "scene-a",
                "added_node_ids": ["zone-a"],
                "revised_node_ids": [],
                "added_edge_ids": ["edge-a"],
                "revised_edge_ids": [],
                "resolution_event_ids": [],
                "relation_descriptors": [relation_descriptor],
                "relation_descriptors_complete": True,
            }
        ],
    }
    json_field = lambda value: json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return {
        "revision_id": revision_id,
        "case_id": "case-library-a",
        "revision_index": revision_index,
        "revision_stage": revision_stage,
        "stage_identity": f"{revision_stage}:{revision_id}",
        "stage_observed_at": asof,
        "input_fingerprint": "",
        "market_epoch_id": "epoch-a",
        "context_thesis_id": "context-a",
        "entry_episode_id": "episode-a",
        "candidate_id": "candidate-a",
        "playbook": "LIQUIDITY_SWEEP_REVERSAL",
        "direction": "LONG",
        "observable_regime": "sweep_failure",
        "mechanism_label": "liquidity_sweep_reversal",
        "asof": asof,
        "decision_at": asof,
        "admitted_at": pd.Timestamp("2022-01-03 09:31", tz="UTC"),
        "symbol": "NQ",
        "instrument_id": 1,
        "decision_price": 100.0,
        "source_path": "/tmp/canonical.parquet",
        "source_sha256": "a" * 64,
        "source_role": "development",
        "split_role": "development",
        "prefix_refs_json": json_field(prefixes),
        "normalization_cutoff_at": asof - timedelta(minutes=1),
        "normalization_policy": "bars_with_end_strictly_before_decision_asof",
        "added_event_ids_json": json_field(["zone-a"]),
        "invalidated_event_ids_json": json_field([]),
        "observation_transition_json": json_field(transition),
        "scene_graph_delta_json": json_field(graph),
        "context_thesis_json": json_field(
            {
                "lifecycle": "active",
                "source_displacement_id": "disp-a",
                "updated_at": asof.isoformat(),
            }
        ),
        "entry_episode_json": json_field(
            {
                "lifecycle": "active",
                "source_displacement_id": "disp-a",
                "updated_at": asof.isoformat(),
            }
        ),
        "brain_response_json": json_field(
            {"selected_action": "ABSTAIN", "candidate_count": 1}
        ),
        "authority_json": json_field(
            {"authority_direction": "LONG", "authority_timeframe": "4H"}
        ),
        "scale_relations_json": json_field(
            {
                "4H": {"direction": "LONG"},
                "1H": {"direction": "LONG"},
            }
        ),
        "draw_json": json_field({"context_draw": {"lifecycle": "active"}}),
        "blockers_json": json_field({"episode_path_blocker_ids": []}),
        "ambiguities_json": json_field({"belief_unresolved": []}),
        "supporting_evidence_json": json_field(["event-a"]),
        "opposing_evidence_json": json_field([]),
        "entry_location_id": "zone-a",
        "entry_path_id": "path-a",
        "planned_entry": None,
        "entry_zone_lower": None,
        "entry_zone_upper": None,
        "invalidation_price": None,
        "invalidation_source_id": None,
        "primary_target_price": None,
        "primary_target_id": None,
        "planned_deadline_at": asof + timedelta(hours=1),
        "selected_trigger_id": None,
        "selected_trigger_kind": None,
        "selected_trigger_at": None,
        "observed_terminal_reason": None,
        "model_versions_json": json_field({"runtime": "test"}),
    }


_NEUTRAL_TRANSITION_COLLECTIONS = (
    "liquidity_inventory_transitions_this_update",
    "liquidity_pool_transitions_this_update",
    "group3_fvg_transitions_this_update",
    "group3_order_block_transitions_this_update",
    "group4_range_transitions_this_update",
    "group4_manipulation_transitions_this_update",
    "interaction_zone_interactions",
    "interaction_reacceptance_interactions",
    "interaction_micro_break_facts",
    "interaction_paths",
    "interaction_path_transitions",
    "interaction_reacceptance_transitions",
    "interaction_milestone_transitions",
    "interaction_cold_source_ids",
    "interaction_boundary_reasons",
)
_CAUSAL_AGGREGATE_TRANSITION_COLLECTIONS = (
    *_NEUTRAL_TRANSITION_COLLECTIONS[:6],
    *_NEUTRAL_TRANSITION_COLLECTIONS[10:],
)


def test_next_event_vocab_has_only_raw_interaction_authority() -> None:
    raw_interaction_names = _NEUTRAL_TRANSITION_COLLECTIONS[6:]
    assert set(raw_interaction_names).issubset(NEXT_EVENT_TYPE_VOCAB)
    assert tuple(
        NEXT_EVENT_TYPE_VOCAB[name] for name in raw_interaction_names
    ) == tuple(range(11, 20))
    assert {
        name for name in NEXT_EVENT_TYPE_VOCAB if name.startswith("group5_")
    } == set()
    assert {
        name: NEXT_EVENT_TYPE_VOCAB[name]
        for name in ("scene_node_delta", "scene_edge_delta", "scene_resolution")
    } == {
        "scene_node_delta": 20,
        "scene_edge_delta": 21,
        "scene_resolution": 22,
    }


def _canonical_json_text(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _neutral_global_context(
    *,
    asof: pd.Timestamp,
    epoch_id: str,
    location_id: str,
) -> GlobalMarketContext:
    return GlobalMarketContext(
        updated_at=asof,
        scene_revision_id="scene:neutral",
        market_epoch_id=epoch_id,
        authority_stack=(
            AuthorityLayer(
                timeframe=Timeframe.H4,
                direction=Direction.LONG,
                structure_id="authority:neutral",
                confirmed_at=asof - pd.Timedelta(minutes=30),
                protected_level_id=None,
                structural_scope="external",
                acceptance_state="confirmed",
                status="intact",
                source_ids=("authority-source:neutral",),
            ),
        ),
        market_mode=MarketMode.BALANCED,
        scale_relation_details={
            timeframe.value: ScaleRelationState(
                timeframe=timeframe,
                relation=ScaleRelation.UNKNOWN,
                direction=None,
                authority_layer_id=None,
                evidence_ids=(),
                evidence_kind=None,
                structural_scope=None,
                acceptance_state=None,
                since=None,
                age_bars=0,
                graph_connected=False,
                ambiguous=False,
            )
            for timeframe in Timeframe
        },
        external_draw_candidates={"above": (), "below": ()},
        obstruction_views={
            direction.value: DirectionalObstructionView(
                direction=direction,
                nearest_draw_id=None,
                nearest_draw_price=None,
                hard_barriers=(),
                soft_frictions=(),
            )
            for direction in Direction
        },
        material_conflicts=(),
        unknown_evidence=(),
        ambiguous_evidence=(),
        dislocations_by_scale={
            timeframe.value: () for timeframe in Timeframe
        },
        balance_context=None,
        invalidated_source_ids=(),
        candidate_structured_episode_ids=(),
        unexplained_structured_episode_ids=(),
        open_market_theses=(
            OpenMarketThesis(
                thesis_id="thesis:neutral",
                root_id="root:neutral",
                market_epoch_id=epoch_id,
                formed_at=asof - pd.Timedelta(minutes=2),
                updated_at=asof,
                direction=Direction.LONG,
                source_timeframe=Timeframe.M5,
                structural_scale="intermediate",
                mechanism="zone_return",
                authority_relation="aligned",
                authority_source_ids=("authority:neutral",),
                mechanism_event_ids=("root:neutral",),
                entry_location_ids=(location_id,),
            ),
        ),
    )


def _neutral_market_case_row(
    *,
    asof: pd.Timestamp = pd.Timestamp(
        "2024-01-08 09:35",
        tz="America/New_York",
    ),
    epoch_id: str = "market-epoch:test",
) -> dict[str, object]:
    location_id = "location:neutral"
    path_id = "path:neutral"
    episode_id = market_episode_id(
        epoch_id,
        location_id,
        path_id,
        Direction.LONG,
    )
    collections = {name: [] for name in _NEUTRAL_TRANSITION_COLLECTIONS}
    collections["group3_fvg_transitions_this_update"] = [
        to_primitive(
            replace(
                _fvg(asof, identity="fvg:neutral"),
                symbol="NQH4",
                instrument_id=750,
            )
        )
    ]
    observation = {
        "asof": asof.isoformat(),
        "replay_update_ordinal": 5,
        "typed_transition_delta_available": True,
        "collections": collections,
    }
    scene = {
        "asof": asof.isoformat(),
        "replay_update_ordinal": 5,
        "revision_id": "scene:neutral",
        "added_node_ids": ["node:neutral"],
        "revised_node_ids": [],
        "added_edge_ids": [],
        "revised_edge_ids": [],
        "resolution_event_ids": [],
        "relation_descriptors": [],
        "relation_descriptors_complete": True,
    }
    context = to_primitive(
        _neutral_global_context(
            asof=asof,
            epoch_id=epoch_id,
            location_id=location_id,
        )
    )
    prefixes = [
        {
            "timeframe": timeframe,
            "frame_row_start": 0,
            "frame_row_end_exclusive": 8,
            "replay_view_1m_row_start": 0,
            "replay_view_1m_row_end_exclusive": 10,
            "cutoff": asof.isoformat(),
        }
        for timeframe in ("4H", "1H", "15m", "5m", "1m")
    ]
    row: dict[str, object] = {
        "revision_id": "",
        "revision_index": 0,
        "market_epoch_id": epoch_id,
        "market_episode_id": episode_id,
        "asof": asof,
        "direction": "long",
        "lifecycle": "registered",
        "entry_location_id": location_id,
        "entry_path_id": path_id,
        "revision_stage": "market_episode_transition",
        "transition_kinds_json": _canonical_json_text(
            ["episode_created", "zone_registered"]
        ),
        "observation_transition_json": _canonical_json_text(observation),
        "scene_graph_delta_json": _canonical_json_text(scene),
        "neutral_global_context_json": _canonical_json_text(context),
        "ohlcv_prefix_refs_json": _canonical_json_text(prefixes),
        "source_replay_ordinal": 9,
        "replay_update_ordinal": 5,
        "source_bar_synthetic": False,
    }
    row["revision_id"] = market_cases_module._expected_revision_id(row)
    return row


def _neutral_run_manifest(tmp_path: Path) -> dict[str, object]:
    profile_registry = (
        Path(__file__).resolve().parents[2]
        / "shares/configs/market_case_input_profiles_v2.json"
    )
    return {
        "schema_version": 1,
        "runner": "continuous_replay",
        "mode": "market_case_input",
        "runtime_state_schema_version": 8,
        "repository": {"commit": "a" * 40},
        "data_continuity": dict(
            market_cases_module.MARKET_CASE_INPUT_DATA_CONTINUITY_POLICY
        ),
        "profile": {"name": "diagnostic", "identity": "c" * 64},
        "profile_registry": {
            "path": str(profile_registry),
            "sha256": hashlib.sha256(
                profile_registry.read_bytes()
            ).hexdigest(),
            "schema_version": 1,
        },
        "source": {
            "path": str((tmp_path / "source.parquet").resolve()),
            "sha256": "a" * 64,
            "rows": 400,
            "first": pd.Timestamp(
                "2024-01-08 09:00",
                tz="America/New_York",
            ).isoformat(),
            "last": pd.Timestamp(
                "2024-01-08 15:59",
                tz="America/New_York",
            ).isoformat(),
            "last_completed_asof": pd.Timestamp(
                "2024-01-08 16:00",
                tz="America/New_York",
            ).isoformat(),
            "role": "diagnostic",
            "symbol": "NQH4",
            "instrument_id": 750,
        },
        "model_config": {
            "path": str((tmp_path / "model.json").resolve()),
            "sha256": "b" * 64,
            "schema_version": 1,
            "tick_size": 0.25,
            "timezone": "America/New_York",
        },
        "market_case_input_identity": expected_market_case_run_identity(),
        "window": {
            "start": pd.Timestamp(
                "2024-01-08 09:30",
                tz="America/New_York",
            ).isoformat(),
            "end_exclusive": pd.Timestamp(
                "2024-01-08 16:00",
                tz="America/New_York",
            ).isoformat(),
            "role": "diagnostic",
            "warmup_days": 5,
            "observation_clock": "completed_1m_bar_end",
            "capture_interval": "[start,end_exclusive)",
        },
        "output": {
            "stream_families": ["market_case_input_shards"],
            "shard_rows": 256,
            "checkpoint_bars": 100,
        },
    }


def _neutral_checkpoint_metadata() -> dict[str, object]:
    profiles = (
        ("neutral_representation_train_2021_02", "train"),
        ("neutral_representation_validation_2022_05", "validation"),
    )
    input_runs = [
        {
            "profile_name": profile,
            "split_role": role,
            "input_manifest_path": f"/registered/{profile}/input_manifest.json",
            "input_manifest_sha256": input_character * 64,
            "run_manifest_path": f"/registered/{profile}/run_manifest.json",
            "run_manifest_sha256": run_character * 64,
            "repository_commit": "d" * 40,
        }
        for (profile, role), input_character, run_character in zip(
            profiles,
            ("a", "b"),
            ("d", "e"),
            strict=True,
        )
    ]
    embargo = [
        {
            "left_profile": profiles[0][0],
            "right_profile": profiles[1][0],
            "purge_end": "2021-03-15T00:00:00-04:00",
            "next_prefix_start": "2022-04-17T00:00:00-04:00",
            "observed_session_count": 250,
            "first_observed_session": "2021-03-16",
            "last_observed_session": "2022-04-15",
        },
    ]
    return {
        "member_id": "member-000",
        "seed": 73,
        "split_counts": {"train": 100, "validation": 40},
        "lineage": {
            "input_runs": input_runs,
            "source_identity": {"sha256": "1" * 64},
            "model_config_identity": {
                "sha256": "2" * 64,
                "timezone": "America/New_York",
            },
            "market_case_protocol": expected_market_case_run_identity(),
            "representation_feature_schema_version": FEATURE_SCHEMA_VERSION,
            "split_protocol": {
                "registry_sha256": "3" * 64,
                "protocol_version": "neutral-representation-splits-1.0.0",
                "warmup_calendar_days": 14,
                "purge_calendar_days": 14,
                "embargo_trading_days": 5,
                "market_episode_split_key": [
                    "run_manifest_sha256",
                    "market_epoch_id",
                    "market_episode_id",
                ],
                "actual_prefix_exposure_verified": True,
                "observed_completed_session_embargo": embargo,
            },
        },
        "outcome_fields_used": False,
        "model_capability_validated": False,
    }


def _neutral_revision_row(
    *,
    asof: pd.Timestamp,
    revision_index: int,
    source_replay_ordinal: int,
    replay_update_ordinal: int,
    lifecycle: str,
    transition_kinds: tuple[str, ...],
    epoch_id: str = "market-epoch:test",
    location_id: str = "location:neutral",
    path_id: str = "path:neutral",
) -> dict[str, object]:
    row = _neutral_market_case_row(asof=asof, epoch_id=epoch_id)
    row["revision_index"] = revision_index
    row["source_replay_ordinal"] = source_replay_ordinal
    row["replay_update_ordinal"] = replay_update_ordinal
    row["lifecycle"] = lifecycle
    row["transition_kinds_json"] = _canonical_json_text(transition_kinds)
    row["entry_location_id"] = location_id
    row["entry_path_id"] = path_id
    row["market_episode_id"] = market_episode_id(
        epoch_id,
        location_id,
        path_id,
        Direction.LONG,
    )

    observation = json.loads(str(row["observation_transition_json"]))
    observation["replay_update_ordinal"] = replay_update_ordinal
    row["observation_transition_json"] = _canonical_json_text(observation)

    scene = json.loads(str(row["scene_graph_delta_json"]))
    scene_revision_id = f"scene:neutral:{replay_update_ordinal}"
    scene["replay_update_ordinal"] = replay_update_ordinal
    scene["revision_id"] = scene_revision_id
    row["scene_graph_delta_json"] = _canonical_json_text(scene)

    context = json.loads(str(row["neutral_global_context_json"]))
    context["scene_revision_id"] = scene_revision_id
    row["neutral_global_context_json"] = _canonical_json_text(context)

    prefixes = json.loads(str(row["ohlcv_prefix_refs_json"]))
    for prefix in prefixes:
        prefix["replay_view_1m_row_end_exclusive"] = source_replay_ordinal + 1
    row["ohlcv_prefix_refs_json"] = _canonical_json_text(prefixes)
    row["revision_id"] = market_cases_module._expected_revision_id(row)
    return row


def _with_neutral_scale_details(
    row: dict[str, object],
    *,
    directions: tuple[str, ...],
    ambiguous: bool = False,
) -> dict[str, object]:
    result = dict(row)
    context = json.loads(str(result["neutral_global_context_json"]))
    detail_timeframes = ("4H", "1H", "15m", "5m", "1m")
    reference_direction = directions[0] if directions else None
    context["scale_relation_details"].update({
        timeframe: {
            "timeframe": timeframe,
            "relation": (
                "unknown"
                if ambiguous
                else (
                    "aligned"
                    if direction == reference_direction
                    else "normal_pullback"
                )
            ),
            "direction": None if ambiguous else direction,
            "authority_layer_id": "authority:neutral",
            "evidence_ids": [f"evidence:{timeframe}"],
            "evidence_kind": "structure",
            "structural_scope": "intermediate",
            "acceptance_state": "confirmed",
            "since": (
                pd.Timestamp(result["asof"]) - pd.Timedelta(minutes=1)
            ).isoformat(),
            "age_bars": 0,
            "graph_connected": not ambiguous,
            "ambiguous": ambiguous,
        }
        for timeframe, direction in zip(
            detail_timeframes,
            directions,
            strict=False,
        )
    })
    result["neutral_global_context_json"] = _canonical_json_text(context)
    result["revision_id"] = market_cases_module._expected_revision_id(result)
    return result


def _neutral_store(case: RepresentationCase) -> CanonicalOHLCVStore:
    frequencies = {
        "4h": pd.Timedelta(hours=4),
        "1h": pd.Timedelta(hours=1),
        "15m": pd.Timedelta(minutes=15),
        "5m": pd.Timedelta(minutes=5),
        "1m": pd.Timedelta(minutes=1),
    }
    frames = {}
    ticks = {}
    availability = {}
    for timeframe, frequency in frequencies.items():
        key = CanonicalSourceKey(
            case.market_epoch_id,
            timeframe,
            "a" * 64,
        )
        index = pd.date_range(
            end=case.asof + 2 * frequency,
            periods=12,
            freq=frequency,
        )
        frames[key] = _frame(timeframe).set_axis(index)
        ticks[key] = 0.25
        availability[key] = BAR_END_INDEX_BINDING
    return CanonicalOHLCVStore(
        frames,
        tick_sizes=ticks,
        availability_bindings=availability,
        normalization_window=3,
    )


def _with_complete_interval_ledger(
    row: dict[str, object],
    *,
    previous_asof: pd.Timestamp,
    collections: dict[str, list[dict[str, object]]],
    scene_added_node_ids: tuple[str, ...] = (),
    scene_resolution_event_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    result = dict(row)
    asof = pd.Timestamp(result["asof"])
    unknown_collections = set(collections).difference(
        _NEUTRAL_TRANSITION_COLLECTIONS
    )
    if unknown_collections:
        raise AssertionError(
            f"unknown fixture collections: {sorted(unknown_collections)}"
        )
    update_collections = {
        name: list(collections.get(name, ()))
        for name in _NEUTRAL_TRANSITION_COLLECTIONS
    }
    aggregate_collections = {
        name: list(update_collections[name])
        for name in _CAUSAL_AGGREGATE_TRANSITION_COLLECTIONS
    }
    eventful = bool(
        any(aggregate_collections.values())
        or scene_added_node_ids
        or scene_resolution_event_ids
    )
    end_ordinal = int(result["revision_index"])
    start_ordinal = max(0, end_ordinal - 1)
    coverage = {
        "complete": True,
        "coverage_start_at": previous_asof.isoformat(),
        "coverage_start_exclusive": True,
        "coverage_end_at": asof.isoformat(),
        "coverage_start_replay_update_ordinal": start_ordinal,
        "coverage_end_replay_update_ordinal": end_ordinal,
        "observed_update_count": 1,
        "eventful_update_count": 1 if eventful else 0,
        "last_observed_update_at": asof.isoformat(),
        "last_observed_replay_update_ordinal": end_ordinal,
        "all_typed_deltas_available": True,
        "gap_free": True,
    }
    transition_updates = (
        [
            {
                "asof": asof.isoformat(),
                "replay_update_ordinal": end_ordinal,
                "typed_transition_delta_available": True,
                "collections": update_collections,
            }
        ]
        if eventful
        else []
    )
    scene_updates = (
        [
            {
                "asof": asof.isoformat(),
                "replay_update_ordinal": end_ordinal,
                "revision_id": "scene-interval",
                "added_node_ids": list(scene_added_node_ids),
                "revised_node_ids": [],
                "added_edge_ids": [],
                "revised_edge_ids": [],
                "resolution_event_ids": list(
                    scene_resolution_event_ids
                ),
                "relation_descriptors": [],
                "relation_descriptors_complete": True,
            }
        ]
        if eventful
        else []
    )
    result["observation_transition_json"] = _canonical_json_text(
        {
            "typed_transition_delta_available": True,
            "coverage": coverage,
            "collections": aggregate_collections,
            "updates": transition_updates,
        },
    )
    result["scene_graph_delta_json"] = _canonical_json_text(
        {
            "asof": asof.isoformat(),
            "revision_id": "scene-interval",
            "coverage": coverage,
            "added_node_ids": list(scene_added_node_ids),
            "revised_node_ids": [],
            "added_edge_ids": [],
            "revised_edge_ids": [],
            "resolution_event_ids": list(scene_resolution_event_ids),
            "relation_descriptors": [],
            "relation_descriptors_complete": True,
            "updates": scene_updates,
        },
    )
    return result


def test_prefix_features_are_causal_and_exclude_absolute_prices() -> None:
    case = _case()
    prepared = prepare_representation_case(case, _store())

    assert set(prepared.timeframe_features) == set(TIMEFRAMES)
    assert set(CAUSAL_CANDLE_FEATURES).isdisjoint(
        {"open", "high", "low", "close", "absolute_price"}
    )
    for matrix in prepared.timeframe_features.values():
        assert matrix.shape[1] == len(CAUSAL_CANDLE_FEATURES)
        assert np.isfinite(matrix).all()


def test_future_bar_mutation_does_not_change_prefix_features() -> None:
    case = _case()
    original = prepare_representation_case(case, _store())
    future_mutated = prepare_representation_case(case, _store(mutate_future=True))

    for timeframe in TIMEFRAMES:
        np.testing.assert_array_equal(
            original.timeframe_features[timeframe],
            future_mutated.timeframe_features[timeframe],
        )


def test_price_scale_does_not_change_normalized_features() -> None:
    case = _case()
    original = prepare_representation_case(case, _store(scale=1.0))
    rescaled = prepare_representation_case(case, _store(scale=10.0))

    for timeframe in TIMEFRAMES:
        np.testing.assert_allclose(
            original.timeframe_features[timeframe],
            rescaled.timeframe_features[timeframe],
            rtol=1e-5,
            atol=1e-5,
        )


def test_prefix_rejects_any_row_available_after_asof() -> None:
    case = _case()
    prefix = case.prefixes["1m"]
    invalid = replace(
        case,
        prefixes={
            **case.prefixes,
            "1m": replace(prefix, row_end_exclusive=prefix.row_end_exclusive + 1),
        },
    )

    with pytest.raises(RepresentationDataError, match="must be <= case.asof"):
        prepare_representation_case(invalid, _store())


def test_case_library_adapter_prepares_without_manual_field_renaming() -> None:
    case = representation_case_from_case_input_row(_case_library_row())
    frames = {}
    ticks = {}
    availability = {}
    for timeframe in TIMEFRAMES:
        key = CanonicalSourceKey("epoch-a", timeframe, "a" * 64)
        frames[key] = _frame(timeframe)
        ticks[key] = 0.25
        availability[key] = BAR_END_INDEX_BINDING
    store = CanonicalOHLCVStore(
        frames,
        tick_sizes=ticks,
        availability_bindings=availability,
        normalization_window=3,
    )

    prepared = prepare_representation_case(case, store)

    assert case.revision_stage == "episode_created"
    assert case.revision_index == 0
    assert case.market_episode_id == case.entry_episode_id
    assert prepared.feature_max_at <= case.asof
    assert prepared.event_type_ids.size > 0


def test_neutral_adapter_binds_market_episode_eye_scene_context_and_prefixes(
    tmp_path: Path,
) -> None:
    row = _neutral_market_case_row()
    manifest = _neutral_run_manifest(tmp_path)

    mapping = market_case_input_to_representation_mapping(row, manifest)
    case = representation_case_from_market_case_input_row(row, manifest)

    assert mapping["canonical_source_id"] == "a" * 64
    assert mapping["source_path"] == manifest["source"]["path"]
    assert mapping["source_role"] == manifest["source"]["role"]
    assert mapping["symbol"] == manifest["source"]["symbol"]
    assert mapping["instrument_id"] == manifest["source"]["instrument_id"]
    assert mapping["model_config_path"] == manifest["model_config"]["path"]
    assert mapping["model_config_sha256"] == "b" * 64
    assert mapping["tick_size"] == 0.25
    assert mapping["timezone"] == "America/New_York"
    assert case.market_episode_id == row["market_episode_id"]
    assert case.entry_episode_id == row["market_episode_id"]
    assert case.entry_location_id == row["entry_location_id"]
    assert case.entry_path_id == row["entry_path_id"]
    assert case.transition_kinds == ("episode_created", "zone_registered")
    assert case.revision_stage == "market_episode_transition"
    assert case.regime == "balance"
    assert case.mechanism_label == "unknown"
    assert case.authority_direction == 1
    assert case.canonical_source_path == manifest["source"]["path"]
    assert case.symbol == "NQH4"
    assert case.instrument_id == 750
    assert set(case.prefixes) == set(TIMEFRAMES)
    assert all(
        prefix.market_epoch_id == row["market_epoch_id"]
        and prefix.canonical_source_id == "a" * 64
        and prefix.row_start == 0
        and prefix.row_end_exclusive == 8
        and prefix.tail_at_or_before == case.asof
        for prefix in case.prefixes.values()
    )
    event_types = {event.event_type for event in case.events}
    assert "market_episode_transition:episode_created" in event_types
    assert "market_episode_transition:zone_registered" in event_types
    assert "group3_fvg_transitions_this_update" in event_types
    assert "graph_delta:added_node_ids:count=1" in event_types
    assert "neutral_global_context:market_mode=balanced" in event_types
    assert all(
        marker not in event_type
        for event_type in event_types
        for marker in ("playbook", "shadow", "outcome", "pnl")
    )


def test_neutral_scene_descriptor_never_tokenizes_brain_vocabulary(
    tmp_path: Path,
) -> None:
    row = _neutral_market_case_row()
    scene = json.loads(str(row["scene_graph_delta_json"]))
    scene["added_edge_ids"] = ["edge:physical"]
    scene["relation_descriptors"] = [
        {
            "change_kind": "added",
            "edge_id": "edge:physical",
            "relation": SceneEdgeKind.SWEEPS.value,
            "lifecycle": "active",
            "observed_at": pd.Timestamp(row["asof"]).isoformat(),
            "source": {
                "node_id": "node:source",
                "kind": "manipulation",
                "role": "manipulation",
                "timeframe": Timeframe.M1.value,
                "structural_scale": StructuralScale.INTERNAL.value,
                "lifecycle": "swept",
            },
            "target": {
                "node_id": "node:target",
                "kind": "liquidity",
                "role": "liquidity",
                "timeframe": Timeframe.M1.value,
                "structural_scale": StructuralScale.INTERNAL.value,
                "lifecycle": "consumed",
            },
        }
    ]
    valid = dict(row)
    valid["scene_graph_delta_json"] = _canonical_json_text(scene)
    valid["revision_id"] = market_cases_module._expected_revision_id(valid)
    case = representation_case_from_market_case_input_row(
        valid,
        _neutral_run_manifest(tmp_path),
    )
    relation_event = next(
        event
        for event in case.events
        if event.event_type.startswith("scene_relation:")
    )
    tokens = "|".join(
        (
            relation_event.event_type,
            relation_event.lifecycle,
            *relation_event.relation_types,
        )
    ).lower()
    assert not any(
        term in tokens
        for term in (
            "playbook",
            "qualified",
            "decision",
            "selected_action",
            "risk",
            "hard_gate",
        )
    )

    attacks = (
        (("relation",), "playbook"),
        (("lifecycle",), "qualified"),
        (("source", "kind"), "decision"),
        (("source", "role"), "selected_action"),
        (("target", "kind"), "risk"),
        (("target", "role"), "hard_gate"),
    )
    for field_path, brain_value in attacks:
        attacked_scene = json.loads(str(valid["scene_graph_delta_json"]))
        descriptor = attacked_scene["relation_descriptors"][0]
        if len(field_path) == 1:
            descriptor[field_path[0]] = brain_value
        else:
            descriptor[field_path[0]][field_path[1]] = brain_value
        attacked = dict(valid)
        attacked["scene_graph_delta_json"] = _canonical_json_text(
            attacked_scene
        )
        attacked["revision_id"] = market_cases_module._expected_revision_id(
            attacked
        )
        with pytest.raises(RepresentationDataError, match="input row is invalid"):
            representation_case_from_market_case_input_row(
                attacked,
                _neutral_run_manifest(tmp_path),
            )


def test_neutral_adapter_requires_repository_identity_without_tokenizing_it(
    tmp_path: Path,
) -> None:
    row = _neutral_market_case_row()
    manifest = _neutral_run_manifest(tmp_path)
    mapping = market_case_input_to_representation_mapping(row, manifest)
    assert "repository" not in mapping

    missing = dict(manifest)
    missing.pop("repository")
    with pytest.raises(
        RepresentationDataError,
        match="runtime/continuity|manifest schema",
    ):
        market_case_input_to_representation_mapping(row, missing)


def test_neutral_adapter_accepts_schema_eight_data_continuity_without_tokenizing_it(
    tmp_path: Path,
) -> None:
    row = _neutral_market_case_row()
    manifest = _neutral_run_manifest(tmp_path)
    mapping = market_case_input_to_representation_mapping(row, manifest)
    assert "data_continuity" not in mapping


@pytest.mark.parametrize(
    "repository",
    (
        {},
        {"commit": "a" * 39},
        {"commit": "A" * 40},
        {"commit": "g" * 40},
        {"commit": "a" * 40, "branch": "main"},
    ),
)
def test_neutral_adapter_rejects_invalid_repository_identity(
    tmp_path: Path,
    repository: object,
) -> None:
    manifest = _neutral_run_manifest(tmp_path)
    manifest["repository"] = repository
    with pytest.raises(RepresentationDataError, match="repository identity"):
        representation_case_from_market_case_input_row(
            _neutral_market_case_row(), manifest
        )


@pytest.mark.parametrize(
    ("runtime_schema", "repository", "data_continuity"),
    (
        (5, {"commit": "a" * 40}, None),
        (6, {"commit": "a" * 40}, None),
        (7, {"commit": "a" * 40}, {}),
        (8, None, {}),
        (8, {"commit": "a" * 40}, None),
    ),
)
def test_neutral_adapter_rejects_unbound_runtime_continuity_versions(
    tmp_path: Path,
    runtime_schema: int,
    repository: object | None,
    data_continuity: object | None,
) -> None:
    manifest = _neutral_run_manifest(tmp_path)
    manifest["runtime_state_schema_version"] = runtime_schema
    if repository is None:
        manifest.pop("repository")
    else:
        manifest["repository"] = repository
    if data_continuity is None:
        manifest.pop("data_continuity")
    else:
        manifest["data_continuity"] = data_continuity
    with pytest.raises(
        RepresentationDataError,
        match="runtime/continuity|manifest schema",
    ):
        representation_case_from_market_case_input_row(
            _neutral_market_case_row(), manifest
        )


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("maximum_no_trade_gap_minutes", 6),
        ("maximum_no_trade_gap_minutes", 5.0),
        ("allow_same_contract_data_gap_reset", False),
        ("allow_same_contract_data_gap_reset", 1),
        ("data_gap_reset_anomaly", "other"),
        ("allow_cross_contract_data_gap_reset", True),
        ("synthesize_over_cap_missing_minutes", True),
        ("unexpected", False),
    ),
)
def test_neutral_adapter_rejects_schema_eight_data_continuity_tamper(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    manifest = _neutral_run_manifest(tmp_path)
    manifest.update(
        {
            "runtime_state_schema_version": 8,
            "repository": {"commit": "a" * 40},
            "data_continuity": {
                **market_cases_module.MARKET_CASE_INPUT_DATA_CONTINUITY_POLICY,
                key: value,
            },
        }
    )
    with pytest.raises(RepresentationDataError, match="continuity policy"):
        representation_case_from_market_case_input_row(
            _neutral_market_case_row(), manifest
        )


@pytest.mark.parametrize(
    ("directions", "alignment"),
    (
        ((Direction.LONG,), "aligned"),
        ((Direction.SHORT,), "opposed"),
        (
            (Direction.LONG, Direction.SHORT),
            "ambiguous_same_clock",
        ),
    ),
)
def test_neutral_adapter_preserves_legal_micro_bos_reference_alignment_as_relation(
    tmp_path: Path,
    directions: tuple[Direction, ...],
    alignment: str,
) -> None:
    update, asof = _strict_interaction(*directions)
    row = _neutral_market_case_row(asof=asof)
    observation = json.loads(str(row["observation_transition_json"]))
    observation["collections"].update(
        to_primitive(dict(interaction_artifact_collections(update)))
    )
    row["observation_transition_json"] = _canonical_json_text(observation)
    row["revision_id"] = market_cases_module._expected_revision_id(row)

    manifest = _neutral_run_manifest(tmp_path)
    manifest["source"].update(
        {
            "first": (asof - pd.Timedelta(minutes=33)).isoformat(),
            "last": (asof + pd.Timedelta(hours=6)).isoformat(),
            "last_completed_asof": (
                asof + pd.Timedelta(hours=6, minutes=1)
            ).isoformat(),
        }
    )
    manifest["window"].update(
        {
            "start": (asof - pd.Timedelta(minutes=3)).isoformat(),
            "end_exclusive": (asof + pd.Timedelta(hours=6)).isoformat(),
        }
    )
    mapping = market_case_input_to_representation_mapping(
        row,
        manifest,
    )
    case = RepresentationCase.from_mapping(mapping)
    reference_id = update.micro_break_facts[0].reference_id
    event = next(
        value for value in case.events if value.event_id == reference_id
    )

    assert (
        f"micro_bos_reference_alignment:{alignment}" in event.relation_types
    )
    assert "\"outcome\"" not in _canonical_json_text(mapping["events"])
    assert "\"outcome\"" not in str(row["observation_transition_json"])


@pytest.mark.parametrize("tamper", ("path", "type", "value"))
def test_neutral_adapter_rejects_non_whitelisted_micro_bos_outcome(
    tmp_path: Path,
    tamper: str,
) -> None:
    row = _neutral_market_case_row()
    observation = json.loads(str(row["observation_transition_json"]))
    collection = (
        "interaction_paths"
        if tamper == "path"
        else "interaction_micro_break_facts"
    )
    outcome: object = (
        {"aligned": True}
        if tamper == "type"
        else ("profitable" if tamper == "value" else "aligned")
    )
    observation["collections"][collection].append(
        {
            "event_id": "tampered:neutral",
            "kind": "micro_bos_reference",
            "lifecycle": "active",
            "observed_at": pd.Timestamp(row["asof"]).isoformat(),
            "direction": "long",
            "timeframe": "1m",
            "outcome": outcome,
        }
    )
    row["observation_transition_json"] = _canonical_json_text(observation)
    row["revision_id"] = market_cases_module._expected_revision_id(row)

    with pytest.raises(RepresentationDataError, match="input row is invalid"):
        representation_case_from_market_case_input_row(
            row,
            _neutral_run_manifest(tmp_path),
        )


def test_neutral_sparse_targets_use_next_same_episode_lifecycle_only(
    tmp_path: Path,
) -> None:
    base = pd.Timestamp("2024-01-08 09:35", tz="America/New_York")
    registered = _with_neutral_scale_details(
        _neutral_revision_row(
            asof=base,
            revision_index=0,
            source_replay_ordinal=9,
            replay_update_ordinal=5,
            lifecycle="registered",
            transition_kinds=("episode_created", "zone_registered"),
        ),
        directions=("long", "long"),
    )
    pullback = _with_neutral_scale_details(
        _neutral_revision_row(
            asof=base + pd.Timedelta(minutes=1),
            revision_index=1,
            source_replay_ordinal=10,
            replay_update_ordinal=6,
            lifecycle="pullback",
            transition_kinds=("first_pullback",),
        ),
        directions=("long", "long"),
    )
    triggered = _with_neutral_scale_details(
        _neutral_revision_row(
            asof=base + pd.Timedelta(minutes=2),
            revision_index=2,
            source_replay_ordinal=11,
            replay_update_ordinal=7,
            lifecycle="triggered",
            transition_kinds=("trigger",),
        ),
        directions=("long", "long"),
    )
    terminal = _with_neutral_scale_details(
        _neutral_revision_row(
            asof=base + pd.Timedelta(minutes=3),
            revision_index=3,
            source_replay_ordinal=12,
            replay_update_ordinal=8,
            lifecycle="terminal",
            transition_kinds=("terminal",),
        ),
        directions=("long", "long"),
    )

    built = build_neutral_market_revision_targets(
        (terminal, registered, triggered, pullback),
        _neutral_run_manifest(tmp_path),
    )

    assert NEUTRAL_MARKET_TRANSITION_KINDS == (
        "episode_created",
        "zone_registered",
        "first_pullback",
        "trigger",
        "successful_pulse",
        "terminal",
    )
    first = built[str(registered["revision_id"])]
    second = built[str(pullback["revision_id"])]
    third = built[str(triggered["revision_id"])]
    last = built[str(terminal["revision_id"])]
    assert first.market_episode_id == registered["market_episode_id"]
    assert first.next_revision_id == pullback["revision_id"]
    assert first.target.next_lifecycle == NEUTRAL_MARKET_LIFECYCLE_TARGETS[
        "pullback"
    ]
    assert second.next_revision_id == triggered["revision_id"]
    assert second.target.next_lifecycle == NEUTRAL_MARKET_LIFECYCLE_TARGETS[
        "triggered"
    ]
    assert third.next_revision_id == terminal["revision_id"]
    assert third.target.next_lifecycle == NEUTRAL_MARKET_LIFECYCLE_TARGETS[
        "terminal"
    ]
    assert last.next_revision_id is None
    assert last.target.next_lifecycle == -100
    assert first.label_max_observed_at == pd.Timestamp(pullback["asof"])
    assert last.label_max_observed_at == pd.Timestamp(terminal["asof"])
    for record in built.values():
        assert record.target.scale_direction_alignment == 1
        assert all(
            getattr(record.target, name) == -100
            for name in NEUTRAL_SPARSE_DISABLED_TARGETS
        )
        assert record.as_dict()["active_tasks"] == list(
            NEUTRAL_SPARSE_ACTIVE_TARGETS
        )


@pytest.mark.parametrize(
    ("directions", "ambiguous", "expected"),
    (
        (("long", "long"), False, 1),
        (("long", "short"), False, 2),
        (("long",), False, 0),
        (("long", "long"), True, 0),
    ),
)
def test_neutral_sparse_scale_alignment_uses_only_active_same_clock_details(
    tmp_path: Path,
    directions: tuple[str, ...],
    ambiguous: bool,
    expected: int,
) -> None:
    row = _with_neutral_scale_details(
        _neutral_market_case_row(),
        directions=directions,
        ambiguous=ambiguous,
    )
    if expected == 1:
        context = json.loads(str(row["neutral_global_context_json"]))
        context["ambiguous_evidence"] = ["historical:not-a-current-scale-fact"]
        row["neutral_global_context_json"] = _canonical_json_text(context)
        row["revision_id"] = market_cases_module._expected_revision_id(row)

    record = build_neutral_market_revision_targets(
        (row,),
        _neutral_run_manifest(tmp_path),
    )[str(row["revision_id"])]

    assert record.target.scale_direction_alignment == expected


def test_neutral_sparse_targets_do_not_cross_episode_or_epoch(
    tmp_path: Path,
) -> None:
    base = pd.Timestamp("2024-01-08 09:35", tz="America/New_York")
    first = _neutral_revision_row(
        asof=base,
        revision_index=0,
        source_replay_ordinal=9,
        replay_update_ordinal=5,
        lifecycle="registered",
        transition_kinds=("episode_created", "zone_registered"),
    )
    same_episode_next = _neutral_revision_row(
        asof=base + pd.Timedelta(minutes=1),
        revision_index=1,
        source_replay_ordinal=10,
        replay_update_ordinal=6,
        lifecycle="pullback",
        transition_kinds=("first_pullback",),
    )
    other_episode = _neutral_revision_row(
        asof=base + pd.Timedelta(minutes=2),
        revision_index=0,
        source_replay_ordinal=11,
        replay_update_ordinal=7,
        lifecycle="registered",
        transition_kinds=("episode_created", "zone_registered"),
        location_id="location:other",
        path_id="path:other",
    )
    other_epoch = _neutral_revision_row(
        asof=base + pd.Timedelta(minutes=3),
        revision_index=0,
        source_replay_ordinal=12,
        replay_update_ordinal=8,
        lifecycle="registered",
        transition_kinds=("episode_created", "zone_registered"),
        epoch_id="market-epoch:other",
    )

    built = build_neutral_market_revision_targets(
        (other_epoch, same_episode_next, other_episode, first),
        _neutral_run_manifest(tmp_path),
    )

    assert built[str(first["revision_id"])].next_revision_id == (
        same_episode_next["revision_id"]
    )
    for row in (same_episode_next, other_episode, other_epoch):
        assert built[str(row["revision_id"])].next_revision_id is None
        assert built[str(row["revision_id"])].target.next_lifecycle == -100


def test_neutral_sparse_targets_reject_discontinuous_episode_revisions(
    tmp_path: Path,
) -> None:
    base = pd.Timestamp("2024-01-08 09:35", tz="America/New_York")
    first = _neutral_revision_row(
        asof=base,
        revision_index=0,
        source_replay_ordinal=9,
        replay_update_ordinal=5,
        lifecycle="registered",
        transition_kinds=("episode_created", "zone_registered"),
    )
    skipped = _neutral_revision_row(
        asof=base + pd.Timedelta(minutes=1),
        revision_index=2,
        source_replay_ordinal=10,
        replay_update_ordinal=6,
        lifecycle="pullback",
        transition_kinds=("first_pullback",),
    )

    with pytest.raises(RepresentationDataError, match="target rows are invalid"):
        build_neutral_market_revision_targets(
            (first, skipped),
            _neutral_run_manifest(tmp_path),
        )


def test_neutral_scale_alignment_tokens_are_shortcut_masked(
    tmp_path: Path,
) -> None:
    row = _with_neutral_scale_details(
        _neutral_market_case_row(),
        directions=("long", "long"),
    )
    case = representation_case_from_market_case_input_row(
        row,
        _neutral_run_manifest(tmp_path),
    )
    prepared = prepare_representation_case(case, _neutral_store(case))
    scale_indices = tuple(
        index
        for index, event in enumerate(case.events)
        if event.event_type.startswith(
            "neutral_global_context:scale_relation_details."
        )
    )

    assert scale_indices
    assert all(
        bool(prepared.direct_label_source_event_mask[index])
        for index in scale_indices
    )
    masked = mask_direct_label_source_tokens(prepared)
    assert masked.label_sources_masked
    assert not masked.direct_label_source_event_mask.any()
    assert len(masked.event_type_ids) < len(prepared.event_type_ids)


def test_neutral_prefix_reloads_snapshot_tail_from_longer_canonical_views(
    tmp_path: Path,
) -> None:
    case = representation_case_from_market_case_input_row(
        _neutral_market_case_row(),
        _neutral_run_manifest(tmp_path),
    )
    frequencies = {
        "4h": pd.Timedelta(hours=4),
        "1h": pd.Timedelta(hours=1),
        "15m": pd.Timedelta(minutes=15),
        "5m": pd.Timedelta(minutes=5),
        "1m": pd.Timedelta(minutes=1),
    }
    frames = {}
    ticks = {}
    availability = {}
    for timeframe, frequency in frequencies.items():
        key = CanonicalSourceKey(
            str(case.market_epoch_id),
            timeframe,
            "a" * 64,
        )
        index = pd.date_range(
            end=case.asof + 2 * frequency,
            periods=12,
            freq=frequency,
        )
        frames[key] = _frame(timeframe).set_axis(index)
        ticks[key] = 0.25
        availability[key] = BAR_END_INDEX_BINDING
    store = CanonicalOHLCVStore(
        frames,
        tick_sizes=ticks,
        availability_bindings=availability,
        normalization_window=3,
    )

    for timeframe in TIMEFRAMES:
        tail_prefix = case.prefixes[timeframe]
        resolved = store.read_prefix(tail_prefix, asof=case.asof)
        direct = store.read_prefix(
            PrefixIndexRange(
                market_epoch_id=case.market_epoch_id,
                timeframe=timeframe,
                canonical_source_id="a" * 64,
                row_start=2,
                row_end_exclusive=10,
            ),
            asof=case.asof,
        )
        assert resolved.row_start == 2
        assert resolved.row_end_exclusive == 10
        assert resolved.last_available_at == case.asof
        np.testing.assert_array_equal(resolved.values, direct.values)


def test_neutral_adapter_does_not_import_legacy_causal_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if str(name).endswith("causal_cases"):
            raise AssertionError("neutral adapter imported causal_cases")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    case = representation_case_from_market_case_input_row(
        _neutral_market_case_row(),
        _neutral_run_manifest(tmp_path),
    )
    assert case.market_episode_id


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("source", "sha256", "not-a-hash"),
        ("source", "symbol", ""),
        ("source", "instrument_id", True),
        ("model_config", "tick_size", None),
        ("window", "observation_clock", "wall_clock"),
        ("root", "market_case_input_identity", {}),
    ),
)
def test_neutral_adapter_fails_closed_on_run_manifest_tamper(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
) -> None:
    manifest = _neutral_run_manifest(tmp_path)
    if section == "root":
        manifest[key] = value
    else:
        manifest[section][key] = value
    with pytest.raises(RepresentationDataError, match="market case run"):
        representation_case_from_market_case_input_row(
            _neutral_market_case_row(),
            manifest,
        )


def test_neutral_row_cannot_supply_source_or_model_config(
    tmp_path: Path,
) -> None:
    row = _neutral_market_case_row()
    row["source_path"] = "/tmp/row-controlled-source.parquet"
    with pytest.raises(RepresentationDataError, match="input row is invalid"):
        representation_case_from_market_case_input_row(
            row,
            _neutral_run_manifest(tmp_path),
        )


@pytest.mark.parametrize(
    "forbidden_key",
    (
        "brain_response",
        "future_outcome",
        "playbook",
        "pnl",
        "qualified",
        "shadow_snapshot",
    ),
)
def test_neutral_adapter_rejects_control_shadow_and_economic_keys(
    tmp_path: Path,
    forbidden_key: str,
) -> None:
    row = _neutral_market_case_row()
    context = json.loads(str(row["neutral_global_context_json"]))
    context["tamper"] = {forbidden_key: "must-not-enter-model"}
    row["neutral_global_context_json"] = _canonical_json_text(context)
    row["revision_id"] = market_cases_module._expected_revision_id(row)

    with pytest.raises(RepresentationDataError):
        representation_case_from_market_case_input_row(
            row,
            _neutral_run_manifest(tmp_path),
        )


@pytest.mark.parametrize(
    "observed_at",
    (
        "2024-01-08T09:34:00",
        "2024-01-08T09:36:00-05:00",
    ),
)
def test_neutral_adapter_rejects_naive_or_future_nested_clocks(
    tmp_path: Path,
    observed_at: str,
) -> None:
    row = _neutral_market_case_row()
    context = json.loads(str(row["neutral_global_context_json"]))
    context["clock_probe"] = {"observed_at": observed_at}
    row["neutral_global_context_json"] = _canonical_json_text(context)
    row["revision_id"] = market_cases_module._expected_revision_id(row)

    with pytest.raises(RepresentationDataError, match="input row is invalid"):
        representation_case_from_market_case_input_row(
            row,
            _neutral_run_manifest(tmp_path),
        )


def test_neutral_adapter_rejects_empty_timeframe_prefix(
    tmp_path: Path,
) -> None:
    row = _neutral_market_case_row()
    prefixes = json.loads(str(row["ohlcv_prefix_refs_json"]))
    prefixes[0]["frame_row_end_exclusive"] = 0
    row["ohlcv_prefix_refs_json"] = _canonical_json_text(prefixes)
    row["revision_id"] = market_cases_module._expected_revision_id(row)

    with pytest.raises(RepresentationDataError, match="prefix row range"):
        representation_case_from_market_case_input_row(
            row,
            _neutral_run_manifest(tmp_path),
        )


def test_neutral_adapter_rejects_cross_timeframe_replay_view_lineage(
    tmp_path: Path,
) -> None:
    row = _neutral_market_case_row()
    prefixes = json.loads(str(row["ohlcv_prefix_refs_json"]))
    prefixes[0]["replay_view_1m_row_start"] = 1
    row["ohlcv_prefix_refs_json"] = _canonical_json_text(prefixes)
    row["revision_id"] = market_cases_module._expected_revision_id(row)

    with pytest.raises(RepresentationDataError, match="replay-view lineage"):
        representation_case_from_market_case_input_row(
            row,
            _neutral_run_manifest(tmp_path),
        )


def test_split_integrity_uses_market_epoch_and_market_episode_pair() -> None:
    first = replace(
        _case(),
        market_episode_id="market-episode:shared",
    )
    second_epoch = "epoch-b"
    second = replace(
        first,
        case_id="case-b",
        revision_id="revision-b",
        market_epoch_id=second_epoch,
        prefixes={
            timeframe: replace(prefix, market_epoch_id=second_epoch)
            for timeframe, prefix in first.prefixes.items()
        },
        events=tuple(
            replace(event, market_epoch_id=second_epoch)
            for event in first.events
        ),
    )
    validate_split_integrity(
        (first, second),
        {first.revision_id: "train", second.revision_id: "validation"},
    )

    same_pair_later = replace(
        first,
        case_id="case-c",
        revision_id="revision-c",
        asof=first.asof + pd.Timedelta(minutes=1),
    )
    with pytest.raises(RepresentationDataError, match="MarketEpisode pair"):
        validate_split_integrity(
            (first, same_pair_later),
            {
                first.revision_id: "train",
                same_pair_later.revision_id: "validation",
            },
        )


def test_case_library_authority_direction_is_not_fabricated_from_thesis() -> None:
    long_case = representation_case_from_case_input_row(_case_library_row())
    short_row = _case_library_row(revision_id="revision-library-short")
    short_row["authority_json"] = _canonical_json_text(
        {"authority_direction": "SHORT", "authority_timeframe": "4H"}
    )
    short_case = representation_case_from_case_input_row(short_row)
    unknown_row = _case_library_row(revision_id="revision-library-unknown")
    unknown_row["authority_json"] = _canonical_json_text({})
    unknown_case = representation_case_from_case_input_row(unknown_row)

    assert long_case.direction == short_case.direction == unknown_case.direction == 1
    assert long_case.authority_direction == 1
    assert short_case.authority_direction == -1
    assert unknown_case.authority_direction == 0


def test_graph_encoder_preserves_typed_relation_topology_not_only_counts() -> None:
    first_row = _case_library_row()
    second_row = dict(first_row)
    second_graph = json.loads(str(first_row["scene_graph_delta_json"]))
    second_graph["relation_descriptors"][0][
        "relation"
    ] = SceneEdgeKind.BLOCKS_PATH_TO.value
    second_graph["relation_descriptors"][0]["target"]["kind"] = "liquidity_pool"
    second_graph["updates"][0]["relation_descriptors"] = second_graph[
        "relation_descriptors"
    ]
    second_row["scene_graph_delta_json"] = _canonical_json_text(second_graph)

    first = representation_case_from_case_input_row(first_row)
    second = representation_case_from_case_input_row(second_row)
    first_relation = tuple(
        event
        for event in first.events
        if event.event_type.startswith("scene_relation:")
    )
    second_relation = tuple(
        event
        for event in second.events
        if event.event_type.startswith("scene_relation:")
    )

    assert len(first_relation) == len(second_relation) == 1
    assert first_relation[0].event_type != second_relation[0].event_type
    assert first_relation[0].relation_types != second_relation[0].relation_types

    incomplete_row = dict(first_row)
    incomplete_graph = json.loads(str(first_row["scene_graph_delta_json"]))
    incomplete_graph["relation_descriptors_complete"] = False
    incomplete_graph["updates"][0]["relation_descriptors_complete"] = False
    incomplete_row["scene_graph_delta_json"] = _canonical_json_text(
        incomplete_graph
    )
    with pytest.raises(RepresentationDataError, match="relation descriptors are incomplete"):
        representation_case_from_case_input_row(incomplete_row)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_opposite_authority_is_contrastive_negative_independent_of_thesis() -> None:
    import torch

    first_case = replace(
        _case(),
        authority_direction=1,
    )
    second_case = replace(
        _case(
            case_id="case-b",
            revision_id="revision-b",
            episode_id="episode-b",
            context_id="context-a",
            direction=1,
        ),
        authority_direction=-1,
    )
    prepared = tuple(
        prepare_representation_case(case, _store())
        for case in (first_case, second_case)
    )
    batch, _ = collate_representation_cases(prepared, mask_probability=0.0)
    embedding = torch.zeros((2, EMBEDDING_DIM), dtype=torch.float32)
    embedding[:, 0] = 1.0

    opposite_loss = supervised_causal_contrastive_loss(embedding, batch)
    unknown_batch, _ = collate_representation_cases(
        (prepared[0], replace(prepared[1], case=replace(second_case, authority_direction=0))),
        mask_probability=0.0,
    )
    unknown_loss = supervised_causal_contrastive_loss(embedding, unknown_batch)

    assert batch.authority_direction.tolist() == [1, -1]
    assert float(opposite_loss) == pytest.approx(0.8)
    assert unknown_batch.authority_direction.tolist() == [1, 0]
    assert float(unknown_loss) == pytest.approx(0.0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_neutral_b1_vicreg_penalizes_collapsed_embeddings_more_than_dispersed() -> None:
    import torch
    from torch.nn import functional as F

    collapsed = torch.zeros((16, EMBEDDING_DIM), dtype=torch.float64)
    collapsed[:, 0] = 1.0
    generator = torch.Generator().manual_seed(751)
    dispersed = F.normalize(
        torch.randn((16, EMBEDDING_DIM), generator=generator, dtype=torch.float64),
        dim=1,
    )

    collapsed_loss = neutral_b1_vicreg_loss(collapsed, collapsed)
    dispersed_loss = neutral_b1_vicreg_loss(dispersed, dispersed)

    assert collapsed_loss.diagnostics["active"] is True
    assert dispersed_loss.diagnostics["active"] is True
    assert float(collapsed_loss.total) > float(dispersed_loss.total)
    assert float(collapsed_loss.components["variance"]) == pytest.approx(0.99)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_neutral_b1_vicreg_has_finite_gradients_and_exact_weighted_total() -> None:
    import torch
    from torch.nn import functional as F

    generator = torch.Generator().manual_seed(752)
    first_source = torch.randn(
        (16, EMBEDDING_DIM), generator=generator, dtype=torch.float64,
        requires_grad=True,
    )
    second_source = torch.randn(
        (16, EMBEDDING_DIM), generator=generator, dtype=torch.float64,
        requires_grad=True,
    )
    result = neutral_b1_vicreg_loss(
        F.normalize(first_source, dim=1),
        F.normalize(second_source, dim=1),
    )
    expected_total = (
        5.0 * result.components["invariance"]
        + 5.0 * result.components["variance"]
        + 0.2 * result.components["covariance"]
    )

    torch.testing.assert_close(result.total, expected_total, rtol=0.0, atol=0.0)
    assert all(torch.isfinite(value) for value in result.components.values())
    assert all(value.requires_grad for value in result.components.values())
    for name, value in result.diagnostics.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value)
            assert value.requires_grad, name

    result.total.backward()
    assert first_source.grad is not None
    assert second_source.grad is not None
    assert bool(torch.isfinite(first_source.grad).all())
    assert bool(torch.isfinite(second_source.grad).all())


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_neutral_b1_vicreg_skips_short_final_batch_but_activates_at_sixteen() -> None:
    import torch
    from torch.nn import functional as F

    generator = torch.Generator().manual_seed(753)
    short_source = torch.randn(
        (15, EMBEDDING_DIM), generator=generator, requires_grad=True
    )
    short_view = F.normalize(short_source, dim=1)
    short_result = neutral_b1_vicreg_loss(short_view, short_view)

    assert short_result.diagnostics["active"] is False
    assert short_result.diagnostics["batch_size"] == 15
    assert float(short_result.total.detach()) == 0.0
    assert all(
        float(value.detach()) == 0.0 for value in short_result.components.values()
    )
    assert short_result.total.requires_grad
    short_result.total.backward()
    assert short_source.grad is not None
    torch.testing.assert_close(short_source.grad, torch.zeros_like(short_source.grad))

    active_source = torch.randn(
        (16, EMBEDDING_DIM), generator=generator, requires_grad=True
    )
    active_view = F.normalize(active_source, dim=1)
    active_result = neutral_b1_vicreg_loss(active_view, active_view.roll(1, dims=0))
    assert active_result.diagnostics["active"] is True
    assert active_result.diagnostics["batch_size"] == 16
    assert float(active_result.total) > 0.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_neutral_b1_vicreg_rejects_invalid_embedding_contract() -> None:
    import torch
    from torch.nn import functional as F

    valid = F.normalize(torch.ones((16, EMBEDDING_DIM)), dim=1)
    invalid_cases = (
        (torch.ones((16, EMBEDDING_DIM - 1)), valid, "shape"),
        (valid, valid[:15], "identical shapes"),
        (torch.full_like(valid, float("nan")), valid, "finite"),
        (torch.zeros_like(valid), valid, "L2-normalized"),
        (torch.ones((16, EMBEDDING_DIM), dtype=torch.long), valid, "floating"),
        (F.normalize(torch.ones((17, EMBEDDING_DIM)), dim=1),) * 2
        + ("batch size",),
    )
    for first, second, message in invalid_cases:
        with pytest.raises(RepresentationDataError, match=message):
            neutral_b1_vicreg_loss(first, second)

    with pytest.raises(TypeError):
        neutral_b1_vicreg_loss(valid, valid, torch.zeros(16))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        neutral_b1_vicreg_loss(  # type: ignore[call-arg]
            valid, valid, targets=torch.zeros(16)
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_neutral_b1_vicreg_uses_frozen_sample_correction_and_reductions() -> None:
    import torch
    from torch.nn import functional as F

    generator = torch.Generator().manual_seed(754)
    first = F.normalize(
        torch.randn((16, EMBEDDING_DIM), generator=generator, dtype=torch.float64),
        dim=1,
    )
    second = F.normalize(
        torch.randn((16, EMBEDDING_DIM), generator=generator, dtype=torch.float64),
        dim=1,
    )
    result = neutral_b1_vicreg_loss(first, second)
    first_q = first * EMBEDDING_DIM**0.5
    second_q = second * EMBEDDING_DIM**0.5
    first_std = torch.sqrt(first_q.var(dim=0, correction=1) + 1e-4)
    second_std = torch.sqrt(second_q.var(dim=0, correction=1) + 1e-4)
    expected_invariance = (first_q - second_q).square().mean()
    expected_variance = (
        torch.relu(1.0 - first_std).mean()
        + torch.relu(1.0 - second_std).mean()
    ) / 2.0

    def expected_covariance(values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / 15
        off_diagonal = ~torch.eye(EMBEDDING_DIM, dtype=torch.bool)
        return covariance[off_diagonal].square().sum() / EMBEDDING_DIM

    expected_covariance_mean = (
        expected_covariance(first_q) + expected_covariance(second_q)
    ) / 2.0
    population_std = torch.sqrt(first_q.var(dim=0, correction=0) + 1e-4)

    torch.testing.assert_close(
        result.components["invariance"], expected_invariance, rtol=0.0, atol=1e-12
    )
    torch.testing.assert_close(
        result.components["variance"], expected_variance, rtol=0.0, atol=1e-12
    )
    torch.testing.assert_close(
        result.components["covariance"],
        expected_covariance_mean,
        rtol=0.0,
        atol=1e-12,
    )
    assert not torch.allclose(
        result.diagnostics["first_view_mean_std"], population_std.mean()
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_brain_response_and_control_fields_cannot_change_market_embedding() -> None:
    import torch

    first_row = _case_library_row()
    first_row["entry_episode_json"] = _canonical_json_text(
        {
            "lifecycle": "active",
            "playbook": "LIQUIDITY_SWEEP_REVERSAL",
            "updated_at": pd.Timestamp("2022-01-03 09:38", tz="UTC").isoformat(),
        }
    )
    second_row = dict(first_row)
    second_row["brain_response_json"] = _canonical_json_text(
        {
            "selected_action": "SUBMIT",
            "risk_action": "RAISE_SIZE",
            "phase": "plan_formed",
            "nested": {"decision": "BUY", "hard_gate": False},
        }
    )
    second_row["entry_episode_json"] = _canonical_json_text(
        {
            "lifecycle": "active",
            "playbook": "DIFFERENT_LABEL_ONLY",
            "updated_at": pd.Timestamp("2022-01-03 09:38", tz="UTC").isoformat(),
        }
    )
    first_case = representation_case_from_case_input_row(first_row)
    second_case = representation_case_from_case_input_row(second_row)
    frames = {}
    ticks = {}
    availability = {}
    for timeframe in TIMEFRAMES:
        key = CanonicalSourceKey("epoch-a", timeframe, "a" * 64)
        frames[key] = _frame(timeframe)
        ticks[key] = 0.25
        availability[key] = BAR_END_INDEX_BINDING
    store = CanonicalOHLCVStore(
        frames,
        tick_sizes=ticks,
        availability_bindings=availability,
        normalization_window=3,
    )
    first = prepare_representation_case(first_case, store)
    second = prepare_representation_case(second_case, store)

    assert first_case.events == second_case.events
    np.testing.assert_array_equal(first.event_type_ids, second.event_type_ids)
    assert bool(first.direct_label_source_event_mask.any())
    masked_first = mask_direct_label_source_tokens(first)
    masked_second = mask_direct_label_source_tokens(second)
    assert masked_first.label_sources_masked is True
    batch_a, _ = collate_representation_cases((masked_first,), mask_probability=0.0)
    batch_b, _ = collate_representation_cases((masked_second,), mask_probability=0.0)
    model = MarketRepresentationModel()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model.encode(batch_a), model.encode(batch_b))


def test_case_library_local_frame_rows_are_resolved_by_time_for_each_asof() -> None:
    early = representation_case_from_case_input_row(
        _case_library_row(asof=pd.Timestamp("2022-01-03 09:33", tz="UTC"))
    )
    late = representation_case_from_case_input_row(
        _case_library_row(
            asof=pd.Timestamp("2022-01-03 09:36", tz="UTC"),
            revision_id="revision-library-b",
            revision_index=1,
            revision_stage="zone_registered",
        )
    )
    source = _frame("1m")
    key = CanonicalSourceKey("epoch-a", "1m", "a" * 64)
    store = CanonicalOHLCVStore(
        {key: source},
        tick_sizes={key: 0.25},
        availability_bindings={key: BAR_END_INDEX_BINDING},
        normalization_window=3,
    )

    early_prefix = store.read_prefix(early.prefixes["1m"], asof=early.asof)
    late_prefix = store.read_prefix(late.prefixes["1m"], asof=late.asof)

    assert early_prefix.row_end_exclusive == 3
    assert late_prefix.row_end_exclusive == 6
    assert late_prefix.row_end_exclusive != 999
    assert tuple(event.event_type for event in early.events) == tuple(
        event.event_type for event in late.events
    )


def test_raw_bar_start_index_is_not_trusted_without_availability_binding() -> None:
    key = CanonicalSourceKey("epoch-a", "1m", "canonical-a")
    with pytest.raises(RepresentationDataError, match="explicit availability"):
        CanonicalOHLCVStore(
            {key: _frame("1m")},
            tick_sizes={key: 0.25},
            availability_bindings={},
        )


def test_training_store_binds_each_view_to_source_and_content_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shares.scripts import train_market_representation as trainer

    source_path = tmp_path / "canonical-source.csv"
    source_path.write_text("canonical parent source\n", encoding="utf-8")
    source_id = hashlib.sha256(source_path.read_bytes()).hexdigest()
    case = _case()
    case = replace(
        case,
        prefixes={
            timeframe: replace(prefix, canonical_source_id=source_id)
            for timeframe, prefix in case.prefixes.items()
        },
        canonical_source_path=str(source_path),
        symbol="NQ",
        instrument_id=1,
    )
    sibling_revision = replace(
        case,
        case_id="case-sibling",
        revision_id="revision-sibling",
        entry_episode_id="episode-sibling",
    )
    view_args: list[str] = []
    hash_args: list[str] = []
    manifest_views: list[dict[str, object]] = []
    for timeframe in TIMEFRAMES:
        path = tmp_path / f"{timeframe}.csv"
        _frame(timeframe).to_csv(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        view_args.append(f"{case.market_epoch_id}:{timeframe}:{source_id}={path}")
        hash_args.append(f"{case.market_epoch_id}:{timeframe}:{source_id}={digest}")
        manifest_views.append(
            {
                "market_epoch_id": case.market_epoch_id,
                "parent_source_sha256": source_id,
                "parent_source_path": str(source_path),
                "symbol": "NQ",
                "instrument_id": 1,
                "timeframe": timeframe,
                "view_path": str(path),
                "view_sha256": digest,
                "aggregation_protocol": "smc-existing-causal-aggregation-v1",
                "availability_binding": BAR_END_INDEX_BINDING,
            }
        )
    manifest_path = tmp_path / "lineage.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "smc-canonical-mtf-lineage-v1",
                "aggregation_protocol": "smc-existing-causal-aggregation-v1",
                "views": manifest_views,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    tick_args = [f"{timeframe}=0.25" for timeframe in TIMEFRAMES]
    availability_args = [
        f"{timeframe}={BAR_END_INDEX_BINDING}" for timeframe in TIMEFRAMES
    ]

    original_hash_file = trainer._sha256_file
    parent_hash_calls = 0

    def counted_hash(path: str) -> str:
        nonlocal parent_hash_calls
        if Path(path).resolve() == source_path.resolve():
            parent_hash_calls += 1
        return original_hash_file(path)

    monkeypatch.setattr(trainer, "_sha256_file", counted_hash)
    store = trainer._build_store(
        (case, sibling_revision),
        view_args,
        hash_args,
        tick_args,
        availability_args,
        str(manifest_path),
        manifest_sha,
    )
    prepared = prepare_representation_case(case, store)
    assert prepared.feature_max_at <= case.asof
    assert parent_hash_calls == 1

    # A reset epoch using the same immutable parent source must bind its own
    # derived five-timeframe views rather than inheriting rows from epoch-a.
    other_epoch = replace(
        case,
        market_epoch_id="epoch-b",
        prefixes={
            timeframe: replace(prefix, market_epoch_id="epoch-b")
            for timeframe, prefix in case.prefixes.items()
        },
        events=tuple(
            replace(event, market_epoch_id="epoch-b") for event in case.events
        ),
    )
    two_epoch_view_args = list(view_args)
    two_epoch_hash_args = list(hash_args)
    two_epoch_manifest_views = list(manifest_views)
    for timeframe in TIMEFRAMES:
        path = tmp_path / f"epoch-b-{timeframe}.csv"
        _frame(timeframe, scale=1.5).to_csv(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        two_epoch_view_args.append(f"epoch-b:{timeframe}:{source_id}={path}")
        two_epoch_hash_args.append(f"epoch-b:{timeframe}:{source_id}={digest}")
        two_epoch_manifest_views.append(
            {
                "market_epoch_id": "epoch-b",
                "parent_source_sha256": source_id,
                "parent_source_path": str(source_path),
                "symbol": "NQ",
                "instrument_id": 1,
                "timeframe": timeframe,
                "view_path": str(path),
                "view_sha256": digest,
                "aggregation_protocol": "smc-existing-causal-aggregation-v1",
                "availability_binding": BAR_END_INDEX_BINDING,
            }
        )
    two_epoch_manifest = tmp_path / "lineage-two-epochs.json"
    two_epoch_manifest.write_text(
        json.dumps(
            {
                "schema": "smc-canonical-mtf-lineage-v1",
                "aggregation_protocol": "smc-existing-causal-aggregation-v1",
                "views": two_epoch_manifest_views,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    two_epoch_store = trainer._build_store(
        (case, other_epoch),
        two_epoch_view_args,
        two_epoch_hash_args,
        tick_args,
        availability_args,
        str(two_epoch_manifest),
        hashlib.sha256(two_epoch_manifest.read_bytes()).hexdigest(),
    )
    epoch_a_features = two_epoch_store.read_prefix(
        case.prefixes["1m"], asof=case.asof
    ).values
    epoch_b_features = two_epoch_store.read_prefix(
        other_epoch.prefixes["1m"], asof=other_epoch.asof
    ).values
    assert not np.array_equal(epoch_a_features, epoch_b_features)
    assert epoch_b_features[-1, CAUSAL_CANDLE_FEATURES.index("body_ticks")] == (
        pytest.approx(
            epoch_a_features[-1, CAUSAL_CANDLE_FEATURES.index("body_ticks")] * 1.5
        )
    )

    unqualified = [
        view_args[0].replace(f"{case.market_epoch_id}:", "", 1),
        *view_args[1:],
    ]
    with pytest.raises(RepresentationDataError, match="MARKET_EPOCH_ID"):
        trainer._build_store(
            (case,),
            unqualified,
            hash_args,
            tick_args,
            availability_args,
            str(manifest_path),
            manifest_sha,
        )

    # A wrong-but-entirely-past view cannot be admitted by supplying its path;
    # the pre-registered manifest still binds the original bytes.
    _frame("4h", scale=2.0).to_csv(tmp_path / "4h.csv")
    with pytest.raises(RepresentationDataError, match="content hash mismatch"):
        trainer._build_store(
            (case,),
            view_args,
            hash_args,
            tick_args,
            availability_args,
            str(manifest_path),
            manifest_sha,
        )


def test_real_case_stream_requires_final_hash_bound_library_manifest(
    tmp_path: Path,
) -> None:
    from shares.scripts import train_market_representation as trainer
    from shares.core.causal_cases import (
        CAUSAL_CASE_INPUT_FIELD_TYPES,
        CAUSAL_CASE_OUTCOME_FIELD_TYPES,
        CAUSAL_CASE_PROTOCOL,
        CAUSAL_CASE_RECORDER_SCHEMA_VERSION,
        expected_causal_case_run_identity,
    )

    shard = tmp_path / "causal_case_input_shards-000000.parquet"
    shard.write_bytes(b"committed-case-input-shard")
    run_manifest = tmp_path / "run_manifest.json"
    run_payload = {
        "schema_version": 1,
        "runner": "continuous_replay",
        "causal_case_identity": expected_causal_case_run_identity(),
        "output": {
            "causal_case_library": True,
            "stream_families": [
                "causal_case_input_shards",
                "causal_case_outcome_shards",
            ],
        },
    }

    def write_run(payload: object) -> None:
        run_manifest.write_text(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    write_run(run_payload)
    input_manifest = tmp_path / "causal_case_input_shards.manifest.json"
    input_payload = {
        "format_version": 1,
        "artifact": "case_inputs",
        "status": "complete",
        "stream": "causal_case_input_shards",
        "rows": 1,
        "shards": [
            {
                "index": 0,
                "path": shard.name,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                "rows": 1,
            }
        ],
        "bindings": {"run_manifest": run_manifest.name},
        "field_types": dict(CAUSAL_CASE_INPUT_FIELD_TYPES),
    }
    input_manifest.write_text(
        json.dumps(input_payload, sort_keys=True), encoding="utf-8"
    )
    input_sha = hashlib.sha256(input_manifest.read_bytes()).hexdigest()
    run_sha = hashlib.sha256(run_manifest.read_bytes()).hexdigest()

    outcome_manifest = tmp_path / "causal_case_outcome_shards.manifest.json"
    outcome_payload = {
        "format_version": 1,
        "artifact": "case_outcomes",
        "status": "complete",
        "stream": "causal_case_outcome_shards",
        "rows": 1,
        "shards": [
            {
                "index": 0,
                "path": "causal_case_outcome_shards-000000.parquet",
                "sha256": "c" * 64,
                "rows": 1,
            }
        ],
        "bindings": {"run_manifest": run_manifest.name},
        "field_types": dict(CAUSAL_CASE_OUTCOME_FIELD_TYPES),
    }
    outcome_manifest.write_text(
        json.dumps(outcome_payload, sort_keys=True), encoding="utf-8"
    )
    outcome_sha = hashlib.sha256(outcome_manifest.read_bytes()).hexdigest()

    observed_sha, observed_rows, observed_run = (
        trainer._validate_case_input_stream_manifest(
            (shard,), str(input_manifest), input_sha, run_sha
        )
    )
    assert (observed_sha, observed_rows, observed_run) == (
        input_sha,
        1,
        run_manifest.resolve(),
    )

    library_manifest = tmp_path / "causal_case_library.manifest.json"
    library_payload = {
        "format_version": 1,
        "artifact": "entry_episode_causal_case_library",
        "status": "complete",
        "recorder_schema_version": CAUSAL_CASE_RECORDER_SCHEMA_VERSION,
        "protocol": dict(CAUSAL_CASE_PROTOCOL),
        "grain": "entry_episode",
        "input_stream": {
            "manifest": input_manifest.name,
            "manifest_sha256": input_sha,
            "rows": 1,
        },
        "future_outcome_stream": {
            "manifest": outcome_manifest.name,
            "manifest_sha256": outcome_sha,
            "rows": 1,
        },
        "bindings": {
            "run_manifest": run_manifest.name,
            "run_manifest_sha256": run_sha,
        },
        "leakage_contract": {
            "outcome_fields_in_input_schema": False,
            "embedding_source": "input_stream_only",
            "episode_split_disjoint_required": True,
            "normalization_prefix_only": True,
        },
    }
    library_manifest.write_text(
        json.dumps(library_payload, sort_keys=True), encoding="utf-8"
    )
    library_sha = hashlib.sha256(library_manifest.read_bytes()).hexdigest()
    assert trainer._validate_case_library_manifest(
        str(library_manifest),
        library_sha,
        input_manifest_path=str(input_manifest),
        input_manifest_sha256=input_sha,
        input_rows=1,
        run_manifest_path=run_manifest,
        run_manifest_sha256=run_sha,
    ) == library_sha

    def assert_run_laundering_rejected(damaged_run: object) -> None:
        write_run(damaged_run)
        damaged_run_sha = hashlib.sha256(run_manifest.read_bytes()).hexdigest()
        laundered_library = json.loads(json.dumps(library_payload))
        laundered_library["bindings"]["run_manifest_sha256"] = damaged_run_sha
        library_manifest.write_text(
            json.dumps(laundered_library, sort_keys=True),
            encoding="utf-8",
        )
        with pytest.raises(
            RepresentationDataError,
            match="run manifest identity",
        ):
            trainer._validate_case_library_manifest(
                str(library_manifest),
                hashlib.sha256(library_manifest.read_bytes()).hexdigest(),
                input_manifest_path=str(input_manifest),
                input_manifest_sha256=input_sha,
                input_rows=1,
                run_manifest_path=run_manifest,
                run_manifest_sha256=damaged_run_sha,
            )

    missing_identity = {
        key: value
        for key, value in run_payload.items()
        if key != "causal_case_identity"
    }
    schema_six = json.loads(json.dumps(run_payload))
    schema_six["causal_case_identity"]["recorder_schema_version"] = 6
    schema_six["causal_case_identity"]["protocol"]["protocol_version"] = (
        "entry-episode-causal-case-1.5.0"
    )
    tampered_future_blind = json.loads(json.dumps(run_payload))
    tampered_future_blind["causal_case_identity"][
        "future_visible_to_input"
    ] = True
    disabled_output = json.loads(json.dumps(run_payload))
    disabled_output["output"]["causal_case_library"] = False
    missing_outcome_family = json.loads(json.dumps(run_payload))
    missing_outcome_family["output"]["stream_families"] = [
        "causal_case_input_shards"
    ]
    duplicate_input_family = json.loads(json.dumps(run_payload))
    duplicate_input_family["output"]["stream_families"].append(
        "causal_case_input_shards"
    )
    for damaged_run in (
        missing_identity,
        schema_six,
        tampered_future_blind,
        disabled_output,
        missing_outcome_family,
        duplicate_input_family,
    ):
        assert_run_laundering_rejected(damaged_run)
    write_run(run_payload)
    library_manifest.write_text(
        json.dumps(library_payload, sort_keys=True), encoding="utf-8"
    )

    unconserved_outcome = dict(outcome_payload)
    unconserved_outcome["shards"] = []
    outcome_manifest.write_text(
        json.dumps(unconserved_outcome, sort_keys=True), encoding="utf-8"
    )
    unconserved_library = json.loads(json.dumps(library_payload))
    unconserved_library["future_outcome_stream"]["manifest_sha256"] = (
        hashlib.sha256(outcome_manifest.read_bytes()).hexdigest()
    )
    library_manifest.write_text(
        json.dumps(unconserved_library, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(RepresentationDataError, match="finalization metadata"):
        trainer._validate_case_library_manifest(
            str(library_manifest),
            hashlib.sha256(library_manifest.read_bytes()).hexdigest(),
            input_manifest_path=str(input_manifest),
            input_manifest_sha256=input_sha,
            input_rows=1,
            run_manifest_path=run_manifest,
            run_manifest_sha256=run_sha,
        )
    outcome_manifest.write_text(
        json.dumps(outcome_payload, sort_keys=True), encoding="utf-8"
    )
    library_manifest.write_text(
        json.dumps(library_payload, sort_keys=True), encoding="utf-8"
    )

    outcome_manifest.unlink()
    with pytest.raises(RepresentationDataError, match="future outcome stream"):
        trainer._validate_case_library_manifest(
            str(library_manifest),
            library_sha,
            input_manifest_path=str(input_manifest),
            input_manifest_sha256=input_sha,
            input_rows=1,
            run_manifest_path=run_manifest,
            run_manifest_sha256=run_sha,
        )

    forged_outcome = dict(outcome_payload)
    forged_outcome["status"] = "incomplete"
    outcome_manifest.write_text(
        json.dumps(forged_outcome, sort_keys=True), encoding="utf-8"
    )
    forged_library = json.loads(json.dumps(library_payload))
    forged_library["future_outcome_stream"]["manifest_sha256"] = (
        hashlib.sha256(outcome_manifest.read_bytes()).hexdigest()
    )
    library_manifest.write_text(
        json.dumps(forged_library, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(RepresentationDataError, match="finalization metadata"):
        trainer._validate_case_library_manifest(
            str(library_manifest),
            hashlib.sha256(library_manifest.read_bytes()).hexdigest(),
            input_manifest_path=str(input_manifest),
            input_manifest_sha256=input_sha,
            input_rows=1,
            run_manifest_path=run_manifest,
            run_manifest_sha256=run_sha,
        )

    outcome_manifest.write_text(
        json.dumps(outcome_payload, sort_keys=True), encoding="utf-8"
    )
    library_manifest.write_text(
        json.dumps(library_payload, sort_keys=True), encoding="utf-8"
    )

    damaged = dict(library_payload)
    damaged["status"] = "incomplete"
    library_manifest.write_text(json.dumps(damaged, sort_keys=True), encoding="utf-8")
    with pytest.raises(RepresentationDataError, match="finalization"):
        trainer._validate_case_library_manifest(
            str(library_manifest),
            hashlib.sha256(library_manifest.read_bytes()).hexdigest(),
            input_manifest_path=str(input_manifest),
            input_manifest_sha256=input_sha,
            input_rows=1,
            run_manifest_path=run_manifest,
            run_manifest_sha256=run_sha,
        )


def test_external_observable_target_cache_is_exactly_identity_and_clock_bound(
    tmp_path: Path,
) -> None:
    from shares.scripts import train_market_representation as trainer

    first = _case_library_row()
    second = _case_library_row(
        asof=pd.Timestamp("2022-01-03 09:40", tz="UTC"),
        revision_id="revision-cache-next",
        revision_index=1,
        revision_stage="trigger",
    )
    second = _with_complete_interval_ledger(
        second,
        previous_asof=pd.Timestamp(first["asof"]),
        collections={},
    )
    internally_built = build_observable_revision_targets((first, second))
    rows = [record.as_dict() for record in internally_built.values()]
    cache = tmp_path / "observable-targets.jsonl"
    cache.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    trainer._validate_external_target_cache(
        trainer._target_map(cache), internally_built
    )

    wrong_episode = [dict(row) for row in rows]
    wrong_episode[0]["entry_episode_id"] = "other-episode"
    cache.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in wrong_episode),
        encoding="utf-8",
    )
    with pytest.raises(RepresentationDataError, match="disagrees"):
        trainer._validate_external_target_cache(
            trainer._target_map(cache), internally_built
        )

    wrong_source = [dict(row) for row in rows]
    wrong_source[0]["label_source"] = "shadow_frozen_outcome"
    cache.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in wrong_source),
        encoding="utf-8",
    )
    with pytest.raises(RepresentationDataError, match="label_source"):
        trainer._target_map(cache)


def test_explicit_completion_column_excludes_future_incomplete_bar() -> None:
    frame = _frame("1m")
    frame["completed_at"] = frame.index + timedelta(minutes=1)
    mutated = frame.copy()
    mutated.iloc[3, mutated.columns.get_loc("close")] += 10_000
    mutated.iloc[3, mutated.columns.get_loc("high")] += 10_000
    key = CanonicalSourceKey("epoch-a", "1m", "canonical-a")
    prefix = PrefixIndexRange(
        market_epoch_id="epoch-a",
        timeframe="1m",
        canonical_source_id="canonical-a",
        row_start=0,
        row_end_exclusive=999,
        start_at=pd.Timestamp("2022-01-03 09:30", tz="UTC"),
        end_at=pd.Timestamp("2022-01-03 09:33", tz="UTC"),
        resolve_external_rows_by_time=True,
    )
    first = CanonicalOHLCVStore(
        {key: frame},
        tick_sizes={key: 0.25},
        availability_bindings={key: "completed_at"},
        normalization_window=3,
    ).read_prefix(prefix, asof=pd.Timestamp("2022-01-03 09:33", tz="UTC"))
    second = CanonicalOHLCVStore(
        {key: mutated},
        tick_sizes={key: 0.25},
        availability_bindings={key: "completed_at"},
        normalization_window=3,
    ).read_prefix(prefix, asof=pd.Timestamp("2022-01-03 09:33", tz="UTC"))

    np.testing.assert_array_equal(first.values, second.values)
    assert first.row_end_exclusive == 2


def test_event_and_prefix_cannot_cross_market_epoch() -> None:
    case = _case()
    with pytest.raises(RepresentationDataError, match="cannot cross"):
        replace(
            case,
            prefixes={
                **case.prefixes,
                "1m": replace(case.prefixes["1m"], market_epoch_id="epoch-b"),
            },
        )
    with pytest.raises(RepresentationDataError, match="cannot cross"):
        replace(
            case,
            events=(replace(case.events[0], market_epoch_id="epoch-b"),),
        )


def test_outcome_fields_are_rejected_from_model_input_mapping() -> None:
    payload = {
        "case_id": "case-a",
        "revision_id": "revision-a",
        "market_epoch_id": "epoch-a",
        "context_thesis_id": "context-a",
        "entry_episode_id": "episode-a",
        "decision_at": "2022-01-03T10:00:00+00:00",
        "direction": "long",
        "future_outcome": {"mfe_r": 2.0},
        "prefixes": {},
        "events": [],
    }

    with pytest.raises(RepresentationDataError, match="future outcome fields"):
        RepresentationCase.from_mapping(payload)


def test_nested_outcome_field_is_rejected_before_event_tokenization() -> None:
    case = _case()
    payload = {
        "case_id": case.case_id,
        "revision_id": case.revision_id,
        "revision_stage": case.revision_stage,
        "market_epoch_id": case.market_epoch_id,
        "context_thesis_id": case.context_thesis_id,
        "entry_episode_id": case.entry_episode_id,
        "asof": case.asof,
        "direction": case.direction,
        "prefixes": {
            timeframe: {
                "market_epoch_id": prefix.market_epoch_id,
                "timeframe": timeframe,
                "canonical_source_id": prefix.canonical_source_id,
                "row_start": prefix.row_start,
                "row_end_exclusive": prefix.row_end_exclusive,
            }
            for timeframe, prefix in case.prefixes.items()
        },
        "events": {
            "observation_transition": {"collections": {}},
            "brain_response": {"diagnostics": {"mfe_r": 2.5}},
        },
    }

    with pytest.raises(RepresentationDataError, match="events.brain_response"):
        RepresentationCase.from_mapping(payload)


def test_outcome_target_changes_do_not_change_causal_input_fingerprint() -> None:
    case = _case()
    loss_label_a = SelfSupervisedTarget(draw_consumed=0)
    loss_label_b = SelfSupervisedTarget(draw_consumed=1)

    assert causal_input_fingerprint(case) == causal_input_fingerprint(case)
    assert loss_label_a != loss_label_b


def test_identical_inputs_are_deduplicated_independent_of_episode_identity() -> None:
    original = _case()
    duplicate = replace(
        original,
        case_id="case-b",
        revision_id="revision-b",
        entry_episode_id="episode-b",
        regime="sweep_failure",
        mechanism_label="different_evaluation_label",
    )

    assert causal_input_fingerprint(original) == causal_input_fingerprint(duplicate)
    assert len(deduplicate_causal_inputs((original, duplicate))) == 1


def test_split_keeps_episode_and_identical_prefix_input_together() -> None:
    first = _case()
    same_episode = replace(first, case_id="case-b", revision_id="revision-b")
    same_input_other_episode = replace(
        first,
        case_id="case-c",
        revision_id="revision-c",
        entry_episode_id="episode-c",
    )
    independent = replace(
        _case(
            case_id="case-d",
            revision_id="revision-d",
            episode_id="episode-d",
            context_id="context-d",
            regime="balance",
        ),
        asof=first.asof - timedelta(minutes=1),
    )
    cases = (first, same_episode, same_input_other_episode, independent)

    split = assign_leakage_safe_splits(cases, seed=991)

    assert split[first.revision_id] == split[same_episode.revision_id]
    assert split[first.revision_id] == split[same_input_other_episode.revision_id]
    validate_split_integrity(cases, split)


def test_observable_target_builder_uses_next_typed_transition_not_case_stage() -> None:
    first = _case_library_row()
    second_asof = pd.Timestamp("2022-01-03 09:40", tz="UTC")
    fvg_next = to_primitive(
        replace(
            _fvg(
                second_asof - pd.Timedelta(minutes=5),
                identity="fvg-next-a",
            ),
            symbol="NQ",
            instrument_id=1,
            lifecycle=FairValueGapLifecycle.INVALIDATED,
            state_started_at=second_asof,
            last_updated_at=second_asof,
            invalidated_at=second_asof,
            transition_reason="close_beyond_frozen_far_edge",
        )
    )
    second = _case_library_row(
        asof=second_asof,
        revision_id="revision-library-b",
        revision_index=1,
        revision_stage="first_pullback",
    )
    second["observation_transition_json"] = _canonical_json_text(
        {
            "typed_transition_delta_available": True,
            "coverage": {
                "complete": True,
                "coverage_start_at": first["asof"].isoformat(),
                "coverage_start_exclusive": True,
                "coverage_end_at": second_asof.isoformat(),
                "coverage_start_replay_update_ordinal": 0,
                "coverage_end_replay_update_ordinal": 1,
                "observed_update_count": 1,
                "eventful_update_count": 1,
                "last_observed_update_at": second_asof.isoformat(),
                "last_observed_replay_update_ordinal": 1,
                "all_typed_deltas_available": True,
                "gap_free": True,
            },
            "collections": {
                **{
                    name: []
                    for name in _CAUSAL_AGGREGATE_TRANSITION_COLLECTIONS
                },
                "group3_fvg_transitions_this_update": [
                    fvg_next
                ]
            },
            "updates": [
                {
                    "asof": second_asof.isoformat(),
                    "replay_update_ordinal": 1,
                    "typed_transition_delta_available": True,
                    "collections": {
                        **{
                            name: []
                            for name in _NEUTRAL_TRANSITION_COLLECTIONS
                        },
                        "group3_fvg_transitions_this_update": [
                            fvg_next
                        ]
                    },
                }
            ],
        },
    )
    second["scene_graph_delta_json"] = _canonical_json_text(
        {
            "asof": second_asof.isoformat(),
            "revision_id": "scene-b",
            "coverage": {
                "complete": True,
                "coverage_start_at": first["asof"].isoformat(),
                "coverage_start_exclusive": True,
                "coverage_end_at": second_asof.isoformat(),
                "coverage_start_replay_update_ordinal": 0,
                "coverage_end_replay_update_ordinal": 1,
                "observed_update_count": 1,
                "eventful_update_count": 1,
                "last_observed_update_at": second_asof.isoformat(),
                "last_observed_replay_update_ordinal": 1,
                "all_typed_deltas_available": True,
                "gap_free": True,
            },
            "added_node_ids": [],
            "revised_node_ids": [],
            "added_edge_ids": [],
            "revised_edge_ids": [],
            "resolution_event_ids": [],
            "relation_descriptors": [],
            "relation_descriptors_complete": True,
            "updates": [
                {
                    "asof": second_asof.isoformat(),
                    "replay_update_ordinal": 1,
                    "revision_id": "scene-b",
                    "added_node_ids": [],
                    "revised_node_ids": [],
                    "added_edge_ids": [],
                    "revised_edge_ids": [],
                    "resolution_event_ids": [],
                    "relation_descriptors": [],
                    "relation_descriptors_complete": True,
                }
            ],
        },
    )
    third_asof = pd.Timestamp("2022-01-03 10:00", tz="UTC")
    third = _case_library_row(
        asof=third_asof,
        revision_id="revision-library-c",
        revision_index=2,
        revision_stage="terminal",
    )
    third["draw_json"] = _canonical_json_text(
        {"context_draw": {"lifecycle": "consumed"}}
    )
    third["entry_episode_json"] = _canonical_json_text(
        {
            "lifecycle": "closed",
            "terminal_reason": "source_invalidated",
            "source_displacement_id": "disp-a",
            "updated_at": third_asof.isoformat(),
        },
    )

    built = build_observable_revision_targets((third, first, second))
    target = built["revision-library-a"]

    assert target.next_revision_id == "revision-library-b"
    assert target.target.next_event_type == NEXT_EVENT_TYPE_VOCAB[
        "group3_fvg_transitions_this_update"
    ]
    assert target.target.next_lifecycle == NEXT_LIFECYCLE_VOCAB["invalidated"]
    assert target.target.next_event_time_bucket == 2
    # Episode/source persistence and a later draw snapshot are not exact entity
    # lifecycle evidence, so neither task may use those sparse-state proxies.
    assert target.target.displacement_state == -100
    assert target.target.draw_consumed == -100
    assert target.target.scale_direction_alignment == 1
    # Unsupported later sparse snapshots are not consulted merely to extend a
    # label clock; the latest exact typed evidence is the second revision.
    assert target.label_max_observed_at == second_asof

    uncovered_second = dict(second)
    uncovered_transition = json.loads(second["observation_transition_json"])
    uncovered_transition["coverage"]["complete"] = False
    uncovered_transition["typed_transition_delta_available"] = False
    uncovered_transition["updates"][0]["typed_transition_delta_available"] = False
    uncovered_transition["coverage"]["all_typed_deltas_available"] = False
    uncovered_second["observation_transition_json"] = _canonical_json_text(
        uncovered_transition
    )
    uncovered_scene = json.loads(second["scene_graph_delta_json"])
    uncovered_scene["coverage"]["complete"] = False
    uncovered_scene["coverage"]["all_typed_deltas_available"] = False
    uncovered_second["scene_graph_delta_json"] = _canonical_json_text(
        uncovered_scene
    )
    uncovered = build_observable_revision_targets((first, uncovered_second, third))
    assert uncovered["revision-library-a"].target.next_event_type == -100
    assert uncovered["revision-library-a"].target.next_lifecycle == -100
    assert uncovered["revision-library-a"].target.next_event_time_bucket == -100


def test_observable_target_builder_is_input_order_invariant_and_epoch_bounded() -> None:
    first = _case_library_row()
    second = _case_library_row(
        asof=pd.Timestamp("2022-01-03 09:40", tz="UTC"),
        revision_id="revision-library-b",
        revision_index=1,
        revision_stage="trigger",
    )
    forward = build_observable_revision_targets((first, second))
    reverse = build_observable_revision_targets((second, first))

    assert forward["revision-library-a"].as_dict() == reverse[
        "revision-library-a"
    ].as_dict()
    assert forward["revision-library-a"].label_max_observed_at >= first["asof"]

    other_epoch = dict(second)
    other_epoch["revision_id"] = "revision-other-epoch"
    other_epoch["market_epoch_id"] = "epoch-b"
    other_epoch["revision_index"] = 0
    other_epoch["stage_identity"] = "trigger:other-epoch"
    bounded = build_observable_revision_targets((first, other_epoch))
    assert bounded["revision-library-a"].next_revision_id is None
    assert bounded["revision-library-a"].target.next_event_type == -100


def test_retained_interaction_current_view_cannot_change_next_delta_target() -> None:
    interaction, next_asof = _strict_interaction(Direction.LONG)
    first_asof = next_asof - pd.Timedelta(minutes=2)
    first = _case_library_row(asof=first_asof)
    second = _case_library_row(
        asof=next_asof,
        revision_id="revision-library-with-future-fvg",
        revision_index=1,
        revision_stage="trigger",
    )
    fvg_delta = to_primitive(
        replace(
            _fvg(next_asof, identity="fvg:future-delta"),
            symbol="NQ",
            instrument_id=1,
        )
    )
    without_current = _with_complete_interval_ledger(
        second,
        previous_asof=first_asof,
        collections={
            "group3_fvg_transitions_this_update": [fvg_delta],
        },
    )
    current_interaction = interaction_artifact_collections(interaction)
    with_current = _with_complete_interval_ledger(
        second,
        previous_asof=first_asof,
        collections={
            "group3_fvg_transitions_this_update": [fvg_delta],
            **{
                name: [to_primitive(value) for value in values]
                for name, values in current_interaction.items()
                if name in _NEUTRAL_TRANSITION_COLLECTIONS[6:10]
            },
        },
    )

    baseline = build_observable_revision_targets((first, without_current))[
        "revision-library-a"
    ].target
    retained = build_observable_revision_targets((first, with_current))[
        "revision-library-a"
    ].target
    assert baseline == retained
    assert baseline.next_event_type == NEXT_EVENT_TYPE_VOCAB[
        "group3_fvg_transitions_this_update"
    ]


def test_entity_targets_require_exact_ledger_identity_not_episode_proxy() -> None:
    first = _case_library_row()
    first_asof = pd.Timestamp(first["asof"])
    next_asof = pd.Timestamp("2022-01-03 09:40", tz="UTC")

    def displacement_revision(
        lifecycle: str, episode_lifecycle: str
    ) -> dict[str, object]:
        row = _case_library_row(
            asof=next_asof,
            revision_id=f"revision-displacement-{lifecycle}",
            revision_index=1,
            revision_stage="terminal" if episode_lifecycle == "closed" else "trigger",
        )
        row["entry_episode_json"] = _canonical_json_text(
            {
                "lifecycle": episode_lifecycle,
                "terminal_reason": (
                    "zone_local_failure" if episode_lifecycle == "closed" else None
                ),
                "source_displacement_id": "disp-a",
                "updated_at": next_asof.isoformat(),
            },
        )
        return _with_complete_interval_ledger(
            row,
            previous_asof=first_asof,
            collections={},
            scene_added_node_ids=(
                ("disp-a",) if lifecycle == "active" else ()
            ),
            scene_resolution_event_ids=(
                ("disp-a",) if lifecycle == "exhausted" else ()
            ),
        )

    episode_failed_but_displacement_active = build_observable_revision_targets(
        (first, displacement_revision("active", "closed"))
    )["revision-library-a"].target.displacement_state
    episode_active_but_displacement_exhausted = build_observable_revision_targets(
        (first, displacement_revision("exhausted", "active"))
    )["revision-library-a"].target.displacement_state

    assert episode_failed_but_displacement_active == 1
    assert episode_active_but_displacement_exhausted == 0

    draw_first = dict(first)
    draw_first["draw_json"] = _canonical_json_text(
        {"context_draw": {"draw_id": "draw-a", "lifecycle": "active"}},
    )
    consumed = _case_library_row(
        asof=next_asof,
        revision_id="revision-draw-consumed",
        revision_index=1,
        revision_stage="context_changed",
    )
    consumed = _with_complete_interval_ledger(
        consumed,
        previous_asof=first_asof,
        collections={
            "liquidity_inventory_transitions_this_update": [
                to_primitive(
                    replace(
                        _draw(
                            identity="draw-a",
                            side="above",
                            kind="swing",
                            price=101.0,
                            confirmed_at=(
                                next_asof - pd.Timedelta(minutes=1)
                            ),
                        ),
                        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
                        consumed_at=next_asof,
                        lifecycle_reason="swing_swept",
                    )
                )
            ]
        },
    )
    assert build_observable_revision_targets(
        (draw_first, consumed)
    )["revision-library-a"].target.draw_consumed == 1

    after_horizon_asof = pd.Timestamp("2022-01-03 10:40", tz="UTC")
    after_horizon = _case_library_row(
        asof=after_horizon_asof,
        revision_id="revision-draw-horizon",
        revision_index=1,
        revision_stage="context_changed",
    )
    after_horizon = _with_complete_interval_ledger(
        after_horizon,
        previous_asof=first_asof,
        collections={},
    )
    assert build_observable_revision_targets(
        (draw_first, after_horizon)
    )["revision-library-a"].target.draw_consumed == 0


def test_first_causal_stage_selector_is_order_invariant_without_hindsight() -> None:
    base = _case(revision_id="revision-0")
    first_change = replace(
        base,
        revision_id="revision-1",
        revision_stage="context_changed",
        revision_index=1,
        stage_identity="context_changed:first",
    )
    later_change = replace(
        base,
        revision_id="revision-2",
        revision_stage="context_changed",
        revision_index=2,
        stage_identity="context_changed:later",
    )

    selected_a = select_first_causal_stage_revisions(
        (later_change, first_change), decision_stage="context_changed"
    )
    selected_b = select_first_causal_stage_revisions(
        (first_change, later_change), decision_stage="context_changed"
    )

    assert [item.revision_id for item in selected_a] == ["revision-1"]
    assert selected_a == selected_b


def test_stage_export_selection_precedes_training_input_deduplication() -> None:
    episode_created = _case(revision_id="revision-created")
    trigger = replace(
        episode_created,
        revision_id="revision-trigger",
        revision_stage="trigger",
        revision_index=1,
        stage_identity="trigger:episode-a",
    )
    # Simultaneous stages can share encoder inputs. Training may deduplicate
    # them, but explicit export selection must operate on strict raw revisions.
    assert causal_input_fingerprint(episode_created) == causal_input_fingerprint(
        trigger
    )
    assert len(deduplicate_causal_inputs((episode_created, trigger))) == 1
    assert select_first_causal_stage_revisions(
        (trigger, episode_created), decision_stage="trigger"
    ) == (trigger,)


def test_outcome_blind_embedding_evaluation_measures_regime_and_cross_date_retrieval() -> None:
    regimes = ("continuation", "sweep_failure", "balance", "unknown")

    def sample(
        regime_index: int,
        serial: int,
        day: int,
        direction: int,
        *,
        label_sources_masked: bool = True,
    ) -> EmbeddingEvaluationSample:
        vector = np.zeros(EMBEDDING_DIM, dtype=np.float64)
        vector[regime_index] = 1.0
        vector[8 + (0 if direction < 0 else 8) + serial] = 0.01
        vector /= np.linalg.norm(vector)
        return EmbeddingEvaluationSample(
            revision_id=f"revision-{day}-{regime_index}-{direction}-{serial}",
            entry_episode_id=f"episode-{day}-{regime_index}-{direction}-{serial}",
            asof=pd.Timestamp(f"2022-01-{day:02d} 10:00", tz="UTC"),
            direction=direction,
            regime=regimes[regime_index],
            mechanism_label=f"mechanism-{regime_index}",
            embedding=tuple(float(value) for value in vector),
            label_sources_masked=label_sources_masked,
        )

    reference = tuple(
        sample(regime_index, serial, 3, direction)
        for regime_index in range(4)
        for direction in (-1, 1)
        for serial in range(2)
    )
    queries = tuple(
        sample(regime_index, serial, 4, direction)
        for regime_index in range(4)
        for direction in (-1, 1)
        for serial in (2, 3)
    )

    metrics = evaluate_outcome_blind_embedding_space(
        reference, queries, k=1, minimum_regime_samples=4
    )

    assert metrics["status"] == "measured"
    assert metrics["criteria_met"] is True
    assert metrics["regime_centroid_accuracy"] == 1.0
    assert metrics["cross_date_mechanism_retrieval_at_k"] == 1.0
    assert metrics["cross_date_mechanism_lift_over_chance"] >= 0.1
    assert metrics["direction_regime_coverage_sufficient"] is True
    assert metrics["direction_regime_noncollapse"] is True
    assert metrics["label_source_shortcut_audit"]["masked_label_probe_used"] is True
    assert metrics["outcome_fields_used"] is False
    assert metrics["trading_edge_claimed"] is False

    uncontrolled = evaluate_outcome_blind_embedding_space(
        tuple(replace(item, label_sources_masked=False) for item in reference),
        tuple(replace(item, label_sources_masked=False) for item in queries),
        k=1,
        minimum_regime_samples=4,
    )
    assert uncontrolled["geometry_criteria_met"] is True
    assert uncontrolled["criteria_met"] is False
    assert uncontrolled["label_source_shortcut_audit"][
        "criteria_blocked_by_uncontrolled_shortcut"
    ] is True

    held_out_missing_group = tuple(
        item
        for item in queries
        if not (item.regime == "balance" and item.direction == -1)
    )
    missing_metrics = evaluate_outcome_blind_embedding_space(
        reference,
        held_out_missing_group,
        k=1,
        minimum_regime_samples=4,
    )
    assert missing_metrics["reference_direction_regime_coverage_sufficient"] is True
    assert missing_metrics["held_out_direction_regime_coverage_sufficient"] is False
    assert missing_metrics["criteria_met"] is False


def test_embedding_evaluation_requires_both_directions_in_every_regime() -> None:
    regimes = ("continuation", "sweep_failure", "balance", "unknown")

    def sample(regime_index: int, serial: int, day: int) -> EmbeddingEvaluationSample:
        vector = np.zeros(EMBEDDING_DIM, dtype=np.float64)
        vector[regime_index] = 1.0
        vector[8 + serial] = 0.01
        vector /= np.linalg.norm(vector)
        return EmbeddingEvaluationSample(
            revision_id=f"revision-{day}-{regime_index}-{serial}",
            entry_episode_id=f"episode-{day}-{regime_index}-{serial}",
            asof=pd.Timestamp(f"2022-01-{day:02d} 10:00", tz="UTC"),
            direction=1,
            regime=regimes[regime_index],
            mechanism_label=f"mechanism-{regime_index}",
            embedding=tuple(float(value) for value in vector),
            label_sources_masked=True,
        )

    reference = tuple(
        sample(regime_index, serial, 3)
        for regime_index in range(4)
        for serial in range(2)
    )
    queries = tuple(
        sample(regime_index, serial, 4)
        for regime_index in range(4)
        for serial in (2, 3)
    )

    metrics = evaluate_outcome_blind_embedding_space(
        reference, queries, k=1, minimum_regime_samples=2
    )

    assert metrics["status"] == "insufficient_evidence"
    assert metrics["direction_regime_coverage_sufficient"] is False
    assert metrics["criteria_met"] is False


def test_embedding_evaluation_rejects_one_collapsed_direction_regime_group() -> None:
    regimes = ("continuation", "sweep_failure", "balance", "unknown")

    def sample(
        regime_index: int, serial: int, day: int, direction: int
    ) -> EmbeddingEvaluationSample:
        vector = np.zeros(EMBEDDING_DIM, dtype=np.float64)
        vector[regime_index] = 1.0
        vector[8 + (0 if direction < 0 else 8) + serial] = 0.01
        vector /= np.linalg.norm(vector)
        return EmbeddingEvaluationSample(
            revision_id=f"revision-{day}-{regime_index}-{direction}-{serial}",
            entry_episode_id=f"episode-{day}-{regime_index}-{direction}-{serial}",
            asof=pd.Timestamp(f"2022-01-{day:02d} 10:00", tz="UTC"),
            direction=direction,
            regime=regimes[regime_index],
            mechanism_label=f"mechanism-{regime_index}",
            embedding=tuple(float(value) for value in vector),
            label_sources_masked=True,
        )

    reference = list(
        sample(regime_index, serial, 3, direction)
        for regime_index in range(4)
        for direction in (-1, 1)
        for serial in range(2)
    )
    collapsed_vector = next(
        item.embedding
        for item in reference
        if item.regime == "continuation" and item.direction == 1
    )
    reference = [
        replace(item, embedding=collapsed_vector)
        if item.regime == "continuation" and item.direction == 1
        else item
        for item in reference
    ]
    queries = tuple(
        sample(regime_index, serial, 4, direction)
        for regime_index in range(4)
        for direction in (-1, 1)
        for serial in (2, 3)
    )

    metrics = evaluate_outcome_blind_embedding_space(
        tuple(reference), queries, k=1, minimum_regime_samples=4
    )

    assert metrics["direction_regime_coverage_sufficient"] is True
    assert metrics["direction_regime_noncollapse"] is False
    assert metrics["status"] == "insufficient_evidence"
    assert metrics["criteria_met"] is False


def test_baseline_comparison_requires_fixed_nontrivial_margin_and_masked_probe() -> None:
    task_names = (
        "next_event_type",
        "next_lifecycle",
        "next_event_time_bucket",
        "displacement_state",
        "draw_consumed",
        "scale_direction_alignment",
    )
    baseline = {f"{name}_nll": 1.0 for name in task_names}
    weak = compare_validation_to_baseline(
        {f"{name}_nll": 0.99 for name in task_names},
        baseline,
        shortcut_sensitive_tasks_masked=True,
    )
    strong_unmasked = compare_validation_to_baseline(
        {f"{name}_nll": 0.90 for name in task_names},
        baseline,
    )
    strong_masked = compare_validation_to_baseline(
        {f"{name}_nll": 0.90 for name in task_names},
        baseline,
        shortcut_sensitive_tasks_masked=True,
    )
    missing_task = compare_validation_to_baseline(
        {f"{name}_nll": 0.90 for name in task_names[:-1]},
        baseline,
        shortcut_sensitive_tasks_masked=True,
    )

    assert weak["all_measured_tasks_better"] is True
    assert weak["pre_registered_criteria_met"] is False
    assert strong_unmasked["pre_registered_criteria_met"] is False
    assert strong_masked["pre_registered_criteria_met"] is True
    assert missing_task["all_tasks_measured"] is False
    assert missing_task["pre_registered_criteria_met"] is False

    reconstruction = compare_reconstruction_to_baselines(
        {"candle_reconstruction": 0.8, "event_reconstruction": 0.8},
        {"masked_candle_zero_loss": 1.0, "masked_event_uniform_nll": 1.0},
    )
    missing_reconstruction = compare_reconstruction_to_baselines(
        {"candle_reconstruction": 0.8},
        {"masked_candle_zero_loss": 1.0, "masked_event_uniform_nll": 1.0},
    )
    assert reconstruction["pre_registered_criteria_met"] is True
    assert missing_reconstruction["pre_registered_criteria_met"] is False


def test_embedding_evaluation_reports_insufficient_groups_instead_of_passing() -> None:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    vector[0] = 1.0
    sample = EmbeddingEvaluationSample(
        revision_id="revision-a",
        entry_episode_id="episode-a",
        asof=pd.Timestamp("2022-01-03 10:00", tz="UTC"),
        direction=1,
        regime="continuation",
        mechanism_label="mechanism-a",
        embedding=tuple(float(value) for value in vector),
    )

    metrics = evaluate_outcome_blind_embedding_space(
        (sample,), (replace(sample, revision_id="revision-b", entry_episode_id="episode-b"),)
    )

    assert metrics["status"] == "insufficient_evidence"
    assert metrics["criteria_met"] is False


def test_embedding_index_cannot_mix_encoder_checkpoint_spaces() -> None:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    vector[0] = 1.0

    def record(checkpoint_id: str, revision_id: str) -> DecisionTimeEmbeddingRecord:
        return DecisionTimeEmbeddingRecord(
            case_id=f"case-{revision_id}",
            revision_id=revision_id,
            revision_stage="episode_created",
            revision_index=0,
            stage_identity=f"episode_created:{revision_id}",
            stage_occurrence=0,
            context_thesis_id="context-a",
            entry_episode_id=f"episode-{revision_id}",
            market_epoch_id="epoch-a",
            direction=1,
            regime="continuation",
            embedding_model_version="test-model",
            embedding_dim=EMBEDDING_DIM,
            embedding_clock="decision_time",
            embedding_asof=pd.Timestamp("2022-01-03 10:00", tz="UTC"),
            feature_max_at=pd.Timestamp("2022-01-03 10:00", tz="UTC"),
            data_split="train",
            split_role="train",
            outcome_fields_used=False,
            embedding_checkpoint_id=checkpoint_id,
            decision_embedding=tuple(float(value) for value in vector),
            mechanism_label="displacement_first_pullback",
        )

    first = record("a" * 64, "revision-a")
    second = record("a" * 64, "revision-b")
    assert validate_single_embedding_checkpoint((first, second)) == "a" * 64
    with pytest.raises(RepresentationDataError, match="different encoder checkpoints"):
        validate_single_embedding_checkpoint((first, record("b" * 64, "revision-c")))


def test_missing_torch_fails_closed() -> None:
    if TORCH_AVAILABLE:
        pytest.skip("environment has optional PyTorch")
    with pytest.raises(TorchUnavailableError, match="requires optional dependency"):
        MarketRepresentationModel()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_market_episode_exporters_roundtrip_into_retrieval_contract(
    tmp_path: Path,
) -> None:
    case = representation_case_from_market_case_input_row(
        _neutral_market_case_row(),
        _neutral_run_manifest(tmp_path),
    )
    unprocessed = prepare_representation_case(case, _neutral_store(case))
    unprocessed_batch, _ = collate_representation_cases(
        (unprocessed,), mask_probability=0.0
    )
    with pytest.raises(RepresentationDataError, match="B0 protocol"):
        encode_market_episode_records(
            MarketRepresentationModel(),
            unprocessed_batch,
            (unprocessed,),
            split_roles={case.revision_id: "train"},
            material_kind="zone_registered",
        )
    prepared = prepare_neutral_representation_case(case, _neutral_store(case))
    batch, _ = collate_representation_cases((prepared,), mask_probability=0.0)
    model = MarketRepresentationModel()
    model.train()
    masked_batch, _ = collate_representation_cases((prepared,), mask_probability=0.5)
    with pytest.raises(RepresentationDataError, match="mask_probability=0.0"):
        encode_market_episode_active_head_records(
            model, masked_batch, (prepared,), member_id="masked-member"
        )

    records = encode_market_episode_records(
        model,
        batch,
        (prepared,),
        split_roles={case.revision_id: "train"},
        material_kind="zone_registered",
    )
    assert records[0]["transition_kinds"] == list(case.transition_kinds)
    index = MarketEpisodeCaseIndex.from_mappings(
        records,
        artifact_lineage={
            "stream_manifest_sha256": "d" * 64,
            "run_manifest_sha256": "e" * 64,
            "selection_contract": (
                "first_online_market_episode_material_kind_by_revision_index_v1"
            ),
        },
    )
    assert tuple(record.material_kind for record in index.records) == (
        "zone_registered",
    )
    parsed = next(record for record in index.records if record.material_kind == "zone_registered")
    assert parsed.market_episode_id == case.market_episode_id
    assert parsed.entry_location_id == case.entry_location_id
    assert parsed.entry_path_id == case.entry_path_id
    assert parsed.decision_at == case.asof.tz_convert("UTC")
    assert "outcome" not in records[0] and "frozen_outcome" not in records[0]

    heads = encode_market_episode_active_head_records(
        model,
        batch,
        (prepared,),
        member_id="member-neutral-000",
    )
    assert set(heads[0]["head_predictions"]) == set(
        NEUTRAL_SPARSE_ACTIVE_TARGETS
    )
    assert set(heads[0]["head_predictions"]).isdisjoint(
        NEUTRAL_SPARSE_DISABLED_TARGETS
    )
    assert heads[0]["market_epoch_id"] == case.market_epoch_id
    count, disagreement, by_head = MarketEpisodeCaseIndex._ensemble(parsed, heads)
    assert count == 1 and disagreement == 0.0
    assert set(by_head) == set(NEUTRAL_SPARSE_ACTIVE_TARGETS)
    assert model.training is True


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_market_episode_exporters_reject_mixed_duplicate_and_missing_physical(
    tmp_path: Path,
) -> None:
    manifest = _neutral_run_manifest(tmp_path)
    first = representation_case_from_market_case_input_row(
        _neutral_market_case_row(), manifest
    )
    other = representation_case_from_market_case_input_row(
        _neutral_revision_row(
            asof=pd.Timestamp("2024-01-08 09:36", tz="America/New_York"),
            revision_index=0,
            source_replay_ordinal=10,
            replay_update_ordinal=6,
            lifecycle="pullback",
            transition_kinds=("first_pullback",),
            location_id="location:other",
            path_id="path:other",
        ),
        manifest,
    )

    def inference_batch(*cases: RepresentationCase):
        examples = tuple(
            prepare_neutral_representation_case(case, _neutral_store(case))
            for case in cases
        )
        batch, _ = collate_representation_cases(examples, mask_probability=0.0)
        return batch, examples

    model = MarketRepresentationModel()
    mixed_batch, mixed_examples = inference_batch(first, other)
    with pytest.raises(RepresentationDataError, match="mixes.*material"):
        encode_market_episode_records(
            model,
            mixed_batch,
            mixed_examples,
            split_roles={case.revision_id: "train" for case in (first, other)},
            material_kind="zone_registered",
        )

    duplicate = replace(
        first,
        case_id="market-case:duplicate",
        revision_id="market-revision:duplicate",
        revision_index=1,
    )
    duplicate_batch, duplicate_examples = inference_batch(first, duplicate)
    with pytest.raises(RepresentationDataError, match="duplicate MarketEpisode"):
        encode_market_episode_records(
            model,
            duplicate_batch,
            duplicate_examples,
            split_roles={case.revision_id: "train" for case in (first, duplicate)},
            material_kind="zone_registered",
        )

    missing = replace(first, entry_location_id="", entry_path_id="")
    missing_batch, missing_examples = inference_batch(missing)
    with pytest.raises(RepresentationDataError, match="physical fields"):
        encode_market_episode_active_head_records(
            model,
            missing_batch,
            missing_examples,
            member_id="member-neutral-000",
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_forward_loss_budget_and_outcome_blind_embedding() -> None:
    import torch

    cases = (
        _case(),
        _case(
            case_id="case-b",
            revision_id="revision-b",
            episode_id="episode-b",
            context_id="context-a",
            epoch_id="epoch-a",
            direction=1,
            regime="continuation",
        ),
        _case(
            case_id="case-c",
            revision_id="revision-c",
            episode_id="episode-c",
            context_id="context-c",
            epoch_id="epoch-a",
            direction=-1,
            regime="sweep_failure",
        ),
    )
    assert all(
        not case.entry_location_id and not case.entry_path_id and not case.transition_kinds
        for case in cases
    )
    prepared = tuple(prepare_representation_case(case, _store()) for case in cases)
    targets_a = (
        SelfSupervisedTarget(2, 3, 1, 0, 0, 1),
        SelfSupervisedTarget(2, 3, 1, 1, 1, 1),
        SelfSupervisedTarget(4, 2, 2, 0, 1, 2),
    )
    targets_b = tuple(replace(item, draw_consumed=1 - item.draw_consumed) for item in targets_a)
    batch, target_batch_a = collate_representation_cases(
        prepared, targets=targets_a, mask_probability=0.25, seed=7
    )
    _, target_batch_b = collate_representation_cases(
        prepared, targets=targets_b, mask_probability=0.25, seed=7
    )
    assert target_batch_a is not None and target_batch_b is not None
    model = MarketRepresentationModel()
    model.eval()
    with torch.no_grad():
        output = model(batch)
        embedding_before = model.encode(batch)
        loss_a = representation_multitask_loss(output, batch, target_batch_a)
        embedding_after = model.encode(batch)

    assert output.embedding.shape == (3, EMBEDDING_DIM)
    assert torch.isfinite(loss_a.total)
    assert model.parameter_count() < PARAMETER_BUDGET
    torch.testing.assert_close(embedding_before, embedding_after)
    # Changing separated training targets cannot alter decision-time encoding.
    assert not torch.equal(target_batch_a.draw_consumed, target_batch_b.draw_consumed)
    torch.testing.assert_close(model.encode(batch), embedding_before)
    with pytest.raises(RepresentationDataError, match="mask_probability=0.0"):
        encode_decision_time_head_records(
            model, batch, prepared, member_id="member-000"
        )
    inference_batch, _ = collate_representation_cases(
        prepared, mask_probability=0.0, seed=7
    )
    model.train()
    head_records = encode_decision_time_head_records(
        model, inference_batch, prepared, member_id="member-000"
    )
    repeated_heads = encode_decision_time_head_records(
        model, inference_batch, prepared, member_id="member-000"
    )
    embedding_records = encode_decision_time_records(
        model,
        inference_batch,
        prepared,
        split_roles={case.revision_id: "train" for case in cases},
        decision_stage="episode_created",
    )
    repeated_embeddings = encode_decision_time_records(
        model,
        inference_batch,
        prepared,
        split_roles={case.revision_id: "train" for case in cases},
        decision_stage="episode_created",
    )
    assert model.training is True
    assert repeated_heads == head_records
    assert repeated_embeddings == embedding_records
    with pytest.raises(RepresentationDataError, match="mask_probability=0.0"):
        encode_decision_time_records(
            model,
            batch,
            prepared,
            split_roles={case.revision_id: "train" for case in cases},
            decision_stage="episode_created",
        )
    assert all(record.decision_at == example.case.asof for record, example in zip(head_records, prepared))
    assert all(record.feature_max_at <= record.decision_at for record in head_records)
    assert all(record.outcome_fields_used is False for record in head_records)
    assert all(
        record.embedding_input_protocol == "inference_unmasked_v1"
        for record in embedding_records
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_future_bar_mutation_does_not_change_embedding() -> None:
    import torch

    case = _case()
    first = prepare_representation_case(case, _store())
    second = prepare_representation_case(case, _store(mutate_future=True))
    batch_a, _ = collate_representation_cases((first,), mask_probability=0.0)
    batch_b, _ = collate_representation_cases((second,), mask_probability=0.0)
    model = MarketRepresentationModel()
    model.eval()

    with torch.no_grad():
        torch.testing.assert_close(model.encode(batch_a), model.encode(batch_b))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_checkpoint_is_content_bound_and_rejects_nested_outcome_metadata(
    tmp_path: Path,
) -> None:
    model = MarketRepresentationModel()
    path = tmp_path / "representation.pt"

    save_representation_checkpoint(path, model, metadata={"member_id": "member-000"})
    restored = load_representation_checkpoint(path)

    assert representation_checkpoint_id(restored) == representation_checkpoint_id(model)
    assert not (tmp_path / ".representation.pt.tmp").exists()
    with pytest.raises(RepresentationDataError, match="future outcomes"):
        save_representation_checkpoint(
            path,
            model,
            metadata={"nested": {"frozen_outcome": {"mfe_r": 2.0}}},
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_neutral_checkpoint_v2_binds_exact_typed_metadata(tmp_path: Path) -> None:
    import torch

    model = MarketRepresentationModel()
    path = tmp_path / "neutral-v2.pt"
    supplied = _neutral_checkpoint_metadata()

    save_neutral_representation_checkpoint(path, model, metadata=supplied)
    restored = load_neutral_representation_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert representation_checkpoint_id(restored) == representation_checkpoint_id(model)
    assert set(payload["metadata"]) == {
        "training_contract",
        "direct_source_preprocessing",
        "inference_input_protocol",
        "member_id",
        "seed",
        "split_counts",
        "lineage",
        "outcome_fields_used",
        "model_capability_validated",
    }
    assert all(payload["metadata"][name] == value for name, value in supplied.items())

    payload["metadata"]["lineage"]["split_protocol"][
        "holdout_used_for_selection"
    ] = False
    torch.save(payload, path)
    with pytest.raises(RepresentationDataError, match="neutral checkpoint"):
        load_neutral_representation_checkpoint(path)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
@pytest.mark.parametrize(
    "mutation",
    (
        "missing_member",
        "member_type",
        "seed_type",
        "commit_sha256",
        "three_run_two_roles",
        "ten_run_two_roles",
        "split_count_type",
        "split_count_extra",
        "action_authority",
        "outcome_claim",
        "holdout_selected",
        "capability_claim",
        "outcome_true",
        "capability_true",
    ),
)
def test_neutral_checkpoint_save_rejects_open_or_untyped_metadata(
    tmp_path: Path,
    mutation: str,
) -> None:
    model = MarketRepresentationModel()
    metadata = copy.deepcopy(_neutral_checkpoint_metadata())
    if mutation == "missing_member":
        metadata.pop("member_id")
    elif mutation == "member_type":
        metadata["member_id"] = 0
    elif mutation == "seed_type":
        metadata["seed"] = True
    elif mutation == "commit_sha256":
        metadata["lineage"]["input_runs"][0]["repository_commit"] = "d" * 64
    elif mutation in {"three_run_two_roles", "ten_run_two_roles"}:
        total = 3 if mutation == "three_run_two_roles" else 10
        runs = metadata["lineage"]["input_runs"]
        embargo = metadata["lineage"]["split_protocol"][
            "observed_completed_session_embargo"
        ]
        while len(runs) < total:
            runs.append(copy.deepcopy(runs[len(runs) % 2]))
            embargo.append(copy.deepcopy(embargo[0]))
    elif mutation == "split_count_type":
        metadata["split_counts"]["train"] = True
    elif mutation == "split_count_extra":
        metadata["split_counts"]["test"] = 1
    elif mutation == "action_authority":
        metadata["lineage"]["source_identity"]["action_authority"] = "trade"
    elif mutation == "outcome_claim":
        metadata["outcome_summary"] = {"profitable": True}
    elif mutation == "holdout_selected":
        metadata["lineage"]["split_protocol"]["holdout_selected"] = False
    elif mutation == "capability_claim":
        metadata["lineage"]["input_runs"][0]["capability_score"] = 0.9
    elif mutation == "outcome_true":
        metadata["outcome_fields_used"] = True
    else:
        metadata["model_capability_validated"] = True

    path = tmp_path / f"invalid-{mutation}.pt"
    with pytest.raises(RepresentationDataError, match="neutral checkpoint"):
        save_neutral_representation_checkpoint(path, model, metadata=metadata)
    assert not path.exists()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch is not installed")
def test_synthetic_smoke_trains_three_independent_ensemble_members(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from shares.scripts.train_market_representation import main

    embedding_output = tmp_path / "embeddings.jsonl"
    head_output = tmp_path / "heads.jsonl"
    assert main(
        (
            "--synthetic-smoke",
            "--ensemble-size",
            "3",
            "--epochs",
            "1",
            "--batch-size",
            "12",
            "--seed",
            "73",
            "--embedding-output",
            str(embedding_output),
            "--head-output",
            str(head_output),
            "--embedding-stage",
            "episode_created",
        )
    ) == 0
    raw_output = capsys.readouterr().out
    assert "NaN" not in raw_output
    assert "Infinity" not in raw_output
    payload = json.loads(raw_output)

    assert payload["ensemble_size"] == 3
    assert len(payload["independent_checkpoint_ids"]) == 3
    assert len(set(payload["independent_checkpoint_ids"])) == 3
    assert payload["head_probability_interface"] == (
        "decision_time_head_probabilities(output)"
    )
    assert payload["outcome_fields_used"] is False
    assert payload["ensemble_stability"]["member_count"] == 3
    assert payload["ensemble_stability"]["criteria_met"] is False
    embedding_rows = [
        json.loads(line) for line in embedding_output.read_text().splitlines()
    ]
    head_rows = [json.loads(line) for line in head_output.read_text().splitlines()]
    assert len(embedding_rows) == 12
    assert len(head_rows) == 36
    assert {row["embedding_input_protocol"] for row in embedding_rows} == {
        "inference_unmasked_v1"
    }
    assert {row["input_protocol"] for row in head_rows} == {
        "inference_unmasked_v1"
    }
    assert len({row["embedding_checkpoint_id"] for row in embedding_rows}) == 1
    assert len({row["checkpoint_id"] for row in head_rows}) == 3
    for artifact_path, expected_schema in (
        (embedding_output, "smc-decision-time-embeddings-v1"),
        (head_output, "smc-decision-time-self-supervised-heads-v1"),
    ):
        manifest_path = artifact_path.with_name(
            f"{artifact_path.name}.manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema"] == expected_schema
        assert manifest["status"] == "complete"
        assert manifest["input_protocol"] == "inference_unmasked_v1"
        assert manifest["artifact_sha256"] == hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()
        assert len(manifest["record_identity_sha256"]) == 64
        assert "decision_at" in manifest["record_identity_fields"]
        assert manifest["decision_stage"] == "episode_created"
        assert manifest["outcome_fields_used"] is False
        assert not (tmp_path / f".{artifact_path.name}.tmp").exists()
