from __future__ import annotations

from dataclasses import replace
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import shares.core.market_cases as market_cases_module
from shares.core.artifact_stream import (
    canonical_json,
    new_stream_state,
    write_stream_shard,
)
from eyes.core.interaction import INTERACTION_ARTIFACT_COLLECTION_NAMES
from shares.core.market_cases import (
    BASE_TRANSITION_ARTIFACT_COLLECTION_NAMES,
    MARKET_CASE_INPUT_FIELD_TYPES,
    MARKET_CASE_PROTOCOL,
    MarketEpisodeCaseRecorder,
    expected_market_case_run_identity,
    validate_market_case_input_row,
    validate_market_case_rows,
)
from shares.core.model import (
    Direction,
    DirectionalObstructionView,
    GlobalMarketContext,
    InteractionUpdate,
    MarketMode,
    NEUTRAL_MARKET_STATE_SCHEMA_VERSION,
    ScaleRelation,
    ScaleRelationState,
    Timeframe,
)
from shares.core.scene_graph import (
    SceneEdgeKind,
    StructuralScale,
    market_episode_id,
)

from eyes.tests.test_v3_group5_primitives import _fvg


NY = "America/New_York"


def _at(minute: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-08 09:30", tz=NY) + pd.Timedelta(minutes=minute)


def _claim(*, evidence: str = "evidence:1") -> SimpleNamespace:
    return SimpleNamespace(
        thesis_id="thesis:1",
        root_id="root:1",
        relation="aligned",
        thesis_direction="long",
        mechanism="displacement_continuation",
        authority_relation="aligned_authority",
        evidence_revision_id=evidence,
        lifecycle="active",
    )


def _episode(
    minute: int,
    *,
    epoch: str = "epoch:1",
    formed_minute: int = 0,
    location: str = "location:1",
    path: str = "path:1",
    source_zone_id: str = "fvg:1",
    binding_status: str = "unique",
    claims: tuple[SimpleNamespace, ...] = (),
    active_claim_ids: tuple[str, ...] = (),
    claim_status: str = "unbound",
    pullback_minute: int | None = None,
    trigger_minute: int | None = None,
    successful: bool = False,
    terminal_reason: str | None = None,
) -> SimpleNamespace:
    terminal = terminal_reason is not None
    pullback = pullback_minute is not None
    trigger = trigger_minute is not None
    return SimpleNamespace(
        episode_id=market_episode_id(epoch, location, path, Direction.LONG),
        market_epoch_id=epoch,
        symbol="MES",
        instrument_id=1,
        direction="long",
        entry_location_id=location,
        entry_path_id=path,
        source_zone_id=source_zone_id,
        source_displacement_id="displacement:1",
        entry_location_protocol_hash="entry-location-protocol:1",
        source_zone_detector_protocol_hash="group3-protocol:1",
        source_zone_kind="fvg",
        source_zone_protocol_hash="fvg-protocol:1",
        source_bos_id=None,
        lower_bound=5000.0,
        upper_bound=5001.0,
        midpoint=5000.5,
        near_edge=5001.0,
        far_edge=5000.0,
        failure_boundary=5000.0,
        formed_at=_at(formed_minute),
        updated_at=_at(minute),
        binding_status=binding_status,
        claims=claims,
        active_claim_ids=active_claim_ids,
        claim_status=claim_status,
        first_pullback_step_id="step:pullback" if pullback else None,
        first_pullback_at=_at(pullback_minute) if pullback else None,
        trigger_step_id="step:trigger" if trigger else None,
        trigger_event_id="micro-bos:1" if trigger else None,
        trigger_at=_at(trigger_minute) if trigger else None,
        successful_pulse_at=_at(trigger_minute) if successful else None,
        successful_pulse_reason="micro_bos_aligned" if successful else None,
        lifecycle=(
            "terminal"
            if terminal
            else "triggered"
            if trigger
            else "pullback"
            if pullback
            else "registered"
        ),
        terminal_at=_at(minute) if terminal else None,
        terminal_reason=terminal_reason,
    )


def _global_context(minute: int, *, epoch: str) -> GlobalMarketContext:
    return GlobalMarketContext(
        updated_at=_at(minute),
        scene_revision_id=f"scene:{minute}",
        market_epoch_id=epoch,
        authority_stack=(),
        market_mode=MarketMode.UNCERTAIN,
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
        external_draw_candidates={"above": ("draw:1",), "below": ()},
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
        open_market_theses=(),
    )


def _snapshot(
    minute: int,
    *,
    epoch: str = "epoch:1",
    episodes: tuple[SimpleNamespace, ...] = (),
    transitions: tuple[SimpleNamespace, ...] = (),
    retired: tuple[tuple[str, str], ...] = (),
    anomalies: tuple[str, ...] = (),
    eventful: bool = False,
    belief_diagnostic: object | None = None,
) -> SimpleNamespace:
    asof = _at(minute)
    context = _global_context(minute, epoch=epoch)
    frames = {
        timeframe: SimpleNamespace(cutoff=asof, bars=20 + max(minute, 0))
        for timeframe in (
            Timeframe.H4,
            Timeframe.H1,
            Timeframe.M15,
            Timeframe.M5,
            Timeframe.M1,
        )
    }
    event = replace(
        _fvg(asof, identity=f"fvg:{minute}"),
        symbol="MES",
        instrument_id=1,
    )
    observation = SimpleNamespace(
        asof=asof,
        symbol="MES",
        instrument_id=1,
        price=5000.0,
        frames=frames,
        anomalies=anomalies,
        typed_transition_delta_available=True,
        scene_revision_id=f"scene:{minute}",
        scene_added_node_ids=((f"node:{minute}",) if eventful else ()),
        scene_revised_node_ids=(),
        scene_added_edge_ids=((f"edge:{minute}",) if eventful else ()),
        scene_revised_edge_ids=(),
        scene_resolution_event_ids=(),
        liquidity_inventory_transitions_this_update=(),
        liquidity_pool_transitions_this_update=(),
        group3_fvg_transitions_this_update=((event,) if eventful else ()),
        group3_order_block_transitions_this_update=(),
        group4_range_transitions_this_update=(),
        group4_manipulation_transitions_this_update=(),
        interaction_update=InteractionUpdate(
            zone_interactions=(),
            reacceptance_interactions=(),
            micro_break_facts=(),
            interaction_paths=(),
        ),
    )
    neutral = SimpleNamespace(
        schema_version=NEUTRAL_MARKET_STATE_SCHEMA_VERSION,
        asof=asof,
        market_epoch_id=epoch,
        scene_revision_id=f"scene:{minute}",
        global_context=context,
        market_episodes=episodes,
        episode_transitions_this_update=transitions,
        retired_episode_ids_this_update=tuple(item[0] for item in retired),
        retirement_reasons_this_update=retired,
    )
    return SimpleNamespace(
        observation=observation,
        belief=belief_diagnostic,
        neutral_market_state=neutral,
    )


def _bar(snapshot: SimpleNamespace, *, synthetic: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        start=snapshot.observation.asof - pd.Timedelta(minutes=1),
        end=snapshot.observation.asof,
        symbol="MES",
        instrument_id=1,
        synthetic_no_trade=synthetic,
    )


def _recorder(*, capture_minute: int = 0) -> MarketEpisodeCaseRecorder:
    return MarketEpisodeCaseRecorder(capture_start=_at(capture_minute))


def _observe(
    recorder: MarketEpisodeCaseRecorder,
    snapshot: SimpleNamespace,
    *,
    source_ordinal: int,
    replay_ordinal: int,
    synthetic: bool = False,
) -> None:
    recorder.observe(
        snapshot,
        source_bar=_bar(snapshot, synthetic=synthetic),
        source_row_ordinal=source_ordinal,
        replay_update_ordinal=replay_ordinal,
        scene_graph=None,
    )


def _row(recorder: MarketEpisodeCaseRecorder) -> dict[str, object]:
    rows = recorder.drain_input_rows()
    assert len(rows) == 1
    return rows[0].to_dict()


def test_input_only_schema_has_one_stable_revision_and_no_legacy_dependencies() -> None:
    assert set(MARKET_CASE_INPUT_FIELD_TYPES) == {
        "revision_id",
        "revision_index",
        "market_epoch_id",
        "market_episode_id",
        "asof",
        "direction",
        "lifecycle",
        "entry_location_id",
        "entry_path_id",
        "revision_stage",
        "transition_kinds_json",
        "observation_transition_json",
        "scene_graph_delta_json",
        "neutral_global_context_json",
        "ohlcv_prefix_refs_json",
        "source_replay_ordinal",
        "replay_update_ordinal",
        "source_bar_synthetic",
    }
    assert MARKET_CASE_PROTOCOL["input_only"] is True
    assert MARKET_CASE_PROTOCOL["protocol_version"] == (
        "market-episode-input-only-1.4.0"
    )
    assert MARKET_CASE_PROTOCOL["neutral_runtime_schema_version"] == 2
    assert MARKET_CASE_PROTOCOL["runtime_source"] == (
        "NeutralEngineSnapshot.neutral_market_state"
    )
    assert MARKET_CASE_PROTOCOL["interaction_update_schema_version"] == 2
    assert MARKET_CASE_PROTOCOL["interaction_authority"] == (
        "raw_eye_physical_facts_only_no_brain_interpretation"
    )
    assert MARKET_CASE_PROTOCOL["interaction_collections"] == list(
        INTERACTION_ARTIFACT_COLLECTION_NAMES
    )
    assert market_cases_module.MARKET_CASE_INPUT_DATA_CONTINUITY_POLICY == {
        "maximum_no_trade_gap_minutes": 5,
        "allow_same_contract_data_gap_reset": True,
        "data_gap_reset_anomaly": "data_gap_history_reset",
        "allow_cross_contract_data_gap_reset": False,
        "synthesize_over_cap_missing_minutes": False,
    }
    assert "data_continuity" not in MARKET_CASE_PROTOCOL
    assert expected_market_case_run_identity() == {
        "recorder_schema_version": 2,
        "protocol": dict(MARKET_CASE_PROTOCOL),
        "input_stream": "market_case_input_shards",
        "input_only": True,
        "output_affects_model": False,
    }
    for removed in (
        "MARKET_CASE_OUTCOME_FIELD_TYPES",
        "write_market_episode_case_library_manifest",
        "write_dual_case_parity_manifest",
        "market_input_fingerprint",
        "market_record_fingerprint",
    ):
        assert not hasattr(market_cases_module, removed)
    source = Path(market_cases_module.__file__).read_text(encoding="utf-8")
    assert "from .causal_cases" not in source
    assert "from .shadow_outcome" not in source


def test_schema_one_neutral_snapshot_is_rejected_fail_closed() -> None:
    episode = _episode(0)
    snapshot = _snapshot(
        0,
        episodes=(episode,),
        transitions=(episode,),
    )
    snapshot.neutral_market_state.schema_version = 1
    with pytest.raises(
        ValueError,
        match="neutral runtime schema version differs",
    ):
        _observe(
            _recorder(),
            snapshot,
            source_ordinal=0,
            replay_ordinal=0,
        )


def test_formation_emits_one_input_row_with_same_clock_eye_scene_context_and_prefix() -> None:
    episode = _episode(0)
    snapshot = _snapshot(
        0,
        episodes=(episode,),
        transitions=(episode,),
        eventful=True,
        belief_diagnostic=SimpleNamespace(playbook="liquidity_sweep_reversal"),
    )
    recorder = _recorder()
    _observe(recorder, snapshot, source_ordinal=0, replay_ordinal=0)
    row = _row(recorder)
    validate_market_case_input_row(row)
    assert row["market_epoch_id"] == "epoch:1"
    assert row["market_episode_id"] == episode.episode_id
    assert row["revision_index"] == 0
    assert json.loads(row["transition_kinds_json"]) == [
        "episode_created",
        "zone_registered",
    ]
    eye = json.loads(row["observation_transition_json"])
    scene = json.loads(row["scene_graph_delta_json"])
    context = json.loads(row["neutral_global_context_json"])
    prefixes = json.loads(row["ohlcv_prefix_refs_json"])
    assert eye["asof"] == _at(0).isoformat()
    assert eye["collections"]["group3_fvg_transitions_this_update"][0][
        "fvg_id"
    ] == "fvg:0"
    assert scene["added_node_ids"] == ["node:0"]
    assert scene["added_edge_ids"] == ["edge:0"]
    assert scene["relation_descriptors_complete"] is False
    assert context["market_epoch_id"] == "epoch:1"
    assert [item["timeframe"] for item in prefixes] == [
        "4H",
        "1H",
        "15m",
        "5m",
        "1m",
    ]
    assert all(item["replay_view_1m_row_end_exclusive"] == 1 for item in prefixes)
    assert "playbook" not in repr(row)


def test_playbook_diagnostics_do_not_change_input_row_or_revision_identity() -> None:
    episode = _episode(0)
    first = _snapshot(
        0,
        episodes=(episode,),
        transitions=(episode,),
        belief_diagnostic=SimpleNamespace(playbook="dfp"),
    )
    second = _snapshot(
        0,
        episodes=(episode,),
        transitions=(episode,),
        belief_diagnostic=SimpleNamespace(playbook="lsr"),
    )
    left = _recorder()
    right = _recorder()
    _observe(left, first, source_ordinal=0, replay_ordinal=0)
    _observe(right, second, source_ordinal=0, replay_ordinal=0)
    assert _row(left) == _row(right)


def test_same_clock_milestones_collapse_to_one_ordered_transition_row() -> None:
    episode = _episode(
        0,
        formed_minute=-2,
        pullback_minute=-1,
        trigger_minute=0,
        successful=True,
        terminal_reason="physical_path_terminal",
    )
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(0, episodes=(episode,), transitions=(episode,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    row = _row(recorder)
    assert json.loads(row["transition_kinds_json"]) == [
        "episode_created",
        "zone_registered",
        "first_pullback",
        "trigger",
        "successful_pulse",
        "terminal",
    ]
    assert row["lifecycle"] == "terminal"
    assert recorder.summary["terminal_rows"] == 1


def test_relation_only_churn_updates_custody_without_rows_or_revision() -> None:
    formed = _episode(0)
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(0, episodes=(formed,), transitions=(formed,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    recorder.drain_input_rows()
    latest = formed
    for minute in range(1, 101):
        has_claim = minute % 2 == 1
        latest = _episode(
            minute,
            binding_status=("ambiguous" if minute % 3 == 0 else "unique"),
            claims=((_claim(evidence=f"evidence:{minute}"),) if has_claim else ()),
            active_claim_ids=(("thesis:1",) if has_claim else ()),
            claim_status=("unique" if has_claim else "unbound"),
        )
        _observe(
            recorder,
            _snapshot(minute, episodes=(latest,), transitions=(latest,)),
            source_ordinal=minute,
            replay_ordinal=minute,
        )
        assert recorder.drain_input_rows() == ()
        assert recorder.summary["pending_row_count"] == 0

    _observe(
        recorder,
        _snapshot(101, episodes=(latest,), transitions=()),
        source_ordinal=101,
        replay_ordinal=101,
    )
    assert recorder.drain_input_rows() == ()
    milestone = _episode(
        102,
        binding_status="unique",
        claims=(_claim(evidence="evidence:physical"),),
        active_claim_ids=("thesis:1",),
        claim_status="unique",
        pullback_minute=102,
    )
    _observe(
        recorder,
        _snapshot(102, episodes=(milestone,), transitions=(milestone,)),
        source_ordinal=102,
        replay_ordinal=102,
    )
    row = _row(recorder)
    assert row["revision_index"] == 1
    assert json.loads(row["transition_kinds_json"]) == ["first_pullback"]
    assert set(recorder.summary["transition_kind_counts"]) == {
        "episode_created",
        "zone_registered",
        "first_pullback",
        "trigger",
        "successful_pulse",
        "terminal",
    }
    assert recorder.summary["rows_emitted"] == 2


def test_terminal_is_one_row_and_cannot_transition_or_revive() -> None:
    formed = _episode(0)
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(0, episodes=(formed,), transitions=(formed,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    recorder.drain_input_rows()
    terminal = _episode(1, terminal_reason="invalidation")
    _observe(
        recorder,
        _snapshot(1, episodes=(terminal,), transitions=(terminal,)),
        source_ordinal=1,
        replay_ordinal=1,
    )
    assert json.loads(_row(recorder)["transition_kinds_json"]) == ["terminal"]
    _observe(
        recorder,
        _snapshot(2, episodes=(terminal,), transitions=()),
        source_ordinal=2,
        replay_ordinal=2,
    )
    revived = _episode(0)
    with pytest.raises(ValueError, match="terminal MarketEpisode custody changed"):
        _observe(
            recorder,
            _snapshot(3, episodes=(revived,), transitions=()),
            source_ordinal=3,
            replay_ordinal=3,
        )


def test_success_retirement_is_bounded_and_reappearance_fails_closed() -> None:
    successful = _episode(
        0,
        formed_minute=-2,
        pullback_minute=-1,
        trigger_minute=0,
        successful=True,
    )
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(0, episodes=(successful,), transitions=(successful,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    recorder.drain_input_rows()
    _observe(
        recorder,
        _snapshot(
            1,
            episodes=(),
            transitions=(),
            retired=((successful.episode_id, "upstream_compacted_after_success"),),
        ),
        source_ordinal=1,
        replay_ordinal=1,
    )
    assert recorder.summary["active_episode_count"] == 0
    assert recorder.summary["closed_identity_count"] == 1
    with pytest.raises(ValueError, match="retired MarketEpisode reappeared"):
        _observe(
            recorder,
            _snapshot(2, episodes=(successful,), transitions=()),
            source_ordinal=2,
            replay_ordinal=2,
        )


def test_reset_requires_new_epoch_and_reused_physical_ids_restart_revision_index() -> None:
    first = _episode(0)
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(0, episodes=(first,), transitions=(first,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    recorder.drain_input_rows()
    with pytest.raises(ValueError, match="did not advance market epoch"):
        _observe(
            recorder,
            _snapshot(
                1,
                episodes=(first,),
                transitions=(),
                anomalies=("data_gap_history_reset",),
            ),
            source_ordinal=1,
            replay_ordinal=1,
        )
    reset = _episode(1, epoch="epoch:2", formed_minute=1)
    _observe(
        recorder,
        _snapshot(
            1,
            epoch="epoch:2",
            episodes=(reset,),
            transitions=(reset,),
            anomalies=("data_gap_history_reset",),
        ),
        source_ordinal=1,
        replay_ordinal=1,
    )
    row = _row(recorder)
    assert row["market_epoch_id"] == "epoch:2"
    assert row["revision_index"] == 0
    prefixes = json.loads(row["ohlcv_prefix_refs_json"])
    assert all(item["replay_view_1m_row_start"] == 1 for item in prefixes)
    assert recorder.summary["epoch_resets"] == 1


def test_reset_accepts_prior_epoch_terminals_and_current_epoch_transitions() -> None:
    recorder = _recorder()
    previous = _episode(0)
    _observe(
        recorder,
        _snapshot(0, episodes=(previous,), transitions=(previous,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    recorder.drain_input_rows()

    prior_terminal = _episode(1, terminal_reason="data_gap_reset")
    current = _episode(
        1,
        epoch="epoch:2",
        formed_minute=1,
        location="location:2",
        path="path:2",
    )
    duplicate_snapshot = _snapshot(
        1,
        epoch="epoch:2",
        episodes=(current,),
        transitions=(prior_terminal, prior_terminal, current),
        anomalies=("data_gap_history_reset",),
    )
    with pytest.raises(ValueError, match="duplicates MarketEpisode transition"):
        _observe(
            _recorder(),
            duplicate_snapshot,
            source_ordinal=0,
            replay_ordinal=0,
        )
    _observe(
        recorder,
        _snapshot(
            1,
            epoch="epoch:2",
            episodes=(current,),
            transitions=(prior_terminal, current),
            anomalies=("data_gap_history_reset",),
        ),
        source_ordinal=1,
        replay_ordinal=1,
    )

    row = _row(recorder)
    assert row["market_epoch_id"] == "epoch:2"
    assert row["market_episode_id"] == current.episode_id
    assert row["revision_index"] == 0
    assert recorder.summary["epoch_resets"] == 1


def test_same_epoch_current_episode_transition_completeness_stays_fail_closed() -> None:
    current = _episode(1)
    with pytest.raises(
        ValueError,
        match="current MarketEpisode transitions are incomplete",
    ):
        _observe(
            _recorder(),
            _snapshot(1, episodes=(current,), transitions=()),
            source_ordinal=0,
            replay_ordinal=0,
        )


def test_data_gap_epoch_cannot_revive_terminal_identity_and_rows_are_future_safe() -> None:
    recorder = _recorder()
    formed = _episode(0)
    _observe(
        recorder,
        _snapshot(0, episodes=(formed,), transitions=(formed,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    rows = [item.to_dict() for item in recorder.drain_input_rows()]

    terminal = _episode(1, terminal_reason="invalidation")
    _observe(
        recorder,
        _snapshot(1, episodes=(terminal,), transitions=(terminal,)),
        source_ordinal=1,
        replay_ordinal=1,
    )
    rows.extend(item.to_dict() for item in recorder.drain_input_rows())

    restarted = _episode(2, epoch="epoch:2", formed_minute=2)
    _observe(
        recorder,
        _snapshot(
            2,
            epoch="epoch:2",
            episodes=(restarted,),
            transitions=(restarted,),
            anomalies=("data_gap_history_reset",),
        ),
        source_ordinal=2,
        replay_ordinal=2,
    )
    rows.extend(item.to_dict() for item in recorder.drain_input_rows())

    assert terminal.episode_id != restarted.episode_id
    assert [row["market_episode_id"] for row in rows].count(
        terminal.episode_id
    ) == 2
    assert all(
        row["market_episode_id"] != terminal.episode_id
        for row in rows
        if row["market_epoch_id"] == "epoch:2"
    )
    assert recorder.summary["epoch_resets"] == 1
    for row in rows:
        validate_market_case_input_row(row)
        asof = pd.Timestamp(row["asof"])
        assert all(
            pd.Timestamp(prefix["cutoff"]) <= asof
            for prefix in json.loads(row["ohlcv_prefix_refs_json"])
        )

    with pytest.raises(ValueError, match="MarketEpisode crossed market epoch"):
        _observe(
            recorder,
            _snapshot(
                3,
                epoch="epoch:2",
                episodes=(restarted, terminal),
                transitions=(),
            ),
            source_ordinal=3,
            replay_ordinal=3,
        )


def test_future_episode_clock_and_recursive_future_key_are_rejected() -> None:
    future = _episode(0, pullback_minute=1)
    recorder = _recorder()
    with pytest.raises(ValueError, match="future-dated"):
        _observe(
            recorder,
            _snapshot(0, episodes=(future,), transitions=(future,)),
            source_ordinal=0,
            replay_ordinal=0,
        )
    valid = _episode(0)
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(0, episodes=(valid,), transitions=(valid,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    row = _row(recorder)
    context = json.loads(row["neutral_global_context_json"])
    context["unknown_evidence"] = [{"future_profit": 99.0}]
    row["neutral_global_context_json"] = canonical_json(context).decode("utf-8")
    row["revision_id"] = market_cases_module._expected_revision_id(row)
    with pytest.raises(ValueError, match="future/outcome key"):
        validate_market_case_input_row(row)


@pytest.mark.parametrize(
    "outcome",
    ("aligned", "opposed", "simultaneous_unknown", "ambiguous_same_clock"),
)
def test_raw_interaction_artifact_rejects_brain_outcome(outcome: str) -> None:
    episode = _episode(0)
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(0, episodes=(episode,), transitions=(episode,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    row = _row(recorder)
    observation = json.loads(row["observation_transition_json"])
    micro_bos = {
        "reference_id": "micro-bos-reference:1",
        "protocol_hash": "group5-protocol:1",
        "context_kind": "fvg",
        "context_id": "fvg:1",
        "expected_direction": "long",
        "anchor_at": (
            _at(0).isoformat()
            if outcome == "simultaneous_unknown"
            else _at(-2).isoformat()
        ),
        "bos_id": "bos:1",
        "bos_direction": "long" if outcome == "aligned" else "short",
        "target_swing_id": "swing:1",
        "scope": "local",
        "pending_at": _at(-1).isoformat(),
        "resolved_at": _at(0).isoformat(),
        "relation": (
            "same_clock_unknown"
            if outcome == "simultaneous_unknown"
            else "strictly_after"
        ),
        "outcome": outcome,
        "qualified": outcome == "aligned",
        "strength": 0.75,
    }
    for invalid_payload in (
        micro_bos,
        {**micro_bos, "outcome": "future_target"},
        {**micro_bos, "outcome": {"value": outcome}},
    ):
        tampered = dict(row)
        invalid = {
            **observation,
            "collections": {
                **observation["collections"],
                "interaction_micro_break_facts": [invalid_payload],
            },
        }
        tampered["observation_transition_json"] = canonical_json(invalid).decode(
            "utf-8"
        )
        tampered["revision_id"] = market_cases_module._expected_revision_id(tampered)
        with pytest.raises(
            ValueError,
            match="interaction artifact|future/outcome key",
        ):
            validate_market_case_input_row(tampered)


def test_synthetic_transition_uses_last_real_prefix_and_is_explicit() -> None:
    formed = _episode(0)
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(0, episodes=(formed,), transitions=(formed,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    recorder.drain_input_rows()
    pullback = _episode(1, pullback_minute=1)
    _observe(
        recorder,
        _snapshot(1, episodes=(pullback,), transitions=(pullback,)),
        source_ordinal=1,
        replay_ordinal=1,
        synthetic=True,
    )
    row = _row(recorder)
    assert row["source_bar_synthetic"] is True
    assert all(
        item["replay_view_1m_row_end_exclusive"] == 1
        for item in json.loads(row["ohlcv_prefix_refs_json"])
    )


def test_prime_seeds_warmup_without_rows_then_emits_first_material_revision_zero() -> None:
    recorder = _recorder(capture_minute=0)
    warmup = _episode(-1, formed_minute=-2)
    snapshot = _snapshot(-1, episodes=(warmup,), transitions=(warmup,))
    recorder.prime(
        snapshot,
        source_bar=_bar(snapshot),
        source_row_ordinal=0,
        replay_update_ordinal=0,
    )
    assert recorder.drain_input_rows() == ()
    claimed = _episode(
        0,
        formed_minute=-2,
        claims=(_claim(),),
        active_claim_ids=("thesis:1",),
        claim_status="unique",
    )
    _observe(
        recorder,
        _snapshot(0, episodes=(claimed,), transitions=(claimed,)),
        source_ordinal=1,
        replay_ordinal=1,
    )
    assert recorder.drain_input_rows() == ()
    pullback = _episode(
        1,
        formed_minute=-2,
        claims=(_claim(),),
        active_claim_ids=("thesis:1",),
        claim_status="unique",
        pullback_minute=1,
    )
    _observe(
        recorder,
        _snapshot(1, episodes=(pullback,), transitions=(pullback,)),
        source_ordinal=2,
        replay_ordinal=2,
    )
    row = _row(recorder)
    assert row["revision_index"] == 0
    assert json.loads(row["transition_kinds_json"]) == ["first_pullback"]
    with pytest.raises(ValueError, match="reached capture start"):
        recorder.prime(
            _snapshot(2, episodes=(pullback,), transitions=()),
            source_bar=_bar(_snapshot(2, episodes=(pullback,), transitions=())),
            source_row_ordinal=3,
            replay_update_ordinal=3,
        )


def test_checkpoint_resume_is_deterministic_and_drain_releases_pending_rows() -> None:
    formed = _episode(0)
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(0, episodes=(formed,), transitions=(formed,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    first = recorder.drain_input_rows()
    assert first and recorder.summary["pending_row_count"] == 0
    restored = pickle.loads(pickle.dumps(recorder))
    pullback = _episode(1, pullback_minute=1)
    for candidate in (recorder, restored):
        _observe(
            candidate,
            _snapshot(1, episodes=(pullback,), transitions=(pullback,)),
            source_ordinal=1,
            replay_ordinal=1,
        )
    assert recorder.drain_input_rows() == restored.drain_input_rows()
    assert recorder.summary["active_episode_count"] == 1
    assert recorder.summary["pending_row_count"] == 0
    assert not hasattr(recorder, "_outcome_rows")
    assert not hasattr(recorder, "_all_rows")


@pytest.mark.parametrize(
    "mutate",
    (
        lambda state: state.pop("_protocol_version"),
        lambda state: state.__setitem__("_extra", True),
        lambda state: state.__setitem__("_recorder_schema_version", 1),
        lambda state: state.__setitem__(
            "_protocol_version", "market-episode-input-only-1.2.0"
        ),
    ),
)
def test_recorder_pickle_identity_rejects_invalid_state_failure_atomically(
    mutate: object,
) -> None:
    recorder = _recorder()
    before = dict(recorder.__dict__)
    damaged = dict(recorder.__getstate__())
    mutate(damaged)

    with pytest.raises(ValueError, match="state identity changed"):
        recorder.__setstate__(damaged)

    assert recorder.__dict__ == before


def test_unavailable_typed_delta_and_nonexact_raw_payloads_fail_closed() -> None:
    episode = _episode(0)
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(
            0,
            episodes=(episode,),
            transitions=(episode,),
            eventful=True,
        ),
        source_ordinal=0,
        replay_ordinal=0,
    )
    row = _row(recorder)

    unavailable = dict(row)
    observation = json.loads(unavailable["observation_transition_json"])
    observation["typed_transition_delta_available"] = False
    unavailable["observation_transition_json"] = canonical_json(observation).decode(
        "utf-8"
    )
    unavailable["revision_id"] = market_cases_module._expected_revision_id(
        unavailable
    )
    with pytest.raises(ValueError, match="unavailable typed transition"):
        validate_market_case_input_row(unavailable)

    for injected_key in ("brain_response", "qualified"):
        injected = dict(row)
        context = json.loads(injected["neutral_global_context_json"])
        context[injected_key] = True
        injected["neutral_global_context_json"] = canonical_json(context).decode(
            "utf-8"
        )
        injected["revision_id"] = market_cases_module._expected_revision_id(
            injected
        )
        with pytest.raises(ValueError, match="future/outcome key"):
            validate_market_case_input_row(injected)

    for mutation in ("missing", "extra"):
        malformed = dict(row)
        observation = json.loads(malformed["observation_transition_json"])
        fvg = observation["collections"][
            "group3_fvg_transitions_this_update"
        ][0]
        if mutation == "missing":
            fvg.pop("qualification")
        else:
            fvg["brain_response"] = True
        malformed["observation_transition_json"] = canonical_json(
            observation
        ).decode("utf-8")
        malformed["revision_id"] = market_cases_module._expected_revision_id(
            malformed
        )
        with pytest.raises(
            ValueError,
            match="future/outcome key|artifact shape changed",
        ):
            validate_market_case_input_row(malformed)


@pytest.mark.parametrize("collection", BASE_TRANSITION_ARTIFACT_COLLECTION_NAMES)
@pytest.mark.parametrize("payload", ({}, {"unexpected": True}))
def test_each_base_eye_collection_requires_an_exact_dto(
    collection: str,
    payload: dict[str, object],
) -> None:
    episode = _episode(0)
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(0, episodes=(episode,), transitions=(episode,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    malformed = _row(recorder)
    observation = json.loads(malformed["observation_transition_json"])
    observation["collections"][collection] = [payload]
    malformed["observation_transition_json"] = canonical_json(observation).decode(
        "utf-8"
    )
    malformed["revision_id"] = market_cases_module._expected_revision_id(
        malformed
    )

    with pytest.raises(ValueError, match="artifact shape changed"):
        validate_market_case_input_row(malformed)


def test_scene_descriptors_are_exact_complete_and_raw_only() -> None:
    episode = _episode(0)
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(
            0,
            episodes=(episode,),
            transitions=(episode,),
            eventful=True,
        ),
        source_ordinal=0,
        replay_ordinal=0,
    )
    row = _row(recorder)
    scene = json.loads(row["scene_graph_delta_json"])
    descriptor = {
        "change_kind": "added",
        "edge_id": scene["added_edge_ids"][0],
        "relation": SceneEdgeKind.CREATES.value,
        "lifecycle": "active",
        "observed_at": row["asof"].isoformat(),
        "source": {
            "node_id": "node:source",
            "kind": "structure",
            "role": "structure",
            "timeframe": Timeframe.H1.value,
            "structural_scale": StructuralScale.EXTERNAL.value,
            "lifecycle": "active",
        },
        "target": {
            "node_id": "node:target",
            "kind": "fvg",
            "role": "fvg",
            "timeframe": Timeframe.M5.value,
            "structural_scale": StructuralScale.INTERNAL.value,
            "lifecycle": "active",
        },
    }
    scene["relation_descriptors"] = [descriptor]
    scene["relation_descriptors_complete"] = True
    valid = dict(row)
    valid["scene_graph_delta_json"] = canonical_json(scene).decode("utf-8")
    valid["revision_id"] = market_cases_module._expected_revision_id(valid)
    validate_market_case_input_row(valid)

    brain_injected = json.loads(valid["scene_graph_delta_json"])
    brain_injected["relation_descriptors"][0]["source"]["playbook"] = "LSR"
    injected = dict(valid)
    injected["scene_graph_delta_json"] = canonical_json(brain_injected).decode(
        "utf-8"
    )
    injected["revision_id"] = market_cases_module._expected_revision_id(
        injected
    )
    with pytest.raises(ValueError, match="future/outcome key"):
        validate_market_case_input_row(injected)

    brain_value_mutations = (
        (("relation",), "playbook"),
        (("lifecycle",), "qualified"),
        (("source", "kind"), "decision"),
        (("source", "role"), "selected_action"),
        (("target", "kind"), "risk"),
        (("target", "role"), "hard_gate"),
    )
    for field_path, brain_value in brain_value_mutations:
        damaged_scene = json.loads(valid["scene_graph_delta_json"])
        descriptor_state = damaged_scene["relation_descriptors"][0]
        if len(field_path) == 1:
            descriptor_state[field_path[0]] = brain_value
        else:
            descriptor_state[field_path[0]][field_path[1]] = brain_value
        damaged = dict(valid)
        damaged["scene_graph_delta_json"] = canonical_json(
            damaged_scene
        ).decode("utf-8")
        damaged["revision_id"] = market_cases_module._expected_revision_id(
            damaged
        )
        with pytest.raises(ValueError, match="Scene"):
            validate_market_case_input_row(damaged)

    for mutation in ("wrong_change_kind", "duplicate", "missing_endpoint_key"):
        damaged_scene = json.loads(valid["scene_graph_delta_json"])
        if mutation == "wrong_change_kind":
            damaged_scene["relation_descriptors"][0]["change_kind"] = "revised"
        elif mutation == "duplicate":
            damaged_scene["relation_descriptors"].append(
                dict(damaged_scene["relation_descriptors"][0])
            )
        else:
            damaged_scene["relation_descriptors"][0]["target"].pop("role")
        damaged = dict(valid)
        damaged["scene_graph_delta_json"] = canonical_json(damaged_scene).decode(
            "utf-8"
        )
        damaged["revision_id"] = market_cases_module._expected_revision_id(
            damaged
        )
        with pytest.raises(ValueError, match="Scene"):
            validate_market_case_input_row(damaged)


def test_generic_arrow_shard_roundtrip_preserves_utc_revision_identity(
    tmp_path: Path,
) -> None:
    import pyarrow.parquet as pq

    episode = _episode(0)
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(0, episodes=(episode,), transitions=(episode,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    rows = [item.to_dict() for item in recorder.drain_input_rows()]
    expected_revision_id = rows[0]["revision_id"]
    state = new_stream_state(MARKET_CASE_INPUT_FIELD_TYPES)
    write_stream_shard(
        tmp_path,
        "market_case_input_shards",
        rows,
        state,
        key_column="revision_id",
        field_types=MARKET_CASE_INPUT_FIELD_TYPES,
    )
    shard = tmp_path / state["committed_shards"][0]["path"]
    materialized = pq.read_table(shard).to_pylist()[0]
    assert str(materialized["asof"].tzinfo) == "UTC"
    assert materialized["revision_id"] == expected_revision_id
    validate_market_case_input_row(materialized)


def test_physical_custody_mutation_and_change_without_transition_fail_closed() -> None:
    formed = _episode(0)
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(0, episodes=(formed,), transitions=(formed,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    recorder.drain_input_rows()
    changed = _episode(1, source_zone_id="fvg:changed", claims=(_claim(),))
    with pytest.raises(ValueError, match="physical custody mutated"):
        _observe(
            recorder,
            _snapshot(1, episodes=(changed,), transitions=(changed,)),
            source_ordinal=1,
            replay_ordinal=1,
        )
    silent = _episode(
        0,
        claims=(_claim(),),
        active_claim_ids=("thesis:1",),
        claim_status="unique",
    )
    with pytest.raises(ValueError, match="changed without a transition"):
        _observe(
            recorder,
            _snapshot(1, episodes=(silent,), transitions=()),
            source_ordinal=1,
            replay_ordinal=1,
        )


def test_row_and_sparse_stream_validators_reject_tamper_and_discontinuity() -> None:
    formed = _episode(0)
    recorder = _recorder()
    _observe(
        recorder,
        _snapshot(0, episodes=(formed,), transitions=(formed,)),
        source_ordinal=0,
        replay_ordinal=0,
    )
    rows = [item.to_dict() for item in recorder.drain_input_rows()]
    pullback = _episode(1, pullback_minute=1)
    _observe(
        recorder,
        _snapshot(1, episodes=(pullback,), transitions=(pullback,)),
        source_ordinal=1,
        replay_ordinal=1,
    )
    rows.extend(item.to_dict() for item in recorder.drain_input_rows())
    validate_market_case_rows(rows)
    broken = dict(rows[1])
    broken["revision_index"] = 2
    broken["revision_id"] = market_cases_module._expected_revision_id(broken)
    with pytest.raises(ValueError, match="indexes are discontinuous"):
        validate_market_case_rows((rows[0], broken))
    malformed = dict(rows[0])
    malformed["transition_kinds_json"] = '["zone_registered", "episode_created"]'
    malformed["revision_id"] = market_cases_module._expected_revision_id(malformed)
    with pytest.raises(ValueError, match="not canonical JSON|transition kinds"):
        validate_market_case_input_row(malformed)
    relation_kind = dict(rows[0])
    relation_kind["transition_kinds_json"] = '["claim_relation_changed"]'
    relation_kind["revision_id"] = market_cases_module._expected_revision_id(
        relation_kind
    )
    with pytest.raises(ValueError, match="transition kinds"):
        validate_market_case_input_row(relation_kind)
