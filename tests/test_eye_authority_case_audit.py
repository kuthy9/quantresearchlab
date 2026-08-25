from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.audit_eye_authority_cases import (
    _audit_summary,
    _case_reason_allowed,
    _case_shape_valid,
    _case_transport_contract,
    _diagnostic_gate_snapshot_present,
    _full_observer_config,
    _load_case_index,
    _path_step_transport,
    merge_case_windows,
    transmission_record,
)
from scripts.run_eye_authority_scan import _registered_payload
from smc_trader.model import Timeframe
from smc_trader.visualization import VisualArtifact


BASE = pd.Timestamp("2023-03-10T10:00:00-05:00")


def _case(index: int, clock: pd.Timestamp) -> dict[str, object]:
    return {
        "case_id": f"case-{index:02d}",
        "stratum": "group5_interrupted_path",
        "group": "group3",
        "primitive": "fvg",
        "entity_id": f"fvg:{index}",
        "lifecycle_or_outcome": "open",
        "event_clock": clock.isoformat(),
        "strata": {"timeframe": "5m"},
        "source_ids": [f"displacement:{index}"],
    }


def test_case_audit_is_full_view_eye_only() -> None:
    config = _full_observer_config(_registered_payload())

    assert config.eye_authority_mode is True
    assert config.materialize_event_view is True
    assert config.project_scene_graph is True
    assert config.range_auction_projection_only is False


def test_case_transport_contract_is_clock_and_stratum_specific() -> None:
    diagnostic = {
        **_case(1, BASE),
        "stratum": "near_mature_single_gate",
        "primitive": "range_maturity_evaluation",
    }
    reset_terminal = {
        **_case(2, BASE),
        "stratum": "source_identity_or_reset_failure",
        "primitive": "dealing_range",
    }
    interrupted = {
        **_case(3, BASE),
        "stratum": "group5_interrupted_path",
        "primitive": "path_sequence",
    }
    manipulation = {
        **_case(4, BASE),
        "stratum": "mature_range_manipulation",
        "primitive": "manipulation",
    }

    diagnostic_contract = _case_transport_contract(diagnostic)
    assert diagnostic_contract["event_required"] is True
    assert diagnostic_contract["node_required"] is True
    assert diagnostic_contract["relations"] == ()
    assert "diagnostic" in diagnostic_contract["event_semantics"]

    reset_contract = _case_transport_contract(reset_terminal)
    assert reset_contract["relations"] == ()
    assert "reset" in reset_contract["relation_applicability"]

    assert _case_transport_contract(interrupted)["relations"] == ()
    assert _case_transport_contract(manipulation)["relations"] == (
        "SWEEPS",
    )


@pytest.mark.parametrize(
    ("outcome", "reason"),
    (
        ("deadline_censored", "deadline_elapsed"),
    ),
)
def test_manipulation_censor_reasons_match_producer_contract(
    outcome: str,
    reason: str,
) -> None:
    case = {
        **_case(5, BASE),
        "stratum": "mature_range_manipulation",
        "group": "group4",
        "primitive": "manipulation",
        "lifecycle_or_outcome": outcome,
    }

    assert _case_reason_allowed(case, reason) is True
    assert _case_reason_allowed(case, "unrelated_reason") is False


def test_boundary_censored_manipulation_is_not_a_materialized_case_contract() -> None:
    case = {
        **_case(5, BASE),
        "stratum": "mature_range_manipulation",
        "group": "group4",
        "primitive": "manipulation",
        "lifecycle_or_outcome": "hard_boundary_censored",
    }

    assert _case_shape_valid(case) is False


def test_group5_step_transport_requires_ordered_memory_and_graph_steps() -> None:
    first = SimpleNamespace(
        step_id="step:1",
        kind="pool_swept",
        observed_at=BASE,
        predecessor_step_ids=(),
        source_entity_id="pool:1",
    )
    second = SimpleNamespace(
        step_id="step:2",
        kind="reacceptance_held",
        observed_at=BASE + pd.Timedelta(minutes=1),
        predecessor_step_ids=("step:1",),
        source_entity_id="manip:1",
    )
    path = SimpleNamespace(sequence_id="path:1", steps=(first, second))

    def event(step):
        return SimpleNamespace(
            kind=SimpleNamespace(value="entry_path_step"),
            observed_at=step.observed_at,
            details={
                "sequence_id": "path:1",
                "step_id": step.step_id,
                "kind": step.kind,
                "predecessor_step_ids": step.predecessor_step_ids,
                "source_entity_id": step.source_entity_id,
            },
        )

    def node(step):
        return SimpleNamespace(
            kind="path_step",
            market_epoch_id="epoch:1",
            source_ids=(
                "path:1",
                step.step_id,
                step.source_entity_id,
                *step.predecessor_step_ids,
            ),
            observed_at=step.observed_at,
            semantic_attributes=(("path_step_kind", step.kind),),
        )

    observation = SimpleNamespace(
        path_sequences=(),
        group5_boundary_path_transitions=(path,),
    )
    complete = _path_step_transport(
        observation=observation,
        events=(event(first), event(second)),
        nodes=(node(first), node(second)),
        current_epoch_id="epoch:1",
        sequence_id="path:1",
    )
    missing_node = _path_step_transport(
        observation=observation,
        events=(event(first), event(second)),
        nodes=(node(first),),
        current_epoch_id="epoch:1",
        sequence_id="path:1",
    )
    missing_predecessor = _path_step_transport(
        observation=observation,
        events=(event(first), event(second)),
        nodes=(
            node(first),
            SimpleNamespace(
                **{
                    **vars(node(second)),
                    "source_ids": ("path:1", "step:2", "manip:1"),
                }
            ),
        ),
        current_epoch_id="epoch:1",
        sequence_id="path:1",
    )

    assert complete is not None and complete["complete"] is True
    assert missing_node is not None
    assert missing_node["event_steps_complete"] is True
    assert missing_node["scene_steps_complete"] is False
    assert missing_node["complete"] is False
    assert missing_predecessor is not None
    assert missing_predecessor["scene_steps_complete"] is False
    assert missing_predecessor["complete"] is False


@pytest.mark.parametrize(
    ("stratum", "gates"),
    (
        ("near_mature_single_gate", ("compression",)),
        (
            "multiple_gate_rejected",
            ("duration", "bilateral_touches", "compression"),
        ),
    ),
)
def test_range_diagnostic_requires_exact_clock_identity_and_gate_set(
    stratum: str,
    gates: tuple[str, ...],
) -> None:
    case = {
        **_case(10, BASE),
        "stratum": stratum,
        "group": "group4",
        "primitive": "range_maturity_evaluation",
        "entity_id": "range:diagnostic",
        "lifecycle_or_outcome": "+".join(gates),
    }
    matching = SimpleNamespace(
        observed_at=BASE,
        maturity_range_id="range:diagnostic",
        unmet_maturity_gates=gates,
    )
    observation = SimpleNamespace(group4_range_funnel=(matching,))

    assert _diagnostic_gate_snapshot_present(observation, case, BASE) is True
    wrong_clock = SimpleNamespace(
        observed_at=BASE - pd.Timedelta(hours=1),
        maturity_range_id="range:diagnostic",
        unmet_maturity_gates=gates,
    )
    observation.group4_range_funnel = (wrong_clock,)
    assert _diagnostic_gate_snapshot_present(observation, case, BASE) is False
    wrong_identity = SimpleNamespace(
        observed_at=BASE,
        maturity_range_id="range:other",
        unmet_maturity_gates=gates,
    )
    observation.group4_range_funnel = (wrong_identity,)
    assert _diagnostic_gate_snapshot_present(observation, case, BASE) is False
    wrong_gates = SimpleNamespace(
        observed_at=BASE,
        maturity_range_id="range:diagnostic",
        unmet_maturity_gates=(*gates, "width"),
    )
    observation.group4_range_funnel = (wrong_gates,)
    assert _diagnostic_gate_snapshot_present(observation, case, BASE) is False

    invalid_cardinality = (
        ("duration", "width")
        if stratum == "near_mature_single_gate"
        else ("compression",)
    )
    invalid_case = {
        **case,
        "lifecycle_or_outcome": "+".join(invalid_cardinality),
    }
    observation.group4_range_funnel = (
        SimpleNamespace(
            observed_at=BASE,
            maturity_range_id="range:diagnostic",
            unmet_maturity_gates=invalid_cardinality,
        ),
    )
    assert (
        _diagnostic_gate_snapshot_present(
            observation,
            invalid_case,
            BASE,
        )
        is False
    )


def test_case_index_is_future_blind_and_overlapping_windows_merge(
    tmp_path: Path,
) -> None:
    cases = [
        {
            **_case(index, BASE + pd.Timedelta(hours=index % 2)),
            "stratum": "group5_interrupted_path",
            "group": "group5",
            "primitive": "path_sequence",
            "entity_id": f"path:{index}",
            "lifecycle_or_outcome": "closed",
            "source_ids": [f"context:{index}"],
            "strata": {
                "context_kind": "zone_return",
                "direction": "long",
            },
        }
        for index in range(20)
    ]
    payload = {
        "schema_version": 1,
        "profile": "eye_group1_5_natural_authority_2023_full_year",
        "scan_identity": "a" * 64,
        "future_hidden_on_first_review": True,
        "cases": cases,
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    profile = _registered_payload()["profile"]

    loaded = _load_case_index(path, profile=profile)
    windows = merge_case_windows(
        loaded["cases"],
        warmup_calendar_days=7,
        timezone="America/New_York",
    )

    assert len(windows) == 1
    assert len(windows[0].cases) == 20
    assert windows[0].start == BASE - pd.DateOffset(days=7)
    assert windows[0].end == BASE + pd.Timedelta(hours=1)

    payload["cases"][0]["future_return"] = 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected field contract"):
        _load_case_index(path, profile=profile)

    payload["cases"][0].pop("future_return")
    payload["cases"][0]["stratum"] = "unregistered_category"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity, source or clock"):
        _load_case_index(path, profile=profile)

    payload["cases"][0]["primitive"] = "path_sequence"
    payload["cases"][0]["strata"] = {
        "context_kind": "zone_return",
        "direction": "long",
        "future_return": 9.0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity, source or clock"):
        _load_case_index(path, profile=profile)

    payload["cases"][0]["stratum"] = "group5_interrupted_path"
    payload["cases"][0]["primitive"] = "fvg"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity, source or clock"):
        _load_case_index(path, profile=profile)


def test_transmission_record_links_observation_memory_graph_and_image(
    tmp_path: Path,
) -> None:
    entity_id = "fvg:7"
    frame = SimpleNamespace(
        swings=(),
        structures=(),
        structure_breaks=(),
        support_resistance=(),
        liquidity_pools=(),
        fair_value_gaps=(SimpleNamespace(fvg_id=entity_id),),
        order_blocks=(),
        dealing_ranges=(),
    )
    event = SimpleNamespace(
        event_id="event:fvg:7",
        entity_id=entity_id,
        kind="fvg_state",
        lifecycle="open",
        observed_at=BASE,
        ended_at=None,
        transition_reason=None,
    )
    observation = SimpleNamespace(
        asof=BASE,
        frames={Timeframe.M5: frame},
        liquidity_inventory=(),
        liquidity_pool_states=(),
        group3_boundary_fvg_transitions=(),
        group3_boundary_order_block_transitions=(),
        group4_boundary_range_transitions=(),
        manipulations=(),
        group4_boundary_manipulation_transitions=(),
        entry_locations=(),
        qualified_reacceptances=(),
        micro_bos_references=(),
        path_sequences=(),
        group5_boundary_path_transitions=(),
        group5_boundary_reacceptance_transitions=(),
        displacement=None,
        anomalies=(),
        group4_range_funnel=(),
        recent_events=(event,),
        retained_entity_timelines={},
        incomplete_entity_timeline_keys=(),
        scene_revision_id="scene:r000000000001",
        execution=SimpleNamespace(source="not_evaluated"),
    )
    node = SimpleNamespace(
        node_id="node:fvg:7",
        kind="fvg",
        entity_id=entity_id,
        source_ids=(entity_id,),
        lifecycle="open",
        observed_at=BASE,
        market_epoch_id="epoch:0",
        resolution_reason=None,
    )
    source = SimpleNamespace(
        node_id="node:displacement:7",
        kind="displacement",
        entity_id="displacement:7",
        source_ids=("displacement:7",),
        lifecycle="active",
        observed_at=BASE,
        market_epoch_id="epoch:0",
        resolution_reason=None,
    )
    source_context = SimpleNamespace(
        node_id="node:source-context:7",
        kind="context",
        entity_id="source-context:7",
        source_ids=("displacement:7",),
        lifecycle="active",
        observed_at=BASE,
        market_epoch_id="epoch:0",
        resolution_reason=None,
    )
    edge = SimpleNamespace(
        edge_id="edge:creates:7",
        source_node_id=source.node_id,
        target_node_id=node.node_id,
        relation=SimpleNamespace(value="CREATES"),
        lifecycle="active",
        first_observed_at=BASE,
        source_ids=("displacement:7", entity_id),
    )
    context_only_edge = SimpleNamespace(
        edge_id="edge:context:7",
        source_node_id=source.node_id,
        target_node_id=source_context.node_id,
        relation=SimpleNamespace(value="RETURNS_TO"),
        lifecycle="active",
        first_observed_at=BASE,
        source_ids=("displacement:7", "source-context:7"),
    )
    reversed_edge = SimpleNamespace(
        edge_id="edge:reversed-creates:7",
        source_node_id=node.node_id,
        target_node_id=source.node_id,
        relation=SimpleNamespace(value="CREATES"),
        lifecycle="active",
        first_observed_at=BASE,
        source_ids=(entity_id, "displacement:7"),
    )

    class Graph:
        revision_id = "scene:r000000000001"

        @staticmethod
        def current_candle_structure_nodes(*args, **kwargs):
            return (
                SimpleNamespace(
                    kind="candle_structure",
                    timeframe="1m",
                    lifecycle="observed",
                    observed_at=BASE,
                    market_epoch_id="epoch:0",
                ),
            )

        @staticmethod
        def nodes_asof(asof):
            return (source, source_context, node)

        @staticmethod
        def edges_asof(asof):
            return (edge, context_only_edge, reversed_edge)

    image = tmp_path / "images/case-07.png"
    artifact = VisualArtifact(
        path=image,
        kind="eye_observation",
        decision_id="case-07",
        maximum_market_time=BASE,
        hypothesis_key=None,
        setup_id=None,
        entry_location_id=None,
        entry_path_id=None,
    )
    record = transmission_record(
        _case(7, BASE),
        observation,
        Graph(),
        artifact,
        output=tmp_path,
    )

    assert record["future_hidden"] is True
    assert record["observation"]["case_clock_matches_observation"] is True
    assert (
        record["observation"]["execution_reality_status"]
        == "not_evaluated"
    )
    assert record["observation"]["case_identity_present"] is True
    assert record["event_memory"]["case_event_present"] is True
    assert record["event_memory"]["case_event_state_present"] is True
    assert record["scene_graph"]["case_node_present"] is True
    assert record["scene_graph"]["case_node_state_present"] is True
    assert record["scene_graph"]["key_edge_coverage"]["CREATES"] is True
    assert record["scene_graph"]["connected_relation_counts"]["RETURNS_TO"] == 1
    assert record["scene_graph"]["key_edge_requirement_met"] is True
    assert record["scene_graph"]["key_edge_exact_relation_counts"] == {
        "CREATES": 1
    }
    assert record["scene_graph"]["key_edge_frozen_source_coverage"] == {
        "CREATES": ["displacement:7"]
    }
    summary = _audit_summary((record,), (_case(7, BASE),))
    assert summary["event_memory_identity_requirement_met"] == 1
    assert summary["scene_graph_identity_requirement_met"] == 1
    assert summary["complete_transport_contract_met"] == 1
    assert summary["key_edge_required_cases"] == 1
    assert summary["key_edge_evaluated_cases"] == 1
    assert summary["key_edge_requirement_met"] == 1

    broad_only = {
        **record,
        "scene_graph": {
            **record["scene_graph"],
            "case_relation_counts": {"CREATES": 99},
            "key_edge_exact_relation_counts": {},
            "key_edge_requirement_met": False,
        },
    }
    assert (
        _audit_summary((broad_only,), (_case(7, BASE),))[
            "key_edge_requirement_met"
        ]
        == 0
    )
    assert (
        record["observation"]["transport_by_typed_kind"]["fvg"]
        == {
            "typed": 1,
            "event_memory": 1,
            "scene_graph": 1,
            "missing_from_event_memory": [],
            "missing_from_scene_graph": [],
        }
    )
    assert record["image_path"] == "images/case-07.png"

    late_clock = BASE + pd.Timedelta(minutes=1)
    late_observation = SimpleNamespace(**vars(observation))
    late_observation.asof = late_clock

    class LateGraph(Graph):
        @staticmethod
        def current_candle_structure_nodes(*args, **kwargs):
            return (
                SimpleNamespace(
                    kind="candle_structure",
                    timeframe="1m",
                    lifecycle="observed",
                    observed_at=late_clock,
                    market_epoch_id="epoch:0",
                ),
            )

    late_record = transmission_record(
        _case(7, BASE),
        late_observation,
        LateGraph(),
        None,
    )
    assert late_record["observation"]["case_clock_matches_observation"] is False
    assert late_record["complete_transport_contract_met"] is False

    incomplete_observation = SimpleNamespace(**vars(observation))
    incomplete_observation.incomplete_entity_timeline_keys = (
        "fvg:fvg:7",
    )
    incomplete_record = transmission_record(
        _case(7, BASE),
        incomplete_observation,
        Graph(),
        None,
    )
    assert incomplete_record["event_memory"][
        "incomplete_entity_timeline"
    ] is True
    assert incomplete_record["complete_transport_contract_met"] is False

    wrong_endpoint = SimpleNamespace(
        node_id="node:not-the-displacement",
        kind="displacement",
        entity_id="displacement:wrong",
        source_ids=("displacement:wrong",),
        lifecycle="active",
        observed_at=BASE,
        market_epoch_id="epoch:0",
        resolution_reason=None,
    )
    metadata_only_edge = SimpleNamespace(
        edge_id="edge:metadata-only-creates",
        source_node_id=wrong_endpoint.node_id,
        target_node_id=node.node_id,
        relation=SimpleNamespace(value="CREATES"),
        lifecycle="active",
        first_observed_at=BASE,
        source_ids=("displacement:7", entity_id),
    )

    class MetadataOnlyGraph(Graph):
        @staticmethod
        def nodes_asof(asof):
            return (wrong_endpoint, node)

        @staticmethod
        def edges_asof(asof):
            return (metadata_only_edge,)

    metadata_only_record = transmission_record(
        _case(7, BASE),
        observation,
        MetadataOnlyGraph(),
        None,
    )
    assert metadata_only_record["scene_graph"][
        "key_edge_requirement_met"
    ] is False
    assert metadata_only_record["complete_transport_contract_met"] is False


def test_transmission_record_rejects_old_epoch_or_wrong_state() -> None:
    entity_id = "fvg:8"
    frame = SimpleNamespace(
        swings=(),
        structures=(),
        structure_breaks=(),
        support_resistance=(),
        liquidity_pools=(),
        fair_value_gaps=(SimpleNamespace(fvg_id=entity_id),),
        order_blocks=(),
        dealing_ranges=(),
    )
    stale_event = SimpleNamespace(
        event_id="event:fvg:8:partial",
        entity_id=entity_id,
        kind="fvg_state",
        lifecycle="partial",
        observed_at=BASE,
        ended_at=None,
        transition_reason=None,
    )
    observation = SimpleNamespace(
        asof=BASE,
        frames={Timeframe.M5: frame},
        liquidity_inventory=(),
        liquidity_pool_states=(),
        group3_boundary_fvg_transitions=(),
        group3_boundary_order_block_transitions=(),
        group4_boundary_range_transitions=(),
        manipulations=(),
        group4_boundary_manipulation_transitions=(),
        entry_locations=(),
        qualified_reacceptances=(),
        micro_bos_references=(),
        path_sequences=(),
        group5_boundary_path_transitions=(),
        group5_boundary_reacceptance_transitions=(),
        displacement=None,
        anomalies=(),
        group4_range_funnel=(),
        recent_events=(stale_event,),
        retained_entity_timelines={},
        incomplete_entity_timeline_keys=(),
        scene_revision_id="scene:r000000000002",
        execution=SimpleNamespace(source="not_evaluated"),
    )
    old_case_node = SimpleNamespace(
        node_id="epoch:0:fvg:8",
        kind="fvg",
        entity_id=entity_id,
        source_ids=(entity_id,),
        lifecycle="open",
        observed_at=BASE,
        market_epoch_id="epoch:0",
        resolution_reason=None,
    )
    current_context = SimpleNamespace(
        node_id="epoch:1:context",
        kind="context",
        entity_id="context",
        source_ids=(),
        lifecycle="active",
        observed_at=BASE,
        market_epoch_id="epoch:1",
        resolution_reason=None,
    )

    class Graph:
        revision_id = "scene:r000000000002"

        @staticmethod
        def current_candle_structure_nodes(*args, **kwargs):
            return (
                SimpleNamespace(
                    kind="candle_structure",
                    timeframe="1m",
                    lifecycle="observed",
                    observed_at=BASE,
                    market_epoch_id="epoch:1",
                ),
            )

        @staticmethod
        def nodes_asof(asof):
            return (old_case_node, current_context)

        @staticmethod
        def edges_asof(asof):
            return ()

    record = transmission_record(
        _case(8, BASE),
        observation,
        Graph(),
        None,
    )

    assert record["event_memory"]["case_event_present"] is True
    assert record["event_memory"]["case_event_state_present"] is False
    assert record["event_memory"]["case_event_requirement_met"] is False
    assert record["scene_graph"]["historical_case_node_count"] == 1
    assert record["scene_graph"]["case_node_present"] is False
    assert record["scene_graph"]["case_node_requirement_met"] is False


def test_transmission_record_accepts_terminal_at_hard_epoch_boundary() -> None:
    entity_id = "range:contract-reset"
    case = {
        **_case(9, BASE),
        "stratum": "source_identity_or_reset_failure",
        "group": "group4",
        "primitive": "dealing_range",
        "entity_id": entity_id,
        "lifecycle_or_outcome": "broken",
        "source_ids": ["sr:lower", "sr:upper"],
    }
    frame = SimpleNamespace(
        swings=(),
        structures=(),
        structure_breaks=(),
        support_resistance=(),
        liquidity_pools=(),
        fair_value_gaps=(),
        order_blocks=(),
        dealing_ranges=(),
    )
    terminal_state = SimpleNamespace(range_id=entity_id)
    terminal_event = SimpleNamespace(
        event_id="event:range:contract-reset:broken",
        entity_id=entity_id,
        kind="dealing_range_state",
        lifecycle="broken",
        observed_at=BASE,
        ended_at=BASE,
        transition_reason="contract_change_reset",
    )
    observation = SimpleNamespace(
        asof=BASE,
        frames={Timeframe.H1: frame},
        liquidity_inventory=(),
        liquidity_pool_states=(),
        group3_boundary_fvg_transitions=(),
        group3_boundary_order_block_transitions=(),
        group4_boundary_range_transitions=(terminal_state,),
        manipulations=(),
        group4_boundary_manipulation_transitions=(),
        entry_locations=(),
        qualified_reacceptances=(),
        micro_bos_references=(),
        path_sequences=(),
        group5_boundary_path_transitions=(),
        group5_boundary_reacceptance_transitions=(),
        displacement=None,
        anomalies=("contract_change_history_reset",),
        group4_range_funnel=(),
        recent_events=(terminal_event,),
        retained_entity_timelines={"range:contract-reset": (terminal_event,)},
        incomplete_entity_timeline_keys=(),
        scene_revision_id="scene:r000000000003",
        execution=SimpleNamespace(source="not_evaluated"),
    )
    old_terminal = SimpleNamespace(
        node_id="epoch:0:range:contract-reset",
        kind="range",
        entity_id=entity_id,
        source_ids=("sr:lower", "sr:upper"),
        lifecycle="censored",
        observed_at=BASE,
        market_epoch_id="epoch:0",
        resolution_reason="contract_change_history_reset",
    )
    current_context = SimpleNamespace(
        node_id="epoch:1:context",
        kind="context",
        entity_id="context",
        source_ids=(),
        lifecycle="active",
        observed_at=BASE,
        market_epoch_id="epoch:1",
        resolution_reason=None,
    )

    class Graph:
        revision_id = "scene:r000000000003"

        @staticmethod
        def current_candle_structure_nodes(*args, **kwargs):
            return (
                SimpleNamespace(
                    kind="candle_structure",
                    timeframe="1m",
                    lifecycle="observed",
                    observed_at=BASE,
                    market_epoch_id="epoch:1",
                ),
            )

        @staticmethod
        def nodes_asof(asof):
            return (old_terminal, current_context)

        @staticmethod
        def edges_asof(asof):
            return ()

    record = transmission_record(case, observation, Graph(), None)

    assert record["event_memory"]["case_event_requirement_met"] is True
    assert record["scene_graph"]["hard_boundary_case"] is True
    assert record["scene_graph"]["case_node_present"] is False
    assert record["scene_graph"]["boundary_no_revival"] is True
    assert record["scene_graph"]["case_node_requirement_met"] is True
    assert record["complete_transport_contract_met"] is True

    class MissingEpochGraph(Graph):
        @staticmethod
        def current_candle_structure_nodes(*args, **kwargs):
            return ()

    missing_epoch = transmission_record(
        case,
        observation,
        MissingEpochGraph(),
        None,
    )
    assert missing_epoch["scene_graph"]["current_market_epoch_id"] is None
    assert missing_epoch["scene_graph"]["case_node_requirement_met"] is False
    assert missing_epoch["complete_transport_contract_met"] is False

    revived = SimpleNamespace(
        node_id="epoch:1:range:contract-reset",
        kind="range",
        entity_id=entity_id,
        source_ids=("sr:lower", "sr:upper"),
        lifecycle="censored",
        observed_at=BASE,
        market_epoch_id="epoch:1",
        resolution_reason="contract_change_history_reset",
    )

    class RevivedGraph(Graph):
        @staticmethod
        def nodes_asof(asof):
            return (old_terminal, current_context, revived)

    revived_record = transmission_record(
        case,
        observation,
        RevivedGraph(),
        None,
    )
    assert revived_record["scene_graph"]["boundary_no_revival"] is False
    assert revived_record["scene_graph"]["case_node_requirement_met"] is False
    assert revived_record["complete_transport_contract_met"] is False
