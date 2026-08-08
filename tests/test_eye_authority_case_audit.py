from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.audit_eye_authority_cases import (
    _load_case_index,
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


def test_case_index_is_future_blind_and_overlapping_windows_merge(
    tmp_path: Path,
) -> None:
    cases = [
        _case(index, BASE + pd.Timedelta(hours=index % 2))
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
    event = SimpleNamespace(event_id="event:fvg:7", entity_id=entity_id)
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
        recent_events=(event,),
        retained_entity_timelines={},
        incomplete_entity_timeline_keys=(),
        scene_revision_id="scene:r000000000001",
    )
    node = SimpleNamespace(
        node_id="node:fvg:7",
        entity_id=entity_id,
        source_ids=(entity_id,),
    )
    source = SimpleNamespace(
        node_id="node:displacement:7",
        entity_id="displacement:7",
        source_ids=("displacement:7",),
    )
    edge = SimpleNamespace(
        source_node_id=source.node_id,
        target_node_id=node.node_id,
        relation=SimpleNamespace(value="CREATES"),
    )

    class Graph:
        revision_id = "scene:r000000000001"

        @staticmethod
        def nodes_asof(asof):
            return (source, node)

        @staticmethod
        def edges_asof(asof):
            return (edge,)

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
    assert record["observation"]["case_identity_present"] is True
    assert record["event_memory"]["case_event_present"] is True
    assert record["scene_graph"]["case_node_present"] is True
    assert record["scene_graph"]["key_edge_coverage"]["CREATES"] is True
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
