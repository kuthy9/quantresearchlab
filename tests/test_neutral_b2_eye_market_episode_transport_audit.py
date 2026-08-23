from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.audit_neutral_b2_eye_market_episode_transport import (
    PathSnapshot,
    TrajectoryAnchor,
    _protocol,
    _scale_diagnostics,
    _trajectory,
    _transport,
    _transport_summary_from_rows,
    _verdict,
)


pytestmark = pytest.mark.historical_frozen


def _path(kinds: list[str], *, identity: str = "path:1") -> dict:
    return {
        "sequence_id": identity,
        "context_kind": "zone_return",
        "context_id": "zone:1",
        "direction": "long",
        "lifecycle": "active",
        "steps": [
            {
                "step_id": f"step:{identity}:{index}",
                "kind": kind,
                "observed_at": f"2021-02-01T10:{index:02d}:00-05:00",
            }
            for index, kind in enumerate(kinds)
        ],
    }


def _snapshot(ordinal: int, kinds: list[str], *, identity: str = "path:1"):
    return PathSnapshot(
        source_ordinal=ordinal,
        replay_update_ordinal=ordinal,
        asof=pd.Timestamp("2021-02-01T10:00:00-05:00")
        + pd.Timedelta(int(ordinal), unit="min"),
        epoch=0,
        path=_path(kinds, identity=identity),
    )


def _anchor(label: str, *, surface: str, identity: str = "path:1"):
    return TrajectoryAnchor(
        surface=surface,
        profile="neutral_representation_train_2021_02",
        path_id=identity,
        context_id="zone:1",
        epoch=0,
        anchor_source_ordinal=10,
        anchor_clock="2021-02-01T10:02:00-05:00",
        anchor_roles=("zone_visible", "departure_confirmed", "first_pullback"),
        horizon_step_ids=(
            f"step:{identity}:0",
            f"step:{identity}:1",
            f"step:{identity}:2",
            f"step:{identity}:3",
        ),
        horizon_step_kinds=(
            "zone_visible",
            "departure_confirmed",
            "first_pullback",
            "micro_bos_confirmed",
        ),
        trajectory=label,
        right_censored=False,
        market_episode_id=("episode:1" if surface == "market_episode" else None),
    )


def test_frozen_four_class_trajectory_rule_is_shared_by_both_layers():
    contract = _protocol()["trajectory_definition_under_test"]
    prefix = ["zone_visible", "departure_confirmed", "first_pullback"]
    cases = {
        "confirmed_continuation": prefix + ["micro_bos_confirmed"],
        "failed_or_opposed": prefix + ["location_left"],
        "returned_to_range_unconfirmed": prefix
        + ["reference_reclaimed", "reacceptance_held"],
        "continued_unresolved": prefix + ["wick_rejection"],
    }
    for expected, final in cases.items():
        history = (_snapshot(10, prefix), _snapshot(20, final))
        actual = _trajectory(
            history,
            0,
            horizon=60,
            epoch_end=100,
            contract=contract,
        )
        assert actual[0] == expected
        assert actual[1] is False
    censored = _trajectory(
        (_snapshot(50, prefix),),
        0,
        horizon=60,
        epoch_end=100,
        contract=contract,
    )
    assert censored[:2] == (None, True)


def test_transport_reports_missing_path_steps_and_label_mismatch():
    raw = _anchor("confirmed_continuation", surface="raw_eye")
    market = _anchor("failed_or_opposed", surface="market_episode")
    market_history = {
        ("neutral_representation_train_2021_02", "path:1"): [
            _snapshot(10, ["zone_visible", "departure_confirmed", "first_pullback"])
        ]
    }
    summary, rows = _transport(
        (raw,),
        (market,),
        market_history,
        60,
        (
            "confirmed_continuation",
            "failed_or_opposed",
            "returned_to_range_unconfirmed",
            "continued_unresolved",
        ),
    )
    assert summary["raw_path_linked"] == 1
    assert summary["raw_steps_complete_by_horizon"] == 0
    assert summary["trajectory_matches"] == 0
    assert rows[0]["missing_raw_step_kinds"] == ["micro_bos_confirmed"]


def test_right_censored_anchor_is_not_a_transport_connection_failure():
    raw = _anchor("confirmed_continuation", surface="raw_eye")
    market = _anchor("confirmed_continuation", surface="market_episode")
    raw = TrajectoryAnchor(**{**raw.__dict__, "trajectory": None, "right_censored": True})
    market = TrajectoryAnchor(
        **{**market.__dict__, "trajectory": None, "right_censored": True}
    )
    summary, rows = _transport(
        (raw,),
        (market,),
        {("neutral_representation_train_2021_02", "path:1"): []},
        60,
        (
            "confirmed_continuation",
            "failed_or_opposed",
            "returned_to_range_unconfirmed",
            "continued_unresolved",
        ),
    )
    rebuilt = _transport_summary_from_rows(rows, 1, tuple(summary["per_raw_class"]))
    assert rebuilt["raw_steps_evaluable_by_horizon"] == 0
    assert rebuilt["raw_steps_complete_by_horizon"] == 0
    assert rebuilt["raw_path_link_rate"] == 1.0


def test_unknown_scale_taxonomy_separates_graph_and_aggregation_loss():
    details = {
        "4H": {
            "relation": "aligned", "graph_connected": True,
            "ambiguous": False, "evidence_ids": ["h4"],
            "evidence_kind": "structure", "authority_layer_id": "h4",
            "direction": "long",
        },
        "1H": {
            "relation": "unknown", "graph_connected": False,
            "ambiguous": False, "evidence_ids": ["h1"],
            "evidence_kind": "bos", "authority_layer_id": "h4",
            "direction": "short",
        },
        "15m": {
            "relation": "aligned", "graph_connected": True,
            "ambiguous": False, "evidence_ids": ["m15"],
            "evidence_kind": "bos", "authority_layer_id": "h4",
            "direction": "long",
        },
        "5m": {
            "relation": "normal_pullback", "graph_connected": True,
            "ambiguous": False, "evidence_ids": ["m5"],
            "evidence_kind": "bos", "authority_layer_id": "h4",
            "direction": "short",
        },
        "1m": {
            "relation": "unknown", "graph_connected": False,
            "ambiguous": False, "evidence_ids": [],
            "evidence_kind": None, "authority_layer_id": "h4",
            "direction": None,
        },
    }
    row = {
        "market_epoch_id": "epoch:0",
        "market_episode_id": "episode:1",
        "revision_id": "revision:1",
        "asof": pd.Timestamp("2021-02-01T10:00:00-05:00"),
        "neutral_global_context_json": json.dumps(
            {
                "scale_relation_details": details,
                "unknown_evidence": [
                    "scale:1H:graph_disconnected",
                    "scale:1m:no_structural_evidence",
                ],
            }
        ),
    }
    summary, rows = _scale_diagnostics(
        ({"profile_name": "train", "rows": [row]},), _protocol()
    )
    assert summary["graph_unconnected_rows"] == 1
    assert summary["genuinely_uncertain_rows"] == 1
    assert summary["field_missing_rows"] == 0
    assert summary["audit_aggregation_loss_rows"] == 1
    assert {item["reason_category"] for item in rows} == {
        "graph_unconnected",
        "genuinely_uncertain",
        "audit_aggregation_loss",
    }
    assert all(isinstance(item["decision_at"], str) for item in rows)


def test_verdict_keeps_model_out_of_scope_until_layers_are_complete():
    classes = (
        "confirmed_continuation",
        "failed_or_opposed",
        "returned_to_range_unconfirmed",
        "continued_unresolved",
    )
    verdict = _verdict(
        {value: (1 if value != "returned_to_range_unconfirmed" else 0) for value in classes},
        {value: 0 for value in classes},
        {
            "raw_anchors": 3,
            "raw_path_linked": 2,
            "raw_steps_evaluable_by_horizon": 3,
            "raw_steps_complete_by_horizon": 2,
            "market_episode_only_anchors": 0,
            "trajectory_matches": 1,
            "trajectory_comparable": 2,
        },
        {"reference_reclaimed": 4, "reacceptance_held": 4},
        classes,
    )
    assert verdict["primary_localization"] == "trajectory_definition_or_eye_expression_gap"
    assert verdict["required_roles_absent_for_missing_raw_classes"] == {
        "returned_to_range_unconfirmed": []
    }
    assert verdict["eye_semantic_sufficiency_resolved"] is False
    assert verdict["b2_model_in_scope"] is False
    assert verdict["b2_training_authorized"] is False
