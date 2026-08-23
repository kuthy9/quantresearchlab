from __future__ import annotations

import json

import pytest

from smc_trader.market_representation import RepresentationDataError
from scripts.audit_neutral_b2_mechanism_semantics import (
    MechanismAnchor,
    _local_scale_role,
    _linked_graph_tokens,
    _load_protocol,
    _oof_nb,
    _pairs,
    _path_snapshot,
    _trajectory,
)


def _row(path_id: str = "path-1", *, roles: tuple[str, ...] = ("zone_visible",)):
    path = {
        "sequence_id": path_id,
        "context_kind": "zone_return",
        "direction": "long",
        "lifecycle": "active",
        "last_updated_at": "2021-02-01T10:00:00-05:00",
        "steps": [{"kind": role} for role in roles],
    }
    return {
        "entry_path_id": path_id,
        "direction": "long",
        "source_replay_ordinal": 100,
        "observation_transition_json": json.dumps(
            {"collections": {"group5_path_transitions_this_update": [path]}}
        ),
    }, path


def _anchor(
    identity: int,
    *,
    window: str,
    date: str,
    template: str,
    higher: str = "material_opposition",
    local: str = "normal_pullback",
    trajectory: str = "confirmed_continuation",
    phase: str = "rth",
    direction: str = "long",
    signal: str = "signal=yes",
) -> MechanismAnchor:
    return MechanismAnchor(
        window=window,
        run_sha256=f"{identity:064x}",
        market_epoch_id="epoch:0",
        market_episode_id=f"episode:{identity}",
        revision_id=f"revision:{identity}",
        entry_path_id=f"path:{identity}",
        et_date=date,
        session_phase=phase,
        absolute_direction=direction,
        source_ordinal=identity,
        template=template,
        higher_timeframe_role=higher,
        local_scale_role=local,
        path_roles=("zone_visible", "departure_confirmed", "first_pullback"),
        token_groups=(
            ("path_roles", frozenset({f"path:template={template}"})),
            ("linked_eye", frozenset({signal})),
            ("connected_graph", frozenset()),
            ("scale_relations", frozenset({f"higher={higher}", f"local={local}"})),
            ("observable_context", frozenset({f"phase={phase}"})),
        ),
        trajectory=trajectory,
        right_censored=False,
    )


def test_current_episode_path_binding_selects_advanced_same_row_prefix():
    row, first = _row(roles=("zone_visible",))
    advanced = {
        **first,
        "last_updated_at": "2021-02-01T10:01:00-05:00",
        "steps": [
            {"kind": "zone_visible"},
            {"kind": "departure_confirmed"},
        ],
    }
    unrelated = {**advanced, "sequence_id": "other-path"}
    row["observation_transition_json"] = json.dumps(
        {
            "collections": {
                "group5_path_transitions_this_update": [unrelated, first, advanced]
            }
        }
    )
    selected, _ = _path_snapshot(row)
    assert [step["kind"] for step in selected["steps"]] == [
        "zone_visible",
        "departure_confirmed",
    ]


def test_current_episode_path_binding_fails_when_transport_is_missing():
    row, path = _row()
    path["sequence_id"] = "other"
    row["observation_transition_json"] = json.dumps(
        {"collections": {"group5_path_transitions_this_update": [path]}}
    )
    with pytest.raises(RepresentationDataError, match="not transported"):
        _path_snapshot(row)


def test_entry_endpoint_is_binding_only_and_never_a_graph_feature():
    protocol = _load_protocol()
    graph = {
        "relation_descriptors": [
            {
                "change_kind": "added",
                "lifecycle": "active",
                "relation": "SOURCED_FROM",
                "source": {
                    "node_id": "epoch:0:path_sequence:1m:path-1",
                    "kind": "path_sequence",
                    "lifecycle": "active",
                    "role": "path_sequence",
                    "structural_scale": "internal",
                    "timeframe": "1m",
                },
                "target": {
                    "node_id": "epoch:0:entry_location:1m:location-1",
                    "kind": "entry_location",
                    "lifecycle": "in_zone",
                    "role": "entry_location",
                    "structural_scale": "internal",
                    "timeframe": "1m",
                },
            }
        ]
    }
    assert not _linked_graph_tokens(
        graph, path_id="path-1", location_id="location-1", protocol=protocol
    )


@pytest.mark.parametrize(
    ("relations", "expected"),
    [
        (("normal_pullback", "normal_pullback", "unknown"), "normal_pullback"),
        (("aligned", "aligned", "unknown"), "aligned"),
        (("aligned", "material_opposition", "aligned"), "material_opposition"),
        (("aligned", "normal_pullback", "unknown"), "unknown"),
    ],
)
def test_local_scale_role_is_relational_not_exact_vector(relations, expected):
    assert (
        _local_scale_role(dict(zip(("15m", "5m", "1m"), relations, strict=True)))
        == expected
    )


def test_fixed_horizon_trajectory_prefers_first_causal_resolution():
    anchor_row, anchor_path = _row(
        roles=("zone_visible", "departure_confirmed", "first_pullback", "wick_rejection")
    )
    failed_row, failed_path = _row(
        roles=(
            "zone_visible",
            "departure_confirmed",
            "first_pullback",
            "wick_rejection",
            "location_left",
            "micro_bos_confirmed",
        )
    )
    failed_row["source_replay_ordinal"] = 110
    assert _trajectory(
        0,
        ((anchor_row, anchor_path, {}), (failed_row, failed_path, {})),
        horizon=60,
        epoch_end=1000,
    ) == ("failed_or_opposed", False)


def test_fixed_horizon_trajectory_distinguishes_return_and_right_censoring():
    anchor_row, anchor_path = _row(
        roles=("zone_visible", "departure_confirmed", "first_pullback", "reference_left")
    )
    returned_row, returned_path = _row(
        roles=(
            "zone_visible",
            "departure_confirmed",
            "first_pullback",
            "reference_left",
            "reference_reclaimed",
            "reacceptance_held",
        )
    )
    returned_row["source_replay_ordinal"] = 130
    ordered = ((anchor_row, anchor_path, {}), (returned_row, returned_path, {}))
    assert _trajectory(0, ordered, horizon=60, epoch_end=1000) == (
        "returned_to_range_unconfirmed",
        False,
    )
    assert _trajectory(0, ordered[:1], horizon=60, epoch_end=150) == (None, True)


def test_pair_rules_create_cross_date_positives_and_matched_hard_negatives():
    anchors = (
        _anchor(1, window="w1", date="2021-02-01", template="pullback_rejection"),
        _anchor(2, window="w1", date="2021-02-02", template="pullback_rejection"),
        _anchor(3, window="w1", date="2021-02-03", template="pullback_reference_left"),
    )
    pairs, summary = _pairs(anchors)
    assert summary["positive_query_coverage"] == pytest.approx(2 / 3)
    assert summary["hard_negative_query_coverage"] == 1.0
    assert summary["per_window"]["w1"]["within_window_positive_query_coverage"] == pytest.approx(2 / 3)
    assert any(kind == "positive" for kind, _, _ in pairs)
    assert any(kind == "hard_negative_template_conflict" for kind, _, _ in pairs)


def test_leave_one_window_out_classifier_can_exceed_template_prior_without_ids():
    anchors = []
    labels = ("confirmed_continuation", "failed_or_opposed")
    for window_index in range(6):
        for label_index, label in enumerate(labels):
            for repeat in range(8):
                identity = window_index * 100 + label_index * 10 + repeat
                anchors.append(
                    _anchor(
                        identity,
                        window=f"w{window_index}",
                        date=f"2021-02-{repeat + 1:02d}",
                        template="pullback_rejection",
                        trajectory=label,
                        signal=f"semantic={label}",
                    )
                )
    result = _oof_nb(tuple(anchors), minimum_document_frequency=5)
    assert result["nll_relative_improvement"] > 0.5
    assert result["balanced_accuracy_lift"] > 0.4
