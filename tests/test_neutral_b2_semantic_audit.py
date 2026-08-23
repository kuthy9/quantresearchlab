from pathlib import Path

import pytest

from scripts.audit_neutral_b2_semantic_signal import (
    MaterialRecord,
    ROOT,
    _load_protocol,
    _preflight_roots,
    _session_phase,
    _signature_conditional_mi,
    _structural_tokens,
)
from smc_trader.market_representation import RepresentationDataError


TRAIN_ROOTS = (
    "outputs/market_case_input_2019_02_20260818_run03",
    "outputs/market_case_input_2019_08_20260818_run01",
    "outputs/market_case_input_2020_02_20260818_run02",
    "outputs/market_case_input_2020_08_20260818_run01",
    "outputs/market_case_input_2021_02_20260817_smoke_run01",
    "outputs/market_case_input_2021_08_20260818_run01",
)


@pytest.mark.historical_frozen
def test_b2_semantic_audit_preflight_accepts_only_six_frozen_train_inputs():
    protocol = _load_protocol()
    loaded = _preflight_roots(
        [str(ROOT / path) for path in reversed(TRAIN_ROOTS)], protocol
    )
    assert [item[1]["name"] for item in loaded] == [
        item["name"] for item in protocol["population"]["profiles"]
    ]

    validation = ROOT / "outputs/market_case_input_2022_05_20260817_smoke_run01"
    with pytest.raises(RepresentationDataError, match="non-train profile"):
        _preflight_roots(
            [str(ROOT / path) for path in TRAIN_ROOTS[:-1]] + [str(validation)],
            protocol,
        )


def test_b2_structural_tokens_exclude_entry_and_reinterpret_only_reference_alignment():
    protocol = _load_protocol()
    observation = {
        "collections": {
            "group5_entry_location_transitions_this_update": [
                {"lifecycle": "in_zone", "entry_mode": "crossed_near_edge"}
            ],
            "group5_micro_bos_transitions_this_update": [
                {
                    "bos_direction": "long",
                    "context_kind": "zone_return",
                    "expected_direction": "long",
                    "outcome": "aligned",
                    "qualified": True,
                    "relation": "strictly_after",
                    "scope": "opposed",
                }
            ],
            "liquidity_inventory_transitions_this_update": [
                {
                    "kind": "swing", "lifecycle": "visible", "side": "above",
                    "structural_rank": "external", "timeframe": "1h",
                    "is_protected_swing": True,
                }
            ],
        }
    }
    graph = {
        "relation_descriptors": [
            {
                "change_kind": "added", "lifecycle": "active",
                "relation": "SOURCED_FROM",
                "source": {
                    "kind": "swing", "lifecycle": "confirmed", "role": "swing",
                    "structural_scale": "external", "timeframe": "1h",
                },
                "target": {
                    "kind": "entry_location", "lifecycle": "approaching",
                    "role": "entry_location", "structural_scale": "internal",
                    "timeframe": "1m",
                },
            }
        ]
    }
    tokens, audit = _structural_tokens(observation, graph, protocol)
    assert "event:group5_micro_bos_transitions_this_update:reference_alignment=aligned" in tokens
    assert "event:liquidity_inventory_transitions_this_update:kind=swing" in tokens
    assert not any("entry" in token or "outcome" in token for token in tokens)
    assert audit["scene_descriptors_excluded_by_endpoint"] == 1

    observation["collections"]["group5_micro_bos_transitions_this_update"][0][
        "outcome"
    ] = "profitable"
    with pytest.raises(RepresentationDataError, match="reference alignment"):
        _structural_tokens(observation, graph, protocol)


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("2021-02-01T08:00:00-05:00", "overnight"),
        ("2021-02-01T09:30:00-05:00", "rth"),
        ("2021-02-01T16:15:00-05:00", "post_rth"),
        ("2021-02-01T18:00:00-05:00", "evening"),
    ],
)
def test_b2_session_phase_is_frozen_to_et(timestamp, expected):
    assert _session_phase(timestamp)[0] == expected


def _record(index: int, *, suffix: str, label: int) -> MaterialRecord:
    return MaterialRecord(
        window=f"window-{index % 2}", run_sha256="a" * 64,
        market_epoch_id="epoch:0", market_episode_id=f"episode:{index}",
        revision_id=f"revision:{index}", et_date=f"2021-02-{index + 1:02d}",
        material_kind="first_pullback", base=("first_pullback", "long", "pullback"),
        signature=("first_pullback", "long", "pullback", suffix),
        signature_suffix=(suffix,), structural_tokens=frozenset(),
        next_lifecycle=label, scale_direction_alignment=1,
    )


def test_b2_conditional_information_shuffle_is_deterministic():
    records = tuple(
        _record(index, suffix="a" if index < 10 else "b", label=5 if index < 10 else 7)
        for index in range(20)
    )
    first = _signature_conditional_mi(
        records, label=lambda record: record.next_lifecycle, shuffle_seed=170021
    )
    second = _signature_conditional_mi(
        records, label=lambda record: record.next_lifecycle, shuffle_seed=170021
    )
    assert first == second
    assert first["conditional_mutual_information_bits"] > 0.9
    assert first["above_shuffle_p95"] is True
