from __future__ import annotations

import json
from pathlib import Path

import pytest

from shares.core.model import Playbook
from brain.core.playbook_registry import (
    PlaybookRegistryError,
    load_playbook_registry,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "brain/configs/playbooks.json"


def _write_registry(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "playbooks.json"
    path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    return path


def test_context_episode_lifecycle_contract_has_protocol_versions() -> None:
    registry = load_playbook_registry(REGISTRY_PATH)
    dfp = registry.for_playbook(Playbook.DISPLACEMENT_FIRST_PULLBACK)
    lsr = registry.for_playbook(Playbook.LIQUIDITY_SWEEP_REVERSAL)
    favr = registry.for_playbook(Playbook.FAILED_AUCTION_VALUE_RETURN)

    assert registry.schema_version == 1
    assert dfp.schema_version == 5
    assert lsr.schema_version == 8
    assert favr.schema_version == 1
    assert dfp.targets["thesis_context_terminal_draw"] == {
        "source": (
            "decision-time visible same-direction H4-timeframe draw; "
            "higher-timeframe/external relative to the M5 setup without "
            "an added structural-rank gate"
        ),
        "role": (
            "establish and retain the directional thesis and terminal "
            "delivery context"
        ),
        "eligible_as_primary_target": False,
    }
    assert "planned target R >= 1" in dfp.targets["selection"]
    assert "current remaining path R >= 1" in dfp.targets["selection"]
    assert dfp.targets["primary_deliverable_target"]["freeze_at"] == (
        "risk_approval"
    )
    assert dfp.invalidation["context_terminal_authority"] == {
        "immutable_roles": [
            "frozen_h4_structure",
            "frozen_protected_swing",
            "frozen_context_draw",
        ],
        "mutable_evidence_only": [
            "latest_h4_high_projection",
            "latest_h4_low_projection",
        ],
        "rule": (
            "a mutable higher-timeframe leg or swing projection may revise "
            "supporting evidence but cannot close the Context Thesis, an "
            "Entry Episode, or a managed position; only an exact typed "
            "terminal of an immutable role may close the Context and "
            "cascade its children"
        ),
    }
    assert (
        dfp.targets["delivery_outcome"]
        == "frozen_primary_deliverable_target_before_frozen_entry_zone_"
        "invalidation_and_deadline"
    )
    root_absence = dfp.deadline["root_absence_lifecycle"]
    assert "identical frozen setup, location and active path" in root_absence
    assert "retain it as dormant without action authority" in root_absence
    assert "completed or compacted Group5 Context path" in (
        lsr.deadline["root_absence_lifecycle"]
    )
    assert "never borrowed from a sibling" in (
        lsr.deadline["root_absence_lifecycle"]
    )
    assert "uniquely earliest completed-clock child" in (
        lsr.deadline["execution_owner"]
    )
    assert "complete entry/stop/primary-target/deadline/route/trigger" in (
        lsr.deadline["execution_owner"]
    )
    assert "sole plan and first-trigger source after completed or invalidated" in (
        lsr.deadline["execution_owner"]
    )
    assert lsr.targets["execution_owner_freeze"] == {
        "clock": (
            "the first uniquely timed completed bar on which one Entry "
            "Episode simultaneously has its own first pullback, same-zone "
            "trigger, every hard gate, and a valid plan"
        ),
        "frozen_fields": [
            "entry_episode_id",
            "entry_location_id",
            "entry_path_id",
            "planned_entry",
            "original_sweep_extreme_stop",
            "primary_target",
            "deadline",
            "liquidity_route",
            "selected_trigger",
            "first_executable_at",
        ],
        "dynamic_diagnostics_only": [
            "current_remaining_path_R",
            "current_frozen_target_visibility",
            "current_hard_obstruction_before_frozen_target",
        ],
        "retrospective_rewrite": False,
        "same_clock_multiple_children": (
            "fail_closed_until_a_new_manipulation_context"
        ),
    }
    assert "accepted_outside" not in (
        lsr.invalidation["scope"]["context_terminal"]
    )
    assert "accepted_outside" not in lsr.deadline["expire_on"]
    assert lsr.invalidation["scope"]["entry_episode_terminal_only"] == [
        "owned_entry_zone_left_or_failed",
        "owned_zone_return_path_terminal",
        "owned_entry_deadline_elapsed",
    ]


def test_dfp_protocol_major_version_changes_registry_fingerprint(
    tmp_path: Path,
) -> None:
    current = load_playbook_registry(REGISTRY_PATH)
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    dfp = next(
        value
        for value in payload["playbooks"]
        if value["id"] == Playbook.DISPLACEMENT_FIRST_PULLBACK.value
    )
    dfp["schema_version"] = 4

    prior_version = load_playbook_registry(_write_registry(tmp_path, payload))

    assert prior_version.for_playbook(
        Playbook.DISPLACEMENT_FIRST_PULLBACK
    ).schema_version == 4
    assert prior_version.fingerprint != current.fingerprint


def test_lsr_protocol_major_version_changes_registry_fingerprint(
    tmp_path: Path,
) -> None:
    current = load_playbook_registry(REGISTRY_PATH)
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    lsr = next(
        value
        for value in payload["playbooks"]
        if value["id"] == Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    )
    lsr["schema_version"] = 4

    prior_version = load_playbook_registry(_write_registry(tmp_path, payload))

    assert prior_version.for_playbook(
        Playbook.LIQUIDITY_SWEEP_REVERSAL
    ).schema_version == 4
    assert prior_version.fingerprint != current.fingerprint


def test_playbook_schema_version_must_be_positive_integer(
    tmp_path: Path,
) -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["playbooks"][0]["schema_version"] = 0

    with pytest.raises(
        PlaybookRegistryError,
        match=r"playbooks\[0\]\.schema_version must be a positive integer",
    ):
        load_playbook_registry(_write_registry(tmp_path, payload))
