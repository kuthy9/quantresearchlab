from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.check_phase8_execution_readiness import (
    DEFAULT_TEMPLATE,
    EXPECTED_BLOCKERS,
    Phase8ReadinessError,
    audit_execution_readiness,
)


ROOT = Path(__file__).resolve().parents[1]


def test_phase8_template_reports_blockers_without_opening_inputs() -> None:
    report = audit_execution_readiness()

    assert report["ready"] is False
    assert report["opened_input_ledgers"] is False
    assert report["artifacts_written"] == []
    assert report["formal_runner_implemented"] is False
    assert tuple(report["readiness_blockers"]) == EXPECTED_BLOCKERS


def test_phase8_checker_cli_uses_blocked_exit_code() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_phase8_execution_readiness.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout)["ready"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__(
            "status", "frozen_research_only_authorized_to_run"
        ),
        lambda value: value["authority"].__setitem__("order_submission", True),
        lambda value: value["identity_bindings"]["trade_intent_ledger"].update(
            {"path": "fixture.jsonl", "sha256": "a" * 64}
        ),
        lambda value: value["readiness_blockers"].clear(),
        lambda value: value["outputs"].__setitem__(
            "summary_json", "outputs/fixture.json"
        ),
    ),
)
def test_phase8_template_cannot_self_promote(mutation, tmp_path: Path) -> None:
    payload = deepcopy(json.loads(DEFAULT_TEMPLATE.read_text(encoding="utf-8")))
    mutation(payload)
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Phase8ReadinessError):
        audit_execution_readiness(changed)
