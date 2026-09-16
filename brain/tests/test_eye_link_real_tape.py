"""The real Eye on the real 2022-01 tape drives the controller.

research_orchestration: minutes of Eye time and needs the materialized tape."""
from __future__ import annotations

from pathlib import Path

import pytest

from brain.core.eye_view import EvidenceRule, build_eye_context
from brain.core.object_registry import ObjectRegistry
from brain.core.sleep_controller import ControllerConfig, Decision, decide
from brain.scripts._run_identity import DEFAULT_SOURCE, RunWindow, drive

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / DEFAULT_SOURCE
pytestmark = pytest.mark.research_orchestration


@pytest.mark.skipif(not SOURCE.exists(), reason="materialized NQ tape not linked")
def test_controller_wakes_on_the_real_tape_but_never_on_1m_alone() -> None:
    controller = ControllerConfig.from_json(ROOT / "brain" / "configs" / "sleep_controller.json")
    registry = ObjectRegistry()
    wakes = 0
    one_minute_only_wakes = 0
    emitted = 0
    aliased = 0

    def on_obs(obs, emitting: bool) -> None:
        nonlocal wakes, one_minute_only_wakes, emitted, aliased
        if not emitting:
            return
        emitted += 1
        decision = decide(obs.events_this_update, active=False, config=controller)
        if decision.decision is Decision.WAKE:
            wakes += 1
            if all(event.timeframe.value == "1m" for event in obs.events_this_update):
                one_minute_only_wakes += 1
            context = build_eye_context(obs, registry, rule=controller.evidence)
            aliased = max(aliased, len(context.objects))
            assert context.known_at == obs.asof

    drive(
        RunWindow(SOURCE, "2021-12-30", "2022-01-04", "2022-01-04 12:00"),
        model_path=ROOT / "configs" / "model.json", root=ROOT, on_observation=on_obs,
    )
    assert emitted > 100 and wakes > 0 and one_minute_only_wakes == 0 and aliased > 0
