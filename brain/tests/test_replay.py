"""A journal written by the runtime replays from the Eye to identical states."""
from __future__ import annotations

from pathlib import Path

import pytest

from brain.core.journal import BrainJournal, JournalReader
from brain.core.llm_client import EchoClient
from brain.core.main_brain import MainBrain, MainBrainConfig
from brain.core.position_ledger import InMemoryPositionLedger
from brain.core.runtime import BrainRuntime
from brain.core.sleep_controller import ControllerConfig
from brain.scripts.replay_journal import replay_run
from shares.core.eye_factory import build_eye
from shares.tests.helpers import session_bars

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ControllerConfig.from_json(ROOT / "brain" / "configs" / "sleep_controller.json")
CONFIG = MainBrainConfig.from_json(ROOT / "brain" / "configs" / "main_brain.json")


@pytest.fixture(scope="module")
def observations():
    reader, observer = build_eye(ROOT / "configs" / "model.json", root=ROOT, audit_journal_dir=None)
    return [o for o in (observer.observe(reader.on_bar(b)) for b in session_bars(2)) if o.market_snapshot is not None]


def _record(tmp_path: Path, observations, client) -> JournalReader:
    journal = BrainJournal(tmp_path, run_id="synthetic")
    journal.write_run({"run_id": "synthetic", "synthetic_sessions": 2})
    ledger = InMemoryPositionLedger()
    runtime = BrainRuntime(
        controller=CONTROLLER,
        brain=MainBrain(client=client, config=CONFIG, ledger=ledger, sleep=lambda s: None),
        journal=journal, ledger=ledger, tick=0.25,
    )
    for obs in observations:
        runtime.step(obs)
    return JournalReader(tmp_path)


def test_scripted_run_replays_to_identical_states(tmp_path: Path, observations) -> None:
    reader = _record(tmp_path, observations, EchoClient(sleep_after=4))
    assert reader.episode_ids()
    verdict = replay_run(tmp_path, observations=observations, controller=CONTROLLER, config=CONFIG, tick=0.25)
    assert verdict.mismatches == () and verdict.ok
    assert verdict.episodes == len(reader.episode_ids()) and verdict.llm_calls > 0 and verdict.revisions > verdict.llm_calls


def test_incidents_replay_too(tmp_path: Path, observations) -> None:
    reader = _record(tmp_path, observations, EchoClient(sleep_after=4, fail_calls=range(2, 7)))
    assert reader.recorded_incidents()
    verdict = replay_run(tmp_path, observations=observations, controller=CONTROLLER, config=CONFIG, tick=0.25)
    assert verdict.mismatches == () and verdict.ok


def test_replay_detects_a_tampered_state(tmp_path: Path, observations) -> None:
    reader = _record(tmp_path, observations, EchoClient())
    ep = reader.episode_ids()[0]
    path = tmp_path / "episodes" / f"{ep}.jsonl"
    text = path.read_text().replace('"market_understanding":"echo', '"market_understanding":"TAMPERED echo', 1)
    assert text != path.read_text()
    path.write_text(text)
    verdict = replay_run(tmp_path, observations=observations, controller=CONTROLLER, config=CONFIG, tick=0.25)
    assert not verdict.ok and any("chain broken" in m for m in verdict.mismatches)


def test_replay_detects_a_missing_revision(tmp_path: Path, observations) -> None:
    reader = _record(tmp_path, observations, EchoClient())
    ep = reader.episode_ids()[0]
    path = tmp_path / "episodes" / f"{ep}.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")  # drop the last record
    verdict = replay_run(tmp_path, observations=observations, controller=CONTROLLER, config=CONFIG, tick=0.25)
    assert not verdict.ok
