"""The frozen week backtest as a regression test.

``brain/docs/evidence/regression_baselines.json`` names the baseline runs
and the summary each produced.  For every run whose journal is present
under ``outputs/brain_journal/`` the replay must reproduce (the Eye is
re-driven, every LLM input's sha, every state revision and every trade
record must match) and the summary computed afresh from the journal must
equal the committed one on its deterministic fields.  A change to the
reducer, the controller, the plan builder, the gate or the executor that
alters a decision fails here; a change to the summarizer shows as a diff.

research_orchestration: minutes of Eye time per run and needs the tape."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.core.journal import JournalReader
from brain.core.main_brain import MainBrainConfig
from brain.core.sleep_controller import ControllerConfig
from brain.scripts._run_identity import DEFAULT_MODEL, DEFAULT_SOURCE
from brain.scripts.replay_journal import replay_run
from brain.scripts.summarize_run import load_pricing, summarize
from execution.core.simulated_executor import SimulatedExecutor, SimulatorConfig
from risk.core.gate import RiskConfig, RiskGate

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "brain" / "docs" / "evidence" / "regression_baselines.json"
JOURNALS = ROOT / "outputs" / "brain_journal"
# Latency and wall-clock fields differ between a run and its replay; everything else must not.
VOLATILE = {("run", "minutes"), ("run", "started_at"), ("run", "finished_at"), ("run", "git_revision")}
pytestmark = pytest.mark.research_orchestration


def _baselines() -> list[dict]:
    if not BASELINES.exists() or not (ROOT / DEFAULT_SOURCE).exists():
        return []
    return [item for item in json.loads(BASELINES.read_text(encoding="utf-8"))["runs"] if (JOURNALS / item["run_id"]).is_dir()]


def _deterministic(summary: dict) -> dict:
    out = {}
    for section, value in summary.items():
        if section in ("timings", "coverage") or not isinstance(value, dict):
            out[section] = value if section not in ("timings", "coverage") else None
            continue
        out[section] = {key: item for key, item in value.items() if (section, key) not in VOLATILE and not (section == "llm" and key == "latency_ms")}
    return out


@pytest.mark.parametrize("baseline", _baselines(), ids=lambda item: item["run_id"])
def test_the_baseline_run_replays_and_summarizes_identically(baseline: dict) -> None:
    run_dir = JOURNALS / baseline["run_id"]
    run = JournalReader(run_dir).run()
    controller = ControllerConfig.from_json(ROOT / "brain" / "configs" / "sleep_controller.json")
    config = MainBrainConfig.from_json(ROOT / run.get("main_brain_config", "brain/configs/main_brain.json"), root=ROOT)
    model = json.loads((ROOT / DEFAULT_MODEL).read_text(encoding="utf-8"))
    risk = RiskConfig.from_json(ROOT / "risk" / "configs" / "risk.json")
    sim = SimulatorConfig.from_json(ROOT / "execution" / "configs" / "simulated_executor.json")
    # The configs may have gained fields since the run (their shas differ); the replay's
    # trade records are the check that no decision changed.
    broker = SimulatedExecutor(sim, tick_size=risk.contract.tick_size, point_value=risk.contract.point_value, equity=float(run["sim_equity"]))
    verdict = replay_run(
        run_dir, observations=None, controller=controller, config=config, tick=float(model["tick_size"]),
        model_path=ROOT / DEFAULT_MODEL, broker=broker, gate=RiskGate(risk),
    )
    assert verdict.ok, verdict.mismatches[:5]
    fresh = summarize(run_dir, pricing=load_pricing(ROOT / "brain" / "configs" / "llm_pricing.json"), bars=None)
    assert _deterministic(fresh) == _deterministic(baseline["summary"])


def test_the_baseline_file_lists_runs_when_present() -> None:
    if not BASELINES.exists():
        pytest.skip("no baseline file yet")
    payload = json.loads(BASELINES.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1 and payload["runs"] and all("run_id" in item and "summary" in item for item in payload["runs"])
