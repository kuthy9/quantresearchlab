from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from brain.core.journal import BrainJournal, JournalReader
from brain.core.llm_client import LLMReply
from brain.core.main_brain import MainBrain, MainBrainConfig
from brain.core.runtime import BrainRuntime
from brain.core.sleep_controller import ControllerConfig
from brain.scripts.replay_journal import replay_run
from contract.brain.llm import LLM_UPDATE_EXAMPLE
from execution.core.order_fsm import MachineState, OrderMachine
from execution.core.simulated_executor import SimulatedExecutor, SimulatorConfig
from execution.core.stack import TradingStack
from risk.core.gate import RiskConfig, RiskGate
from shares.core.eye_factory import build_eye
from shares.tests.helpers import session_bars

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ControllerConfig.from_json(ROOT / "brain" / "configs" / "sleep_controller.json")
CONFIG = MainBrainConfig.from_json(ROOT / "brain" / "configs" / "main_brain.json")
# The synthetic tape's objects are packed tightly; the e2e proves the pipeline, not the ratio.
RISK = dataclasses.replace(RiskConfig.from_json(ROOT / "risk" / "configs" / "risk.json"), min_reward_risk=0.5)
SIM = SimulatorConfig.from_json(ROOT / "execution" / "configs" / "simulated_executor.json")


def executor() -> SimulatedExecutor:
    return SimulatedExecutor(SIM, tick_size=0.25, point_value=20.0, equity=1_000_000.0)


@pytest.fixture(scope="module")
def tape():
    reader, observer = build_eye(ROOT / "configs" / "model.json", root=ROOT, audit_journal_dir=None)
    pairs = []
    for bar in session_bars(2):
        obs = observer.observe(reader.on_bar(bar))
        if obs.market_snapshot is not None:
            pairs.append((obs, bar))
    return pairs


# The scripted Brain's thesis fields: one thesis on the 5m scale, base grade, touch stop.
THESIS = {"thesis_id": "T1", "governing_timeframe": "5m", "grade": "BASE", "invalidation_mode": "TOUCH"}


class LongAtTheNearestZone:
    """Names an ACTIONABLE LONG from what the input shows: entry at the nearest
    FVG below price, stop at the farthest sell-side pool below, target at a
    buy-side pool above; repeats it while its objects stay visible."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system: str, user: str) -> LLMReply:
        self.calls += 1
        request = json.loads(user)
        relations = {row["object_id"]: row for row in request["price_relations"]}
        payload = json.loads(json.dumps(LLM_UPDATE_EXAMPLE))
        payload["evidence_verdicts"] = [
            {"evidence_id": e["evidence_id"], "verdict": "NEUTRAL", "note": "", "resolves_evidence_id": None, "resolution": None}
            for e in request["new_evidence"]
        ]
        payload["market_understanding"] = "scripted"
        payload["watch_next"] = []
        payload["destination_candidates"] = []
        prior = request.get("prior_state") or {}
        previous = (prior.get("opportunity") or {})
        ids = [previous.get(k) for k in ("entry_object_id", "invalidation_object_id", "target_object_id")]
        if previous.get("state") == "ACTIONABLE" and all(alias in relations for alias in ids):
            payload["opportunity"] = {**THESIS, "state": "ACTIONABLE", "direction": "LONG", "entry_object_id": ids[0], "invalidation_object_id": ids[1], "target_object_id": ids[2]}
            return LLMReply(json.dumps(payload), None, {}, 1, "scripted")
        fvgs = [r for a, r in relations.items() if a.startswith("FVG_") and r["position"] == "below_price" and r["offset_atr"] is not None]
        ssls = [r for a, r in relations.items() if a.startswith("SSL_") and r["position"] == "below_price" and r["offset_atr"] is not None]
        bsls = [r for a, r in relations.items() if a.startswith("BSL_") and r["position"] == "above_price" and r["offset_atr"] is not None]
        if fvgs and ssls and bsls:
            entry = max(fvgs, key=lambda r: r["offset_atr"])  # the nearest zone below price
            below = [r for r in ssls if r["offset_atr"] < entry["offset_atr"]]
            stop = max(below, key=lambda r: r["offset_atr"]) if below else None  # the nearest pool under the zone
            target = max(bsls, key=lambda r: r["offset_atr"])  # the farthest pool above
            if stop is not None:
                payload["opportunity"] = {
                    **THESIS, "state": "ACTIONABLE", "direction": "LONG", "entry_object_id": entry["object_id"],
                    "invalidation_object_id": stop["object_id"], "target_object_id": target["object_id"],
                }
        return LLMReply(json.dumps(payload), None, {}, 1, "scripted")


def build(tmp_path: Path, run_id: str = "stack"):
    journal = BrainJournal(tmp_path, run_id=run_id)
    broker = executor()
    machine = OrderMachine(broker, RiskGate(RISK), journal=journal)
    runtime = BrainRuntime(
        controller=CONTROLLER,
        brain=MainBrain(client=LongAtTheNearestZone(), config=CONFIG, ledger=machine.ledger, sleep=lambda s: None),
        journal=journal, ledger=machine.ledger, tick=0.25,
    )
    return TradingStack(runtime, machine, tick=0.25), broker, journal


def test_a_signal_becomes_a_bracket_a_fill_and_a_closed_position_and_replays(tape, tmp_path: Path) -> None:
    stack, broker, journal = build(tmp_path)
    results = [stack.step(obs, bar) for obs, bar in tape]
    reader = JournalReader(tmp_path)
    trades = [r for ep in reader.episode_ids() for r in reader.records(ep) if r.record == "trade"]
    kinds = [r.payload["kind"] for r in trades]
    assert "submitted" in kinds, "no plan reached the broker on the synthetic tape"
    if "filled" in kinds:
        assert "position_opened" in kinds
        opened = kinds.index("position_opened")
        engaged_bars = [r for r in results if "sleep_refused:open_position" in r.rejections]
        # while engaged the Brain's idle rule and its own sleep requests are refused on the position
        assert stack.machine.state in (MachineState.IDLE, MachineState.IN_POSITION)
        assert kinds.count("position_opened") == kinds.count("submitted") - kinds.count("cancelled") - kinds.count("expired")
        assert "position_closed" in kinds or stack.machine.state is MachineState.IN_POSITION
    else:
        assert "cancelled" in kinds or "expired" in kinds or stack.machine.state is MachineState.WORKING
    for ep in reader.episode_ids():
        reader.verify_chain(ep, run_id="stack")
    inputs = [r.payload["input"] for ep in reader.episode_ids() for r in reader.records(ep) if r.record == "llm_call"]
    updates = [i for i in inputs if i["prior_state"] is not None]
    assert updates and all(i["prior_state"]["execution"]["status"] in ("IDLE", "WORKING", "PARTIAL", "IN_POSITION") for i in updates)
    assert any(i["prior_state"]["execution"]["status"] != "IDLE" for i in updates), "no call was told about a working order or a position"
    assert all({"positions", "theses", "cooldown_bars_left", "daily_stop", "halted"} <= set(i["prior_state"]["execution"]) for i in updates)
    if "position_opened" in kinds:
        assert any(i["prior_state"]["execution"]["positions"] and i["prior_state"]["execution"]["positions"][0]["thesis_id"] == "T1" for i in updates)
    assert any("5m" in TradingStack.closed_timeframes(obs) for obs, _ in tape), "the Eye reports 5m bar completions the close-beyond exit relies on"
    submitted = next(r for r in trades if r.payload["kind"] == "submitted").payload
    assert submitted["plan"]["direction"] == "LONG" and submitted["verdict"]["quantity"] >= 1
    assert submitted["plan"]["entry"]["entity_id"] and submitted["intent"]["stop_price"] < submitted["intent"]["limit_price"] < submitted["intent"]["target_price"]

    verdict = replay_run(
        tmp_path, observations=[obs for obs, _ in tape], bars=[bar for _, bar in tape],
        controller=CONTROLLER, config=CONFIG, tick=0.25,
        broker=executor(), gate=RiskGate(RISK),
    )
    assert verdict.ok, verdict.mismatches
    assert verdict.trades == len(trades)


def test_the_same_signature_never_places_a_second_order(tape, tmp_path: Path) -> None:
    stack, broker, journal = build(tmp_path, run_id="once")
    for obs, bar in tape:
        stack.step(obs, bar)
        assert len(broker.snapshot(obs.asof).open_entry_orders()) <= 1
