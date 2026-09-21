"""Replay a Brain journal from the Eye alone and prove it reproduces.

    .venv/bin/python -m brain.scripts.replay_journal --run-dir outputs/brain_journal/<run_id>

The Eye is re-driven over the run's window; every ``LLMInput`` is rebuilt
from the current bar and compared byte-for-byte (by sha) with the journal's
``llm_call``; the recorded replies are fed back through the reducer, and every
``state`` and ``tick`` revision must match.  Nothing the journal holds is read
by the replay before the bar that produced it — the only thing it takes from
the future is the LLM's own answer, keyed by the input it answered.  Exit 0
when every episode reproduces, 1 on the first mismatch."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys

from brain.core.journal import JournalError, JournalReader, JournalRecord
from brain.core.llm_client import LLMClientError, RecordedClient
from brain.core.main_brain import MainBrain, MainBrainConfig
from brain.core.position_ledger import InMemoryPositionLedger
from brain.core.runtime import BrainRuntime, StepResult
from brain.core.sleep_controller import ControllerConfig, Decision
from brain.scripts._run_identity import DEFAULT_MODEL, ROOT, RunWindow, drive
from contract.brain.llm import LLMInput, canonical_json
from contract.brain.state import BrainState
from contract.eye import MarketObservation
from contract.market.primitives import Bar
from execution.core.broker import Broker
from execution.core.order_fsm import OrderMachine
from execution.core.simulated_executor import SimulatedExecutor, SimulatorConfig
from execution.core.stack import TradingStack
from risk.core.gate import RiskConfig, RiskGate

DEFAULT_CONTROLLER = "brain/configs/sleep_controller.json"
DEFAULT_CONFIG = "brain/configs/main_brain.json"
DEFAULT_RISK = "risk/configs/risk.json"
DEFAULT_SIM = "execution/configs/simulated_executor.json"


@dataclass(frozen=True)
class ReplayVerdict:
    ok: bool
    episodes: int
    llm_calls: int
    revisions: int
    mismatches: tuple[str, ...]
    trades: int = 0


class _ReplayDiverged(RuntimeError):
    """The replay asked the recorded client for an input the run never sent."""


class _Expectation:
    """The journal's records of one episode, consumed in order."""

    def __init__(self, records: Sequence[JournalRecord]) -> None:
        self.calls = [r for r in records if r.record == "llm_call"]
        self.states = [r for r in records if r.record in ("state", "tick")]
        self.trades = [str(r.payload["kind"]) for r in records if r.record == "trade"]
        self.slept = any(r.record == "sleep" for r in records)
        self.call_cursor = 0
        self.state_cursor = 0
        self.trade_cursor = 0


def replay_run(
    run_dir: Path,
    *,
    observations: Sequence[MarketObservation] | None,
    controller: ControllerConfig,
    config: MainBrainConfig,
    tick: float,
    model_path: Path | None = None,
    log=lambda message: None,
    bars: Sequence[Bar] | None = None,
    broker: Broker | None = None,
    gate: RiskGate | None = None,
) -> ReplayVerdict:
    """``broker`` and ``gate`` re-drive the order machine of a ``sim`` run so
    its ``trade`` records and its engaged-Brain steps reproduce; an ``ibkr``
    run replays its Brain path only."""
    reader = JournalReader(run_dir)
    run = reader.run()
    run_id = str(run.get("run_id", ""))
    mismatches: list[str] = []
    expectations: dict[str, _Expectation] = {}
    for episode_id in reader.episode_ids():
        try:
            reader.verify_chain(episode_id, run_id=run_id or None)
        except JournalError as error:
            mismatches.append(f"{episode_id}: chain broken — {error}")
        expectations[episode_id] = _Expectation(reader.records(episode_id))

    client = RecordedClient(reader.recorded_replies(), reader.recorded_incidents())
    machine = None if broker is None or gate is None else OrderMachine(broker, gate, journal=None)
    ledger = InMemoryPositionLedger() if machine is None else machine.ledger
    calls = 0
    revisions = 0
    trades = 0

    def on_step(result: StepResult, state: BrainState | None, llm_input: LLMInput | None) -> None:
        nonlocal calls, revisions
        if result.decision is Decision.STAY_ASLEEP:
            return
        expectation = expectations.get(result.episode_id or "")
        if expectation is None:
            mismatches.append(f"{result.episode_id}: the journal has no such episode")
            return
        if llm_input is not None:
            calls += 1
            if expectation.call_cursor >= len(expectation.calls):
                mismatches.append(f"{result.episode_id}: extra llm_call at {result.known_at}")
            else:
                recorded = expectation.calls[expectation.call_cursor]
                expectation.call_cursor += 1
                if recorded.payload["input_sha"] != llm_input.input_sha:
                    mismatches.append(
                        f"{result.episode_id}: input sha differs at {result.known_at} "
                        f"(journal {recorded.payload['input_sha'][:12]}, replay {llm_input.input_sha[:12]})"
                    )
        if state is not None:
            revisions += 1
            if expectation.state_cursor >= len(expectation.states):
                mismatches.append(f"{result.episode_id}: extra revision {state.revision} at {result.known_at}")
            else:
                recorded = expectation.states[expectation.state_cursor]
                expectation.state_cursor += 1
                if recorded.payload["revision"] != state.revision:
                    mismatches.append(
                        f"{result.episode_id}: revision {state.revision} where the journal has {recorded.payload['revision']}"
                    )
                elif recorded.record == "state" and canonical_json(recorded.payload["state"]) != state.to_json():
                    mismatches.append(f"{result.episode_id}: state revision {state.revision} differs")
        if result.slept and not expectation.slept:
            mismatches.append(f"{result.episode_id}: replay slept but the journal did not")

    runtime = BrainRuntime(
        controller=controller,
        brain=MainBrain(client=client, config=config, ledger=ledger, sleep=lambda seconds: None),
        journal=None, ledger=ledger, tick=tick, on_step=on_step,
    )
    stack = TradingStack(runtime, machine, tick=tick)

    def step(observation: MarketObservation, bar: Bar | None) -> None:
        nonlocal trades
        try:
            result = stack.step(observation, bar)
        except LLMClientError as error:
            # The journal holds neither a reply nor an incident for this input:
            # the replay diverged from the run before this call.
            raise _ReplayDiverged(f"{runtime.state.episode_id if runtime.state else '?'}: {error}") from None
        if machine is None or result.episode_id is None:
            return
        expectation = expectations.get(result.episode_id)
        for kind in stack.last_trade_kinds:
            trades += 1
            if expectation is None:
                continue
            if expectation.trade_cursor >= len(expectation.trades):
                mismatches.append(f"{result.episode_id}: extra trade record {kind} at {result.known_at}")
            else:
                recorded = expectation.trades[expectation.trade_cursor]
                expectation.trade_cursor += 1
                if recorded != kind:
                    mismatches.append(f"{result.episode_id}: trade {kind} where the journal has {recorded} at {result.known_at}")

    try:
        if observations is not None:
            paired = zip(observations, bars if bars is not None else [None] * len(observations))
            for observation, bar in paired:
                if stack.halted:
                    break  # the runner stops at the drawdown halt; so does the replay (2026-09-21)
                step(observation, bar)
        else:
            window = RunWindow(Path(run["window"]["source"]), run["window"]["warmup_start"], run["window"]["emit_start"], run["window"]["end"])
            source = window.source if window.source.is_absolute() else ROOT / window.source
            window = RunWindow(source, window.warmup_start, window.emit_start, window.end)
            drive(
                window, model_path=model_path or ROOT / DEFAULT_MODEL, root=ROOT,
                on_observation=lambda observation, emitting, bar: step(observation, bar) if emitting else None,
                progress_every=2000, log=log, stop=lambda: stack.halted,
            )
    except _ReplayDiverged as error:
        mismatches.append(str(error))

    for episode_id, expectation in expectations.items():
        if expectation.call_cursor != len(expectation.calls):
            mismatches.append(f"{episode_id}: journal has {len(expectation.calls)} llm_calls, replay made {expectation.call_cursor}")
        if expectation.state_cursor != len(expectation.states):
            mismatches.append(f"{episode_id}: journal has {len(expectation.states)} revisions, replay produced {expectation.state_cursor}")
        if machine is not None and expectation.trade_cursor != len(expectation.trades):
            mismatches.append(f"{episode_id}: journal has {len(expectation.trades)} trade records, replay produced {expectation.trade_cursor}")

    return ReplayVerdict(not mismatches, len(expectations), calls, revisions, tuple(mismatches), trades)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER)
    parser.add_argument("--risk-config", default=DEFAULT_RISK)
    parser.add_argument("--sim-config", default=DEFAULT_SIM)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    controller = ControllerConfig.from_json(ROOT / args.controller)
    config = MainBrainConfig.from_json(ROOT / args.config, root=ROOT)
    model = json.loads((ROOT / args.model_path).read_text(encoding="utf-8"))
    tick = float(model["tick_size"])
    run = JournalReader(run_dir).run()
    broker = gate = None
    if run.get("broker") == "sim":
        risk = RiskConfig.from_json(ROOT / args.risk_config)
        sim = SimulatorConfig.from_json(ROOT / args.sim_config)
        if run.get("simulator_config_sha256") not in (None, sim.sha256):
            print(f"WARNING simulator config sha differs from the run's ({run['simulator_config_sha256'][:12]} vs {sim.sha256[:12]})")
        broker = SimulatedExecutor(sim, tick_size=risk.contract.tick_size, point_value=risk.contract.point_value, equity=float(run["sim_equity"]))
        gate = RiskGate(risk)
    verdict = replay_run(
        run_dir, observations=None, controller=controller, config=config, tick=tick,
        model_path=ROOT / args.model_path, log=print, broker=broker, gate=gate,
    )
    for line in verdict.mismatches:
        print(f"MISMATCH {line}")
    print(
        f"replay {'OK' if verdict.ok else 'FAILED'}: {verdict.episodes} episodes, "
        f"{verdict.llm_calls} llm calls, {verdict.revisions} revisions, {verdict.trades} trade records reproduced"
    )
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    sys.exit(main())
