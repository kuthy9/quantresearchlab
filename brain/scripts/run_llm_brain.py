"""Run the LLM Brain over a window of the tape and write its journal.

    .venv/bin/python -m brain.scripts.run_llm_brain --client deepseek \
        --warmup-start 2021-12-30 --emit-start 2022-01-04 --end "2022-01-04 12:00" \
        --max-llm-calls 60

The Eye is warmed from ``--warmup-start``; the controller and the Brain act
from ``--emit-start``.  ``--client echo`` needs no key and no network: it
answers every call with a contract-valid reply and is for smoke runs only.
``--client deepseek`` reads ``DEEPSEEK_API_KEY`` (and optionally
``DEEPSEEK_BASE_URL``) from the environment, else the gitignored key file.
``--max-llm-calls`` caps spend: past the cap every call is refused and
journaled as an incident while the runtime keeps ticking to the end of the
window.  ``--reasoning-effort low|high|max`` overrides the
config's DeepSeek reasoning budget and enters the run identity as
``deepseek:<model>@<effort>`` (``max`` needs ``brain/configs/main_brain_max.json``:
its replies run past 32 768 tokens).  ``--broker sim`` runs the Risk gate and the order machine against
the local simulated executor (OHLCV matching on the tape's bars, a virtual
account from ``execution/configs/simulated_executor.json``, ``--sim-equity``
overrides its starting cash); ``--broker ibkr`` against the IBKR paper
session described by ``execution/configs/ibkr.json`` — refused over a tape
older than a day or an account that is not flat; the default ``none`` runs
the Brain alone."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

import pandas as pd

from brain.core.journal import BrainJournal
from brain.core.llm_client import DeepSeekClient, EchoClient, LLMClient, LLMReply, LLMRequestRejected
from brain.core.main_brain import MainBrain, MainBrainConfig
from brain.core.position_ledger import InMemoryPositionLedger
from brain.core.runtime import BrainRuntime, StepResult
from brain.core.sleep_controller import ControllerConfig, Decision
from execution.core.broker import Broker
from execution.core.ibkr_broker import IBKRBroker, IBKRConfig, IBKRRefused, require_flat
from execution.core.order_fsm import OrderMachine
from execution.core.simulated_executor import SimulatedExecutor, SimulatorConfig
from execution.core.stack import TradingStack
from risk.core.gate import RiskConfig, RiskGate
from shares.core.timing import Timings
from brain.scripts._run_identity import (
    DEFAULT_MODEL,
    DEFAULT_SOURCE,
    MARKET_TIMEZONE,
    ROOT,
    RunWindow,
    drive,
    eye_identities,
    run_identity,
    sha256_file,
)

DEFAULT_CONTROLLER = "brain/configs/sleep_controller.json"
DEFAULT_CONFIG = "brain/configs/main_brain.json"
DEFAULT_OUTPUT_ROOT = "outputs/brain_journal"
DEFAULT_RISK = "risk/configs/risk.json"
DEFAULT_IBKR = "execution/configs/ibkr.json"
DEFAULT_SIM = "execution/configs/simulated_executor.json"
# ``--broker ibkr`` prices its orders from the tape it is driven over; a tape
# older than this is another market, and its limits would be marketable or
# unreachable at TWS.
IBKR_MAX_TAPE_AGE = pd.Timedelta(days=1)


def model_label(client: str, model: str, reasoning_effort: str | None) -> str:
    """The model as the run identity names it: the effort is part of it."""
    return f"{client}:{model}" if reasoning_effort is None else f"{client}:{model}@{reasoning_effort}"


def tape_is_current(end: str, *, now: pd.Timestamp, max_age: pd.Timedelta = IBKR_MAX_TAPE_AGE) -> bool:
    """Whether a window ending at ``end`` (market time) is recent enough for
    orders priced from it to reach a live session."""
    ends = pd.Timestamp(end, tz=MARKET_TIMEZONE).tz_convert("UTC")
    return pd.Timestamp(now).tz_convert("UTC") - ends <= max_age


class BudgetedClient:
    """Refuses every call past ``max_calls`` so a run cannot overspend."""

    def __init__(self, inner: LLMClient, *, max_calls: int | None) -> None:
        self._inner = inner
        self._max = max_calls
        self.calls = 0

    def complete(self, *, system: str, user: str) -> LLMReply:
        if self._max is not None and self.calls >= self._max:
            raise LLMRequestRejected(f"LLM call budget of {self._max} exhausted")
        self.calls += 1
        return self._inner.complete(system=system, user=user)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--model-path", default=DEFAULT_MODEL, help="the Eye's configs/model.json")
    parser.add_argument("--warmup-start", required=True)
    parser.add_argument("--emit-start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--client", choices=("deepseek", "echo"), default="deepseek")
    parser.add_argument("--model", default=None, help="LLM model id; defaults to main_brain.json")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default=None, help="overrides main_brain.json's reasoning_effort")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=2000)
    parser.add_argument("--broker", choices=("none", "sim", "ibkr"), default="none")
    parser.add_argument("--risk-config", default=DEFAULT_RISK)
    parser.add_argument("--ibkr-config", default=DEFAULT_IBKR)
    parser.add_argument("--sim-config", default=DEFAULT_SIM)
    parser.add_argument("--sim-equity", type=float, default=None, help="starting cash; defaults to the simulator config's initial_equity")
    return parser.parse_args(argv)


def build_broker(
    kind: str, *, risk: RiskConfig, ibkr_config: Path, sim_config: SimulatorConfig | None, sim_equity: float | None, live_execution_allowed: bool
) -> Broker | None:
    if kind == "none":
        return None
    if kind == "sim":
        assert sim_config is not None
        return SimulatedExecutor(sim_config, tick_size=risk.contract.tick_size, point_value=risk.contract.point_value, equity=sim_equity)
    return IBKRBroker.connect(IBKRConfig.from_json(ibkr_config), live_execution_allowed=live_execution_allowed)


def build_client(kind: str, config: MainBrainConfig, model: str) -> LLMClient:
    if kind == "echo":
        return EchoClient(sleep_after=3)
    return DeepSeekClient(
        model=model, timeout_s=config.timeout_s, max_tokens=config.max_tokens, reasoning_effort=config.reasoning_effort,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = ROOT
    controller = ControllerConfig.from_json(root / args.controller)
    config = MainBrainConfig.from_json(root / args.config, root=root)
    if args.reasoning_effort is not None:
        config = dataclasses.replace(config, reasoning_effort=args.reasoning_effort)
    model = args.model or config.model
    window = RunWindow(Path(args.source), args.warmup_start, args.emit_start, args.end)
    source_path = window.source if window.source.is_absolute() else root / window.source
    atomic_identity, registry_id = eye_identities(root / args.model_path, root=root)
    run_id, run = run_identity(
        window=window, model=model_label(args.client, model, config.reasoning_effort), prompt_sha256=config.prompt_sha256,
        controller_sha256=controller.sha256, config_sha256=config.sha256,
        atomic_identity=atomic_identity, scale_registry_id=registry_id,
        source_sha256=sha256_file(source_path), root=root,
    )
    run_dir = root / args.output_root / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"run {run_id} already exists at {run_dir}; refusing to overwrite", file=sys.stderr)
        return 2
    eye_model = json.loads((root / args.model_path).read_text(encoding="utf-8"))
    risk = RiskConfig.from_json(root / args.risk_config)
    if args.broker == "ibkr" and not tape_is_current(args.end, now=pd.Timestamp.now(tz="UTC")):
        print(f"--broker ibkr refused: the window ends at {args.end}, more than {IBKR_MAX_TAPE_AGE} ago; its prices are not this market's", file=sys.stderr)
        return 3
    sim_config = SimulatorConfig.from_json(root / args.sim_config) if args.broker == "sim" else None
    broker = build_broker(
        args.broker, risk=risk, ibkr_config=root / args.ibkr_config, sim_config=sim_config, sim_equity=args.sim_equity,
        live_execution_allowed=bool(eye_model["release_readiness"]["live_execution_allowed"]),
    )
    if args.broker == "ibkr" and broker is not None:
        try:
            require_flat(broker.snapshot(pd.Timestamp.now(tz="UTC")), risk.contract.symbol)
        except IBKRRefused as error:
            print(f"--broker ibkr refused: {error}", file=sys.stderr)
            return 3
    journal = BrainJournal(run_dir, run_id=run_id)
    run.update({
        "client": args.client, "max_llm_calls": args.max_llm_calls, "eye_model": args.model_path,
        "reasoning_effort": config.reasoning_effort, "main_brain_config": args.config,
        "broker": args.broker, "risk_config_sha256": risk.sha256, "max_open_positions": risk.max_open_positions,
        "sim_equity": None if not isinstance(broker, SimulatedExecutor) else broker.account.cash,
        "simulator_config_sha256": None if sim_config is None else sim_config.sha256,
    })
    journal.write_run(run)

    client = BudgetedClient(build_client(args.client, config, model), max_calls=args.max_llm_calls)
    machine = None if broker is None else OrderMachine(broker, RiskGate(risk), journal=journal)
    ledger = InMemoryPositionLedger() if machine is None else machine.ledger
    tick = float(eye_model["tick_size"])
    timings = Timings()
    runtime = BrainRuntime(
        controller=controller,
        brain=MainBrain(client=client, config=config, ledger=ledger, timings=timings),
        journal=journal, ledger=ledger, tick=tick, timings=timings,
    )
    stack = TradingStack(runtime, machine, tick=tick, timings=timings)
    counts = {kind.value: 0 for kind in Decision}
    incidents = 0
    emitted = 0
    trades = 0
    started = time.monotonic()

    def on_observation(observation, emitting: bool, bar) -> None:
        nonlocal emitted, incidents, trades
        if not emitting:
            return
        emitted += 1
        result: StepResult = stack.step(observation, bar)
        counts[result.decision.value] += 1
        if result.incident:
            incidents += 1
        trades += len(stack.last_trade_kinds)
        if result.decision in (Decision.WAKE, Decision.UPDATE) or result.slept or stack.last_trade_kinds:
            print(
                f"{result.known_at.strftime('%Y-%m-%d %H:%M')}Z {result.decision.value:11s} "
                f"{result.episode_id} rev={result.revision} llm={client.calls}"
                f"{' incident=' + result.incident if result.incident else ''}"
                f"{' SLEEP' if result.slept else ''}"
                f"{' trade=' + ','.join(stack.last_trade_kinds) if stack.last_trade_kinds else ''}"
            )

    print(f"run {run_id} → {run_dir}")
    bars = drive(
        window, model_path=root / args.model_path, root=root,
        on_observation=on_observation, progress_every=args.progress_every, timings=timings, stop=lambda: stack.halted,
    )
    finished = {
        **run,
        "finished_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bars_seen": bars,
        "bars_emitted": emitted,
        "llm_calls": client.calls,
        "decisions": counts,
        "incidents": incidents,
        "trade_records": trades,
        "simulated_account": broker.account.summary() if isinstance(broker, SimulatedExecutor) else None,
        "machine_stats": None if machine is None else dict(machine.stats),
        "halted": None if machine is None else machine.halt_record,
        "timings": timings.summary(),
        "minutes": round((time.monotonic() - started) / 60, 2),
    }
    journal.write_run(finished)
    print(json.dumps({k: finished[k] for k in ("bars_emitted", "llm_calls", "decisions", "incidents", "trade_records", "minutes")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
