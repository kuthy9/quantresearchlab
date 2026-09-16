"""Run the LLM Brain over a window of the tape and write its journal.

    .venv/bin/python -m brain.scripts.run_llm_brain --client deepseek \
        --warmup-start 2021-12-30 --emit-start 2022-01-04 --end "2022-01-04 12:00" \
        --max-llm-calls 60

The Eye is warmed from ``--warmup-start``; the controller and the Brain act
from ``--emit-start``.  ``--client echo`` needs no key and no network: it
answers every call with a contract-valid reply and is for smoke runs only.
``--client deepseek`` reads ``DEEPSEEK_API_KEY`` (and optionally
``DEEPSEEK_BASE_URL``) from the environment.  ``--max-llm-calls`` caps spend:
past the cap every call is refused and journaled as an incident while the
runtime keeps ticking to the end of the window."""
from __future__ import annotations

import argparse
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
from brain.scripts._run_identity import (
    DEFAULT_MODEL,
    DEFAULT_SOURCE,
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
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=2000)
    return parser.parse_args(argv)


def build_client(kind: str, config: MainBrainConfig, model: str) -> LLMClient:
    if kind == "echo":
        return EchoClient(sleep_after=3)
    return DeepSeekClient(model=model, timeout_s=config.timeout_s, max_tokens=config.max_tokens)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = ROOT
    controller = ControllerConfig.from_json(root / args.controller)
    config = MainBrainConfig.from_json(root / args.config, root=root)
    model = args.model or config.model
    window = RunWindow(Path(args.source), args.warmup_start, args.emit_start, args.end)
    source_path = window.source if window.source.is_absolute() else root / window.source
    atomic_identity, registry_id = eye_identities(root / args.model_path, root=root)
    run_id, run = run_identity(
        window=window, model=f"{args.client}:{model}", prompt_sha256=config.prompt_sha256,
        controller_sha256=controller.sha256, config_sha256=config.sha256,
        atomic_identity=atomic_identity, scale_registry_id=registry_id,
        source_sha256=sha256_file(source_path), root=root,
    )
    run_dir = root / args.output_root / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"run {run_id} already exists at {run_dir}; refusing to overwrite", file=sys.stderr)
        return 2
    journal = BrainJournal(run_dir, run_id=run_id)
    run.update({"client": args.client, "max_llm_calls": args.max_llm_calls, "eye_model": args.model_path})
    journal.write_run(run)

    client = BudgetedClient(build_client(args.client, config, model), max_calls=args.max_llm_calls)
    ledger = InMemoryPositionLedger()
    runtime = BrainRuntime(
        controller=controller,
        brain=MainBrain(client=client, config=config, ledger=ledger),
        journal=journal, ledger=ledger, tick=json.loads((root / args.model_path).read_text())["tick_size"],
    )
    counts = {kind.value: 0 for kind in Decision}
    incidents = 0
    emitted = 0
    started = time.monotonic()

    def on_observation(observation, emitting: bool) -> None:
        nonlocal emitted, incidents
        if not emitting:
            return
        emitted += 1
        result: StepResult = runtime.step(observation)
        counts[result.decision.value] += 1
        if result.incident:
            incidents += 1
        if result.decision in (Decision.WAKE, Decision.UPDATE) or result.slept:
            print(
                f"{result.known_at.strftime('%Y-%m-%d %H:%M')}Z {result.decision.value:11s} "
                f"{result.episode_id} rev={result.revision} llm={client.calls}"
                f"{' incident=' + result.incident if result.incident else ''}"
                f"{' SLEEP' if result.slept else ''}"
            )

    print(f"run {run_id} → {run_dir}")
    bars = drive(
        window, model_path=root / args.model_path, root=root,
        on_observation=on_observation, progress_every=args.progress_every,
    )
    finished = {
        **run,
        "finished_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bars_seen": bars,
        "bars_emitted": emitted,
        "llm_calls": client.calls,
        "decisions": counts,
        "incidents": incidents,
        "minutes": round((time.monotonic() - started) / 60, 2),
    }
    journal.write_run(finished)
    print(json.dumps({k: finished[k] for k in ("bars_emitted", "llm_calls", "decisions", "incidents", "minutes")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
