"""Run the LLM Brain over the benchmark windows, one ``run_llm_brain`` per window.

    .venv/bin/python -m brain.scripts.run_benchmark --client deepseek --broker sim \
        --reasoning-effort high --label entry-model --max-llm-calls 400 --parallel 3

The windows come from ``brain/configs/benchmark_windows.json`` (market
time); the Eye is warmed ``warmup_days`` before each ``emit_start``.
``--dry-run`` prints the commands and launches nothing; ``--only <text>``
keeps the windows whose name contains it; ``--parallel N`` runs N windows
at a time.  Each run's stdout and stderr go to
``<output-root>/benchmark_logs/<window name>.log``; a window whose run id
already exists is refused by ``run_llm_brain`` itself (exit 2), which this
runner reports and moves past.  Summaries are the summarizer's business:
``summarize_run --run-dir ... --write`` over the run directories."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from brain.scripts._run_identity import ROOT

DEFAULT_WINDOWS = "brain/configs/benchmark_windows.json"
DEFAULT_OUTPUT_ROOT = "outputs/brain_journal"
WINDOWS_SCHEMA_VERSION = 1


def load_windows(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != WINDOWS_SCHEMA_VERSION:
        raise ValueError("unsupported benchmark windows schema_version")
    if int(payload["warmup_days"]) < 1 or not payload["windows"]:
        raise ValueError("benchmark windows need warmup_days >= 1 and at least one window")
    for window in payload["windows"]:
        for key in ("name", "regime", "emit_start", "end"):
            if not isinstance(window.get(key), str) or not window[key]:
                raise ValueError(f"benchmark window {window!r} lacks {key}")
        if pd.Timestamp(window["end"]) <= pd.Timestamp(window["emit_start"]):
            raise ValueError(f"benchmark window {window['name']} ends before it starts")
    return payload


def warmup_start(emit_start: str, warmup_days: int) -> str:
    """The calendar day ``warmup_days`` before the window's first emitted bar."""
    return (pd.Timestamp(emit_start) - pd.Timedelta(days=int(warmup_days))).strftime("%Y-%m-%d")


def build_commands(
    windows: dict, *, client: str, broker: str, reasoning_effort: str | None, label: str | None, max_llm_calls: int | None,
    python: str = sys.executable, output_root: str = DEFAULT_OUTPUT_ROOT, only: str | None = None,
) -> list[tuple[str, list[str]]]:
    """``(window name, argv)`` per window, in the file's order."""
    commands: list[tuple[str, list[str]]] = []
    for window in windows["windows"]:
        if only is not None and only not in window["name"]:
            continue
        argv = [
            python, "-u", "-m", "brain.scripts.run_llm_brain", "--client", client, "--broker", broker,
            "--warmup-start", warmup_start(window["emit_start"], windows["warmup_days"]),
            "--emit-start", window["emit_start"], "--end", window["end"], "--output-root", output_root,
        ]
        if reasoning_effort is not None:
            argv += ["--reasoning-effort", reasoning_effort]
        if label is not None:
            argv += ["--label", label]
        if max_llm_calls is not None:
            argv += ["--max-llm-calls", str(max_llm_calls)]
        commands.append((window["name"], argv))
    return commands


def run_one(name: str, argv: list[str], *, log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{name}.log"
    with log.open("w", encoding="utf-8") as sink:
        sink.write(" ".join(argv) + "\n")
        sink.flush()
        completed = subprocess.run(argv, cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT, check=False)
    print(f"{name}: exit {completed.returncode} (log {log})", flush=True)
    return completed.returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--client", choices=("deepseek", "echo"), default="deepseek")
    parser.add_argument("--broker", choices=("none", "sim"), default="sim")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--only", default=None, help="run the windows whose name contains this text")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    windows = load_windows(ROOT / args.windows if not Path(args.windows).is_absolute() else Path(args.windows))
    commands = build_commands(
        windows, client=args.client, broker=args.broker, reasoning_effort=args.reasoning_effort, label=args.label,
        max_llm_calls=args.max_llm_calls, output_root=args.output_root, only=args.only,
    )
    if not commands:
        print("no window matches", file=sys.stderr)
        return 2
    if args.dry_run:
        for name, command in commands:
            print(f"{name}: {' '.join(command)}")
        return 0
    log_dir = ROOT / args.output_root / "benchmark_logs"
    with ThreadPoolExecutor(max_workers=max(1, int(args.parallel))) as pool:
        codes = list(pool.map(lambda item: run_one(item[0], item[1], log_dir=log_dir), commands))
    failed = [name for (name, _), code in zip(commands, codes) if code != 0]
    print(f"{len(commands) - len(failed)} of {len(commands)} windows finished; failed: {failed or 'none'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
