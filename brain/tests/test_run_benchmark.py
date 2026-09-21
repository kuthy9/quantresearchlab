"""The benchmark runner: one ``run_llm_brain`` command per window, the Eye
warmed ``warmup_days`` before it, nothing launched on a dry run."""
from __future__ import annotations

import sys
from pathlib import Path

from brain.scripts.run_benchmark import build_commands, load_windows, main

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = ROOT / "brain" / "configs" / "benchmark_windows.json"


def test_the_windows_file_builds_one_command_per_window_with_the_warmup() -> None:
    windows = load_windows(WINDOWS)
    assert len(windows["windows"]) == 10 and windows["warmup_days"] == 7
    assert [w["name"][:10] for w in windows["windows"]] == sorted(w["name"][:10] for w in windows["windows"])
    commands = build_commands(windows, client="deepseek", broker="sim", reasoning_effort="high", label="entry-model", max_llm_calls=400)
    assert len(commands) == 10
    name, first = commands[0]
    assert name == "2022-01-24-extreme-reversal"
    assert first[:4] == [sys.executable, "-u", "-m", "brain.scripts.run_llm_brain"]
    assert first[first.index("--warmup-start") + 1] == "2022-01-17" and first[first.index("--emit-start") + 1] == "2022-01-24T12:00"
    assert first[first.index("--end") + 1] == "2022-01-24T16:00" and first[first.index("--label") + 1] == "entry-model"
    assert first[first.index("--max-llm-calls") + 1] == "400" and "--reasoning-effort" in first and "--broker" in first
    without_label = build_commands(windows, client="echo", broker="none", reasoning_effort=None, label=None, max_llm_calls=None)
    assert "--label" not in without_label[0][1] and "--reasoning-effort" not in without_label[0][1] and "--max-llm-calls" not in without_label[0][1]


def test_a_dry_run_prints_the_commands_and_launches_nothing(capsys) -> None:
    assert main(["--windows", str(WINDOWS), "--client", "echo", "--broker", "none", "--dry-run", "--only", "2022-05-10"]) == 0
    out = capsys.readouterr().out
    assert out.count("run_llm_brain") == 1 and "2022-05-03" in out and "2022-05-10T09:30" in out
