"""The bar's ``known_at`` is the only clock a component may read.

The Eye, the controller, the Brain, the Risk gate and the executor are
driven bar by bar from the tape; a wall-clock read inside any of them would
let a replay differ from the run it replays, or let a backtest see the
present.  Only two places may touch the clock: the LLM client measures its
own latency, and the IBKR adapter waits on a live socket."""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CORE_DIRS = ("eyes/core", "brain/core", "execution/core", "risk/core", "shares/core")
WALL_CLOCK = re.compile(r"Timestamp\.now\(|datetime\.now\(|utcnow\(|\btime\.time\(|perf_counter\(|monotonic\(")
ALLOWED = {
    "brain/core/llm_client.py": "reply latency",
    "execution/core/ibkr_broker.py": "socket wait on a live session",
    "shares/core/timing.py": "durations for the timing report, never a timestamp",
}


def test_core_modules_never_read_the_wall_clock() -> None:
    offending = []
    for directory in CORE_DIRS:
        for path in sorted((ROOT / directory).glob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative in ALLOWED:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if WALL_CLOCK.search(line) and not line.lstrip().startswith("#"):
                    offending.append(f"{relative}:{number}: {line.strip()}")
    assert not offending, "\n".join(offending)


def test_the_allow_list_names_only_files_that_exist() -> None:
    for relative in ALLOWED:
        assert (ROOT / relative).exists(), relative
