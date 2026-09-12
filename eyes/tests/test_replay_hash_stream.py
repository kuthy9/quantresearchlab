"""The hash-stream harness is the acceptance test of every Part 1 change."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from eyes.scripts.replay_hash_stream import hash_stream
from shares.tests.helpers import session_bars

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "configs" / "model.json"


def test_two_runs_over_the_same_bars_produce_the_same_stream() -> None:
    bars = session_bars(1)[:300]
    first = hash_stream(bars, model_path=MODEL, root=ROOT)
    second = hash_stream(bars, model_path=MODEL, root=ROOT)
    assert len(first) == 300
    assert first == second


def test_the_stream_changes_when_a_bar_changes() -> None:
    bars = session_bars(1)[:300]
    altered = list(bars)
    altered[150] = replace(altered[150], close=altered[150].close + 0.25)
    reference = hash_stream(bars, model_path=MODEL, root=ROOT)
    changed = hash_stream(altered, model_path=MODEL, root=ROOT)
    assert reference[:150] == changed[:150]
    assert reference[150][1] != changed[150][1]
