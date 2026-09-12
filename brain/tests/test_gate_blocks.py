"""Weekly blocks tile the sessions once, each warmed by the seven days before its open."""
from __future__ import annotations

import pandas as pd

from brain.scripts.build_gate_blocks import globex_weeks


def test_weeks_tile_the_window_without_gaps_or_overlap() -> None:
    blocks = globex_weeks("2022-01-03", "2022-06-06")
    assert blocks[0].week == "2022-01-03"
    assert blocks[-1].week == "2022-06-06"
    # a Globex week opens Sunday 18:00 New York; emit_start is that open.
    assert blocks[0].emit_start == "2022-01-02T18:00"
    assert blocks[0].warmup_start == "2021-12-26T18:00"
    for previous, current in zip(blocks, blocks[1:]):
        assert previous.emit_end == current.emit_start
    # Sundays 2022-01-02 .. 2022-06-05 inclusive
    assert len(blocks) == 23


def test_the_last_block_ends_after_the_last_session_plus_horizon() -> None:
    blocks = globex_weeks("2022-01-03", "2022-01-07")
    assert len(blocks) == 1
    block = blocks[0]
    assert pd.Timestamp(block.end) > pd.Timestamp("2022-01-07T17:00")
    assert block.emit_end == "2022-01-09T18:00"
    assert block.end == "2022-01-09T20:00"
