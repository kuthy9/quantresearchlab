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


def test_recording_paths_changes_the_run_id_and_nothing_else_does() -> None:
    from pathlib import Path

    from brain.scripts.build_gate_blocks import run_id

    root = Path(__file__).resolve().parents[2]
    source = root / "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
    model = root / "configs/model.json"
    plain = run_id(source=source, model=model, first_session="2022-01-03", last_session="2022-06-06")
    again = run_id(source=source, model=model, first_session="2022-01-03", last_session="2022-06-06", recorder=None)
    paths = run_id(source=source, model=model, first_session="2022-01-03", last_session="2022-06-06", recorder="paths_v3")
    assert plain == again and plain != paths and len(paths) == 16


def test_the_eye_revision_is_recorded_beside_the_run_id() -> None:
    # The run id digests the atomic definition identity, not the Eye's
    # code; run.json names the revision the blocks were built under so a
    # receipt can say which Eye it measured.
    from brain.scripts.build_gate_blocks import eye_revision

    revision = eye_revision()
    assert revision is None or (isinstance(revision, str) and len(revision) >= 7)


def test_the_eye_journals_where_the_caller_says(tmp_path) -> None:
    # The model's observer section names a shared journal directory the
    # runtime spills cold events into and never empties. A block build is a
    # bounded pass that keeps nothing of the Eye afterwards, so the builder
    # gives each block its own journal and removes it with the block's Eye.
    from pathlib import Path

    from brain.research.trajectory_dataset import build_eye

    root = Path(__file__).resolve().parents[2]
    _, shared = build_eye(root / "configs/model.json", root=root)
    assert shared.config.audit_journal_dir == str(root / "outputs/eye_journal")
    _, own = build_eye(root / "configs/model.json", root=root, audit_journal_dir=tmp_path / "journal")
    assert own.config.audit_journal_dir == str(tmp_path / "journal")
