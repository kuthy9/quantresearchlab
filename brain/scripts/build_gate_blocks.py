"""Drive the Eye over Globex-week blocks and cache each block for the gate.

Each block is a fresh Eye warmed for the seven calendar days before the
week's Sunday 18:00 New York open and sampled through the week; the tape is
read two hours past the next open, which covers every sampled clock's
sixty-minute future. Blocks are independent, so ``--workers`` builds them in
parallel. The rule is the one ``configs/data_splits.json`` registers for its
fixed development windows (``warmup_calendar_days: 7``); see
brain/docs/specs/2026-09-11-information-gain-gate-design.md §5.2.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.research.event_log import save_block  # noqa: E402
from brain.research.trajectory_dataset import build_dataset  # noqa: E402

DEFAULT_SOURCE = "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
DEFAULT_MODEL = "configs/model.json"
BLOCK_RULE = "globex_week_warmup_7d"
_CLOCK = "%Y-%m-%dT%H:%M"


@dataclass(frozen=True)
class Block:
    week: str          # the Monday session date, used as the directory name
    warmup_start: str  # local time; fed to the Eye, never sampled
    emit_start: str    # the week's Sunday 18:00 open
    emit_end: str      # the next week's open; rows at or after it are dropped
    end: str           # where the tape read stops (emit_end plus two hours)


def _sunday_open(session: str) -> pd.Timestamp:
    day = pd.Timestamp(session).normalize()
    return day - pd.Timedelta(days=(day.weekday() + 1) % 7) + pd.Timedelta(hours=18)


def globex_weeks(first_session: str, last_session: str, *, warmup_days: int = 7) -> list[Block]:
    open_at = _sunday_open(first_session)
    last_open = _sunday_open(last_session)
    blocks: list[Block] = []
    while open_at <= last_open:
        next_open = open_at + pd.Timedelta(days=7)
        blocks.append(
            Block(
                week=(open_at + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                warmup_start=(open_at - pd.Timedelta(days=warmup_days)).strftime(_CLOCK),
                emit_start=open_at.strftime(_CLOCK),
                emit_end=next_open.strftime(_CLOCK),
                # Two hours past the next open: with a contiguous future
                # required, no clock after Friday 16:00 is sampled, so the
                # read stops as soon as the week's own bars are in.
                end=(next_open + pd.Timedelta(hours=2)).strftime(_CLOCK),
            )
        )
        open_at = next_open
    return blocks


def run_id(
    *, source: Path, model: Path, first_session: str, last_session: str, recorder: str | None = None,
) -> str:
    from eyes.core.semantics import load_semantic_selection

    payload = json.loads(model.read_text(encoding="utf-8"))
    selection = load_semantic_selection(payload["semantic_selection"], root=ROOT)
    digest_input = {
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "model": str(model.relative_to(ROOT)) if model.is_relative_to(ROOT) else str(model),
        "first_session": first_session,
        "last_session": last_session,
        "block_rule": BLOCK_RULE,
        "atomic_definition_identity": selection.atomic_definition_identity,
        "data_splits_sha256": hashlib.sha256(
            (ROOT / "configs" / "data_splits.json").read_bytes()
        ).hexdigest(),
    }
    if recorder is not None:
        digest_input["recorder"] = recorder
    return hashlib.sha256(json.dumps(digest_input, sort_keys=True).encode()).hexdigest()[:16]


def _build_one(job: tuple[Block, str, str, str, bool]) -> str:
    block, source, model, out_root, record_paths = job
    out = Path(out_root) / block.week
    cached = (out / "dataset.npz").exists() and (out / "events.parquet").exists()
    if record_paths:
        cached = cached and (out / "paths.parquet").exists()
    if cached:
        return f"{block.week}: cached"
    started = time.monotonic()
    dataset = build_dataset(
        source=Path(source),
        warmup_start=block.warmup_start,
        emit_start=block.emit_start,
        end=block.end,
        model_path=Path(model),
        root=ROOT,
        record_paths=record_paths,
    )
    save_block(dataset, out, emit_end=block.emit_end)
    return (
        f"{block.week}: {len(dataset.index)} clocks, {len(dataset.events)} events, "
        f"{len(dataset.paths)} path steps, {(time.monotonic() - started) / 60:.1f} min"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--first-session", default="2022-01-03")
    parser.add_argument("--last-session", default="2022-06-06")
    parser.add_argument("--output-root", default="outputs/information_gain_gate")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--record-paths", action="store_true",
        help="also log every new Group-5 path step (the Setup gate's unit); changes the run id",
    )
    args = parser.parse_args()
    recorder = "paths_v1" if args.record_paths else None

    source, model = ROOT / args.source, ROOT / args.model
    identity = run_id(
        source=source, model=model,
        first_session=args.first_session, last_session=args.last_session, recorder=recorder,
    )
    out_root = ROOT / args.output_root / identity / "blocks"
    out_root.mkdir(parents=True, exist_ok=True)
    blocks = globex_weeks(args.first_session, args.last_session)
    (out_root.parent / "run.json").write_text(
        json.dumps(
            {
                "run_id": identity,
                "first_session": args.first_session,
                "last_session": args.last_session,
                "source": str(source),
                "model": str(model),
                "block_rule": BLOCK_RULE,
                "recorder": recorder,
                "blocks": [asdict(block) for block in blocks],
            },
            indent=2,
        )
    )
    print(f"run {identity}: {len(blocks)} blocks -> {out_root}", flush=True)
    jobs = [(block, str(source), str(model), str(out_root), args.record_paths) for block in blocks]
    started = time.monotonic()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for line in pool.map(_build_one, jobs):
                print(line, flush=True)
    else:
        for job in jobs:
            print(_build_one(job), flush=True)
    print(f"done in {(time.monotonic() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
