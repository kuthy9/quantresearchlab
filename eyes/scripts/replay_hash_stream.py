"""Per-bar hash stream of the Eye's published output.

One line per completed bar: the clock, ``content_hash`` of the whole
``MarketObservation``, ``content_hash`` of the events published on that bar,
and the event count. Two runs of the same code over the same bars produce the
same stream (the Eye is deterministic), so a change that leaves the stream
identical has not changed what the Eye publishes. This is the acceptance test
for every cost change in
brain/docs/specs/2026-09-11-information-gain-gate-design.md §4.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract.market import Bar, content_hash  # noqa: E402
from eyes.core.causal import CausalMarketReader  # noqa: E402
from eyes.core.observation import CausalObserver  # noqa: E402
from shares.core.io import iter_completed_bars, load_ohlcv  # noqa: E402

DEFAULT_SOURCE = "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"


def build_eye(model_path: Path, *, root: Path) -> tuple[CausalMarketReader, CausalObserver]:
    """The same construction ``shares/core/eye_factory.build_eye`` uses."""

    from shares.core.eye_factory import build_eye as _build_eye

    return _build_eye(model_path, root=root)


def hash_stream(
    bars: Iterable[Bar], *, model_path: Path, root: Path
) -> list[tuple[str, str, str, int]]:
    reader, observer = build_eye(model_path, root=root)
    rows: list[tuple[str, str, str, int]] = []
    for bar in bars:
        observation = observer.observe(reader.on_bar(bar))
        events = observation.semantic_events_this_update
        rows.append(
            (
                observation.asof.isoformat(),
                content_hash(observation),
                content_hash(events),
                len(events),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--model", default="configs/model.json")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many bars")
    parser.add_argument("--output", default=None, help="hash stream CSV; omit with --timing-only")
    parser.add_argument("--timing-every", type=int, default=500)
    parser.add_argument(
        "--timing-only", action="store_true",
        help="skip hashing: the timing lines then measure the Eye alone",
    )
    args = parser.parse_args()
    if args.output is None and not args.timing_only:
        parser.error("--output is required unless --timing-only")

    frame = load_ohlcv(ROOT / args.source, start=args.start, end=args.end).frame
    bars = list(iter_completed_bars(frame))
    if args.limit:
        bars = bars[: args.limit]
    reader, observer = build_eye(ROOT / args.model, root=ROOT)
    out = None if args.output is None else ROOT / args.output
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
    handle = out.open("w", newline="") if out is not None else None
    writer = csv.writer(handle) if handle is not None else None
    if writer is not None:
        writer.writerow(("asof", "observation_hash", "events_hash", "n_events"))
    started = time.monotonic()
    # The timing lines measure the Eye alone: hashing the observation is the
    # harness's cost, and it grows with the snapshot, so it is kept outside.
    eye_seconds = 0.0
    try:
        for number, bar in enumerate(bars, start=1):
            tick = time.perf_counter()
            observation = observer.observe(reader.on_bar(bar))
            eye_seconds += time.perf_counter() - tick
            if writer is not None:
                events = observation.semantic_events_this_update
                writer.writerow(
                    (
                        observation.asof.isoformat(),
                        content_hash(observation),
                        content_hash(events),
                        len(events),
                    )
                )
            if args.timing_every and number % args.timing_every == 0:
                print(
                    f"bars {number - args.timing_every:6d}-{number:6d}: {eye_seconds:6.1f}s "
                    f"({args.timing_every / eye_seconds:5.1f} bars/s)",
                    flush=True,
                )
                eye_seconds = 0.0
    finally:
        if handle is not None:
            handle.close()
    target = out if out is not None else "(timing only)"
    print(f"{len(bars)} bars -> {target}  ({(time.monotonic() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
