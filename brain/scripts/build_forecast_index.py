#!/usr/bin/env python3
"""Build the Brain's retrieval index from one OHLCV window.

Drives the Eye once, pairs every completed bar with the sixty minutes that
followed it, caches that dataset, and fits the retrieval index: the standardized
contexts, the realized future curves, and the principal basis the detrended
shapes are compared in.

The dataset cache keeps the raw future window, not just derived features, so a
change to the trajectory representation never costs another Eye run — which is
almost all of the wall-clock cost of a study.

This is a study runner. It reads the future by construction and its output is
``shadow_only``; it grants no research, empirical or trading authority.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from brain.research.cluster_study import basis_report  # noqa: E402
from brain.research.forecast_index import build_index, save_index  # noqa: E402
from brain.research.trajectory_dataset import build_dataset  # noqa: E402
from brain.scripts._windows import load_dataset, slice_window  # noqa: E402

DEFAULT_SOURCE = "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
DEFAULT_OUTPUT = "outputs/hypothesis_v3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--model", default="configs/model.json")
    parser.add_argument(
        "--warmup-start",
        default="2021-12-20",
        help=(
            "first bar fed to the Eye. Bars before --emit-start never become "
            "observation points; they only warm the registered scales, the "
            "coarsest of which needs sixteen completed 4H bars."
        ),
    )
    parser.add_argument("--emit-start", default="2022-01-02T18:00")
    parser.add_argument("--emit-end", default="2022-01-11T17:00")
    parser.add_argument("--end", default="2022-01-12")
    parser.add_argument(
        "--fit-start",
        default="2022-01-02T18:00",
        help="start of the window the index is fitted on, exchange-local",
    )
    parser.add_argument("--fit-end", default="2022-01-05T17:00")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-every", type=int, default=4000)
    args = parser.parse_args()

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cache = out / "dataset.npz"

    if not cache.exists():
        print(
            f"driving the Eye {args.warmup_start} -> {args.end} "
            f"(emitting {args.emit_start} -> {args.emit_end})",
            flush=True,
        )
        dataset = build_dataset(
            source=ROOT / args.source,
            warmup_start=args.warmup_start,
            emit_start=args.emit_start,
            end=args.end,
            model_path=ROOT / args.model,
            root=ROOT,
            progress_every=args.progress_every,
        )
        local = dataset.index.tz_convert("America/New_York")
        keep = local < pd.Timestamp(args.emit_end, tz="America/New_York")
        np.savez_compressed(
            cache,
            index=dataset.index[keep].tz_convert("UTC").tz_localize(None).to_numpy(),
            features=dataset.features[keep],
            prices=dataset.prices.to_numpy(dtype=float)[keep],
            future_closes=dataset.future_closes[keep],
            future_highs=dataset.future_highs[keep],
            future_lows=dataset.future_lows[keep],
        )
        print(f"cached dataset -> {cache}  ({(time.monotonic()-started)/60:.1f} min)")
    else:
        print(f"reusing cached dataset {cache}")

    data = load_dataset(cache)
    window = slice_window(data, name="fit", start=args.fit_start, end=args.fit_end)
    print(window.describe())

    index, basis = build_index(
        features=window.features,
        anchor_prices=window.prices[:, 0],
        anchor_atrs=window.prices[:, 3],
        future_closes=window.future_closes,
        future_highs=window.future_highs,
        future_lows=window.future_lows,
    )
    print(f"\nindex fingerprint {index.fingerprint[:16]}…  rows {len(index)}")
    print("\n=== principal basis of the detrended shapes ===")
    print(basis_report(basis).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print(
        f"\ncomponent scale {index.component_scale:.4f} "
        f"(typical distance between two unrelated futures)"
    )
    print(
        f"context scale   {index.context_scale:.4f} "
        f"(typical distance between two unrelated contexts)"
    )

    destination = out / "forecast_index.npz"
    save_index(index, basis, destination)
    print(f"wrote {destination}")
    print(f"total {(time.monotonic() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
