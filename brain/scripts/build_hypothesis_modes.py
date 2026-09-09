#!/usr/bin/env python3
"""Fit the Brain's trajectory-mode library from one OHLCV window.

Drives the Eye once, pairs every completed bar with the sixty minutes that
followed it, compares four clustering families on those trajectories, and
writes the fitted HDBSCAN + K-Medoids library.

The dataset is cached alongside the library so the replay runner never has to
re-run the Eye over the same window.

This is a study runner.  It reads the future by construction and its output is
``shadow_only``; it grants no research, empirical or trading authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from brain.research.mode_discovery import (  # noqa: E402
    DiscoveryConfig,
    block_stability,
    compare_algorithms,
    discover_modes,
    library_payload,
)
from brain.research.trajectory_dataset import build_dataset  # noqa: E402

DEFAULT_SOURCE = "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
DEFAULT_OUTPUT = "outputs/hypothesis_modes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--model", default="configs/model.json")
    parser.add_argument(
        "--warmup-start",
        default="2021-12-20",
        help=(
            "first bar fed to the Eye. Bars before --emit-start are never turned "
            "into observation points; they only warm the registered scales, the "
            "coarsest of which needs sixteen completed 4H bars."
        ),
    )
    parser.add_argument("--emit-start", default="2022-01-02T18:00")
    parser.add_argument(
        "--emit-end",
        default="2022-01-05T17:00",
        help=(
            "last observation point, exclusive, in exchange-local time. The "
            "default is the close of the third trading session on or after "
            "2022-01-01: sessions run 18:00 to 17:00 New York, so the three "
            "sessions 2022-01-03/04/05 span 01-02 18:00 to 01-05 17:00. Bars "
            "after this are still loaded, because the last observation points "
            "need sixty minutes of future."
        ),
    )
    parser.add_argument("--end", default="2022-01-07")
    parser.add_argument(
        "--algorithm",
        default="ward",
        choices=("ward", "kmeans", "hdbscan"),
        help=(
            "grouping method. ward is the default because HDBSCAN abstains on "
            "real trajectory vectors: they form one continuous cloud with no "
            "density gaps. See clustering_comparison.csv."
        ),
    )
    parser.add_argument("--n-modes", type=int, default=6)
    parser.add_argument("--min-cluster-size", type=int, default=25)
    parser.add_argument(
        "--decimation",
        type=int,
        default=60,
        help=(
            "stride of the non-overlapping comparison sample; consecutive "
            "observation points share 59 of their 60 future minutes"
        ),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--skip-comparison", action="store_true")
    args = parser.parse_args()

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    cache = out / "dataset.npz"
    if cache.exists():
        print(f"reusing cached dataset {cache}")
        stored = np.load(cache, allow_pickle=False)
        index = pd.DatetimeIndex(pd.to_datetime(stored["index"], utc=True), name="asof")
        features, trajectories = stored["features"], stored["trajectories"]
        prices = pd.DataFrame(
            stored["prices"], index=index, columns=["close", "high", "low", "atr"]
        )
    else:
        print(f"driving the Eye {args.warmup_start} -> {args.end} "
              f"(emitting from {args.emit_start})", flush=True)
        dataset = build_dataset(
            source=ROOT / args.source,
            warmup_start=args.warmup_start,
            emit_start=args.emit_start,
            end=args.end,
            model_path=ROOT / args.model,
            root=ROOT,
            progress_every=args.progress_every,
        )
        index, features, trajectories = dataset.index, dataset.features, dataset.trajectories
        prices = dataset.prices
        np.savez_compressed(
            cache,
            index=index.tz_convert("UTC").tz_localize(None).to_numpy(),
            features=features,
            trajectories=trajectories,
            prices=prices.to_numpy(dtype=float),
        )
        print(f"cached dataset -> {cache}")

    if args.emit_end:
        local = index.tz_convert("America/New_York")
        keep = local < pd.Timestamp(args.emit_end, tz="America/New_York")
        dropped = int((~keep).sum())
        index, features = index[keep], features[keep]
        trajectories, prices = trajectories[keep], prices[keep]
        if dropped:
            print(f"trimmed {dropped} observation points at or after {args.emit_end}")

    local = index.tz_convert("America/New_York")
    print(
        f"{len(index)} observation points "
        f"{local.min()} -> {local.max()} (New York)  "
        f"({(time.monotonic() - started) / 60:.1f} min)"
    )

    if not args.skip_comparison:
        print("\n=== clustering comparison ===")
        table = compare_algorithms(trajectories, decimation=args.decimation)
        pd.set_option("display.width", 200)
        print(table.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
        table.to_csv(out / "clustering_comparison.csv", index=False)

        print("\n=== block-subsample stability (mean ARI, disjoint sample) ===")
        disjoint = trajectories[:: max(1, args.decimation)]
        stability = {}
        for name, parameters in (
            ("kmeans", {"n_clusters": 4}),
            ("gmm", {"n_components": 4}),
            ("ward", {"n_clusters": 4}),
            ("hdbscan", {"min_cluster_size": 15}),
            ("hdbscan", {"min_cluster_size": 25}),
        ):
            score = block_stability(disjoint, algorithm=name, block=20, **parameters)
            key = f"{name}({','.join(f'{k}={v}' for k, v in parameters.items())})"
            stability[key] = score
            print(f"  {key:32s} ARI={score:.3f}")
        (out / "stability.json").write_text(json.dumps(stability, indent=2) + "\n")

    print(f"\n=== fitting the mode library ({args.algorithm} + K-Medoids) ===")
    result = discover_modes(
        trajectories=trajectories,
        features=features,
        fitted_at=pd.Timestamp.now(tz="UTC").floor("s"),
        config=DiscoveryConfig(
            algorithm=args.algorithm,
            n_modes=args.n_modes,
            min_cluster_size=args.min_cluster_size,
        ),
    )
    library = result.library
    print(f"library {library.library_id}  fingerprint {library.fingerprint[:16]}…")
    print(
        f"  {len(library.modes)} modes over {library.observation_count} observations, "
        f"{library.noise_count} left as noise "
        f"({library.noise_count / max(1, library.observation_count):.1%})"
    )
    leaves = [m for m in library.modes if not m.child_mode_ids]
    for mode in leaves:
        print(
            f"  {mode.mode_id:10s} n={mode.support:5d}  "
            f"r15={mode.component('r_15'):+6.2f} r60={mode.component('r_60'):+6.2f}  "
            f"mfe60={mode.component('mfe_60'):+5.2f} mae60={mode.component('mae_60'):+5.2f}"
        )

    payload = library_payload(result)
    artifact = out / "mode_library.json"
    artifact.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {artifact}")
    print(f"total {(time.monotonic() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
