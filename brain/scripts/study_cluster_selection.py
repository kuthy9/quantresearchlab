#!/usr/bin/env python3
"""Sweep the clustering resolution and check the shapes survive out of sample.

There is no natural number of clusters in a continuous conditional future
cloud: silhouette falls with k while explained variance rises with it, and the
two never agree. So this reports four things per (algorithm, k) and leaves the
trade-off visible, with centroid reproduction on an unseen window as the
criterion that matters most.

This is a study runner. Its output is ``shadow_only`` and grants no research,
empirical or trading authority.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from brain.core.trajectory import curve_matrix  # noqa: E402
from brain.research.design_study import path_attribute_rows  # noqa: E402
from brain.research.cluster_study import sweep_cluster_count  # noqa: E402
from brain.research.forecast_index import load_index  # noqa: E402
from brain.scripts._windows import load_dataset, slice_window  # noqa: E402

DEFAULT_ARTIFACTS = "outputs/hypothesis_v3"


def _curves(window) -> np.ndarray:
    return curve_matrix(
        anchor_prices=window.prices[:, 0],
        anchor_atrs=window.prices[:, 3],
        future_closes=window.future_closes,
    )


def _attributes(window) -> list:
    return path_attribute_rows(
        anchor_prices=window.prices[:, 0],
        anchor_atrs=window.prices[:, 3],
        future_closes=window.future_closes,
        future_highs=window.future_highs,
        future_lows=window.future_lows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--fit-start", default="2022-01-02T18:00")
    parser.add_argument("--fit-end", default="2022-01-14T17:00")
    parser.add_argument(
        "--holdout-start",
        default="2022-01-14T18:00",
        help=(
            "start of the unseen window, exchange-local. The default is the "
            "second half of January 2022, so the sweep is judged on sessions "
            "the basis was not fitted on."
        ),
    )
    parser.add_argument("--holdout-end", default="2022-01-28T17:00")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument(
        "--reproduction-gate",
        type=float,
        default=1.0,
        help="per-point RMS curve distance, in ATR units, within which two "
             "representative shapes count as the same shape",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    artifacts = ROOT / args.artifacts
    data = load_dataset(artifacts / "dataset.npz")
    index = load_index(artifacts / "forecast_index.npz")

    fit = slice_window(data, name="fit", start=args.fit_start, end=args.fit_end)
    holdout = slice_window(
        data, name="holdout", start=args.holdout_start, end=args.holdout_end
    )
    print(fit.describe())
    print(holdout.describe())
    print(f"index fingerprint {index.fingerprint[:16]}…\n")

    fit_curves = _curves(fit)
    holdout_curves = _curves(holdout)
    # Both windows are placed in the index's own two-channel space, so the
    # sweep scores the representation the runtime actually uses.
    fit_scores = index.represent(fit_curves, _attributes(fit))
    holdout_scores = index.represent(holdout_curves, _attributes(holdout))

    table = sweep_cluster_count(
        fit_scores=fit_scores,
        fit_curves=fit_curves,
        fit_r60=fit_curves[:, -1],
        holdout_scores=holdout_scores,
        holdout_curves=holdout_curves,
        holdout_r60=holdout_curves[:, -1],
        cluster_counts=tuple(range(args.k_min, args.k_max + 1)),
        reproduction_gate=args.reproduction_gate,
    )
    pd.set_option("display.width", 220)
    print(table.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    destination = ROOT / (args.output or f"{args.artifacts}/cluster_selection.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    print(f"\nwrote {destination}")

    best = table.dropna(subset=["reproduction_matched"]).sort_values(
        ["reproduction_matched", "reproduction_distance"], ascending=[False, True]
    )
    if not best.empty:
        print("\n=== best reproduction out of sample ===")
        print(
            best.head(8)[
                [
                    "algorithm", "k", "silhouette_disjoint", "eta2_r60",
                    "holdout_eta2_r60", "stability_ari",
                    "reproduction_matched", "reproduction_distance",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:8.3f}")
        )
        print(
            "\nReproduction is the share of fitted representative shapes that "
            "found a partner within the gate on the unseen window. A resolution "
            "whose shapes do not reappear is describing one window's noise."
        )


if __name__ == "__main__":
    main()
