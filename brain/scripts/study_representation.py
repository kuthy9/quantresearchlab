#!/usr/bin/env python3
"""Does the two-channel representation earn its place, and does retrieval work?

Two questions, answered on the same cached dataset:

**Raw PCA against Direction + detrended Shape.** Both are fitted on the fit
window and applied to the holdout, and both are judged on prototype geometry,
out-of-sample stability, how often the leading hypothesis flips, and how much of
the clustering is seed artefact.

**Retrieval skill.** For every holdout clock, the conditional future cloud
retrieved by ``X_t`` is scored against the path that actually followed, and the
same is done for a random cloud of the same size and for the unconditional mean
of the whole fit window. This is the test the design does not survive failing:
if the conditional cloud is no closer to what happened than random history, the
hypothesis lifecycle is decoration however well it behaves.

This is a study runner. It reads the future by construction and its output is
``shadow_only``; it grants no research, empirical or trading authority.
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
from brain.research.design_study import (  # noqa: E402
    path_attribute_rows,
    paired_verdict,
    skill_profile,
    prototype_geometry,
    raw_representation,
    representation_report,
    retrieval_skill,
    two_channel_representation,
)
from brain.research.forecast_index import standardize  # noqa: E402
from brain.scripts._windows import Window, load_dataset, slice_window  # noqa: E402

DEFAULT_DATASET = "outputs/hypothesis_v3/dataset.npz"


def _curves(window: Window) -> np.ndarray:
    return curve_matrix(
        anchor_prices=window.prices[:, 0],
        anchor_atrs=window.prices[:, 3],
        future_closes=window.future_closes,
    )


def _attributes(window: Window) -> list:
    return path_attribute_rows(
        anchor_prices=window.prices[:, 0],
        anchor_atrs=window.prices[:, 3],
        future_closes=window.future_closes,
        future_highs=window.future_highs,
        future_lows=window.future_lows,
    )


def _neighbourhoods(
    *, fit: Window, holdout: Window, neighbours: int, stride: int
) -> tuple[list[np.ndarray], np.ndarray]:
    """Retrieve each holdout clock's neighbours *from the fit window*.

    The holdout's own futures are never in the pool being searched, so nothing
    here can retrieve the answer it is about to be scored against.
    """

    reference, centre, scale = standardize(fit.features)
    query = np.nan_to_num(
        (holdout.features - centre) / scale, nan=0.0, posinf=0.0, neginf=0.0
    )
    rows = np.arange(0, holdout.features.shape[0], stride)
    take = min(neighbours, reference.shape[0])
    found = []
    for position in rows:
        distances = np.linalg.norm(reference - query[position], axis=1)
        head = np.argpartition(distances, take - 1)[:take]
        found.append(head[np.argsort(distances[head], kind="stable")])
    return found, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--fit-start", default="2022-01-02T18:00")
    parser.add_argument("--fit-end", default="2022-01-14T17:00")
    parser.add_argument("--holdout-start", default="2022-01-14T18:00")
    parser.add_argument("--holdout-end", default="2022-01-28T17:00")
    parser.add_argument("--neighbours", type=int, default=200)
    parser.add_argument("--cluster-count", type=int, default=4)
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help=(
            "clocks between retrievals. Adjacent observation points share 59 of "
            "their 60 future minutes, so scoring every one of them inflates the "
            "sample without adding information."
        ),
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    data = load_dataset(ROOT / args.dataset)
    fit = slice_window(data, name="fit", start=args.fit_start, end=args.fit_end)
    holdout = slice_window(
        data, name="holdout", start=args.holdout_start, end=args.holdout_end
    )
    print(fit.describe())
    print(holdout.describe())

    fit_curves, holdout_curves = _curves(fit), _curves(holdout)
    fit_attributes, holdout_attributes = _attributes(fit), _attributes(holdout)
    neighbourhoods, sampled = _neighbourhoods(
        fit=fit, holdout=holdout, neighbours=args.neighbours, stride=args.stride
    )
    print(
        f"\nretrieved {len(neighbourhoods)} holdout clocks "
        f"(every {args.stride}th) against {len(fit)} fit contexts"
    )

    print("\n=== 1. raw curve PCA against Direction + detrended Shape ===")
    rows = []
    for build in (raw_representation, two_channel_representation):
        representation = build(fit_curves, fit_attributes)
        holdout_scores = representation.apply(holdout_curves, holdout_attributes)
        rows.append(
            representation_report(
                representation=representation,
                fit_curves=fit_curves,
                holdout_scores=holdout_scores,
                holdout_curves=holdout_curves,
                neighbourhoods=neighbourhoods,
                cluster_count=args.cluster_count,
            )
        )
        print(f"\n  prototypes under {representation.name}:")
        print(
            prototype_geometry(
                representation.scores, fit_curves, cluster_count=args.cluster_count
            ).to_string(index=False, float_format=lambda v: f"{v:8.3f}")
        )
    comparison = pd.DataFrame(rows)
    print("\n  side by side:")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(
        "\n  leading_variance_share near one is the failure mode: one axis "
        "deciding every distance.\n"
        "  shape_span near zero means the prototypes differ only in where they "
        "ended.\n"
        "  oos_reproduction_distance is per-point ATR, lower is better; "
        "cluster_jitter_ari near one means the cut is a property of the cloud.\n"
        f"  leading_flip_rate is measured between sampled clocks {args.stride} "
        "minutes apart, not between adjacent bars, so it is comparable across "
        "the two representations but is not a per-minute rate."
    )

    print("\n=== 4. does the retrieved cloud beat random history? ===")
    skill = retrieval_skill(
        neighbourhoods=neighbourhoods,
        reference_curves=fit_curves,
        realized_curves=holdout_curves[sampled],
    )
    print(
        skill[["conditional_rmse", "random_rmse", "climatology_rmse"]]
        .describe()
        .loc[["mean", "50%", "std"]]
        .to_string(float_format=lambda v: f"{v:8.4f}")
    )
    print(
        f"\n  sign of the sixty-minute move: conditional "
        f"{skill['conditional_sign'].mean():.1%}  random "
        f"{skill['random_sign'].mean():.1%}"
    )
    against_random = paired_verdict(skill, left="conditional_rmse", right="random_rmse")
    against_climatology = paired_verdict(
        skill, left="conditional_rmse", right="climatology_rmse"
    )
    for name, verdict in (
        ("vs random cloud", against_random),
        ("vs climatology", against_climatology),
    ):
        print(
            f"  {name:18s} mean difference {verdict['mean_difference']:+.4f} ATR/point "
            f"[{verdict['ci_low']:+.4f}, {verdict['ci_high']:+.4f}]  "
            f"better on {verdict['share_improved']:.1%} of clocks  "
            f"{'SIGNIFICANT' if verdict['significant'] else 'not significant'}"
        )
    print(
        "\n  the interval is a moving-block bootstrap: consecutive observation "
        "points share 59 of 60 future minutes, so an i.i.d. interval would be "
        "far narrower than the data supports."
    )
    # RMSE alone cannot separate "no information" from "information stated too
    # boldly", and a single split cannot separate "the method does not work"
    # from "these two windows are different regimes". Both need answering
    # before the verdict above means anything.
    print("\n  the same question asked four more ways:")
    header = (
        f"  {'pool -> scored':34s} {'clocks':>6s} {'nbr_dist':>9s} {'scale':>6s} "
        f"{'corr':>7s} {'sign':>6s} {'alpha':>7s} {'cond':>6s} {'clim':>6s}"
    )
    print(header)
    trials = (
        ("fit -> holdout", fit, holdout),
        ("holdout -> fit (reversed)", holdout, fit),
        ("fit -> fit (in-sample, leaks)", fit, fit),
    )
    for name, pool, scored in trials:
        profile = skill_profile(
            pool_features=pool.features,
            pool_curves=_curves(pool),
            scored_features=scored.features,
            scored_curves=_curves(scored),
            neighbours=args.neighbours,
            stride=args.stride,
        )
        print(
            f"  {name:34s} {profile['clocks']:6d} {profile['neighbour_distance']:9.2f} "
            f"{profile['context_scale']:6.2f} {profile['correlation']:+7.3f} "
            f"{profile['sign_agreement']:6.1%} {profile['optimal_alpha']:+7.3f} "
            f"{profile['conditional_rmse']:6.3f} {profile['climatology_rmse']:6.3f}"
        )
    print(
        "\n  alpha is the least-squares optimal scaling of the conditional mean: "
        "at or below zero, the best use of the prediction is to ignore it.\n"
        "  The in-sample row leaks by construction — adjacent observation points "
        "share 59 of their 60 future minutes — so it is the shape of the gap "
        "between it and the others that carries the information, not its level."
    )

    verdict = (
        "RETRIEVAL HAS OUT-OF-SAMPLE SKILL"
        if against_random["significant"]
        else "NO OUT-OF-SAMPLE SKILL — the lifecycle has no trading meaning yet"
    )
    print(f"\n{verdict}")

    if args.output:
        destination = ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(destination.with_suffix(".representation.csv"), index=False)
        skill.to_csv(destination.with_suffix(".retrieval.csv"), index=False)
        print(f"wrote {destination.with_suffix('.representation.csv')}")
        print(f"wrote {destination.with_suffix('.retrieval.csv')}")

    raise SystemExit(0 if against_random["significant"] else 1)


if __name__ == "__main__":
    main()
