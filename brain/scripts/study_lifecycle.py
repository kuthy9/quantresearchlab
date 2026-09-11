#!/usr/bin/env python3
"""Is a surviving hypothesis the same hypothesis, and does the local cut adapt?

Two questions about how the pool maintains its working set, both answered by
replaying the published Brain over a window it was not fitted on.

**Support inheritance.** Every node records the historical observation points
that support it. Between consecutive clocks those sets are compared, so a claim
that kept its coordinates while its evidence was replaced can be told apart from
one that genuinely persisted. Centroid distance reports these two identically,
which is why the pool no longer decides identity on distance alone.

**The local cut.** ``k_t`` is chosen per clock from the cloud's own separation.
This reports the distribution it actually took, how it maps onto the number of
published hypotheses, and how the adaptive rule scores against every fixed k it
could have been pinned to.

This is a study runner. It reads the future by construction and its output is
``shadow_only``; it grants no research, empirical or trading authority.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from brain.core.forecast import ForecastInput, HypothesisForecaster  # noqa: E402
from brain.core.hypothesis_pool import PoolConfig  # noqa: E402
from brain.core.hypothesis_proposer import (  # noqa: E402
    HypothesisProposer,
    ProposerConfig,
    load_hypothesis_protocol,
    protocol_fingerprint,
)
from brain.research.design_study import (  # noqa: E402
    fixed_versus_adaptive,
    local_cut_profile,
)
from brain.research.forecast_index import load_index  # noqa: E402
from brain.scripts._windows import load_dataset, slice_window  # noqa: E402
from contract.brain.forecast import LifecycleOperation, support_overlap  # noqa: E402

DEFAULT_INDEX = "outputs/hypothesis_v3/forecast_index.npz"
DEFAULT_DATASET = "outputs/hypothesis_v3/dataset.npz"
PROTOCOL = "brain/configs/hypothesis_protocol.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--start", default="2022-01-14T18:00")
    parser.add_argument("--end", default="2022-01-28T17:00")
    parser.add_argument("--limit", type=int, default=0, help="0 replays the window")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    index = load_index(ROOT / args.index)
    protocol = load_hypothesis_protocol(ROOT / args.protocol)
    proposer = HypothesisProposer(
        index=index, config=ProposerConfig.from_protocol(protocol)
    )
    forecaster = HypothesisForecaster(
        proposer=proposer,
        protocol_fingerprint=protocol_fingerprint(ROOT / args.protocol),
        pool_config=PoolConfig.from_protocol(protocol),
    )

    data = load_dataset(ROOT / args.dataset)
    window = slice_window(data, name="replay", start=args.start, end=args.end)
    print(window.describe())
    rows = len(window) if args.limit <= 0 else min(args.limit, len(window))

    clouds = []
    per_clock = []
    previous: dict[str, tuple[int, ...]] = {}
    for position in range(rows):
        state = forecaster.observe(
            ForecastInput(
                asof=window.index[position],
                close=float(window.prices[position, 0]),
                high=float(window.prices[position, 1]),
                low=float(window.prices[position, 2]),
                context=window.features[position],
                atr=float(window.prices[position, 3]),
            )
        )
        cloud = state.cloud
        clouds.append(cloud)
        by_node = {node.node_id: node for node in cloud.nodes}
        current: dict[str, tuple[int, ...]] = {}
        carried = []
        for item in state.hypotheses:
            node = by_node.get(item.node_id)
            if node is None:
                continue
            current[item.hypothesis_id] = node.member_ids
            if item.hypothesis_id in previous:
                carried.append(
                    support_overlap(previous[item.hypothesis_id], node.member_ids)
                )
        per_clock.append(
            {
                "asof": window.index[position],
                "live": len(state.hypotheses),
                "k_t": cloud.cluster_count,
                "carried_overlap": float(np.mean(carried)) if carried else float("nan"),
                "survivors": len(carried),
                "mode_ambiguity": state.uncertainty.mode_ambiguity,
                "representation_coverage": state.uncertainty.representation_coverage,
                "retrieval_confidence": state.uncertainty.retrieval_confidence,
                "operations": tuple(
                    record.operation.value for record in state.lifecycle_records
                ),
            }
        )
        previous = current

    frame = pd.DataFrame(per_clock).set_index("asof")

    print("\n=== 2. is a surviving hypothesis the same hypothesis? ===")
    carried = frame["carried_overlap"].dropna()
    if carried.empty:
        print("  no hypothesis survived a clock in this window")
    else:
        print(
            f"  survivors scored: {len(carried)} clocks\n"
            f"  weighted support overlap  mean {carried.mean():.3f}  "
            f"median {carried.median():.3f}  "
            f"p10 {carried.quantile(0.10):.3f}  p90 {carried.quantile(0.90):.3f}"
        )
        for threshold in (0.25, 0.5, 0.75):
            print(
                f"  share of survivals below {threshold:.2f} overlap: "
                f"{(carried < threshold).mean():.1%}"
            )
        print(
            "\n  a survival with low overlap is a claim whose geometry persisted "
            "while its evidence was replaced. Under the v3 pool those are no "
            "longer published as updates; the count above is what the geometry-"
            "only rule would have mislabelled."
        )

    operations = collections.Counter(
        operation for row in per_clock for operation in row["operations"]
    )
    print("\n  lifecycle over the same window:")
    for operation in LifecycleOperation:
        print(f"    {operation.value:8s} {operations.get(operation.value, 0):6d}")

    print("\n=== 3. fixed cut against the cloud's own choice ===")
    profile = local_cut_profile(clouds)
    counts = profile["k_t"].value_counts().sort_index()
    for value, count in counts.items():
        print(f"  k_t = {value}: {count:6d} clocks ({count/len(profile):6.1%})")
    print(
        f"\n  mean k_t {profile['k_t'].mean():.2f}, "
        f"mean published hypotheses {frame['live'].mean():.2f}"
    )
    print("\n  local k against the working set it produced:")
    print(
        pd.crosstab(profile["k_t"], frame["live"].to_numpy())
        .rename_axis(index="k_t", columns="H_t")
        .to_string()
    )

    sampled = [
        proposer.neighbourhood(window.features[position])
        for position in range(0, rows, max(1, rows // 200))
    ]
    scoreboard = fixed_versus_adaptive(
        neighbourhoods=sampled, scores=index.reference_scores, proposer=proposer
    )
    print("\n  separation achieved, adaptive against every fixed alternative:")
    print(
        scoreboard.groupby("policy")["separation"]
        .agg(["mean", "median", "count"])
        .to_string(float_format=lambda v: f"{v:8.4f}")
    )
    adaptive = scoreboard.loc[scoreboard["policy"] == "adaptive", "separation"].mean()
    best_fixed = (
        scoreboard.loc[scoreboard["policy"] != "adaptive"]
        .groupby("policy")["separation"]
        .mean()
        .max()
    )
    print(
        f"\n  adaptive {adaptive:.4f} against the best fixed {best_fixed:.4f} — "
        f"{'the cloud chooses better' if adaptive >= best_fixed else 'a constant would have done as well'}"
    )
    print(
        "  k_t = 1 clocks are the ones a fixed cut would have split anyway; a "
        "silhouette below the floor means the cloud is one mode, not several."
    )

    if args.output:
        destination = ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.drop(columns=["operations"]).to_parquet(destination)
        print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
