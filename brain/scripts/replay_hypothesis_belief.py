#!/usr/bin/env python3
"""Replay the Brain minute by minute and report whether it rebuilt correctly.

Runs the forecaster over every clock in a window and checks five things:

* every clock published a well-formed ``MarketBeliefState``
* the live working set never exceeded three hypotheses
* probabilities and the residual summed to one on every clock
* all five lifecycle operations were exercised
* a second pass reproduced every ``revision_id`` exactly

None of that is evidence the forecast is *right*. It is evidence the machinery
runs, is bounded, and is deterministic.

It also measures churn honestly. Every lifecycle event is recorded against the
clustering jitter and neighbourhood drift it happened under, so a real change of
claim can be told from a re-initialization artefact instead of both being
counted as "a split".
"""
from __future__ import annotations

import argparse
import collections
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from brain.core.belief_updater import BeliefUpdaterConfig  # noqa: E402
from brain.core.forecast import ForecastInput, HypothesisForecaster  # noqa: E402
from brain.core.hypothesis_pool import PoolConfig  # noqa: E402
from brain.core.hypothesis_proposer import (  # noqa: E402
    HypothesisProposer,
    ProposerConfig,
    load_hypothesis_protocol,
)
from brain.research.churn_diagnostics import (  # noqa: E402
    association_distance_profile,
    cloud_drift,
    cluster_jitter,
    summarize_churn,
)
from brain.research.forecast_index import load_index  # noqa: E402
from brain.scripts._windows import load_dataset, slice_window  # noqa: E402
from contract.brain.forecast import MAX_LIVE_HYPOTHESES, LifecycleOperation  # noqa: E402

DEFAULT_ARTIFACTS = "outputs/hypothesis_v2"


def _protocol_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(forecaster, window, *, diagnose_every: int = 0):
    """Replay one window; optionally sample clustering jitter as it goes."""

    forecaster.reset()
    states, jitter_rows = [], []
    previous_rows = None
    for position, asof in enumerate(window.index):
        row = window.prices[position]
        state = forecaster.observe(
            ForecastInput(
                asof=asof,
                close=float(row[0]),
                high=float(row[1]),
                low=float(row[2]),
                context=window.features[position],
                atr=float(row[3]),
            )
        )
        states.append(state)
        if diagnose_every and position % diagnose_every == 0:
            rows = forecaster.proposer.neighbourhood(window.features[position])
            scores = forecaster.proposer.index.reference_scores[rows]
            agreement, shift = cluster_jitter(
                scores, cluster_count=forecaster.proposer.config.cluster_count
            )
            jitter_rows.append(
                {
                    "asof": asof,
                    "jitter_ari": agreement,
                    "centroid_shift": shift,
                    "cloud_drift": (
                        cloud_drift(previous_rows, rows)
                        if previous_rows is not None
                        else 0.0
                    ),
                }
            )
            previous_rows = rows
    return states, pd.DataFrame(jitter_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--protocol", default="brain/configs/hypothesis_protocol.json")
    parser.add_argument("--start", default="2022-01-02T18:00")
    parser.add_argument("--end", default="2022-01-05T17:00")
    parser.add_argument("--label", default="fit")
    parser.add_argument(
        "--diagnose-every",
        type=int,
        default=25,
        help="sample clustering jitter every N clocks; 0 disables the diagnostic",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    artifacts = ROOT / args.artifacts
    data = load_dataset(artifacts / "dataset.npz")
    window = slice_window(data, name=args.label, start=args.start, end=args.end)
    index = load_index(artifacts / "forecast_index.npz")
    protocol_path = ROOT / args.protocol
    protocol = load_hypothesis_protocol(protocol_path)

    proposer = HypothesisProposer(
        index=index, config=ProposerConfig.from_protocol(protocol)
    )
    forecaster = HypothesisForecaster(
        proposer=proposer,
        protocol_fingerprint=_protocol_fingerprint(protocol_path),
        pool_config=PoolConfig.from_protocol(protocol),
        updater_config=BeliefUpdaterConfig.from_protocol(protocol),
    )

    print(window.describe())
    print(f"index fingerprint {index.fingerprint[:16]}…\n")
    states, jitter = _run(forecaster, window, diagnose_every=args.diagnose_every)

    live = [len(state.hypotheses) for state in states]
    residual = [state.residual_probability for state in states]
    sums = [
        sum(h.probability for h in state.hypotheses) + state.residual_probability
        for state in states
    ]
    worst_sum = max(abs(value - 1.0) for value in sums)
    counts: collections.Counter[str] = collections.Counter()
    churn_rows = []
    jitter_lookup = jitter.set_index("asof") if not jitter.empty else None
    for state in states:
        for record in state.lifecycle_records:
            counts[record.operation.value] += 1
            nearest = (
                jitter_lookup.index.asof(record.asof)
                if jitter_lookup is not None and len(jitter_lookup)
                else None
            )
            churn_rows.append(
                {
                    "operation": record.operation.value,
                    "association_distance": record.association_distance,
                    "jitter_ari": (
                        float(jitter_lookup.loc[nearest, "jitter_ari"])
                        if nearest is not None and nearest in jitter_lookup.index
                        else float("nan")
                    ),
                    "cloud_drift": (
                        float(jitter_lookup.loc[nearest, "cloud_drift"])
                        if nearest is not None and nearest in jitter_lookup.index
                        else float("nan")
                    ),
                }
            )

    print("=== working set ===")
    histogram = collections.Counter(live)
    for size in sorted(histogram):
        print(f"  H_t = {size}: {histogram[size]:6d} clocks ({histogram[size]/len(states):6.1%})")
    print(f"  mean live hypotheses {np.mean(live):.2f}, max {max(live)}")

    print("\n=== belief ===")
    print(f"  residual mean {np.mean(residual):.3f}  min {min(residual):.3f}  max {max(residual):.3f}")
    for name in ("entropy", "distribution_ambiguity", "coverage"):
        values = [getattr(state.uncertainty, name) for state in states]
        print(f"  {name:22s} mean {np.mean(values):.3f}  min {min(values):.3f}  max {max(values):.3f}")
    combined = [state.uncertainty.combined for state in states]
    print(f"  {'combined':22s} mean {np.mean(combined):.3f}  min {min(combined):.3f}  max {max(combined):.3f}")
    print(f"  worst |sum(p) + residual - 1| = {worst_sum:.3e}")

    print("\n=== lifecycle ===")
    for operation in LifecycleOperation:
        print(f"  {operation.value:8s} {counts.get(operation.value, 0):6d}")

    if churn_rows and jitter_lookup is not None and not jitter.empty:
        frame = pd.DataFrame(churn_rows).dropna(subset=["jitter_ari"])
        if not frame.empty:
            print("\n=== churn: real change or clustering jitter? ===")
            print(summarize_churn(frame).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
            print(
                "\n  artefact_suspicion is high when an operation fires while the "
                "cloud has barely moved but the clustering is unstable."
            )
        distances = [r["association_distance"] for r in churn_rows if r["association_distance"] > 0]
        if distances:
            print("\n=== where an association gate would sit ===")
            print(association_distance_profile(distances).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print("\n=== determinism ===")
    second, _ = _run(forecaster, window, diagnose_every=0)
    identical = all(a.revision_id == b.revision_id for a, b in zip(states, second))
    print(f"  second pass reproduced every revision_id: {identical}")

    checks = {
        "every_clock_published": len(states) == len(window),
        "working_set_bounded": max(live) <= MAX_LIVE_HYPOTHESES,
        "probabilities_sum_to_one": worst_sum <= 1e-9,
        "all_lifecycle_operations_exercised": all(
            counts.get(op.value, 0) > 0 for op in LifecycleOperation
        ),
        "deterministic": identical,
    }
    print("\n=== rebuild verdict ===")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n{'REBUILD SUCCEEDED' if all(checks.values()) else 'REBUILD INCOMPLETE'}")

    if args.output:
        frame = pd.DataFrame(
            {
                "asof": window.index,
                "live_hypotheses": live,
                "residual_probability": residual,
                "entropy": [s.uncertainty.entropy for s in states],
                "distribution_ambiguity": [s.uncertainty.distribution_ambiguity for s in states],
                "coverage": [s.uncertainty.coverage for s in states],
                "uncertainty": combined,
                "leading_node": [s.leading.node_id if s.leading else None for s in states],
                "leading_probability": [s.leading.probability if s.leading else 0.0 for s in states],
                "leading_r60": [s.leading.terminal_return if s.leading else float("nan") for s in states],
                "revision_id": [s.revision_id for s in states],
            }
        ).set_index("asof")
        destination = ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination)
        print(f"wrote {destination}")

    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
