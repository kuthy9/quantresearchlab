#!/usr/bin/env python3
"""Replay the rebuilt Brain minute by minute over one window.

Loads the cached observation points and the fitted mode library, runs the
forecaster over every bar, and reports whether the Brain rebuilt correctly:

* every clock published a well-formed ``MarketBeliefState``
* the live working set never exceeded three hypotheses
* probabilities and the residual summed to one on every clock
* all five lifecycle operations were exercised
* a second pass reproduced every ``revision_id`` exactly

None of that is evidence the forecast is *right*.  It is evidence the machinery
runs, is bounded, and is deterministic.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from brain.core.belief_updater import BeliefUpdaterConfig  # noqa: E402
from brain.core.hypothesis_pool import PoolConfig  # noqa: E402
from brain.core.hypothesis_proposer import (  # noqa: E402
    HypothesisProposer,
    ProposerConfig,
    load_hypothesis_protocol,
)
from brain.research.mode_discovery import load_library_payload  # noqa: E402
from contract.brain.forecast import MAX_LIVE_HYPOTHESES, LifecycleOperation  # noqa: E402

DEFAULT_ARTIFACTS = "outputs/hypothesis_modes"


def _protocol_fingerprint(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(artifacts: Path) -> tuple[pd.DatetimeIndex, np.ndarray, pd.DataFrame, dict]:
    stored = np.load(artifacts / "dataset.npz", allow_pickle=False)
    index = pd.DatetimeIndex(pd.to_datetime(stored["index"], utc=True), name="asof")
    prices = pd.DataFrame(
        stored["prices"], index=index, columns=["close", "high", "low", "atr"]
    )
    payload = json.loads((artifacts / "mode_library.json").read_text(encoding="utf-8"))
    return index, stored["features"], prices, payload


def _run(forecaster, index, features, prices) -> tuple[list, dict]:
    from brain.core.forecast import ForecastInput

    forecaster.reset()
    states = []
    counts: collections.Counter[str] = collections.Counter()
    for position, asof in enumerate(index):
        row = prices.iloc[position]
        # The context vector was already derived from the live snapshot by the
        # dataset builder, so the replay hands it over rather than re-deriving
        # it — but it still publishes through the ordinary forecast surface.
        state = forecaster.observe(
            ForecastInput(
                asof=asof,
                close=float(row["close"]),
                high=float(row["high"]),
                low=float(row["low"]),
                context=features[position],
                atr=float(row["atr"]),
            )
        )
        for record in state.lifecycle_records:
            counts[record.operation.value] += 1
        states.append(state)
    return states, dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--protocol", default="brain/configs/hypothesis_protocol.json")
    parser.add_argument(
        "--emit-end",
        default="2022-01-05T17:00",
        help="last clock to replay, exclusive, in exchange-local time; must "
             "match the window the library was fitted on",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    artifacts = ROOT / args.artifacts
    index, features, prices, payload = _load(artifacts)
    if args.emit_end:
        keep = index.tz_convert("America/New_York") < pd.Timestamp(
            args.emit_end, tz="America/New_York"
        )
        index, features, prices = index[keep], features[keep], prices[keep]
    library, assignments, centre, scale = load_library_payload(payload)
    protocol_path = ROOT / args.protocol
    protocol = load_hypothesis_protocol(protocol_path)

    proposer = HypothesisProposer(
        library=library,
        reference_features=features,
        reference_modes=assignments,
        center=centre,
        scale=scale,
        config=ProposerConfig.from_protocol(protocol),
    )

    from brain.core.forecast import HypothesisForecaster

    forecaster = HypothesisForecaster(
        proposer=proposer,
        library=library,
        protocol_fingerprint=_protocol_fingerprint(protocol_path),
        pool_config=PoolConfig.from_protocol(protocol),
        updater_config=BeliefUpdaterConfig.from_protocol(protocol),
    )

    print(f"replaying {len(index)} clocks {index.min()} -> {index.max()}")
    states, counts = _run(forecaster, index, features, prices)

    live = [len(state.hypotheses) for state in states]
    residual = [state.residual_probability for state in states]
    uncertainty = [state.uncertainty for state in states]
    sums = [
        sum(h.probability for h in state.hypotheses) + state.residual_probability
        for state in states
    ]
    worst_sum = max(abs(value - 1.0) for value in sums)

    print("\n=== working set ===")
    histogram = collections.Counter(live)
    for size in sorted(histogram):
        share = histogram[size] / len(states)
        print(f"  H_t = {size}: {histogram[size]:6d} clocks ({share:6.1%})")
    print(f"  mean live hypotheses {np.mean(live):.2f}, max {max(live)}")

    print("\n=== belief ===")
    print(f"  residual    mean {np.mean(residual):.3f}  min {min(residual):.3f}  max {max(residual):.3f}")
    print(f"  uncertainty mean {np.mean(uncertainty):.3f}  min {min(uncertainty):.3f}  max {max(uncertainty):.3f}")
    print(f"  worst |sum(p) + residual - 1| = {worst_sum:.3e}")

    print("\n=== lifecycle ===")
    for operation in LifecycleOperation:
        print(f"  {operation.value:8s} {counts.get(operation.value, 0):6d}")

    print("\n=== determinism ===")
    second, _ = _run(forecaster, index, features, prices)
    identical = all(a.revision_id == b.revision_id for a, b in zip(states, second))
    print(f"  second pass reproduced every revision_id: {identical}")

    checks = {
        "every_clock_published": len(states) == len(index),
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
                "asof": index,
                "live_hypotheses": live,
                "residual_probability": residual,
                "uncertainty": uncertainty,
                "leading_mode": [
                    state.leading.mode_id if state.leading else None for state in states
                ],
                "leading_probability": [
                    state.leading.probability if state.leading else 0.0 for state in states
                ],
                "revision_id": [state.revision_id for state in states],
            }
        ).set_index("asof")
        destination = ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination)
        print(f"wrote {destination}")

    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
