"""Which generator finds balance: the Structural Range, or an independent scan?

Two competing accounts of where a Balance Candidate comes from:

* **H1 - the Structural Range is already the right candidate.** A range forms,
  width and overlap develop inside it, bilateral interactions accumulate, and
  balance is observed. If this holds, ``P(balance | structural range)`` should
  beat the base rate of an arbitrary window.
* **H2 - balance must be found on its own terms.** A narrow, overlapping
  episode appears anywhere in the market, is nominated on its shape, and is
  then confirmed bilaterally. If this holds, an independent scan should beat
  both the base rate and the Structural Range.

The baseline is every rolling H1 window: 3.09% show a bilateral revisit, which
is the number any candidate generator has to beat to be worth having.

**Out-of-sample discipline.** The independent generator's shape thresholds are
chosen on 2022-01 and every reported number is measured on 2022-02. Choosing a
cut and scoring it on the same bars would make precision meaningless, and the
whole point of this comparison is that precision is meaningful.

**Ground truth.** A balance episode is a maximal run of H1 bars covered by
windows that reach two revisit generations on both of their own extremes, using
the band the balance claim uses. Overlapping windows collapse into one episode
so that a single stretch of two-sided trade counts once, and the episode is
anchored at the bar where its first window confirmed.

Nothing here changes any semantics, reads the event stream, or writes a
detector into the Eye. It is a measurement that decides whether a detector is
worth building.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from smc_trader.causal import CausalMarketReader  # noqa: E402
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.model import Timeframe  # noqa: E402
from smc_trader.scale_registry import parse_scale_specs  # noqa: E402
from scan_eye_event_statistics import DEFAULT_SOURCE  # noqa: E402
from study_balance_shape_h1_windows import (  # noqa: E402
    ATR_PERIOD,
    WINDOW_LENGTHS,
    h1_candles,
    revisit_generations,
    true_ranges,
    window_features,
)

BILATERAL_STANDARD = 2
# How long after a nomination the confirmation still counts as that
# nomination's. Beyond this the generator did not anticipate anything.
HORIZON_H1_BARS = 24


def wilson(successes: int, total: int) -> tuple[float, float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    d = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    margin = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    )
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


def scan_windows(candles, ranges):
    """Every rolling window with its features and its bilateral outcome."""

    rows = []
    for end in range(ATR_PERIOD, len(candles)):
        atr = statistics.fmean(ranges[end - ATR_PERIOD : end])
        for length in WINDOW_LENGTHS:
            start = end - length + 1
            if start < 0:
                continue
            features = window_features(candles[start : end + 1], atr)
            if features is None:
                continue
            rows.append(
                {
                    "start": start,
                    "end": end,
                    "length": length,
                    "atr": atr,
                    **features,
                }
            )
    return rows


def balance_episodes(rows) -> list[dict]:
    """Collapse overlapping bilateral windows into disjoint episodes."""

    hits = sorted(
        (r for r in rows if r["bilateral_revisits"] >= BILATERAL_STANDARD),
        key=lambda r: (r["end"], r["start"]),
    )
    episodes: list[dict] = []
    for row in hits:
        if episodes and row["start"] <= episodes[-1]["end"]:
            episode = episodes[-1]
            episode["end"] = max(episode["end"], row["end"])
            episode["start"] = min(episode["start"], row["start"])
            episode["windows"] += 1
            continue
        episodes.append(
            {
                "start": row["start"],
                "end": row["end"],
                # The bar at which this stretch first proved two-sided.
                "confirmed_at": row["end"],
                "windows": 1,
            }
        )
    return episodes


def fit_independent_cut(rows) -> dict[str, float]:
    """Pick shape thresholds on the fitting month only.

    The shape study found width (inverted) and overlap ratio carry the signal
    and that directional and path efficiency carry almost none, so the cut uses
    the two that do. Each threshold is the median of the *balanced* windows, so
    it is read off the positive class rather than tuned against a score.
    """

    balanced = [
        r for r in rows if r["bilateral_revisits"] >= BILATERAL_STANDARD
    ]
    if not balanced:
        raise SystemExit("fitting window contains no balance episodes")
    return {
        "max_width_atr": round(
            statistics.median(r["width_atr"] for r in balanced), 4
        ),
        "min_overlap_ratio": round(
            statistics.median(r["overlap_ratio"] for r in balanced), 4
        ),
    }


def independent_nominations(rows, cut) -> list[dict]:
    """Windows the independent generator nominates, on shape alone.

    Nomination reads only the window's own bars, never its outcome. A
    nomination is anchored at its last bar, which is the first moment the
    generator could have made it.
    """

    return [
        r
        for r in rows
        if r["width_atr"] <= cut["max_width_atr"]
        and r["overlap_ratio"] >= cut["min_overlap_ratio"]
    ]


def collapse_nominations(nominations) -> list[dict]:
    """One nomination per contiguous stretch, so stability is measurable."""

    ordered = sorted(nominations, key=lambda r: (r["end"], r["start"]))
    episodes: list[dict] = []
    for row in ordered:
        if episodes and row["end"] <= episodes[-1]["end"] + 1:
            episode = episodes[-1]
            episode["end"] = max(episode["end"], row["end"])
            episode["start"] = min(episode["start"], row["start"])
            episode["windows"] += 1
            continue
        episodes.append(
            {
                "start": row["start"],
                "end": row["end"],
                "nominated_at": row["end"],
                "windows": 1,
            }
        )
    return episodes


def score(nominations, episodes, *, horizon: int) -> dict:
    """Precision, recall and lead time for one generator."""

    if not nominations:
        return {
            "nominations": 0,
            "precision": None,
            "recall": {"hit": 0, "episodes": len(episodes), "rate": None},
            "lead_time_h1_bars": None,
        }
    matched_episodes = set()
    leads: list[int] = []
    hits = 0
    for nomination in nominations:
        anchor = nomination["nominated_at"]
        found = None
        for index, episode in enumerate(episodes):
            if anchor <= episode["confirmed_at"] <= anchor + horizon:
                found = index
                break
        if found is not None:
            hits += 1
            matched_episodes.add(found)
            leads.append(episodes[found]["confirmed_at"] - anchor)
    return {
        "nominations": len(nominations),
        "precision": {
            "hit": hits,
            "total": len(nominations),
            "rate": round(hits / len(nominations), 4),
            "wilson95": wilson(hits, len(nominations)),
        },
        "recall": {
            "hit": len(matched_episodes),
            "episodes": len(episodes),
            "rate": (
                round(len(matched_episodes) / len(episodes), 4)
                if episodes
                else None
            ),
            "wilson95": wilson(len(matched_episodes), len(episodes)),
        },
        "lead_time_h1_bars": (
            {
                "n": len(leads),
                "median": statistics.median(leads),
                "mean": round(statistics.fmean(leads), 2),
                "max": max(leads),
            }
            if leads
            else None
        ),
    }


def structural_range_arm(path: Path, candles) -> dict | None:
    """Score Group 4's Structural Ranges as a candidate generator.

    Read from the width-strata study's raw output rather than replayed here, so
    the arm is scored on exactly the ranges the Eye actually published.
    """

    if not path.exists():
        return None
    data = json.loads(path.read_text())
    ranges = data["ranges"]
    total = len(ranges)
    bilateral = sum(
        1
        for row in ranges.values()
        if min(row["lower"], row["upper"]) >= BILATERAL_STANDARD
    )
    lifetimes = [row["lifetime_h1_bars"] for row in ranges.values()]
    return {
        "candidates": total,
        "p_balance_given_candidate": {
            "hit": bilateral,
            "total": total,
            "rate": round(bilateral / total, 4) if total else None,
            "wilson95": wilson(bilateral, total),
        },
        "episode_stability": {
            "median_lifetime_h1_bars": (
                statistics.median(lifetimes) if lifetimes else None
            ),
            "mean_lifetime_h1_bars": (
                round(statistics.fmean(lifetimes), 2) if lifetimes else None
            ),
        },
        "note": (
            "Lead time is undefined for this arm: a Structural Range nominates "
            "itself at creation and no bilateral confirmation ever followed, so "
            "there is no interval to measure."
        ),
    }


def month(source: str, start: str, end: str, specs):
    candles = h1_candles(source, start, end, specs)
    ranges = true_ranges(candles)
    rows = scan_windows(candles, ranges)
    return candles, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--fit-start", default="2022-01-01")
    parser.add_argument("--fit-end", default="2022-02-01")
    parser.add_argument("--test-start", default="2022-02-01")
    parser.add_argument("--test-end", default="2022-03-01")
    parser.add_argument(
        "--structural-ranges",
        type=Path,
        default=ROOT / "docs/evidence/structural_range_width_strata_2022_02.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = json.loads(
        (ROOT / "configs/model.json").read_text(encoding="utf-8")
    )
    specs = parse_scale_specs(model["scales"])

    _, fit_rows = month(args.source, args.fit_start, args.fit_end, specs)
    test_candles, test_rows = month(
        args.source, args.test_start, args.test_end, specs
    )

    cut = fit_independent_cut(fit_rows)
    episodes = balance_episodes(test_rows)

    baseline_hits = sum(
        1 for r in test_rows if r["bilateral_revisits"] >= BILATERAL_STANDARD
    )
    baseline = {
        "windows": len(test_rows),
        "p_balance": {
            "hit": baseline_hits,
            "total": len(test_rows),
            "rate": round(baseline_hits / len(test_rows), 4),
            "wilson95": wilson(baseline_hits, len(test_rows)),
        },
    }

    nominations = collapse_nominations(
        independent_nominations(test_rows, cut)
    )
    # A nomination must not be scored against the confirmation it already
    # contains, or shape would be credited with reading the outcome.
    forward = [
        dict(row, nominated_at=row["nominated_at"])
        for row in nominations
    ]
    independent = score(forward, episodes, horizon=HORIZON_H1_BARS)
    raw_nominated = independent_nominations(test_rows, cut)
    independent["p_balance_given_nomination"] = {
        "hit": sum(
            1
            for r in raw_nominated
            if r["bilateral_revisits"] >= BILATERAL_STANDARD
        ),
        "total": len(raw_nominated),
        "rate": (
            round(
                sum(
                    1
                    for r in raw_nominated
                    if r["bilateral_revisits"] >= BILATERAL_STANDARD
                )
                / len(raw_nominated),
                4,
            )
            if raw_nominated
            else None
        ),
        "wilson95": wilson(
            sum(
                1
                for r in raw_nominated
                if r["bilateral_revisits"] >= BILATERAL_STANDARD
            ),
            len(raw_nominated),
        ),
    }
    independent["episode_stability"] = {
        "episodes": len(nominations),
        "median_windows_per_episode": (
            statistics.median(n["windows"] for n in nominations)
            if nominations
            else None
        ),
        "median_span_h1_bars": (
            statistics.median(n["end"] - n["start"] + 1 for n in nominations)
            if nominations
            else None
        ),
    }

    structural = structural_range_arm(args.structural_ranges, test_candles)

    # Episode-level precision is only meaningful against the precision a
    # generator would get by nominating at random. With 17 episodes and a
    # 24-bar horizon a large share of the month can sit inside some episode's
    # lead window, and a high hit rate would then say nothing about shape.
    hit_anchors = set()
    for episode in episodes:
        for anchor in range(
            episode["confirmed_at"] - HORIZON_H1_BARS,
            episode["confirmed_at"] + 1,
        ):
            if 0 <= anchor < len(test_candles):
                hit_anchors.add(anchor)
    chance = {
        "anchors_that_would_hit": len(hit_anchors),
        "anchors_available": len(test_candles),
        "random_precision": (
            round(len(hit_anchors) / len(test_candles), 4)
            if test_candles
            else None
        ),
    }

    payload = {
        "fit_window": {"start": args.fit_start, "end": args.fit_end},
        "test_window": {
            "start": args.test_start,
            "end": args.test_end,
            "h1_bars": len(test_candles),
            "windows_scanned": len(test_rows),
        },
        "independent_cut_fitted_on_fit_window": cut,
        "ground_truth_episodes": {
            "count": len(episodes),
            "median_span_h1_bars": (
                statistics.median(e["end"] - e["start"] + 1 for e in episodes)
                if episodes
                else None
            ),
        },
        "chance_precision_control": chance,
        "arms": {
            "baseline_any_window": baseline,
            "h1_structural_range": structural,
            "h2_independent_scan": independent,
        },
        "horizon_h1_bars": HORIZON_H1_BARS,
    }

    out = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(out, file=sys.stderr)
    if args.output:
        args.output.write_text(out)


if __name__ == "__main__":
    main()
