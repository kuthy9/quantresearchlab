"""Does Structural Range width explain why the Balance claim is never reached?

The balance claim is currently a maturity state *of* a Structural Range, so a
range that never balances is recorded as a range that failed. Two months of
data are consistent with a different reading: that the two are separate market
objects, and that the intervals Group 4 selects are simply not the kind of
object a two-sided test can complete inside.

This script does not change any semantics. It replays a window, groups every
Structural Range by the width it froze at formation, and reports how far the
balance evidence got in each stratum. Nothing here tunes a threshold or writes
to the event stream.

The discriminating result is the shape across strata, not any single number:

* If P(bilateral) climbs steeply as ranges get narrower, width is the mechanism
  and the candidate filter is what makes balance unreachable.
* If P(bilateral) is near zero in *every* stratum, including the narrowest,
  then width is not the explanation and the parent -> maturity coupling is
  measuring something the object does not have.

Sample sizes are small, so every proportion is reported with its exact count
and a Wilson 95% interval rather than as a bare percentage.
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

from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.range_auction import (  # noqa: E402
    CausalRangeAuctionTracker,
)
from scan_eye_event_statistics import DEFAULT_SOURCE, build_eye  # noqa: E402


# Width is read from ``width_atr_at_formation``, which is frozen when the range
# is created, so the stratification itself carries no look-ahead.
STRATA = (
    ("<=2 ATR", 0.0, 2.0),
    ("2-3 ATR", 2.0, 3.0),
    ("3-4 ATR", 3.0, 4.0),
    ("4-5 ATR", 4.0, 5.0),
    (">5 ATR", 5.0, math.inf),
)
BILATERAL_STANDARD = 2


def stratum_of(width_atr: float) -> str:
    for name, low, high in STRATA:
        if low <= width_atr < high:
            return name
    return STRATA[-1][0]


def wilson(successes: int, total: int) -> tuple[float, float] | None:
    """95% Wilson score interval.

    Preferred over the normal approximation because these counts are small and
    frequently zero, where the normal interval collapses to a point and would
    read as certainty the data does not contain.
    """

    if total == 0:
        return None
    z = 1.959963984540054
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(phat * (1.0 - phat) / total + z * z / (4 * total * total))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "median": round(statistics.median(ordered), 4),
        "mean": round(statistics.fmean(ordered), 4),
        "max": round(ordered[-1], 4),
    }


def collect(start: str, end: str, source: str, model: str, limit: int):
    """Replay the window and keep each range's peak balance evidence."""

    ranges: dict[str, dict] = {}
    original = CausalRangeAuctionTracker.on_completed_update

    def traced(
        self,
        candle,
        *,
        prior_inventory,
        liquidity_pools,
        completed_h1=None,
        h1_support_resistance=(),
    ):
        output = original(
            self,
            candle,
            prior_inventory=prior_inventory,
            liquidity_pools=liquidity_pools,
            completed_h1=completed_h1,
            h1_support_resistance=h1_support_resistance,
        )
        for state in output.dealing_ranges:
            row = ranges.setdefault(
                state.range_id,
                {
                    "width_atr": float(state.width_atr_at_formation),
                    "width_points": float(
                        state.upper_bound - state.lower_bound
                    ),
                    "formation_atr": float(state.formation_atr),
                    "lower": 0,
                    "upper": 0,
                    "midpoint_crossings": 0,
                    "lifetime_h1_bars": 0,
                    "inside_close_fraction": None,
                    "compression_ratio": None,
                    "lifecycle": None,
                    "transition_reason": None,
                    "kinds": [],
                },
            )
            # Peak evidence over the range's life: the question is whether the
            # range ever got there, not where it happened to be at the end.
            row["lower"] = max(
                row["lower"], int(state.balance_lower_test_generations)
            )
            row["upper"] = max(
                row["upper"], int(state.balance_upper_test_generations)
            )
            row["midpoint_crossings"] = max(
                row["midpoint_crossings"], int(state.midpoint_crossings)
            )
            row["lifetime_h1_bars"] = max(
                row["lifetime_h1_bars"], int(state.candidate_real_h1_bars)
            )
            # These describe the range as it last stood.
            row["inside_close_fraction"] = float(state.inside_close_fraction)
            row["compression_ratio"] = float(state.compression_ratio)
            row["lifecycle"] = state.lifecycle.value
            row["transition_reason"] = state.transition_reason
            row["kinds"] = list(state.balance_lower_test_kinds) + list(
                state.balance_upper_test_kinds
            )
        return output

    CausalRangeAuctionTracker.on_completed_update = traced
    try:
        loaded = load_ohlcv(ROOT / source, start=start, end=end)
        reader, observer = build_eye(ROOT / model)
        bars = 0
        for bar in iter_completed_bars(loaded.frame):
            if limit and bars >= limit:
                break
            observer.observe(reader.on_bar(bar))
            bars += 1
    finally:
        CausalRangeAuctionTracker.on_completed_update = original
    return ranges, bars


def stratify(ranges: dict[str, dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in ranges.values():
        grouped[stratum_of(row["width_atr"])].append(row)

    report: dict[str, dict] = {}
    for name, _, _ in STRATA:
        rows = grouped.get(name, [])
        total = len(rows)
        lower_hits = sum(1 for r in rows if r["lower"] >= BILATERAL_STANDARD)
        upper_hits = sum(1 for r in rows if r["upper"] >= BILATERAL_STANDARD)
        bilateral = sum(
            1
            for r in rows
            if min(r["lower"], r["upper"]) >= BILATERAL_STANDARD
        )
        any_lower = sum(1 for r in rows if r["lower"] >= 1)
        any_upper = sum(1 for r in rows if r["upper"] >= 1)
        report[name] = {
            "ranges": total,
            "p_lower_ge_2": {
                "successes": lower_hits,
                "total": total,
                "wilson95": wilson(lower_hits, total),
            },
            "p_upper_ge_2": {
                "successes": upper_hits,
                "total": total,
                "wilson95": wilson(upper_hits, total),
            },
            "p_bilateral_2_2": {
                "successes": bilateral,
                "total": total,
                "wilson95": wilson(bilateral, total),
            },
            "p_lower_ge_1": {"successes": any_lower, "total": total},
            "p_upper_ge_1": {"successes": any_upper, "total": total},
            "midpoint_crossings": summarize(
                [float(r["midpoint_crossings"]) for r in rows]
            ),
            "inside_close_fraction": summarize(
                [
                    r["inside_close_fraction"]
                    for r in rows
                    if r["inside_close_fraction"] is not None
                ]
            ),
            "lifetime_h1_bars": summarize(
                [float(r["lifetime_h1_bars"]) for r in rows]
            ),
            "width_atr": summarize([r["width_atr"] for r in rows]),
            "width_points": summarize([r["width_points"] for r in rows]),
            "terminal_reasons": dict(
                collections.Counter(
                    r["transition_reason"] or "still_forming" for r in rows
                )
            ),
            "interaction_kinds": dict(
                collections.Counter(k for r in rows for k in r["kinds"])
            ),
        }
    return report


def render(report: dict[str, dict], window: dict) -> str:
    lines = [
        f"window {window['start']}..{window['end']}  "
        f"bars={window['bars_replayed']}  ranges={window['ranges']}",
        "",
        f"{'stratum':<10} {'n':>3} {'lo>=2':>7} {'up>=2':>7} {'both':>7} "
        f"{'lo>=1':>7} {'up>=1':>7} {'life':>6} {'cross':>6} {'inside':>7}",
    ]
    for name, _, _ in STRATA:
        row = report[name]
        n = row["ranges"]

        def frac(key: str) -> str:
            cell = row[key]
            if not cell["total"]:
                return "    -  "
            return f"{cell['successes']:>2}/{cell['total']:<2}"

        life = row["lifetime_h1_bars"]
        cross = row["midpoint_crossings"]
        inside = row["inside_close_fraction"]
        lines.append(
            f"{name:<10} {n:>3} {frac('p_lower_ge_2'):>7} "
            f"{frac('p_upper_ge_2'):>7} {frac('p_bilateral_2_2'):>7} "
            f"{frac('p_lower_ge_1'):>7} {frac('p_upper_ge_1'):>7} "
            f"{(life['median'] if life else 0):>6} "
            f"{(cross['median'] if cross else 0):>6} "
            f"{(inside['median'] if inside else 0):>7}"
        )
    lines.append("")
    lines.append("life/cross/inside are medians. Fractions are successes/n.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--start", default="2022-02-01")
    parser.add_argument("--end", default="2022-03-01")
    parser.add_argument("--model", default="configs/model.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ranges, bars = collect(
        args.start, args.end, args.source, args.model, args.limit
    )
    report = stratify(ranges)
    window = {
        "start": args.start,
        "end": args.end,
        "source": args.source,
        "bars_replayed": bars,
        "ranges": len(ranges),
        "bilateral_standard": BILATERAL_STANDARD,
    }
    payload = {
        "window": window,
        "strata": report,
        "ranges": ranges,
    }
    print(render(report, window), file=sys.stderr)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
