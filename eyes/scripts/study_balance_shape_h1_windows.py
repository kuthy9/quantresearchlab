"""What does balance actually look like in H1 data?

Group 4's Structural Ranges never show a two-sided test, at any width. That
could mean two-sided rejection is rare in this market, or it could mean the
intervals Group 4 selects are the wrong place to look for it. The two are
distinguishable: measure the same evidence on *arbitrary* H1 windows, which no
candidate filter has touched.

This script builds no detector and applies no threshold. It rolls every window
length from 4 to 24 completed H1 bars across the data and reports the
distribution of continuous shape features, so the question "what does balance
look like" is answered before the question "what should trigger".

Features, all continuous and all computed from the window alone:

* ``directional_efficiency`` |C_end - C_start| / sum|dC| -- 0 is pure chop,
  1 is a straight line. The classic trend/chop separator.
* ``path_efficiency`` |C_end - C_start| / sum(high - low) -- net displacement
  per unit of range actually traversed.
* ``overlap_ratio`` mean over consecutive bar pairs of their range
  intersection divided by their union. High means each bar keeps trading where
  the last one did, which is what acceptance looks like.
* ``bilateral_revisits`` the *same* generation rule the balance claim uses,
  applied to the window's own extremes: a visit opens when price reaches within
  ``max(1 tick, 0.25 x ATR)`` of an extreme and closes when it leaves. This is
  the bridge measurement -- if arbitrary windows show bilateral revisits where
  Group 4 ranges do not, the candidate filter is the problem rather than the
  market.
* ``width_atr`` window high-low in ATR, for stratification only.

Nothing here reads or writes the event stream, and no semantics change.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eyes.core.causal import CausalMarketReader  # noqa: E402
from shares.core.io import iter_completed_bars, load_ohlcv  # noqa: E402
from contract.market import Timeframe  # noqa: E402
from shares.core.scale_registry import parse_scale_specs  # noqa: E402
from eyes.scripts.scan_eye_event_statistics import DEFAULT_SOURCE  # noqa: E402

WINDOW_LENGTHS = tuple(range(4, 25))
ATR_PERIOD = 14
# The same band the balance claim uses, so the two measurements are comparable.
BAND_ATR_FRACTION = 0.25
BAND_MINIMUM_TICKS = 1
TICK_SIZE = 0.25
PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


def h1_candles(source: str, start: str, end: str, scale_specs):
    """Aggregate completed H1 bars exactly the way the Eye does."""

    loaded = load_ohlcv(ROOT / source, start=start, end=end)
    reader = CausalMarketReader(scale_specs=scale_specs)
    out = []
    for bar in iter_completed_bars(loaded.frame):
        update = reader.on_bar(bar)
        out.extend(update.newly_completed.get(Timeframe.H1, ()))
    return out


def true_ranges(candles) -> list[float]:
    values = []
    previous_close = None
    for candle in candles:
        high, low = float(candle.high), float(candle.low)
        if previous_close is None:
            values.append(high - low)
        else:
            values.append(
                max(
                    high - low,
                    abs(high - previous_close),
                    abs(low - previous_close),
                )
            )
        previous_close = float(candle.close)
    return values


def revisit_generations(
    candles, extreme: float, band: float, *, upper: bool
) -> int:
    """Count distinct visits to one extreme, by the balance generation rule."""

    generations = 0
    inside = False
    for candle in candles:
        if upper:
            reached = float(candle.high) >= extreme - band
        else:
            reached = float(candle.low) <= extreme + band
        if reached and not inside:
            generations += 1
        inside = reached
    return generations


def window_features(candles, atr: float) -> dict[str, float] | None:
    closes = [float(c.close) for c in candles]
    highs = [float(c.high) for c in candles]
    lows = [float(c.low) for c in candles]
    steps = [abs(b - a) for a, b in zip(closes, closes[1:])]
    traversed = sum(h - l for h, l in zip(highs, lows))
    net = abs(closes[-1] - closes[0])
    total_step = sum(steps)
    if total_step <= 0.0 or traversed <= 0.0 or atr <= 0.0:
        return None

    overlaps = []
    for i in range(1, len(candles)):
        top = min(highs[i], highs[i - 1])
        bottom = max(lows[i], lows[i - 1])
        union_top = max(highs[i], highs[i - 1])
        union_bottom = min(lows[i], lows[i - 1])
        span = union_top - union_bottom
        overlaps.append(max(0.0, top - bottom) / span if span > 0 else 1.0)

    window_high, window_low = max(highs), min(lows)
    band = max(TICK_SIZE * BAND_MINIMUM_TICKS, BAND_ATR_FRACTION * atr)
    upper_visits = revisit_generations(candles, window_high, band, upper=True)
    lower_visits = revisit_generations(candles, window_low, band, upper=False)

    return {
        "directional_efficiency": net / total_step,
        "path_efficiency": net / traversed,
        "overlap_ratio": statistics.fmean(overlaps) if overlaps else 0.0,
        "upper_revisits": float(upper_visits),
        "lower_revisits": float(lower_visits),
        "bilateral_revisits": float(min(upper_visits, lower_visits)),
        "width_atr": (window_high - window_low) / atr,
    }


def auc(positive: list[float], negative: list[float]) -> float | None:
    """Probability a random positive scores above a random negative.

    Reported instead of a threshold: 0.5 means the feature carries no
    information about two-sided revisiting, and the distance from 0.5 in either
    direction is how much it carries.  Rank-based, so it needs no assumption
    about the shape of either distribution.
    """

    if not positive or not negative:
        return None
    merged = sorted(
        [(v, 1) for v in positive] + [(v, 0) for v in negative],
        key=lambda item: item[0],
    )
    rank_sum = 0.0
    index = 0
    while index < len(merged):
        stop = index
        while stop + 1 < len(merged) and merged[stop + 1][0] == merged[index][0]:
            stop += 1
        # Ties share the average rank, else equal values would order by label.
        average_rank = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            if merged[position][1] == 1:
                rank_sum += average_rank
        index = stop + 1
    n_pos, n_neg = len(positive), len(negative)
    return round(
        (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg), 4
    )


def percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    out = {"n": len(ordered)}
    for p in PERCENTILES:
        index = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
        out[f"p{p}"] = round(ordered[index], 4)
    out["mean"] = round(statistics.fmean(ordered), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2022-03-01")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = json.loads(
        (ROOT / "configs/model.json").read_text(encoding="utf-8")
    )
    specs = parse_scale_specs(model["scales"])
    candles = h1_candles(args.source, args.start, args.end, specs)
    ranges = true_ranges(candles)
    print(f"completed H1 bars: {len(candles)}", file=sys.stderr)

    by_length: dict[int, list[dict[str, float]]] = collections.defaultdict(list)
    for end in range(len(candles)):
        if end < ATR_PERIOD:
            continue
        atr = statistics.fmean(ranges[end - ATR_PERIOD : end])
        for length in WINDOW_LENGTHS:
            start = end - length + 1
            if start < 0:
                continue
            features = window_features(candles[start : end + 1], atr)
            if features is not None:
                by_length[length].append(features)

    names = (
        "directional_efficiency",
        "path_efficiency",
        "overlap_ratio",
        "upper_revisits",
        "lower_revisits",
        "bilateral_revisits",
        "width_atr",
    )
    report: dict[str, dict] = {}
    for length in WINDOW_LENGTHS:
        rows = by_length[length]
        if not rows:
            continue
        balanced = [r for r in rows if r["bilateral_revisits"] >= 2]
        rest = [r for r in rows if r["bilateral_revisits"] < 2]
        report[str(length)] = {
            "windows": len(rows),
            "p_bilateral_ge_2": {
                "successes": len(balanced),
                "total": len(rows),
                "rate": round(len(balanced) / len(rows), 4),
            },
            "features": {n: percentiles([r[n] for r in rows]) for n in names},
            # What the two-sided windows look like versus everything else.
            # This is the "what does balance look like" answer; a detector may
            # be built on it later, but nothing here thresholds anything.
            "features_bilateral": {
                n: percentiles([r[n] for r in balanced]) for n in names
            },
            "features_rest": {
                n: percentiles([r[n] for r in rest]) for n in names
            },
            "separability_auc": {
                n: auc([r[n] for r in balanced], [r[n] for r in rest])
                for n in names
            },
        }

    pooled = [r for rows in by_length.values() for r in rows]
    summary = {
        "window": {
            "start": args.start,
            "end": args.end,
            "source": args.source,
            "h1_bars": len(candles),
            "windows_scanned": len(pooled),
            "window_lengths": list(WINDOW_LENGTHS),
            "atr_period": ATR_PERIOD,
            "revisit_band": "max(1 tick, 0.25 * ATR)",
        },
        "pooled_features": {
            n: percentiles([r[n] for r in pooled]) for n in names
        },
        "pooled_p_bilateral_ge_2": {
            "successes": sum(1 for r in pooled if r["bilateral_revisits"] >= 2),
            "total": len(pooled),
        },
        "by_window_length": report,
    }

    header = (
        f"{'len':>4} {'windows':>8} {'bilat>=2':>9} {'dirEff p50':>11} "
        f"{'overlap p50':>12} {'pathEff p50':>12} {'widthATR p50':>13}"
    )
    print(header, file=sys.stderr)
    for length in WINDOW_LENGTHS:
        row = report.get(str(length))
        if not row:
            continue
        f = row["features"]
        print(
            f"{length:>4} {row['windows']:>8} "
            f"{row['p_bilateral_ge_2']['rate']:>9.3f} "
            f"{f['directional_efficiency']['p50']:>11} "
            f"{f['overlap_ratio']['p50']:>12} "
            f"{f['path_efficiency']['p50']:>12} "
            f"{f['width_atr']['p50']:>13}",
            file=sys.stderr,
        )

    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
