#!/usr/bin/env python3
"""Forward outcomes of the Eye's sparse events — the pre-registered edge study.

Reads the tables ``eyes/scripts/scan_sparse_events.py`` wrote, rebuilds the
1m clock of the same tape, and measures every event of the families in
``eyes/docs/plans/2026-09-22-event-edge-2023.md`` §2 at 15 / 30 / 60 / 120 /
240 minutes (signed return, MFE, MAE, two 15m-ATR brackets), against a
baseline of every completed 15m bar matched by New York hour and direction.
Standard errors are clustered by trade date; the candidate-edge rule is
``candidate_edge`` (§5).  Writes ``study/`` beside the scan's tables.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import glob
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MARKET_TIMEZONE = "America/New_York"
HORIZONS = (15, 30, 60, 120, 240)
BRACKETS = ((1, 1), (2, 1))
BRACKET_HORIZON = 240
PRIMARY_HORIZON = 60
TIMEFRAMES = ("5m", "15m", "1H", "4H")
NANOS_PER_MINUTE = 60_000_000_000


# --------------------------------------------------------------------------- pure pieces


@dataclass(frozen=True)
class Tape:
    """The 1m clock: bar-end minutes since the epoch and the bars' prices."""

    minutes: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    def index_of(self, at: pd.Timestamp) -> int | None:
        minute = int(at.value // NANOS_PER_MINUTE)
        i = int(np.searchsorted(self.minutes, minute))
        return i if i < len(self.minutes) and self.minutes[i] == minute else None


def _contiguous_end(tape: Tape, i0: int, span: int) -> int:
    """The last index j ≤ i0 + span such that bars i0..j sit on consecutive minutes."""

    last = min(len(tape.minutes) - 1, i0 + span)
    if last <= i0:
        return i0
    steps = np.diff(tape.minutes[i0:last + 1])
    broken = np.nonzero(steps != 1)[0]
    return i0 + (int(broken[0]) if len(broken) else last - i0)


def outcomes(
    tape: Tape, i0: int, direction: int, *, unit: float, horizons=HORIZONS, brackets=BRACKETS,
    bracket_horizon: int = BRACKET_HORIZON,
) -> dict[str, Any]:
    """Signed return, MFE and MAE (points) per horizon, and first-touch brackets in ``unit``s.

    A horizon whose window misses a minute is ``None``; a bracket that meets a
    gap or its horizon unresolved is ``"none"``; a bar touching both the target
    and the stop is the stop."""

    entry = float(tape.close[i0])
    reach = _contiguous_end(tape, i0, max([*horizons, bracket_horizon]) if (horizons or brackets) else 0)
    out: dict[str, Any] = {}
    for h in horizons:
        if i0 + h > reach:
            out[f"ret_{h}"] = out[f"mfe_{h}"] = out[f"mae_{h}"] = None
            continue
        window_high = float(tape.high[i0 + 1:i0 + h + 1].max())
        window_low = float(tape.low[i0 + 1:i0 + h + 1].min())
        out[f"ret_{h}"] = direction * (float(tape.close[i0 + h]) - entry)
        if direction > 0:
            out[f"mfe_{h}"], out[f"mae_{h}"] = max(0.0, window_high - entry), max(0.0, entry - window_low)
        else:
            out[f"mfe_{h}"], out[f"mae_{h}"] = max(0.0, entry - window_low), max(0.0, window_high - entry)
    last = min(reach, i0 + bracket_horizon)
    for target, stop in brackets:
        key = f"br_{target}_{stop}"
        out[key], out[f"{key}_min"] = "none", None
        if last <= i0:
            continue
        highs, lows = tape.high[i0 + 1:last + 1], tape.low[i0 + 1:last + 1]
        if direction > 0:
            hit_target, hit_stop = highs >= entry + target * unit, lows <= entry - stop * unit
        else:
            hit_target, hit_stop = lows <= entry - target * unit, highs >= entry + stop * unit
        first_stop = int(np.argmax(hit_stop)) if hit_stop.any() else None
        first_target = int(np.argmax(hit_target)) if hit_target.any() else None
        if first_stop is not None and (first_target is None or first_stop <= first_target):
            out[key], out[f"{key}_min"] = "stop", first_stop + 1
        elif first_target is not None:
            out[key], out[f"{key}_min"] = "target", first_target + 1
    return out


def classify_day(o: float, h: float, low: float, c: float, *, t_high: pd.Timestamp, t_low: pd.Timestamp) -> str:
    """Trend / reversal / chop of one Regular-Trading-Hours session (pre-registration §4)."""

    span = h - low
    if span <= 0 or t_high == t_low:
        return "chop"
    first = h if t_high < t_low else low
    move = c - o
    if abs(first - o) >= 0.35 * span and move * (first - o) < 0 and abs(move) >= 0.25 * span:
        return "reversal"
    if abs(move) >= 0.5 * span and ((move > 0 and c >= h - 0.25 * span) or (move < 0 and c <= low + 0.25 * span)):
        return "trend"
    return "chop"


def trade_date(at: pd.Timestamp) -> dt.date:
    """The date of the 17:00 close a New York time belongs to (the 18:00 open starts the next one)."""

    local = at.tz_convert(MARKET_TIMEZONE)
    return (local + pd.Timedelta(days=1)).date() if local.hour >= 18 else local.date()


def clustered_mean(values: np.ndarray, clusters: np.ndarray) -> tuple[float, float, float, int]:
    """Mean, cluster-robust standard error, t and n (NaNs dropped)."""

    values = np.asarray(values, dtype=float)
    keep = ~np.isnan(values)
    x, groups = values[keep], np.asarray(clusters)[keep]
    n = int(len(x))
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(x.mean())
    labels, codes = np.unique(groups, return_inverse=True)
    g = len(labels)
    if n < 2 or g < 2:
        return mean, math.nan, math.nan, n
    sums = np.bincount(codes, weights=x - mean)
    se = math.sqrt(g / (g - 1) * float((sums ** 2).sum()) / n ** 2)
    t = mean / se if se > 0 else (math.inf if mean > 0 else -math.inf if mean < 0 else math.nan)
    return mean, se, t, n


def excess_over_baseline(events: pd.DataFrame, baseline: pd.DataFrame, column: str) -> pd.Series:
    """The event's ``column`` minus the baseline mean for its New York hour and direction."""

    means = baseline.groupby(["hour", "direction"])[column].mean().rename("_base")
    merged = events[["hour", "direction"]].join(means, on=["hour", "direction"])
    return pd.to_numeric(events[column], errors="coerce") - merged["_base"]


def leg_flips(bars: pd.DataFrame, column: str) -> pd.DataFrame:
    """The bars where ``column`` changed from the last known value (a direction per flip)."""

    rows, previous = [], None
    for index, value in bars[column].items():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if previous is not None and value != previous:
            rows.append(index)
        previous = value
    flips = bars.loc[rows].copy()
    flips["direction"] = flips[column]
    return flips


def sweep_then_mss(sweeps: pd.DataFrame, mss: pd.DataFrame, *, minutes: int) -> np.ndarray:
    """For each MSS, whether a same-direction sweep came strictly before it within ``minutes``."""

    window = pd.Timedelta(minutes=minutes)
    keep = []
    for at, direction in zip(mss["known_at"], mss["direction"]):
        same = sweeps[(sweeps["direction"] == direction) & (sweeps["known_at"] < at) & (sweeps["known_at"] >= at - window)]
        keep.append(not same.empty)
    return np.array(keep, dtype=bool)


def candidate_edge(stats: dict[str, Any], *, require_cost: bool = True) -> bool:
    """The pre-registered candidate-edge rule (§5) on one metric's cell statistics."""

    def ok(value) -> bool:
        return value is not None and not (isinstance(value, float) and math.isnan(value))

    if not (stats["n"] >= 150 and stats["n_h1"] >= 60 and stats["n_h2"] >= 60):
        return False
    if not (ok(stats["t"]) and stats["mean"] > 0 and stats["t"] >= 3.0):
        return False
    for half in ("h1", "h2"):
        mean, t = stats[f"mean_{half}"], stats[f"t_{half}"]
        if not (ok(mean) and ok(t) and mean > 0 and t >= 1.5):
            return False
    if stats["months_positive"] < 8:
        return False
    if require_cost and not (ok(stats.get("mean_pts")) and stats["mean_pts"] > 1.0):
        return False
    return True


def negated(stats: dict[str, Any]) -> dict[str, Any]:
    """The same cell read in the opposite direction (a fade), for the same rule."""

    out = dict(stats)
    for key in ("mean", "t", "mean_h1", "t_h1", "mean_h2", "t_h2", "mean_pts"):
        value = stats.get(key)
        out[key] = None if value is None else -value
    out["months_positive"] = stats.get("months_negative", stats["months"] - stats["months_positive"])
    return out


# --------------------------------------------------------------------------- the study


def load_tape(source: Path, start: str, end: str) -> tuple[Tape, pd.DataFrame]:
    from shares.core.io import iter_completed_bars, load_ohlcv

    loaded = load_ohlcv(source, start=start, end=end)
    rows = [(bar.end, bar.open, bar.high, bar.low, bar.close) for bar in iter_completed_bars(loaded.frame)]
    frame = pd.DataFrame(rows, columns=["end", "open", "high", "low", "close"])
    minutes = (frame["end"].map(lambda ts: ts.value) // NANOS_PER_MINUTE).to_numpy(dtype=np.int64)
    tape = Tape(minutes, frame["high"].to_numpy(float), frame["low"].to_numpy(float), frame["close"].to_numpy(float))
    return tape, frame


def day_types(frame: pd.DataFrame) -> dict[dt.date, str]:
    local = frame["end"].map(lambda ts: ts.tz_convert(MARKET_TIMEZONE))
    minute_of_day = local.map(lambda ts: ts.hour * 60 + ts.minute)
    rth = frame[(minute_of_day > 9 * 60 + 30) & (minute_of_day <= 16 * 60)].copy()
    rth["date"] = local[rth.index].map(lambda ts: ts.date())
    types = {}
    for date, day in rth.groupby("date"):
        if len(day) < 300:
            continue
        hi, lo = day["high"].idxmax(), day["low"].idxmin()
        types[date] = classify_day(float(day["open"].iloc[0]), float(day["high"].max()), float(day["low"].min()),
                                   float(day["close"].iloc[-1]), t_high=day.at[hi, "end"], t_low=day.at[lo, "end"])
    return types


def measure(rows: pd.DataFrame, tape: Tape) -> pd.DataFrame:
    """Outcome columns for every row (entry at its ``known_at`` close, unit = its 15m ATR)."""

    records = []
    for known_at, direction, unit in zip(rows["known_at"], rows["direction"], rows["atr_15m"]):
        i0 = tape.index_of(pd.Timestamp(known_at))
        if i0 is None or unit is None or not (unit > 0) or direction not in ("long", "short"):
            records.append({"entry_found": i0 is not None})
            continue
        result = outcomes(tape, i0, 1 if direction == "long" else -1, unit=float(unit))
        result["entry_found"] = True
        records.append(result)
    out = pd.DataFrame(records, index=rows.index)
    for h in HORIZONS:
        for part in ("ret", "mfe", "mae"):
            out[f"{part}_{h}"] = pd.to_numeric(out.get(f"{part}_{h}"), errors="coerce")
            out[f"{part}_{h}_R"] = out[f"{part}_{h}"] / pd.to_numeric(rows["atr_15m"], errors="coerce")
        out[f"edge_{h}_R"] = out[f"mfe_{h}_R"] - out[f"mae_{h}_R"]
    for target, stop in BRACKETS:
        key = f"br_{target}_{stop}"
        column = out.get(key)
        out[f"win_{target}_{stop}"] = np.where(column == "target", 1.0, np.where(column == "stop", 0.0, np.nan)) if column is not None else np.nan
    return out


def annotate(rows: pd.DataFrame, days: dict[dt.date, str]) -> pd.DataFrame:
    rows = rows.copy()
    local = rows["known_at"].map(lambda ts: pd.Timestamp(ts).tz_convert(MARKET_TIMEZONE))
    rows["hour"] = local.map(lambda ts: ts.hour)
    rows["month"] = local.map(lambda ts: ts.month)
    rows["half"] = np.where(rows["month"] <= 6, "H1", "H2")
    rows["trade_date"] = rows["known_at"].map(lambda ts: trade_date(pd.Timestamp(ts)))
    rows["day_type"] = rows["trade_date"].map(days.get)
    minute = local.map(lambda ts: ts.hour * 60 + ts.minute)
    rows["session"] = np.where((minute >= 9 * 60 + 30) & (minute < 16 * 60), "rth", "overnight")

    def htf(row) -> str:
        agree = [row.get("ext_1H") == row["direction"], row.get("ext_4H") == row["direction"]]
        against = [row.get("ext_1H") not in (None, row["direction"]) and isinstance(row.get("ext_1H"), str),
                   row.get("ext_4H") not in (None, row["direction"]) and isinstance(row.get("ext_4H"), str)]
        return "with" if all(agree) else "against" if all(against) else "mixed"

    rows["htf"] = rows.apply(htf, axis=1)
    loc = pd.to_numeric(rows.get("range_loc_15m"), errors="coerce")
    rows["in_range_15m"] = rows.get("range_active_15m").fillna(False).astype(bool) & loc.between(0.0, 1.0)
    return rows


def families(events: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Every family × scale row of §2 (``family``, ``tf``), deduplicated per bar and direction."""

    parts = []
    by_kind = {"MSS": "mss_core_confirmed", "BOS": "qualified_bos", "SWEEP": "sweep_confirmed",
               "ACCEPT": "acceptance_confirmed", "SDC": "structure_direction_confirmed"}
    for tf in TIMEFRAMES:
        on = events[events["tf"] == tf]
        for family, kind in by_kind.items():
            parts.append(on[on["kind"] == kind].assign(family=family))
        disp = on[on["kind"] == "displacement_observed"]
        for life, family in (("active", "DISP"), ("started", "DISP_started"), ("exhausted", "DISP_exhausted")):
            parts.append(disp[disp["lifecycle"] == life].assign(family=family))
        closes = bars[(bars["tf"] == tf) & (bars["real_completed"] != False)].sort_values("known_at")  # noqa: E712
        flips = leg_flips(closes, f"leg_{tf}")
        parts.append(flips.assign(family="LEGFLIP", kind="leg_flip"))
    htf = events[(events["tf"] == "1m") & (events["kind"] == "sweep_confirmed")]
    for level_tf in ("15m", "1H", "4H"):
        parts.append(htf[htf["level_tf"] == level_tf].assign(family="HTF_SWEEP", tf=level_tf))
    m15 = events[(events["tf"] == "15m") & (events["kind"] == "mss_core_confirmed")]
    parts.append(m15[m15["displacement_context_present"] == True].assign(family="C1_MSS_DISP"))  # noqa: E712
    parts.append(m15[m15["prior_sweep"] == True].assign(family="C2_MSS_PRIOR_SWEEP"))  # noqa: E712
    sweeps = events[(events["tf"] == "15m") & (events["kind"] == "sweep_confirmed")]
    parts.append(m15[sweep_then_mss(sweeps, m15, minutes=60)].assign(family="C3_SWEEP_THEN_MSS"))
    rows = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    return rows.drop_duplicates(subset=["family", "tf", "direction", "known_at"]).reset_index(drop=True)


METRICS = {
    "ret60": ("ret_60_R", True),
    "edge60": ("edge_60_R", False),
    "win11": ("win_1_1", False),
}


def cell_stats(rows: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {"n_events": int(len(rows))}
    for name, (column, needs_cost) in METRICS.items():
        x = rows[f"x_{column}"].to_numpy(float)
        mean, se, t, n = clustered_mean(x, rows["trade_date"].astype(str).to_numpy())
        entry = {"n": n, "mean": mean, "t": t}
        for half in ("H1", "H2"):
            part = rows[rows["half"] == half]
            m, _, th, nh = clustered_mean(part[f"x_{column}"].to_numpy(float), part["trade_date"].astype(str).to_numpy())
            entry[f"mean_{half.lower()}"], entry[f"t_{half.lower()}"], entry[f"n_{half.lower()}"] = m, th, nh
        monthly = rows.groupby("month")[f"x_{column}"].mean()
        entry["months"] = int(monthly.notna().sum())
        entry["months_positive"] = int((monthly > 0).sum())
        entry["months_negative"] = int((monthly < 0).sum())
        entry["mean_pts"] = float(np.nanmean(rows["x_ret_60"])) if needs_cost and rows["x_ret_60"].notna().any() else None
        entry["candidate"] = candidate_edge(entry, require_cost=needs_cost)
        entry["fade"] = candidate_edge(negated(entry), require_cost=needs_cost)
        stats[name] = entry
    for h in HORIZONS:
        m, _, t, n = clustered_mean(rows[f"x_ret_{h}_R"].to_numpy(float), rows["trade_date"].astype(str).to_numpy())
        stats[f"h{h}"] = {"n": n, "mean": m, "t": t}
    stats["raw_ret60_R"] = float(np.nanmean(rows["ret_60_R"])) if rows["ret_60_R"].notna().any() else math.nan
    stats["mfe_mae_ratio60"] = (float(np.nanmedian(rows["mfe_60_R"])) / float(np.nanmedian(rows["mae_60_R"]))
                                if rows["mae_60_R"].notna().any() and np.nanmedian(rows["mae_60_R"]) > 0 else math.nan)
    stats["win11_raw"] = float(np.nanmean(rows["win_1_1"])) if rows["win_1_1"].notna().any() else math.nan
    stats["win21_raw"] = float(np.nanmean(rows["win_2_1"])) if rows["win_2_1"].notna().any() else math.nan
    return stats


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True, type=Path, help="the scan's output directory")
    args = parser.parse_args(argv)
    scan = args.scan if args.scan.is_absolute() else ROOT / args.scan
    manifest = json.loads((scan / "manifest.json").read_text())
    window = manifest["window"]
    if "parts" in manifest:
        # A checkpointed scan: only the parts its manifest committed are data.
        from eyes.scripts.scan_sparse_events import read_parts

        events, bars = read_parts(scan, "events"), read_parts(scan, "bars")
    else:
        events = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(scan / "events_*.parquet")))], ignore_index=True)
        bars = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(scan / "bars_*.parquet")))], ignore_index=True)
    tape, frame = load_tape(ROOT / window["source"], window["emit_start"], window["end"])
    days = day_types(frame)

    rows = annotate(families(events, bars), days)
    rows = rows.join(measure(rows, tape))
    base = bars[(bars["tf"] == "15m") & (bars["real_completed"] != False)]  # noqa: E712
    base = pd.concat([base.assign(direction="long"), base.assign(direction="short")], ignore_index=True)
    base = annotate(base, days)
    base = base.join(measure(base, tape))
    for column in [*(f"ret_{h}_R" for h in HORIZONS), "ret_60", "edge_60_R", "win_1_1", "win_2_1"]:
        rows[f"x_{column}"] = excess_over_baseline(rows, base, column)

    study = scan / "study"
    study.mkdir(exist_ok=True)
    rows.drop(columns=[c for c in rows.columns if c.startswith("br_")]).to_parquet(study / "rows.parquet", index=False)

    cells = []

    def add(label: dict[str, Any], part: pd.DataFrame) -> None:
        if len(part) == 0:
            return
        cells.append({**label, **cell_stats(part)})

    for (family, tf), part in rows.groupby(["family", "tf"]):
        add({"family": family, "tf": tf, "stratum": "all", "value": "all"}, part)
        for direction, sub in part.groupby("direction"):
            add({"family": family, "tf": tf, "stratum": "direction", "value": direction}, sub)
        if tf == "15m":
            for stratum in ("day_type", "htf", "in_range_15m", "session"):
                for value, sub in part.groupby(stratum):
                    add({"family": family, "tf": tf, "stratum": stratum, "value": str(value)}, sub)
    baseline = {}
    for direction, part in base.groupby("direction"):
        baseline[direction] = {
            "n": int(part["ret_60_R"].notna().sum()), "ret60_R": float(np.nanmean(part["ret_60_R"])),
            "ret60_pts": float(np.nanmean(part["ret_60"])), "edge60_R": float(np.nanmean(part["edge_60_R"])),
            "mfe_mae_ratio60": float(np.nanmedian(part["mfe_60_R"]) / np.nanmedian(part["mae_60_R"])),
            "win11": float(np.nanmean(part["win_1_1"])), "win21": float(np.nanmean(part["win_2_1"])),
        }
    day_counts = pd.Series(days).value_counts().to_dict()
    flat = []
    for cell in cells:
        flat.append({k: v for k, v in cell.items() if not isinstance(v, dict)} | {
            f"{m}_{k}": v for m in (*METRICS, *(f"h{h}" for h in HORIZONS)) for k, v in cell[m].items()})
    pd.DataFrame(flat).to_csv(study / "cells.csv", index=False)
    summary = {"manifest": manifest, "baseline": baseline, "day_types": {str(k): v for k, v in day_counts.items()},
               "rows": int(len(rows)), "rows_with_entry": int(rows["entry_found"].sum()),
               "primary_cells": sorted({(c["family"], c["tf"]) for c in cells if c["stratum"] == "all"}),
               "candidates": [{"family": c["family"], "tf": c["tf"], "stratum": c["stratum"], "value": c["value"], "metric": m}
                              for c in cells for m in METRICS if c[m]["candidate"]],
               "fades": [{"family": c["family"], "tf": c["tf"], "stratum": c["stratum"], "value": c["value"], "metric": m}
                         for c in cells for m in METRICS if c[m]["fade"]]}
    (study / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    lines = ["# Eye event edge 2023 — study tables", "",
             f"rows {len(rows)}, with an entry bar {int(rows['entry_found'].sum())}; day types {day_counts}", "",
             "Baseline (every 15m close, per direction): " + json.dumps({d: {k: round(v, 4) if isinstance(v, float) else v
                                                                          for k, v in b.items()} for d, b in baseline.items()}), "",
             "| family | tf | stratum | value | n | ret60 excess R (t) | H1 t / H2 t | months + | excess pts | edge60 excess R (t) | win1:1 excess (t) | raw win1:1 | MFE/MAE | cand |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in sorted(cells, key=lambda c: (c["tf"] != "15m", c["family"], c["tf"], c["stratum"] != "all", c["stratum"], c["value"])):
        r, e, w = c["ret60"], c["edge60"], c["win11"]
        cand = ",".join([*(m for m in METRICS if c[m]["candidate"]), *(f"fade:{m}" for m in METRICS if c[m]["fade"])])
        lines.append(f"| {c['family']} | {c['tf']} | {c['stratum']} | {c['value']} | {r['n']} | {_fmt(r['mean'])} ({_fmt(r['t'], 2)}) | "
                     f"{_fmt(r['t_h1'], 2)} / {_fmt(r['t_h2'], 2)} | {r['months_positive']}/{r['months']} | {_fmt(r['mean_pts'], 2)} | "
                     f"{_fmt(e['mean'])} ({_fmt(e['t'], 2)}) | {_fmt(w['mean'])} ({_fmt(w['t'], 2)}) | {_fmt(c['win11_raw'])} | "
                     f"{_fmt(c['mfe_mae_ratio60'], 2)} | {cand} |")
    lines += ["", "## Horizons (mean excess return in 15m ATRs, t)", "",
              "| family | tf | " + " | ".join(f"{h}m" for h in HORIZONS) + " |", "| --- | --- | " + " | ".join("---" for _ in HORIZONS) + " |"]
    for c in cells:
        if c["stratum"] == "all":
            lines.append(f"| {c['family']} | {c['tf']} | " + " | ".join(f"{_fmt(c[f"h{h}"]["mean"])} ({_fmt(c[f"h{h}"]["t"], 2)}, n {c[f"h{h}"]["n"]})" for h in HORIZONS) + " |")
    (study / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in ("rows", "rows_with_entry", "day_types", "candidates", "fades")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
