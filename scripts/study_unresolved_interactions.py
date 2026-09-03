"""Ask whether an unresolved penetration is its own market state.

Every registered crossing generation is expected to end as a sweep or as an
acceptance.  Two complete months left roughly a fifth of penetrations with
neither.  This script replays one month against the frozen Eye, classifies each
penetration by the terminal it actually received, and compares the three groups
against the top-of-book record for the same minute.

It answers one question and refuses to answer more: do UNRESOLVED penetrations
look like a distinct book regime, or like a mixture of the other two?  It does
not fit a model, choose a threshold, or grant any trading authority.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smc_trader.causal import CausalMarketReader  # noqa: E402
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.model import EventKind, Timeframe  # noqa: E402
from smc_trader.observation import CausalObserver, ObserverConfig  # noqa: E402
from smc_trader.scale_registry import parse_scale_specs  # noqa: E402
from smc_trader.semantics import load_semantic_selection  # noqa: E402

# The widest registered resolution window for a crossing generation.  A
# penetration opened inside this many minutes of the window edge cannot be
# called unresolved: its deadline simply had not arrived yet.
CENSOR_TAIL_MINUTES = 5

SWEEP, ACCEPTANCE, UNRESOLVED = "sweep", "acceptance", "unresolved"


def build_eye(model_path: Path) -> tuple[CausalMarketReader, CausalObserver]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    selection = load_semantic_selection(model.get("semantic_selection"), root=ROOT)
    raw = model["observer"]
    specs = parse_scale_specs(model["scales"])
    minimum = raw["minimum_bars"]
    observer = CausalObserver(
        ObserverConfig(
            atr_period=int(raw["atr_period"]),
            memory_events=int(raw["memory_events"]),
            minimum_bars={
                timeframe: int(minimum[timeframe.value])
                for timeframe in (
                    Timeframe.H4, Timeframe.H1, Timeframe.M15,
                    Timeframe.M5, Timeframe.M1,
                )
            },
            tick_size=float(model["tick_size"]),
            point_value=float(model["point_value"]),
            structure_protocol=str(ROOT / raw["structure_protocol"]),
            liquidity_protocol=str(ROOT / raw["liquidity_protocol"]),
            displacement_protocol=str(ROOT / raw["displacement_protocol"]),
            zone_protocol=str(ROOT / raw["zone_protocol"]),
            range_auction_protocol=str(ROOT / raw["range_auction_protocol"]),
            interaction_protocol=str(ROOT / raw["interaction_protocol"]),
            semantic_registry=str(selection.atomic_registry.source_path),
            scale_specs=specs,
            project_scene_graph=False,
            materialize_event_view=False,
            range_auction_projection_only=False,
            eye_authority_mode=True,
            persist_state_projections=False,
        ),
        semantic_registry=selection.atomic_registry,
    )
    reader = CausalMarketReader(
        scale_specs=specs, tick_size=float(model["tick_size"])
    )
    return reader, observer


def classify(store) -> tuple[dict[str, str], dict[str, object]]:
    """Label every penetration by the terminal that actually cited it."""

    penetrations: dict[str, object] = {}
    outcome: dict[str, str] = {}
    for event in store.events():
        if event.kind is EventKind.LEVEL_PENETRATED:
            penetrations[event.event_id] = event
            outcome.setdefault(event.event_id, UNRESOLVED)
        elif event.kind in (EventKind.SWEEP_CONFIRMED, EventKind.ACCEPTANCE_CONFIRMED):
            label = SWEEP if event.kind is EventKind.SWEEP_CONFIRMED else ACCEPTANCE
            for parent_id in event.source_event_ids:
                if parent_id in penetrations:
                    outcome[parent_id] = label
    return outcome, penetrations


def book_features(row: pd.Series, side: str, tick_size: float) -> dict[str, float]:
    """Side-aware top-of-book description at one decision minute.

    ``near`` is the side price is pressing into: an upside penetration lifts
    offers, so its near side is the ask.
    """

    bid, ask = float(row.bid), float(row.ask)
    bid_size, ask_size = float(row.bid_size), float(row.ask_size)
    top5_bid, top5_ask = float(row.top5_bid_size), float(row.top5_ask_size)
    upside = side == "above"
    near_size = ask_size if upside else bid_size
    far_size = bid_size if upside else ask_size
    near_top5 = top5_ask if upside else top5_bid
    far_top5 = top5_bid if upside else top5_ask
    total_top5 = near_top5 + far_top5
    return {
        "spread_ticks": (ask - bid) / tick_size,
        "near_size": near_size,
        "far_size": far_size,
        "near_top5": near_top5,
        "far_top5": far_top5,
        "total_top5": total_top5,
        # >0 means depth sits behind the move, <0 means it opposes it
        "signed_depth_imbalance": (
            (far_top5 - near_top5) / total_top5 if total_top5 > 0 else 0.0
        ),
        "near_far_ratio": near_size / far_size if far_size > 0 else math.nan,
    }


def describe(values: list[float]) -> dict[str, float] | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if len(clean) < 2:
        return None
    clean.sort()
    return {
        "n": len(clean),
        "mean": round(statistics.fmean(clean), 4),
        "p25": round(clean[len(clean) // 4], 4),
        "median": round(clean[len(clean) // 2], 4),
        "p75": round(clean[3 * len(clean) // 4], 4),
        "stdev": round(statistics.pstdev(clean), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-06-01")
    parser.add_argument("--end", default="2024-07-01")
    parser.add_argument(
        "--ohlcv",
        default="data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet",
    )
    parser.add_argument(
        "--mbo", default="data/processed/mbo_execution_dev_202406_v2_3.parquet"
    )
    parser.add_argument("--model", default="configs/model.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tick_size = float(json.loads((ROOT / args.model).read_text())["tick_size"])
    reader, observer = build_eye(ROOT / args.model)
    loaded = load_ohlcv(ROOT / args.ohlcv, start=args.start, end=args.end)

    bars = 0
    last_end = None
    for bar in iter_completed_bars(loaded.frame):
        if args.limit and bars >= args.limit:
            break
        observer.observe(reader.on_bar(bar))
        bars += 1
        last_end = bar.end

    store = observer.audit_store
    outcome, penetrations = classify(store)

    censor_after = last_end - pd.Timedelta(CENSOR_TAIL_MINUTES, unit="min")
    book = pd.read_parquet(ROOT / args.mbo).set_index("decision_time")
    book = book[book["book_valid"]]

    groups: dict[str, dict[str, list[float]]] = {
        label: collections.defaultdict(list)
        for label in (SWEEP, ACCEPTANCE, UNRESOLVED)
    }
    counts = collections.Counter()
    censored = 0
    unmatched = 0
    for event_id, label in outcome.items():
        event = penetrations[event_id]
        if label is UNRESOLVED and event.known_at > censor_after:
            censored += 1
            continue
        counts[label] += 1
        minute = event.known_at.floor("min")
        if minute not in book.index:
            unmatched += 1
            continue
        row = book.loc[minute]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        for name, value in book_features(row, event.side or "above", tick_size).items():
            groups[label][name].append(value)
        groups[label]["timeframe_is_m1"].append(
            1.0 if event.timeframe is Timeframe.M1 else 0.0
        )

    summary = {
        "window": {
            "start": args.start,
            "end": args.end,
            "bars_replayed": bars,
            "last_bar_end": str(last_end),
        },
        "penetrations": {
            "total_classified": sum(counts.values()),
            "by_outcome": dict(counts),
            "right_censored_unresolved": censored,
            "without_a_book_row": unmatched,
        },
        "book_features": {
            label: {
                name: describe(values)
                for name, values in sorted(groups[label].items())
            }
            for label in (SWEEP, ACCEPTANCE, UNRESOLVED)
        },
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        (ROOT / args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
