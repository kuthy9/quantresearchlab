"""Which registered maturity gate actually stops a balance range?

Two complete months registered ~40 structural ranges each and matured none, so
"maturity is unreachable" is established.  This script asks the next question
and refuses to answer more: at the moment each range is terminalized, which of
the six registered gates were unmet, and by how much?

It fits nothing, proposes no threshold, and grants no authority.  It exists so a
threshold discussion can start from the observed distribution rather than from
the fact that the conjunction is empty.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eyes.core.causal import CausalMarketReader  # noqa: E402
from shares.core.io import iter_completed_bars, load_ohlcv  # noqa: E402
from contract.market import Timeframe  # noqa: E402
from contract.eye import DealingRangeLifecycle  # noqa: E402
from eyes.core.observation import CausalObserver, ObserverConfig  # noqa: E402
from eyes.core.range_auction import RangeAuctionProtocol  # noqa: E402
from shares.core.scale_registry import parse_scale_specs  # noqa: E402
from eyes.core.semantics import load_semantic_selection  # noqa: E402

# name -> (statistic attribute, protocol attribute, comparison)
GATES = (
    ("duration", "candidate_real_h1_bars", "minimum_candidate_real_h1_bars", "min"),
    (
        "bilateral_price_tests",
        None,
        "balance_minimum_price_test_generations_each",
        "min",
    ),
    ("midpoint_crossing", "midpoint_crossings", "minimum_midpoint_crossings", "min"),
    (
        "inside_close_fraction",
        "inside_close_fraction",
        "minimum_inside_close_fraction",
        "min",
    ),
    ("width", "width_atr_at_formation", "maximum_width_atr_at_formation", "max"),
    ("compression", "compression_ratio", "maximum_compression_ratio", "max"),
)


def build_eye(model_path: Path):
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
                tf: int(minimum[tf.value])
                for tf in (
                    Timeframe.H4,
                    Timeframe.H1,
                    Timeframe.M15,
                    Timeframe.M5,
                    Timeframe.M1,
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
    reader = CausalMarketReader(scale_specs=specs, tick_size=float(model["tick_size"]))
    protocol = RangeAuctionProtocol.from_file(
        str(ROOT / raw["range_auction_protocol"])
    )
    return reader, observer, protocol


def actual(state, name: str, attribute: str | None) -> float:
    if name == "bilateral_price_tests":
        return float(
            min(
                state.balance_lower_test_generations,
                state.balance_upper_test_generations,
            )
        )
    return float(getattr(state, attribute))


def describe(values: list[float]) -> dict[str, float] | None:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    return {
        "n": len(clean),
        "min": round(clean[0], 4),
        "p25": round(clean[len(clean) // 4], 4),
        "median": round(clean[len(clean) // 2], 4),
        "p75": round(clean[3 * len(clean) // 4], 4),
        "max": round(clean[-1], 4),
        "mean": round(statistics.fmean(clean), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-03-01")
    parser.add_argument("--end", default="2022-04-01")
    parser.add_argument(
        "--ohlcv",
        default="data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet",
    )
    parser.add_argument("--model", default="configs/model.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reader, observer, protocol = build_eye(ROOT / args.model)

    # Capture the frozen range states the emitter is handed.  This observes the
    # existing path; it changes nothing the Eye publishes.
    seen: dict[str, object] = {}
    original = observer._emitter._record_group4_events

    def capture(update, *a, **kw):
        for state in getattr(update, "range_transitions", ()) or ():
            seen[state.range_id] = state
        return original(update, *a, **kw)

    observer._emitter._record_group4_events = capture

    loaded = load_ohlcv(ROOT / args.ohlcv, start=args.start, end=args.end)
    bars = 0
    for bar in iter_completed_bars(loaded.frame):
        observer.observe(reader.on_bar(bar))
        bars += 1

    thresholds = {
        name: float(getattr(protocol, threshold_attribute))
        for name, _, threshold_attribute, _ in GATES
    }
    unmet_counter: collections.Counter[str] = collections.Counter()
    per_gate_values: dict[str, list[float]] = {name: [] for name, *_ in GATES}
    unmet_count_histogram: collections.Counter[int] = collections.Counter()
    terminal_reasons: collections.Counter[str] = collections.Counter()
    sole_blocker: collections.Counter[str] = collections.Counter()

    for state in seen.values():
        terminal_reasons[str(state.transition_reason)] += 1
        unmet: list[str] = []
        for name, attribute, threshold_attribute, sense in GATES:
            value = actual(state, name, attribute)
            per_gate_values[name].append(value)
            limit = thresholds[name]
            if (sense == "min" and value < limit) or (
                sense == "max" and value > limit
            ):
                unmet.append(name)
        for name in unmet:
            unmet_counter[name] += 1
        unmet_count_histogram[len(unmet)] += 1
        if len(unmet) == 1:
            sole_blocker[unmet[0]] += 1

    summary = {
        "window": {"start": args.start, "end": args.end, "bars_replayed": bars},
        "registered_thresholds": thresholds,
        "ranges_observed": len(seen),
        # The range no longer has a maturity state.  What is counted is the
        # balance claim settling, which is what the gates were ever about.
        "balance_claims_confirmed": sum(
            1
            for state in seen.values()
            if state.balance_confirmed_at is not None
        ),
        "terminal_reasons": dict(terminal_reasons),
        "unmet_gate_counts": dict(unmet_counter),
        "unmet_gates_per_range": {
            str(k): v for k, v in sorted(unmet_count_histogram.items())
        },
        "sole_blocking_gate": dict(sole_blocker),
        "observed_distributions": {
            name: describe(values) for name, values in per_gate_values.items()
        },
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        (ROOT / args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
