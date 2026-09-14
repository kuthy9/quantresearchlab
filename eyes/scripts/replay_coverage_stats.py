"""Replay a window through the Eye alone and count what each repair changed.

One streaming pass, bounded memory: nothing from an earlier bar is retained
beyond counters, so the month costs what the Eye costs.  Writes one JSON
under ``outputs/eye_coverage/``.  The sections map onto the 2026-09 repairs:

- ``ranges``          dealing-range coverage per scale (Task 3, 2026-09-14)
- ``duplicates``      exact same-update duplicates; touches per crossing (Task 4)
- ``formation_clock`` known_at - formed_at per kind (Task 1)
- ``target_outcomes`` created / reached / invalidated / retired per scale (Task 5)
- ``event_clock``     events per scale on the main and 1m channels (Task 2)
- ``scales``          facts and reduced-state availability per scale (Task 3)
- ``growth``          seconds per block and state sizes at checkpoints (Task 6)
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract.eye import DealingRangeLifecycle, EventKind  # noqa: E402
from contract.market import Timeframe  # noqa: E402
from eyes.scripts.replay_hash_stream import DEFAULT_SOURCE, build_eye  # noqa: E402
from shares.core.io import iter_completed_bars, load_ohlcv  # noqa: E402

_SCALE_FACT_KINDS = (
    EventKind.DISPLACEMENT_OBSERVED,
    EventKind.FVG_CREATED,
    EventKind.QUALIFIED_ORIGIN_ZONE_CREATED,
    EventKind.STRUCTURAL_LEG_CREATED,
    EventKind.LIQUIDITY_LEVEL_CREATED,
    EventKind.DEALING_RANGE_CREATED,
    EventKind.BALANCE_RANGE_OBSERVED,
    EventKind.DEALING_RANGE_INVALIDATED,
)
_OUTCOME_KINDS = (
    EventKind.LIQUIDITY_LEVEL_CREATED,
    EventKind.LEVEL_TOUCHED,
    EventKind.LEVEL_REACHED,
    EventKind.LEVEL_INVALIDATED,
    EventKind.LIQUIDITY_RETIRED,
)


def _json_ready(value):
    """Every mapping key as text, so no counter key can refuse serialization."""

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _dup_key(event):
    """Two events are the same fact when everything a consumer reads agrees."""

    return (
        event.known_at,
        event.event_time,
        event.kind.value,
        event.timeframe.value,
        None if event.direction is None else event.direction.value,
        event.side,
        event.strength,
        event.price,
        event.entity_id,
        event.lifecycle,
        event.details.get("level_id"),
        # An atomic lifecycle fact (a displacement's started -> active on
        # one bar) carries its lifecycle in details.
        event.details.get("lifecycle"),
        event.source_entity_ids,
        event.source_event_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--model", default="configs/model.json")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--block", type=int, default=500, help="bars per timing block")
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--output", default=None, help="JSON path; default outputs/eye_coverage/<start>_<end>.json")
    args = parser.parse_args()

    frame = load_ohlcv(ROOT / args.source, start=args.start, end=args.end).frame
    bars = list(iter_completed_bars(frame))
    if args.limit:
        bars = bars[: args.limit]
    reader, observer = build_eye(ROOT / args.model, root=ROOT)
    range_scales = tuple(observer._range_timeframes)

    # ranges: per scale, identities seen and their milestones
    range_seen: dict[str, dict[str, dict]] = {tf.value: {} for tf in range_scales}
    range_bars = {tf.value: Counter() for tf in range_scales}
    # duplicates
    total_events = 0
    dup_extra = Counter()
    kind_totals = Counter()
    touches_per_crossing = Counter()
    # formation clock
    lag_count = Counter()
    lag_minutes = Counter()
    lag_zero = Counter()
    formed_missing = Counter()
    # outcomes
    outcome = {kind.value: Counter() for kind in _OUTCOME_KINDS}
    retired_reasons = Counter()
    # event clock
    main_by_scale = Counter()
    micro_by_scale = Counter()
    main_by_kind = Counter()
    micro_by_kind = Counter()
    # scales
    fact_by_scale = {kind.value: Counter() for kind in _SCALE_FACT_KINDS}
    state_available = defaultdict(Counter)
    # growth
    blocks: list[dict] = []
    checkpoints: list[dict] = []
    started = time.perf_counter()
    block_seconds = 0.0
    observation = None
    for number, bar in enumerate(bars, start=1):
        tick = time.perf_counter()
        observation = observer.observe(reader.on_bar(bar))
        block_seconds += time.perf_counter() - tick

        events = observation.events_this_update
        total_events += len(events)
        groups = Counter(_dup_key(event) for event in events)
        for key, count in groups.items():
            kind = key[2]
            kind_totals[kind] += count
            if count > 1:
                dup_extra[kind] += count - 1
        touched = Counter(
            (event.details.get("level_id"), event.event_time, event.timeframe)
            for event in events
            if event.kind is EventKind.LEVEL_TOUCHED
        )
        for count in touched.values():
            touches_per_crossing[min(count, 3)] += 1
        # The same level touched on this clock on its own scale and on 1m is
        # the two channels, not a duplicate; count those pairs apart.
        cross_scale = Counter(
            (level_id, event_time) for level_id, event_time, _ in touched
        )
        touches_per_crossing["cross_scale_pairs"] += sum(
            1 for count in cross_scale.values() if count > 1
        )
        for event in events:
            kind = event.kind.value
            if event.known_at is not None:
                if event.formed_at is None:
                    formed_missing[kind] += 1
                else:
                    lag = (event.known_at - event.formed_at).total_seconds() / 60.0
                    lag_count[kind] += 1
                    lag_minutes[kind] += lag
                    if lag == 0.0:
                        lag_zero[kind] += 1
            if event.kind in outcome:
                outcome[kind][event.timeframe.value] += 1
                if event.kind is EventKind.LIQUIDITY_RETIRED:
                    retired_reasons[str(event.details.get("reason"))] += 1
            if event.kind in fact_by_scale:
                fact_by_scale[kind][event.timeframe.value] += 1
        for event in observation.semantic_events_this_update:
            main_by_scale[event.timeframe.value] += 1
            main_by_kind[event.kind.value] += 1
        for event in observation.microstructure_events_this_update:
            micro_by_scale[event.timeframe.value] += 1
            micro_by_kind[event.kind.value] += 1

        snapshot = observation.market_snapshot
        for timeframe, state in snapshot.timeframe_states.items():
            key = timeframe.value
            state_available[key]["bars"] += 1
            if state.range.range_id is not None:
                state_available[key]["range"] += 1
            if state.delivery.displacement_score is not None:
                state_available[key]["displacement_score"] += 1
            if state.zones.active_fvg:
                state_available[key]["active_fvg"] += 1
            if state.zones.active_ob:
                state_available[key]["active_ob"] += 1
            if state.structural_legs:
                state_available[key]["structural_leg"] += 1
        for timeframe in range_scales:
            key = timeframe.value
            frame_ranges = observation.frames[timeframe].dealing_ranges
            live = [r for r in frame_ranges if r.lifecycle is DealingRangeLifecycle.ACTIVE]
            range_bars[key]["bars"] += 1
            if live:
                range_bars[key]["live"] += 1
                if live[0].balance_confirmed_at is not None:
                    range_bars[key]["live_balance_confirmed"] += 1
            for state in frame_ranges:
                record = range_seen[key].setdefault(
                    state.range_id,
                    {"formed_at": state.formed_at.isoformat(), "balance_confirmed_at": None, "broken_at": None, "age_bars": 0},
                )
                record["age_bars"] = max(record["age_bars"], int(state.age_h1_bars))
                if state.balance_confirmed_at is not None:
                    record["balance_confirmed_at"] = state.balance_confirmed_at.isoformat()
                if state.broken_at is not None:
                    record["broken_at"] = state.broken_at.isoformat()
                    record["break_reason"] = state.transition_reason

        if number % args.block == 0:
            blocks.append({"bars_done": number, "seconds": round(block_seconds, 1), "bars_per_s": round(args.block / block_seconds, 1)})
            print(f"bars {number - args.block:6d}-{number:6d}: {block_seconds:6.1f}s ({args.block / block_seconds:5.1f} bars/s)", flush=True)
            block_seconds = 0.0
        if number % args.checkpoint_every == 0 or number == len(bars):
            interaction = observation.interaction_update
            checkpoints.append(
                {
                    "bars_done": number,
                    "asof": observation.asof.isoformat(),
                    "candidates": {tf.value: len(st.liquidity.candidates) for tf, st in snapshot.timeframe_states.items()},
                    "inventory": len(observation.liquidity_inventory),
                    "inventory_consumed": sum(1 for item in observation.liquidity_inventory if item.lifecycle.value == "consumed"),
                    "interaction_paths": 0 if interaction is None else len(interaction.interaction_paths),
                    "zones": {tf.value: len(tr._zones) for tf, tr in observer._liquidity_trackers.items()},
                    "pools": {tf.value: len(tr._pools) for tf, tr in observer._liquidity_trackers.items()},
                    "structural_legs": {tf.value: len(f.structural_legs) for tf, f in observation.frames.items()},
                    "events_total": total_events,
                }
            )

    ranges_out = {}
    for key, seen in range_seen.items():
        created = len(seen)
        confirmed = sum(1 for r in seen.values() if r["balance_confirmed_at"])
        broken = sum(1 for r in seen.values() if r["broken_at"])
        bars_seen = range_bars[key]["bars"]
        ranges_out[key] = {
            "created": created,
            "balance_confirmed": confirmed,
            "broken": broken,
            "break_reasons": dict(Counter(r.get("break_reason") for r in seen.values() if r["broken_at"])),
            "mean_age_native_bars": round(sum(r["age_bars"] for r in seen.values()) / created, 1) if created else None,
            "bars_with_live_range_share": round(range_bars[key]["live"] / bars_seen, 4) if bars_seen else None,
            "bars_with_live_balance_confirmed_share": round(range_bars[key]["live_balance_confirmed"] / bars_seen, 4) if bars_seen else None,
        }
    formation = {
        kind: {
            "events": lag_count[kind],
            "formed_missing": formed_missing[kind],
            "lag_zero_share": round(lag_zero[kind] / lag_count[kind], 4),
            "mean_lag_minutes": round(lag_minutes[kind] / lag_count[kind], 2),
        }
        for kind in sorted(lag_count)
    }
    result = {
        "window": {"start": args.start, "end": args.end, "bars": len(bars), "events": total_events, "events_per_bar": round(total_events / max(len(bars), 1), 2), "eye_seconds": round(sum(b["seconds"] for b in blocks), 1)},
        "ranges": ranges_out,
        "duplicates": {
            "exact_same_update_extra": sum(dup_extra.values()),
            "exact_same_update_extra_share": round(sum(dup_extra.values()) / max(total_events, 1), 5),
            "extra_by_kind": dict(dup_extra.most_common()),
            "level_touched_per_crossing_on_its_scale": {str(k): v for k, v in sorted(touches_per_crossing.items(), key=str)},
        },
        "formation_clock": formation,
        "target_outcomes": {
            "by_kind_and_scale": {kind: dict(counter) for kind, counter in outcome.items()},
            "retired_reasons": dict(retired_reasons),
        },
        "event_clock": {
            "main_channel_by_scale": dict(main_by_scale),
            "microstructure_channel_by_scale": dict(micro_by_scale),
            "main_channel_by_kind": dict(main_by_kind.most_common()),
            "microstructure_channel_by_kind": dict(micro_by_kind.most_common()),
        },
        "scales": {
            "facts_by_kind_and_scale": {kind: dict(counter) for kind, counter in fact_by_scale.items()},
            "reduced_state_available_share": {
                key: {name: round(value / counter["bars"], 4) for name, value in counter.items() if name != "bars"}
                for key, counter in state_available.items()
            },
        },
        "growth": {"blocks": blocks, "checkpoints": checkpoints},
    }
    out = Path(args.output) if args.output else ROOT / "outputs" / "eye_coverage" / f"{args.start}_{args.end}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_ready(result), indent=1, default=str))
    print(f"{len(bars)} bars, {total_events} events -> {out} ({(time.perf_counter() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
