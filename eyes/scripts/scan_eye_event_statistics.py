#!/usr/bin/env python3
"""Outcome-blind census of what the Eye emitted over one OHLCV window.

The scan reuses the causal reader and observer and deliberately omits the
Brain, actions, PnL, MBO and per-minute trace output.  It counts every event by
kind, timeframe, direction, side, session phase, lifecycle and transition
reason, and reconstructs the market-structure relation graph from the ancestry
each canonical event is required to cite.  It reads only; it grants no
research, empirical, or trading authority.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eyes.core.causal import CausalMarketReader  # noqa: E402
from shares.core.io import iter_completed_bars, load_ohlcv  # noqa: E402
from contract.market import Timeframe  # noqa: E402
from contract.eye import EventOrigin  # noqa: E402
from eyes.core.observation import CausalObserver, ObserverConfig  # noqa: E402
from shares.core.scale_registry import parse_scale_specs  # noqa: E402
from eyes.core.semantics import load_semantic_selection  # noqa: E402

DEFAULT_SOURCE = (
    "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
)


def build_eye(model_path: Path) -> tuple[CausalMarketReader, CausalObserver]:
    """Construct the registered graph-free Eye described by ``model_path``."""

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
    reader = CausalMarketReader(
        scale_specs=specs,
        tick_size=float(model["tick_size"]),
    )
    return reader, observer


def _pairs(counter: collections.Counter) -> dict[str, int]:
    return {
        f"{left} -> {right}": count
        for (left, right), count in sorted(
            counter.items(), key=lambda item: (-item[1], item[0])
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--start", default="2022-02-01")
    parser.add_argument("--end", default="2022-03-01")
    parser.add_argument("--model", default="configs/model.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--progress-bars",
        type=int,
        default=0,
        help=(
            "write a progress line to stderr every N completed bars; a month "
            "otherwise prints nothing at all until it finishes, so a running "
            "scan offers no observable but its resident size"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    loaded = load_ohlcv(ROOT / args.source, start=args.start, end=args.end)
    reader, observer = build_eye(ROOT / args.model)

    total_bars = len(loaded.frame)
    started = time.monotonic()
    bars = 0
    first_end = last_end = None
    for bar in iter_completed_bars(loaded.frame):
        if args.limit and bars >= args.limit:
            break
        observer.observe(reader.on_bar(bar))
        bars += 1
        if first_end is None:
            first_end = bar.end
        last_end = bar.end
        if args.progress_bars and bars % args.progress_bars == 0:
            elapsed = time.monotonic() - started
            print(
                f"{bars}/{total_bars} bars  {bar.end.isoformat()}  "
                f"{elapsed / 60.0:.1f} min elapsed  "
                f"{bars / elapsed:.2f} bars/s",
                file=sys.stderr,
                flush=True,
            )

    store = observer.audit_store
    events = list(store.events())
    by_id = {event.event_id: event for event in events}

    kinds: collections.Counter[str] = collections.Counter()
    canonical: collections.Counter[str] = collections.Counter()
    origins: collections.Counter[str] = collections.Counter()
    by_timeframe = collections.defaultdict(collections.Counter)
    by_direction: collections.Counter = collections.Counter()
    by_side: collections.Counter = collections.Counter()
    by_session: collections.Counter = collections.Counter()
    lifecycles = collections.defaultdict(collections.Counter)
    transitions = collections.defaultdict(collections.Counter)
    entities = collections.defaultdict(set)
    known_lag: collections.Counter = collections.Counter()
    ancestry: collections.Counter = collections.Counter()
    context: collections.Counter = collections.Counter()
    citing: collections.Counter = collections.Counter()
    dangling: collections.Counter[str] = collections.Counter()

    for event in events:
        name = event.kind.value
        kinds[name] += 1
        origins[event.origin.value] += 1
        by_timeframe[name][event.timeframe.value] += 1
        if event.direction is not None:
            by_direction[(name, event.direction.value)] += 1
        if event.side is not None:
            by_side[(name, event.side)] += 1
        phase = event.evidence.get("session_phase")
        if isinstance(phase, str):
            by_session[(name, phase)] += 1
        if event.lifecycle:
            lifecycles[name][event.lifecycle] += 1
        if event.transition_reason:
            transitions[name][event.transition_reason] += 1
        if event.entity_id:
            entities[name].add(event.entity_id)
        if event.known_at is not None and event.event_time is not None:
            known_lag[(name, str(event.known_at - event.event_time))] += 1

        # Only canonical events cite real event ancestry: a legacy transport
        # event carries entity identities in both source namespaces, so
        # following them would fabricate broken references.
        if event.origin is not EventOrigin.SEMANTIC_ATOMIC:
            continue
        canonical[name] += 1
        parents = tuple(event.source_event_ids)
        citing[(name, len(parents))] += 1
        for parent_id in parents:
            parent = by_id.get(parent_id)
            if parent is None:
                dangling[name] += 1
                continue
            ancestry[(parent.kind.value, name)] += 1
        for parent_id in event.context_event_ids:
            parent = by_id.get(parent_id)
            if parent is not None:
                context[(parent.kind.value, name)] += 1

    result = {
        "window": {
            "start": args.start,
            "end": args.end,
            "source": args.source,
            "model": args.model,
            "bars_replayed": bars,
            "first_bar_end": str(first_end),
            "last_bar_end": str(last_end),
        },
        "event_store": {
            "total_events": len(events),
            "distinct_kinds": len(kinds),
            "fingerprint": store.fingerprint(),
            "origins": dict(sorted(origins.items())),
        },
        "event_counts": dict(
            sorted(kinds.items(), key=lambda item: (-item[1], item[0]))
        ),
        "canonical_event_counts": dict(
            sorted(canonical.items(), key=lambda item: (-item[1], item[0]))
        ),
        "counts_by_timeframe": {
            kind: dict(sorted(values.items()))
            for kind, values in sorted(by_timeframe.items())
        },
        "counts_by_direction": _pairs(by_direction),
        "counts_by_side": _pairs(by_side),
        "counts_by_session_phase": _pairs(by_session),
        "lifecycle_counts": {
            kind: dict(sorted(v.items(), key=lambda item: (-item[1], item[0])))
            for kind, v in sorted(lifecycles.items())
        },
        "transition_reason_counts": {
            kind: dict(sorted(v.items(), key=lambda item: (-item[1], item[0])))
            for kind, v in sorted(transitions.items())
        },
        "distinct_entities": {
            kind: len(ids) for kind, ids in sorted(entities.items())
        },
        "structure_relations_source": _pairs(ancestry),
        "structure_relations_context": _pairs(context),
        "parents_cited_histogram": _pairs(citing),
        "known_at_minus_event_time": _pairs(known_lag),
        "unresolved_parent_references": dict(sorted(dangling.items())),
    }
    text = json.dumps(result, indent=2, default=str)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
