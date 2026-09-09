#!/usr/bin/env python3
"""How much of what the Eye sees is a distinct fact, and how fast it says so.

Two questions the v1.3 entity work was meant to answer, measured over the same
OHLCV windows the earlier census used:

*Efficiency* -- how many rows a consumer has to read to learn one thing.  A
relation recomputed every minute, or a structural claim renamed by every break,
inflates the row count without adding facts.  The generation entities collapse
those into episodes, and the compression ratio is how much.

*Precision* -- whether a fact is dated when it became true.  A base origin core
published only once a break qualified it could only ever be counted with
hindsight; published at the impulse, the unqualified ones become visible too.

It also answers the FVG expiry question empirically rather than by threshold:
how soon a gap is first retested, and whether older gaps behave differently.
The scan reads only; it grants no research, empirical or trading authority.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from shares.core.io import iter_completed_bars, load_ohlcv  # noqa: E402
from eyes.core.market_state import session_name_phase  # noqa: E402
from contract.eye import EventKind  # noqa: E402

from eyes.scripts.scan_eye_event_statistics import DEFAULT_SOURCE, build_eye  # noqa: E402


RETEST_BUCKETS = (1, 3, 6, 12, 24)
POPULATION_SAMPLE_BARS = 250
PROGRESS_BARS = 1000


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def _span_summary(values: list[int]) -> dict[str, object]:
    return {
        "count": len(values),
        "observations": sum(values),
        "min": min(values) if values else None,
        "median": _percentile(values, 0.5),
        "p90": _percentile(values, 0.9),
        "max": max(values) if values else None,
    }


def study(start: str, end: str, *, model_path: Path, source: Path) -> dict:
    reader, observer = build_eye(model_path)
    loaded = load_ohlcv(source, start=start, end=end)

    structure: dict[str, object] = {}
    relation: dict[str, object] = {}
    swings: dict[str, object] = {}
    levels: dict[str, object] = {}
    relation_rows = 0
    bars = 0
    last_bar_end = None

    total = len(loaded.frame)
    started = time.perf_counter()
    for bar in iter_completed_bars(loaded.frame):
        observer.observe(reader.on_bar(bar))
        snapshot = observer.last_market_snapshot
        bars += 1
        if bars % PROGRESS_BARS == 0:
            elapsed = time.perf_counter() - started
            rate = bars / elapsed
            remaining = (total - bars) / rate if rate else float("nan")
            print(
                f"[{bars}/{total}] {bars / total:.1%} "
                f"elapsed={elapsed / 60:.1f}min "
                f"rate={rate:.1f}bars/s "
                f"eta={remaining / 60:.1f}min",
                flush=True,
            )
        if snapshot is None:
            continue
        last_bar_end = snapshot.asof
        for generation in snapshot.structure_generations.values():
            structure[generation.generation_id] = generation
        for generation in snapshot.relation_generations.values():
            relation[generation.generation_id] = generation
        relation_rows += len(snapshot.relations)
        # The swing and level populations are large and long-lived, so
        # re-reading them every bar makes the scan quadratic in the window.
        # They are sampled instead, and accumulated across samples: a sample
        # loses only objects that appear and leave entirely between two of
        # them.  The Swing hot set is bounded, but by far more confirmations
        # than one sample interval spans, so the bound costs the census
        # nothing that the sampling did not already cost it.
        if bars % POPULATION_SAMPLE_BARS == 0:
            for state in snapshot.timeframe_states.values():
                for swing in state.swing_hierarchy:
                    swings[swing.swing_id] = swing
                for candidate in state.liquidity.candidates:
                    levels[candidate.candidate_id] = candidate

    final = observer.last_market_snapshot
    if final is not None:
        for state in final.timeframe_states.values():
            for swing in state.swing_hierarchy:
                swings[swing.swing_id] = swing
            for candidate in state.liquidity.candidates:
                levels[candidate.candidate_id] = candidate

    events = observer.audit_store.events()
    by_kind = collections.Counter(event.kind.value for event in events)
    by_id = {event.event_id: event for event in events}

    # --- precision: cores published at the impulse, not at qualification
    cores = [
        event
        for event in events
        if event.kind is EventKind.BASE_ORIGIN_CORE_CREATED
    ]
    qualified = [
        event
        for event in events
        if event.kind is EventKind.QUALIFIED_ORIGIN_ZONE_CREATED
    ]
    qualified_core_ids = {
        str(event.evidence.get("base_origin_core_event_id"))
        for event in qualified
    }
    lead_seconds = [
        int(
            (
                event.known_at
                - by_id[str(event.evidence["base_origin_core_event_id"])].known_at
            ).total_seconds()
        )
        for event in qualified
        if str(event.evidence.get("base_origin_core_event_id")) in by_id
    ]

    # --- FVG first retest: how soon, and does age change the picture
    retests = [
        event for event in events if event.kind is EventKind.FVG_FIRST_RETEST
    ]
    ages = [int(event.evidence["age_bars"]) for event in retests]
    within = {
        f"within_{bucket}_bars": sum(1 for age in ages if age <= bucket)
        for bucket in RETEST_BUCKETS
    }
    created = sum(
        1 for event in events if event.kind is EventKind.FVG_CREATED
    )
    # A gap first retested in a different session than the one that made it
    # is a different animal from one retested minutes later; count them apart
    # rather than folding both into a bar-count threshold.
    cross_session = 0
    for event in retests:
        created_at = event.known_at - pd.Timedelta(
            seconds=int(event.evidence["age_seconds"])
        )
        if session_name_phase(created_at)[0] != event.evidence.get("session"):
            cross_session += 1
    depth_by_age: dict[str, list[float]] = collections.defaultdict(list)
    for event in retests:
        age = int(event.evidence["age_bars"])
        bucket = next(
            (f"<= {edge}" for edge in RETEST_BUCKETS if age <= edge),
            f"> {RETEST_BUCKETS[-1]}",
        )
        depth_by_age[bucket].append(float(event.evidence["fill_depth_at_entry"]))

    # --- structural ranges usable without a balance claim
    range_states = [
        event
        for event in events
        if event.kind
        in {EventKind.DEALING_RANGE_CREATED, EventKind.BALANCE_RANGE_MATURED}
    ]

    relation_spans = [g.observation_count for g in relation.values()]
    structure_breaks = [
        len(g.bos_event_ids) + len(g.mss_event_ids) for g in structure.values()
    ]

    return {
        "window": {"start": start, "end": end},
        "bars_replayed": bars,
        "population_sample_bars": POPULATION_SAMPLE_BARS,
        "last_bar_end": None if last_bar_end is None else str(last_bar_end),
        "efficiency": {
            "structure_generations": {
                **_span_summary(structure_breaks),
                "breaks_emitted": (
                    by_kind.get("qualified_bos", 0)
                    + by_kind.get("mss_core_confirmed", 0)
                ),
                "breaks_absorbed": sum(structure_breaks),
            },
            "relation_generations": {
                **_span_summary(relation_spans),
                "per_bar_rows": relation_rows,
                "compression_ratio": round(
                    relation_rows / max(len(relation), 1), 2
                ),
            },
            "swing_geometry": {
                "swings": len(swings),
                "by_depth": dict(
                    collections.Counter(
                        swing.geometric_depth for swing in swings.values()
                    )
                ),
                "nested": sum(
                    1
                    for swing in swings.values()
                    if swing.geometric_parent_id is not None
                ),
                "depths_per_semantic_rank": {
                    rank: sorted(depths)
                    for rank, depths in sorted(
                        {
                            swing.semantic_rank.value: {
                                other.geometric_depth
                                for other in swings.values()
                                if other.semantic_rank is swing.semantic_rank
                            }
                            for swing in swings.values()
                        }.items()
                    )
                },
            },
            "level_generations": {
                "levels": len(levels),
                "by_generation_ordinal": dict(
                    collections.Counter(
                        level.generation_ordinal for level in levels.values()
                    )
                ),
                "rearmed": sum(
                    1
                    for level in levels.values()
                    if level.generation_ordinal > 1
                ),
                "disarmed_now": sum(
                    1 for level in levels.values() if not level.is_armed
                ),
            },
        },
        "precision": {
            "base_origin_cores_published": len(cores),
            "cores_that_were_qualified": len(
                {
                    event.event_id
                    for event in cores
                    if event.event_id in qualified_core_ids
                }
            ),
            "qualified_origin_zones": len(qualified),
            "qualification_lead_seconds": {
                "min": min(lead_seconds) if lead_seconds else None,
                "median": _percentile(lead_seconds, 0.5),
                "max": max(lead_seconds) if lead_seconds else None,
            },
        },
        "fvg_first_retest": {
            "fvg_created": created,
            "first_retests": len(retests),
            "retested_fraction": (
                round(len(retests) / created, 4) if created else None
            ),
            "age_bars_at_first_retest": {
                "min": min(ages) if ages else None,
                "median": _percentile(ages, 0.5),
                "p90": _percentile(ages, 0.9),
                "max": max(ages) if ages else None,
            },
            "cumulative_within": within,
            "cumulative_within_fraction": {
                key: (round(value / len(retests), 4) if retests else None)
                for key, value in within.items()
            },
            "cross_session_first_retests": cross_session,
            "mean_fill_depth_by_age": {
                bucket: round(sum(values) / len(values), 4)
                for bucket, values in sorted(depth_by_age.items())
                if values
            },
        },
        "structural_range": {
            "created": by_kind.get("dealing_range_created", 0),
            "balance_matured": by_kind.get("balance_range_matured", 0),
            "balance_observed": by_kind.get("balance_range_observed", 0),
            "invalidated": by_kind.get("dealing_range_invalidated", 0),
            "transitions_seen": len(range_states),
        },
        "event_counts": dict(sorted(by_kind.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--model", default="configs/model.json")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = study(
        args.start,
        args.end,
        model_path=ROOT / args.model,
        source=ROOT / args.source,
    )
    destination = ROOT / args.out
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["efficiency"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
