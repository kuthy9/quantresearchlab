#!/usr/bin/env python3
"""One bounded, non-PnL real-data review of displacement episodes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.artifact_stream import sha256_file  # noqa: E402
from smc_trader.causal import CausalMarketReader  # noqa: E402
from smc_trader.displacement import (  # noqa: E402
    DisplacementLifecycle,
    DisplacementProtocol,
)
from smc_trader.displacement_observer import CausalDisplacementEye  # noqa: E402
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.model import (  # noqa: E402
    CORE_TIMEFRAMES,
    Timeframe,
    to_primitive,
)
from smc_trader.visualization import (  # noqa: E402
    BlindCandlePanelRenderer,
    _collision_safe_annotate,
)


DEFAULT_SOURCE = (
    ROOT / "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
)
DEFAULT_PROTOCOL = ROOT / "configs/primitives_displacement.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs/development/displacement_episode_review_202101_v2"
)
FROZEN_START = pd.Timestamp("2021-01-04T18:00:00", tz="America/New_York")
FROZEN_END = pd.Timestamp("2021-02-01T18:00:00", tz="America/New_York")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            to_primitive(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _selection_key(
    protocol_hash: str,
    stratum: str,
    entity_id: str,
    clock: pd.Timestamp,
) -> str:
    value = (
        f"{protocol_hash}|{stratum}|{entity_id}|{clock.isoformat()}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _case_record(
    *,
    kind: str,
    state: Any,
    clock: pd.Timestamp,
    transition_id: str | None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "clock": clock,
        "entity_id": state.entity_id,
        "direction": state.direction.value,
        "lifecycle": state.lifecycle.value,
        "reason": state.terminal_reason,
        "transition_id": transition_id,
        "atr0": float(state.atr0),
        "episode_bars": int(state.real_episode_bar_count),
        "state": to_primitive(state),
    }


def _scan(
    source: Path,
    protocol: DisplacementProtocol,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loaded = load_ohlcv(source, start=FROZEN_START, end=FROZEN_END)
    if not loaded.contract_selection_causal:
        raise RuntimeError("review requires previous-session causal contract selection")

    reader = CausalMarketReader()
    eye = CausalDisplacementEye(protocol)
    lifecycles: Counter[str] = Counter()
    direction_lifecycles: Counter[str] = Counter()
    terminal_reasons: Counter[str] = Counter()
    boundary_reasons: Counter[str] = Counter()
    compound_sequences: Counter[str] = Counter()
    started_ids: set[str] = set()
    active_ids: set[str] = set()
    terminal_ids: set[str] = set()
    activation_bars: list[float] = []
    terminal_bars: list[float] = []
    terminal_age_minutes: list[float] = []
    terminal_bars_by_id: dict[str, int] = {}
    candidates: list[dict[str, Any]] = []
    prior_state = None
    interruption_entries = 0
    resumes = 0
    progress_loss_same_direction_rearm = 0
    qualified_reverse_link_failures = 0
    source_rows = 0
    completed_m5 = 0

    for bar in iter_completed_bars(loaded.frame):
        source_rows += 1
        update = reader.on_bar(bar)
        completed_m5 += len(update.newly_completed.get(Timeframe.M5, ()))
        observation = eye.on_update(update)
        raw = eye.last_update
        if raw is None:
            raise RuntimeError("displacement eye lost its raw update")

        if len(raw.transitions) > 1:
            sequence = ">".join(
                item.state.lifecycle.value for item in raw.transitions
            )
            compound_sequences[sequence] += 1
        transition_states = tuple(item.state for item in raw.transitions)
        for left, right in zip(
            transition_states[:-1],
            transition_states[1:],
        ):
            if (
                left.terminal_reason == "confirmed_progress_loss"
                and right.lifecycle is DisplacementLifecycle.STARTED
                and left.direction is right.direction
            ):
                progress_loss_same_direction_rearm += 1
        for index, state in enumerate(transition_states):
            if state.terminal_reason != "qualified_opposite_displacement":
                continue
            followers = transition_states[index + 1 : index + 3]
            valid = bool(
                len(followers) == 2
                and followers[0].lifecycle is DisplacementLifecycle.STARTED
                and followers[1].lifecycle is DisplacementLifecycle.ACTIVE
                and followers[0].entity_id == followers[1].entity_id
                and followers[0].direction is followers[1].direction
                and followers[0].direction is not state.direction
                and followers[1].activation_gate_count == 6
            )
            qualified_reverse_link_failures += int(not valid)

        for transition in raw.transitions:
            state = transition.state
            lifecycle = state.lifecycle.value
            lifecycles[lifecycle] += 1
            direction_lifecycles[f"{state.direction.value}:{lifecycle}"] += 1
            if state.lifecycle is DisplacementLifecycle.STARTED:
                started_ids.add(state.entity_id)
            elif state.lifecycle is DisplacementLifecycle.ACTIVE:
                active_ids.add(state.entity_id)
                activation_bars.append(float(state.real_episode_bar_count))
            elif state.lifecycle in {
                DisplacementLifecycle.EXHAUSTED,
                DisplacementLifecycle.CENSORED,
            }:
                terminal_ids.add(state.entity_id)
                terminal_bars_by_id[state.entity_id] = int(
                    state.real_episode_bar_count
                )
                terminal_bars.append(float(state.real_episode_bar_count))
                terminal_age_minutes.append(
                    float(state.age_minutes_at_last_admitted)
                )
                reason = state.terminal_reason or "missing"
                terminal_reasons[reason] += 1
                if state.lifecycle is DisplacementLifecycle.CENSORED:
                    boundary_reasons[reason] += 1

            candidates.append(
                _case_record(
                    kind="transition",
                    state=state,
                    clock=observation.asof,
                    transition_id=transition.transition_id,
                )
            )

        state = raw.state
        if (
            state is not None
            and prior_state is not None
            and state.entity_id == prior_state.entity_id
        ):
            if prior_state.interruption_run == 0 and state.interruption_run == 1:
                interruption_entries += 1
            if prior_state.interruption_run > 0 and state.interruption_run == 0:
                resumes += 1
                candidates.append(
                    _case_record(
                        kind="interruption_resumed",
                        state=state,
                        clock=observation.asof,
                        transition_id=None,
                    )
                )
        prior_state = state

    active_atrs = [
        float(item["atr0"])
        for item in candidates
        if item["lifecycle"] == "active"
    ]
    atr_median = _percentile(active_atrs, 0.5)
    never_active_bars = [
        float(value)
        for entity_id, value in terminal_bars_by_id.items()
        if entity_id not in active_ids
    ]
    summary = {
        "role": "bounded_non_pnl_descriptive_displacement_review",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_role": loaded.source_role,
        "contract_selection_causal": loaded.contract_selection_causal,
        "source_warnings": loaded.warnings,
        "window": {
            "start_inclusive": FROZEN_START,
            "end_exclusive": FROZEN_END,
        },
        "protocol_hash": protocol.protocol_hash,
        "protocol_version": protocol.protocol_version,
        "downstream_authoritative_during_review": (
            protocol.downstream_authoritative
        ),
        "source_rows": source_rows,
        "completed_m5": completed_m5,
        "transition_counts": dict(lifecycles),
        "direction_lifecycle_counts": dict(direction_lifecycles),
        "terminal_reason_counts": dict(terminal_reasons),
        "boundary_reason_counts": dict(boundary_reasons),
        "same_update_compound_sequences": dict(compound_sequences),
        "identity_invariants": {
            "progress_loss_same_direction_same_clock_rearm": (
                progress_loss_same_direction_rearm
            ),
            "qualified_reverse_without_linked_opposite_active": (
                qualified_reverse_link_failures
            ),
        },
        "unique_entities": {
            "started": len(started_ids),
            "active": len(active_ids),
            "terminal": len(terminal_ids),
            "started_never_active": len(started_ids - active_ids),
        },
        "interruption": {
            "entered": interruption_entries,
            "resumed": resumes,
        },
        "activation_bars": {
            "min": _percentile(activation_bars, 0.0),
            "median": _percentile(activation_bars, 0.5),
            "p90": _percentile(activation_bars, 0.9),
            "max": _percentile(activation_bars, 1.0),
        },
        "terminal_episode_bars": {
            "min": _percentile(terminal_bars, 0.0),
            "median": _percentile(terminal_bars, 0.5),
            "p90": _percentile(terminal_bars, 0.9),
            "max": _percentile(terminal_bars, 1.0),
        },
        "terminal_age_minutes": {
            "median": _percentile(terminal_age_minutes, 0.5),
            "p90": _percentile(terminal_age_minutes, 0.9),
            "max": _percentile(terminal_age_minutes, 1.0),
        },
        "never_active_episode_bars": {
            "at_least_7_bars_30_minutes": sum(
                int(value >= 7) for value in never_active_bars
            ),
            "at_least_10_bars_45_minutes": sum(
                int(value >= 10) for value in never_active_bars
            ),
            "max": _percentile(never_active_bars, 1.0),
        },
        "active_atr0_median_for_blind_stratification": atr_median,
        "forbidden_fields_used": [],
        "threshold_search_performed": False,
    }
    return summary, candidates


def _strata(
    candidates: list[dict[str, Any]],
    atr_median: float,
) -> dict[str, dict[str, Any]]:
    slots: dict[str, dict[str, Any]] = {}
    for item in candidates:
        if pd.Timestamp(item["clock"]) < FROZEN_START + pd.Timedelta(hours=8):
            continue
        direction = str(item["direction"])
        lifecycle = str(item["lifecycle"])
        reason = item.get("reason")
        kind = str(item["kind"])
        strata: list[str] = []
        if lifecycle == "active":
            regime = "low_atr" if float(item["atr0"]) <= atr_median else "high_atr"
            strata.append(f"{direction}:active:{regime}")
        if kind == "interruption_resumed":
            strata.append(f"{direction}:interruption_resumed")
        if lifecycle == "exhausted" and reason in {
            "confirmed_progress_loss",
            "protection_broken",
            "qualified_opposite_displacement",
        }:
            strata.append(f"{direction}:exhausted:{reason}")
        for stratum in strata:
            priority = _selection_key(
                str(item["state"]["protocol_hash"]),
                stratum,
                str(item["entity_id"]),
                pd.Timestamp(item["clock"]),
            )
            prior = slots.get(stratum)
            if prior is None or priority < prior["selection_key"]:
                slots[stratum] = {
                    **item,
                    "stratum": stratum,
                    "selection_key": priority,
                }
    return slots


def _render_case(
    *,
    histories: dict[Timeframe, tuple[Any, ...]],
    case: dict[str, Any],
    destination: Path,
    reveal: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clock = pd.Timestamp(case["clock"])
    panels = BlindCandlePanelRenderer.panels(histories, clock)
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(15, 12),
        dpi=110,
        constrained_layout=True,
    )
    for axis, timeframe in zip(axes, CORE_TIMEFRAMES):
        first_index, candles = panels[timeframe]
        BlindCandlePanelRenderer.draw_candles(
            axis,
            candles,
            first_history_index=first_index,
            tick_size=0.25,
        )
        axis.axvline(
            len(candles) - 0.5,
            color="#111827",
            linewidth=0.8,
        )
        axis.set_title(
            f"{timeframe.value} completed through {candles[-1].end:%Y-%m-%d %H:%M %Z}",
            loc="left",
            fontsize=8,
        )

    opaque = hashlib.sha256(
        str(case["selection_key"]).encode("utf-8")
    ).hexdigest()[:12]
    if reveal:
        axis = axes[2]
        _, candles = panels[Timeframe.M5]
        original_ylim = tuple(float(value) for value in axis.get_ylim())
        state = case["state"]

        def clock_index(value: Any) -> int | None:
            if value is None:
                return None
            target = pd.Timestamp(value)
            return next(
                (
                    index
                    for index, candle in enumerate(candles)
                    if candle.start < target <= candle.end
                ),
                None,
            )

        for label, value, color in (
            ("START", state["started_at"], "#0369a1"),
            ("ACTIVE", state.get("active_at"), "#15803d"),
            ("FOCUS", case["clock"], "#7c3aed"),
        ):
            index = clock_index(value)
            if index is not None:
                axis.axvline(index, color=color, linestyle="-.", linewidth=0.8)
                _collision_safe_annotate(
                    axis,
                    label,
                    index,
                    candles[index].close,
                    color=color,
                    fontsize=5.2,
                )
        for label, value, color in (
            ("origin", float(state["origin_price"]), "#0369a1"),
            ("protection", float(state["protection_price"]), "#b91c1c"),
        ):
            if original_ylim[0] <= value <= original_ylim[1]:
                axis.hlines(
                    value,
                    0,
                    len(candles) - 1,
                    color=color,
                    linestyle=":" if label == "origin" else "--",
                    linewidth=0.7,
                )
                _collision_safe_annotate(
                    axis,
                    f"{label} {value:.2f}",
                    len(candles) - 1,
                    value,
                    color=color,
                    fontsize=5.0,
                    preferred_side="left",
                )
        axis.set_ylim(*original_ylim)
        figure.suptitle(
            "DISPLACEMENT REVEAL — "
            f"{case['stratum']} · {str(case['entity_id'])[:8]} · "
            f"bars={case['episode_bars']}",
            fontsize=11,
        )
    else:
        figure.suptitle(
            f"BLIND DISPLACEMENT CASE {opaque} — no labels or future path",
            fontsize=11,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def _render_selected(
    *,
    source: Path,
    cases: tuple[dict[str, Any], ...],
    output: Path,
) -> tuple[dict[str, Any], ...]:
    by_clock: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_clock[pd.Timestamp(case["clock"])].append(case)
    last_clock = max(by_clock)
    loaded = load_ohlcv(source, start=FROZEN_START, end=last_clock)
    reader = CausalMarketReader()
    artifacts: list[dict[str, Any]] = []
    for bar in iter_completed_bars(loaded.frame):
        update = reader.on_bar(bar)
        selected = by_clock.get(update.asof, ())
        if not selected:
            continue
        histories = {
            timeframe: reader.window(timeframe, 80)
            for timeframe in CORE_TIMEFRAMES
        }
        for case in selected:
            opaque = hashlib.sha256(
                str(case["selection_key"]).encode("utf-8")
            ).hexdigest()[:12]
            blind = output / "blind" / f"{opaque}.png"
            reveal = output / "reveal" / f"{opaque}.png"
            _render_case(
                histories=histories,
                case=case,
                destination=blind,
                reveal=False,
            )
            _render_case(
                histories=histories,
                case=case,
                destination=reveal,
                reveal=True,
            )
            artifacts.append(
                {
                    "opaque_case_id": opaque,
                    "stratum": case["stratum"],
                    "clock": case["clock"],
                    "entity_id": case["entity_id"],
                    "blind_image": str(blind),
                    "blind_sha256": sha256_file(blind),
                    "reveal_image": str(reveal),
                    "reveal_sha256": sha256_file(reveal),
                    "maximum_market_time": update.asof,
                    "future_visible": False,
                    "state": case["state"],
                }
            )
    if len(artifacts) != len(cases):
        raise RuntimeError(
            f"rendered {len(artifacts)} of {len(cases)} selected cases"
        )
    return tuple(sorted(artifacts, key=lambda item: item["opaque_case_id"]))


def run(source: Path, protocol_path: Path, output: Path) -> None:
    protocol = DisplacementProtocol.from_file(protocol_path)
    if protocol.downstream_authoritative:
        raise RuntimeError("semantic review must run before downstream authority")
    summary, candidates = _scan(source, protocol)
    atr_median = summary["active_atr0_median_for_blind_stratification"]
    if atr_median is None:
        raise RuntimeError("fixed review window produced no active episodes")
    slots = _strata(candidates, float(atr_median))
    selected = tuple(
        slots[key]
        for key in sorted(slots)
    )
    artifacts = _render_selected(
        source=source,
        cases=selected,
        output=output,
    )
    summary["selected_case_count"] = len(artifacts)
    summary["selected_strata"] = [item["stratum"] for item in artifacts]
    _write_json(output / "summary.json", summary)
    _write_json(output / "cases.json", artifacts)
    _write_json(
        output / "BLIND_REVIEW_TEMPLATE.json",
        {
            "status": "pending",
            "instructions": (
                "Review only blind_image first; freeze per-case episode start, "
                "continuity and terminal verdicts before opening reveal_image."
            ),
            "cases": [
                {
                    "opaque_case_id": item["opaque_case_id"],
                    "start_timing": None,
                    "continuity": None,
                    "terminal_timing": None,
                    "systematic_misread": None,
                    "notes": None,
                }
                for item in artifacts
            ],
        },
    )
    print(json.dumps(to_primitive(summary), indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.source, args.protocol, args.output)
