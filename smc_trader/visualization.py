"""Causal scale-registry decision views and price/structure overlays."""
from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path
import textwrap
from typing import Any, Mapping, Sequence

import pandas as pd

from .model import (
    Action,
    Candle,
    DealingRangeLifecycle,
    EngineSnapshot,
    EventKind,
    FVGQualification,
    FairValueGapLifecycle,
    LiquidityInventoryLifecycle,
    ManipulationLifecycle,
    OrderBlockLifecycle,
    PlaybookPhase,
    Timeframe,
)
from .market_clock import scheduled_gap_kind


@dataclass(frozen=True)
class VisualArtifact:
    path: Path
    kind: str
    decision_id: str
    maximum_market_time: pd.Timestamp
    hypothesis_key: str | None
    setup_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None


@dataclass(frozen=True)
class _ObservationSnapshot:
    """Adapter for reusing causal overlays without constructing the Brain."""

    observation: Any


def _candles(axis, candles: Sequence[Candle]) -> None:
    from matplotlib.patches import Rectangle

    for index, candle in enumerate(candles):
        up = candle.close >= candle.open
        color = "#0f766e" if up else "#b91c1c"
        axis.vlines(index, candle.low, candle.high, color="#334155", linewidth=0.7)
        bottom = min(candle.open, candle.close)
        height = abs(candle.close - candle.open)
        if height < 1e-12:
            axis.hlines(candle.open, index - 0.30, index + 0.30, color=color, linewidth=1)
        else:
            axis.add_patch(
                Rectangle(
                    (index - 0.30, bottom),
                    0.60,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.4,
                )
            )
    if candles:
        ticks = sorted(set([0, len(candles) // 2, len(candles) - 1]))
        axis.set_xticks(ticks)
        axis.set_xticklabels(
            [candles[index].start.strftime("%m-%d\n%H:%M") for index in ticks],
            fontsize=7,
        )
    axis.grid(True, color="#dbe4ee", linewidth=0.4, alpha=0.7)


def active_causal_timeframes(observation: Any) -> tuple[Timeframe, ...]:
    """Return the ordered scale registry bound to this observation."""

    active = tuple(
        getattr(observation, "active_timeframes", ())
        or tuple(observation.frames)
    )
    if (
        not active
        or len(active) != len(set(active))
        or set(active) != set(observation.frames)
        or Timeframe.M1 not in active
    ):
        raise ValueError(
            "observation active timeframes disagree with its frame mapping"
        )
    return active


def validate_causal_histories(
    snapshot: EngineSnapshot,
    histories: Mapping[Timeframe, Sequence[Any]],
) -> dict[Timeframe, tuple[Candle, ...]]:
    """Reject incomplete, future, wrong-contract or discontinuous panels."""

    active = active_causal_timeframes(snapshot.observation)
    if set(histories) != set(active):
        expected = ", ".join(timeframe.value for timeframe in active)
        raise ValueError(
            "causal history must match the enabled scale registry: "
            f"{expected}"
        )
    output: dict[Timeframe, tuple[Candle, ...]] = {}
    for timeframe in active:
        values = tuple(histories[timeframe])
        frame = snapshot.observation.frame(timeframe)
        if any(
            not isinstance(item, Candle)
            or item.timeframe is not timeframe
            or not item.complete
            or item.end > snapshot.observation.asof
            or (item.symbol, item.instrument_id)
            != (
                snapshot.observation.symbol,
                snapshot.observation.instrument_id,
            )
            for item in values
        ):
            raise ValueError(
                "causal history contains a wrong, incomplete or future candle"
            )
        if (
            len({(item.start, item.end) for item in values}) != len(values)
            or tuple(sorted(values, key=lambda item: (item.start, item.end)))
            != values
            or any(
                right.start < left.end
                for left, right in zip(values[:-1], values[1:])
            )
            or any(
                right.start != left.end
                and scheduled_gap_kind(left.end, right.start) is None
                for left, right in zip(values[:-1], values[1:])
            )
        ):
            raise ValueError(
                "causal history is duplicated, unordered or has an "
                "unexplained gap"
            )
        if not values:
            if (
                frame.bars != 0
                or frame.ready
                or frame.cutoff != snapshot.observation.asof
            ):
                raise ValueError(
                    "empty causal history disagrees with observation frame"
                )
            output[timeframe] = values
            continue
        if values[-1].end != frame.cutoff:
            raise ValueError(
                "causal history cutoff disagrees with observation"
            )
        output[timeframe] = values
    if (
        not output[Timeframe.M1]
        or output[Timeframe.M1][-1].end != snapshot.observation.asof
    ):
        raise ValueError("causal M1 history does not reach the decision clock")
    return output


def _blank_causal_panel(
    axis: Any,
    timeframe: Timeframe,
    cutoff: pd.Timestamp,
    *,
    title_suffix: str = "",
) -> None:
    """Render an explicit no-history state after a causal reset."""

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(())
    axis.set_yticks(())
    axis.grid(False)
    axis.text(
        0.5,
        0.5,
        (
            "BLANK / WARMUP\n"
            f"0 completed {timeframe.value} candles in current causal history"
        ),
        ha="center",
        va="center",
        fontsize=9,
        color="#64748b",
        transform=axis.transAxes,
    )
    axis.set_title(
        (
            f"{timeframe.value}{title_suffix} · blank/warmup · "
            f"causal cutoff {cutoff:%Y-%m-%d %H:%M %Z}"
        ),
        loc="left",
        fontsize=9,
    )


def _selected_hypothesis(snapshot: EngineSnapshot):
    selected_key = snapshot.decision.best_hypothesis_key
    if selected_key is not None:
        belief = snapshot.belief.resolve_hypothesis(selected_key)
        if belief is None:
            raise ValueError("selected visual hypothesis identity is absent")
        return belief
    ranked = snapshot.belief.ranked()
    return None if not ranked else ranked[0]


def _belief_identity(
    belief: Any,
) -> tuple[str | None, str | None, str | None]:
    if belief is None:
        return None, None, None
    plan = belief.plan
    sequence_setup = (
        None if belief.sequence is None else belief.sequence.setup_id
    )
    plan_setup = None if plan is None else plan.setup_id
    if (
        sequence_setup is not None
        and plan_setup is not None
        and sequence_setup != plan_setup
    ):
        raise ValueError("visual hypothesis setup identities disagree")
    belief_location = belief.entry_location_id
    plan_location = None if plan is None else plan.entry_location_id
    if (
        belief_location is not None
        and plan_location is not None
        and belief_location != plan_location
    ):
        raise ValueError(
            "visual hypothesis entry-location identities disagree"
        )
    return (
        sequence_setup or plan_setup,
        belief_location or plan_location,
        None if plan is None else plan.entry_path_id,
    )


def _plan_display_context(
    snapshot: EngineSnapshot,
    belief: Any,
) -> tuple[str, bool]:
    """Describe plan actionability and whether price panels may show its levels."""

    if belief is None or belief.plan is None:
        return "NO COMPLETE CAUSAL PLAN", False
    if belief.phase in {
        PlaybookPhase.COMPLETED,
        PlaybookPhase.INVALIDATED,
        PlaybookPhase.INACTIVE,
    }:
        return (
            f"HISTORICAL PLAN — NON-ACTIONABLE ({belief.phase.value})",
            False,
        )
    if snapshot.risk.final_action is Action.ENTER:
        return "RISK-APPROVED ENTRY PLAN", True
    if snapshot.risk.final_action in {
        Action.HOLD,
        Action.PROTECT,
        Action.EXIT,
    } or belief.phase in {
        PlaybookPhase.ENTERED,
        PlaybookPhase.WEAKENING,
        PlaybookPhase.DELIVERING,
    }:
        return (
            f"POSITION-MANAGEMENT FROZEN PLAN ({snapshot.risk.final_action.value})",
            True,
        )
    if snapshot.decision.selected_action is Action.WAIT:
        return f"MONITORED SETUP PLAN — WAIT ({belief.phase.value})", True
    return (
        f"NON-EXECUTED HYPOTHESIS PLAN — "
        f"{snapshot.risk.final_action.value.upper()} ({belief.phase.value})",
        True,
    )


def _format_evidence(
    snapshot: EngineSnapshot,
) -> str:
    belief = _selected_hypothesis(snapshot)
    if belief is None:
        return "No selected playbook hypothesis"
    supporting = "\n".join(
        f"+ {item.primitive}: {item.value:.2f}" for item in belief.supporting[:3]
    ) or "+ none"
    contradicting = "\n".join(
        f"− {item.primitive}: {item.value:.2f}" for item in belief.contradicting[:3]
    ) or "− none"
    typed = belief.thesis_strength is not None
    score_text = (
        f"thesis: {belief.thesis_strength:.3f}\n"
        f"effective/action readiness: {belief.effective_probability:.3f}"
        if typed
        else (
            f"legacy probability: {belief.probability:.3f} "
            f"(raw {belief.raw_probability:.3f})"
            if belief.raw_probability is not None
            else f"legacy probability: {belief.probability:.3f}"
        )
    )
    qualities = "\n".join(
        f"{name}: {'n/a' if value is None else f'{value:.3f}'}"
        for name, value in (
            ("thesis", belief.thesis_strength),
            ("sequence", belief.sequence_progress),
            ("location", belief.location_quality),
            ("readiness", belief.entry_readiness),
            ("delivery", belief.delivery_quality),
        )
    )
    groups = (
        "\n".join(
            f"{name}: {value:.3f}"
            for name, value in belief.evidence_group_scores.items()
        )
        or "legacy/untyped"
    )
    gates = (
        "\n".join(
            f"{'✓' if passed else '×'} {name}"
            for name, passed in belief.hard_gate_results.items()
        )
        or "legacy/untyped"
    )
    return (
        f"{belief.playbook.value} / {belief.direction.value}\n"
        f"phase: {belief.phase.value}\n"
        f"{score_text}\n"
        f"uncertainty: {belief.uncertainty:.3f}\n"
        f"{qualities}\n\n"
        f"EVIDENCE GROUPS\n{groups}\n\n"
        f"HARD GATES\n{gates}\n\n"
        f"SUPPORT\n{supporting}\n\nAGAINST\n{contradicting}"
    )


def _selected_group5_entities(
    snapshot: EngineSnapshot,
    belief: Any,
    focus_entity_id: str | None = None,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    if belief is None:
        # Eye-only audits have no Brain hypothesis to select a setup.  The
        # renderer must still show the typed states actually emitted by the
        # Observer; this changes presentation only and grants no action
        # authority to Group 5.
        locations = tuple(snapshot.observation.entry_locations)
        paths = tuple(snapshot.observation.path_sequences)
        reacceptances = tuple(snapshot.observation.qualified_reacceptances)
        micro_bos = tuple(snapshot.observation.micro_bos_references)
        if focus_entity_id is None:
            return locations, paths, reacceptances, micro_bos

        def promote(values: tuple[Any, ...], predicate: Any) -> tuple[Any, ...]:
            return (
                *(value for value in values if not predicate(value)),
                *(value for value in values if predicate(value)),
            )

        focused_paths = tuple(
            path
            for path in paths
            if (
                path.sequence_id == focus_entity_id
                or path.context_id == focus_entity_id
            )
        )
        focus_context_ids = {
            focus_entity_id,
            *(path.context_id for path in focused_paths),
        }
        locations = promote(
            locations,
            lambda item: item.location_id in focus_context_ids,
        )
        paths = promote(
            paths,
            lambda item: (
                item.sequence_id == focus_entity_id
                or item.context_id in focus_context_ids
            ),
        )
        # A context may own several live or terminal paths.  Context peers are
        # useful surrounding evidence, but the exact frozen audit identity must
        # remain the final item consumed by the compact overlay/text views.
        paths = promote(
            paths,
            lambda item: item.sequence_id == focus_entity_id,
        )
        reacceptances = promote(
            reacceptances,
            lambda item: item.context_id in focus_context_ids,
        )
        micro_bos = promote(
            micro_bos,
            lambda item: item.context_id in focus_context_ids,
        )
        return locations, paths, reacceptances, micro_bos
    plan = belief.plan
    location_ids = {
        value
        for value in (
            belief.entry_location_id,
            None if plan is None else plan.entry_location_id,
        )
        if value is not None
    }
    setup_context_ids = {
        value
        for value in (
            belief.setup_context_id,
            *location_ids,
        )
        if value is not None
    }
    entry_path_id = None if plan is None else plan.entry_path_id
    locations = tuple(
        item
        for item in snapshot.observation.entry_locations
        if item.location_id in location_ids
    )
    paths = tuple(
        item
        for item in snapshot.observation.path_sequences
        if (
            (entry_path_id is not None and item.sequence_id == entry_path_id)
            or item.sequence_id in setup_context_ids
            or item.context_id in setup_context_ids
        )
    )
    context_ids = setup_context_ids | {
        item.context_id for item in paths
    }
    reacceptances = tuple(
        item
        for item in snapshot.observation.qualified_reacceptances
        if item.context_id in context_ids
    )
    micro_bos = tuple(
        item
        for item in snapshot.observation.micro_bos_references
        if item.context_id in context_ids
    )
    return locations, paths, reacceptances, micro_bos


def _metric_text(
    snapshot: EngineSnapshot,
    timeframe: Timeframe,
    belief: Any,
    focus_entity_id: str | None = None,
) -> str:
    metrics = snapshot.observation.frame(timeframe).metrics
    if (
        timeframe is Timeframe.M5
        and snapshot.observation.displacement is not None
    ):
        displacement = snapshot.observation.displacement
        rows = [
            "typed displacement: "
            + (
                "none"
                if displacement is None
                else (
                    f"{displacement.lifecycle} / "
                    f"{'none' if displacement.current_direction is None else displacement.current_direction.value}"
                )
            ),
            "descriptive proxy: compression (not mature range): "
            f"{metrics['compression']:+.2f}",
        ]
        if displacement is not None:
            rows.extend(
                f"{name}: {value:+.2f}"
                for name, value in tuple(
                    displacement.current_metrics or ()
                )[:5]
            )
        locations, _, _, _ = _selected_group5_entities(
            snapshot,
            belief,
            focus_entity_id,
        )
        if locations:
            latest = locations[-1]
            rows.append(
                f"frozen zone: {latest.source_zone_kind} / "
                f"{latest.lifecycle.value}"
            )
            rows.append(
                "first pullback: "
                + (
                    "pending"
                    if latest.first_entered_at is None
                    else latest.first_entered_at.strftime("%H:%M")
                )
            )
        return "\n".join(rows)
    if timeframe is Timeframe.M1 and snapshot.observation.group5_typed_available:
        _, paths, reacceptances, micro_bos_references = (
            _selected_group5_entities(
                snapshot,
                belief,
                focus_entity_id,
            )
        )
        path = None if not paths else paths[-1]
        reacceptance = (
            None if not reacceptances else reacceptances[-1]
        )
        micro_bos = (
            None
            if not micro_bos_references
            else micro_bos_references[-1]
        )
        return "\n".join(
            (
                "typed path: "
                + (
                    "none"
                    if path is None
                    else " → ".join(step.kind for step in path.steps[-4:])
                ),
                f"acceleration: {metrics['acceleration']:+.2f}",
                f"counter pressure: {metrics['counter_pressure']:+.2f}",
                "reacceptance: "
                + (
                    "none"
                    if reacceptance is None
                    else reacceptance.lifecycle.value
                ),
                "micro BOS: "
                + (
                    "none"
                    if micro_bos is None
                    else (
                        f"{micro_bos.outcome} / "
                        f"{'qualified' if micro_bos.qualified else 'not-qualified'}"
                    )
                ),
            )
        )
    if timeframe is Timeframe.H4:
        values = (
            (
                "descriptive proxy: directional move "
                "(not typed displacement)",
                metrics["directional_displacement"],
            ),
            ("efficiency", metrics["path_efficiency"]),
            ("structure progression", metrics["structure_direction"]),
            ("structure age", metrics["structure_age_bars"]),
            ("range position", metrics["range_position"]),
            ("external above ATR", metrics["external_above_distance_atr"]),
            ("external below ATR", metrics["external_below_distance_atr"]),
        )
    elif timeframe in {Timeframe.H1, Timeframe.M15}:
        values = (
            ("swing progression", metrics["swing_progression"]),
            (
                "descriptive proxy: acceptance",
                metrics["acceptance_direction"],
            ),
            (
                "descriptive proxy: rejection",
                metrics["rejection_direction"],
            ),
            (
                "rolling envelope position",
                metrics["rolling_range_position"],
            ),
            ("up obstruction ATR", metrics["up_path_obstruction_atr"]),
            ("down obstruction ATR", metrics["down_path_obstruction_atr"]),
        )
    elif timeframe is Timeframe.M5:
        values = (
            (
                "descriptive proxy: compression (not mature range)",
                metrics["compression"],
            ),
        )
        return "\n".join(
            (
                "typed displacement: unavailable",
                *(f"{name}: {value:+.2f}" for name, value in values),
            )
        )
    else:
        values = (
            ("acceleration", metrics["acceleration"]),
            ("counter pressure", metrics["counter_pressure"]),
        )
        return "\n".join(
            (
                "typed path: unavailable (no swing-progression proxy)",
                *(f"{name}: {value:+.2f}" for name, value in values),
            )
        )
    return "\n".join(f"{name}: {value:+.2f}" for name, value in values)


def _event_markers(
    axis: Any,
    snapshot: EngineSnapshot,
    timeframe: Timeframe,
    candles: Sequence[Candle],
) -> None:
    if not candles:
        return
    colors = {
        EventKind.SWING_FORMED: "#7c3aed",
        EventKind.SWING_STATE: "#7c3aed",
        EventKind.STRUCTURE_STATE: "#1d4ed8",
        EventKind.BOS_STATE: "#60a5fa",
        EventKind.BOS_POST_BREAK_STATE: "#3b82f6",
        EventKind.SUPPORT_RESISTANCE_STATE: "#0f766e",
        EventKind.LIQUIDITY_POOL_STATE: "#a21caf",
        EventKind.LIQUIDITY_SWEEP: "#db2777",
        EventKind.LIQUIDITY_CONSUMED: "#9d174d",
        EventKind.LIQUIDITY_RETIRED: "#94a3b8",
        EventKind.STRUCTURE_BREAK: "#2563eb",
        EventKind.STRUCTURE_BREAK_FAILED: "#94a3b8",
        EventKind.FVG_STATE: "#0284c7",
        EventKind.ORDER_BLOCK_STATE: "#9333ea",
        EventKind.DEALING_RANGE_STATE: "#ca8a04",
        EventKind.MANIPULATION_STATE: "#c2410c",
        EventKind.ENTRY_PATH_STATE: "#0369a1",
    }
    visible = [
        event
        for event in snapshot.observation.recent_events
        if event.timeframe is timeframe
        and candles[0].start <= event.observed_at <= candles[-1].end
    ][-8:]
    # Mark every visible event, but keep dense execution panels to one direct
    # annotation. Full labels and durations remain in the ordered-memory panel.
    annotation_count = 1 if timeframe in {Timeframe.M1, Timeframe.M5} else 2
    annotation_ids = {event.event_id for event in visible[-annotation_count:]}
    visible_low = min(candle.low for candle in candles)
    visible_high = max(candle.high for candle in candles)
    for event in visible:
        index = next(
            (
                number
                for number, candle in enumerate(candles)
                if candle.start < event.observed_at <= candle.end
            ),
            None,
        )
        if index is None:
            continue
        candle = candles[index]
        price = candle.close if event.price is None else event.price
        if not visible_low <= price <= visible_high:
            continue
        duration = snapshot.observation.event_durations_minutes.get(event.event_id, 0)
        lifecycle = (
            ""
            if event.lifecycle is None
            else f" · {event.lifecycle}"
        )
        label = (
            f"{event.kind.value}{lifecycle} "
            f"[{_short_identity(event.event_id)}]\n"
            f"persist {duration}m"
        )
        color = colors.get(event.kind, "#475569")
        axis.scatter(
            [index],
            [price],
            s=18,
            color=color,
            zorder=5,
        )
        if event.event_id not in annotation_ids:
            continue
        _collision_safe_annotate(
            axis,
            label,
            index,
            price,
            color=color,
            fontsize=5.5,
            priority=True,
        )


def _candle_index(
    candles: Sequence[Candle],
    clock: pd.Timestamp | None,
) -> int | None:
    if clock is None:
        return None
    return next(
        (
            index
            for index, candle in enumerate(candles)
            if candle.start < clock <= candle.end
        ),
        None,
    )


def _short_identity(*values: Any) -> str:
    """Return one compact, stable object identity for chart annotations."""

    for value in values:
        if isinstance(value, str) and value:
            prefix, separator, identifier = value.rpartition(":")
            if separator and prefix and identifier:
                return f"{prefix}:{identifier[:8]}"
            return value[:8]
    return "unknown"


def _optional_score(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _wrap_panel_text(value: str, *, width: int) -> str:
    """Wrap long identities/reasons before constrained-layout measures them."""

    rows: list[str] = []
    for raw in value.splitlines():
        if not raw:
            rows.append("")
            continue
        leading = raw[: len(raw) - len(raw.lstrip())]
        rows.extend(
            textwrap.wrap(
                raw,
                width=width,
                initial_indent="",
                subsequent_indent=f"{leading}  ",
                break_long_words=True,
                break_on_hyphens=False,
                replace_whitespace=False,
            )
            or [""]
        )
    return "\n".join(rows)


def _collision_safe_annotate(
    axis: Any,
    text: str,
    x: float,
    y: float,
    *,
    color: str,
    fontsize: float = 5.1,
    preferred_side: str | None = None,
    priority: bool = False,
) -> None:
    """Place a label in a shared axes lane without covering earlier labels.

    All overlays use the same registry on the axes, so BOS, zones, paths,
    events and plan levels cannot independently choose the same right-edge
    location.  The data point remains exact; only the label is displaced and
    joined to it by a leader line.
    """

    budget = getattr(axis, "_smc_annotation_budget", None)
    annotation_count = int(
        getattr(axis, "_smc_annotation_count", 0)
    )
    if budget is not None and annotation_count >= int(budget) and not priority:
        axis._smc_annotation_omitted = int(
            getattr(axis, "_smc_annotation_omitted", 0)
        ) + 1
        return

    point = axis.transAxes.inverted().transform(
        axis.transData.transform((float(x), float(y)))
    )
    x_fraction = min(1.0, max(0.0, float(point[0])))
    y_fraction = min(1.0, max(0.0, float(point[1])))
    side = preferred_side or ("left" if x_fraction >= 0.64 else "right")
    longest = max((len(line) for line in text.splitlines()), default=1)
    width = min(0.40, max(0.11, longest * 0.0064))
    height = min(0.22, max(0.045, len(text.splitlines()) * 0.038))
    if side == "left":
        text_x = min(0.96, max(width + 0.02, x_fraction - 0.025))
        ha = "right"
        horizontal_lanes = tuple(
            max(width + 0.02, text_x - lane * (width + 0.025))
            for lane in range(3)
        )
    else:
        text_x = min(0.96 - width, max(0.02, x_fraction + 0.025))
        ha = "left"
        horizontal_lanes = tuple(
            min(0.96 - width, text_x + lane * (width + 0.025))
            for lane in range(3)
        )

    registry = getattr(axis, "_smc_annotation_boxes", None)
    if registry is None:
        registry = []
        setattr(axis, "_smc_annotation_boxes", registry)
    vertical_offsets = (
        0.045,
        -0.055,
        0.115,
        -0.125,
        0.185,
        -0.195,
        0.255,
        -0.265,
        0.325,
        -0.335,
    )
    chosen_y = min(0.97 - height / 2.0, max(0.03 + height / 2.0, y_fraction))
    chosen_box = None
    for candidate_x in dict.fromkeys(horizontal_lanes):
        for offset in vertical_offsets:
            candidate_y = min(
                0.97 - height / 2.0,
                max(0.03 + height / 2.0, y_fraction + offset),
            )
            left = candidate_x - width if ha == "right" else candidate_x
            candidate = (
                left - 0.008,
                candidate_y - height / 2.0 - 0.008,
                left + width + 0.008,
                candidate_y + height / 2.0 + 0.008,
            )
            if not any(
                candidate[0] < prior[2]
                and candidate[2] > prior[0]
                and candidate[1] < prior[3]
                and candidate[3] > prior[1]
                for prior in registry
            ):
                text_x = candidate_x
                chosen_y = candidate_y
                chosen_box = candidate
                break
        if chosen_box is not None:
            break
    if chosen_box is None:
        # The local lanes can fill on dense 1m/5m charts.  Search the complete
        # in-axes label grid before using the deterministic exterior overflow
        # gutter.  No unchecked overlap is ever accepted.
        minimum_y = 0.03 + height / 2.0
        maximum_y = 0.97 - height / 2.0
        y_step = height + 0.018
        y_positions: list[float] = []
        candidate_y = minimum_y
        while candidate_y <= maximum_y + 1e-12:
            y_positions.append(candidate_y)
            candidate_y += y_step
        if maximum_y not in y_positions:
            y_positions.append(maximum_y)

        x_step = width + 0.018
        left_positions: list[float] = []
        candidate_left = 0.02
        maximum_left = 0.98 - width
        while candidate_left <= maximum_left + 1e-12:
            left_positions.append(candidate_left)
            candidate_left += x_step
        if maximum_left not in left_positions:
            left_positions.append(maximum_left)

        grid_candidates = sorted(
            (
                (candidate_left, candidate_y)
                for candidate_left in dict.fromkeys(left_positions)
                for candidate_y in dict.fromkeys(y_positions)
            ),
            key=lambda candidate: (
                abs((candidate[0] + width / 2.0) - x_fraction)
                + abs(candidate[1] - y_fraction),
                abs(candidate[1] - y_fraction),
                candidate[1],
                candidate[0],
            ),
        )
        for candidate_left, candidate_y in grid_candidates:
            candidate = (
                candidate_left - 0.008,
                candidate_y - height / 2.0 - 0.008,
                candidate_left + width + 0.008,
                candidate_y + height / 2.0 + 0.008,
            )
            if not any(
                candidate[0] < prior[2]
                and candidate[2] > prior[0]
                and candidate[1] < prior[3]
                and candidate[3] > prior[1]
                for prior in registry
            ):
                text_x = candidate_left
                chosen_y = candidate_y
                ha = "left"
                chosen_box = candidate
                break

    if chosen_box is None and budget is not None:
        axis._smc_annotation_omitted = int(
            getattr(axis, "_smc_annotation_omitted", 0)
        ) + 1
        return
    overflow = chosen_box is None
    if overflow:
        # A fully occupied plot uses a left-side overflow gutter.  Its running
        # edge is derived from prior label widths, so differently sized labels
        # remain disjoint and the saved figure can expand via bbox_inches=tight.
        overflow_right = float(
            getattr(axis, "_smc_annotation_overflow_left", -0.08)
        )
        text_x = overflow_right
        chosen_y = min(
            0.97 - height / 2.0,
            max(0.03 + height / 2.0, y_fraction),
        )
        ha = "right"
        chosen_box = (
            overflow_right - width - 0.008,
            chosen_y - height / 2.0 - 0.008,
            overflow_right + 0.008,
            chosen_y + height / 2.0 + 0.008,
        )
        setattr(
            axis,
            "_smc_annotation_overflow_left",
            chosen_box[0] - 0.025,
        )
    registry.append(chosen_box)
    axis._smc_annotation_count = annotation_count + 1
    axis.annotate(
        text,
        xy=(x, y),
        xycoords="data",
        xytext=(text_x, chosen_y),
        textcoords=axis.transAxes,
        ha=ha,
        va="center",
        fontsize=fontsize,
        color=color,
        annotation_clip=not overflow,
        clip_on=not overflow,
        bbox={
            "facecolor": "white",
            "edgecolor": color,
            "alpha": 0.78,
            "pad": 0.8,
        },
        arrowprops={
            "arrowstyle": "-",
            "lw": 0.4,
            "color": color,
            "alpha": 0.8,
        },
        zorder=9,
    )


def _typed_structure_overlay(
    axis: Any,
    snapshot: EngineSnapshot,
    timeframe: Timeframe,
    candles: Sequence[Candle],
) -> None:
    """Directly mark frozen swings, BOS, S/R and liquidity pools."""

    if not candles:
        return
    frame = snapshot.observation.frame(timeframe)
    visible_low, visible_high = axis.get_ylim()
    swings = [
        item
        for item in frame.swings
        if item.confirmed_at is not None
        and visible_low <= item.price <= visible_high
        and _candle_index(candles, item.confirmed_at) is not None
    ][-6:]
    for offset, item in enumerate(swings):
        index = _candle_index(candles, item.confirmed_at)
        assert index is not None
        marker = "^" if item.side.value == "high" else "v"
        color = "#7c3aed" if item.lifecycle.value != "broken" else "#94a3b8"
        axis.scatter(
            [index],
            [item.price],
            marker=marker,
            s=24,
            facecolors="none",
            edgecolors=color,
            linewidths=0.8,
            zorder=6,
        )
        if offset >= max(0, len(swings) - 2):
            _collision_safe_annotate(
                axis,
                (
                    f"{item.relation.value.upper()} "
                    f"{item.lifecycle.value} · {item.age_bars}b "
                    f"[{_short_identity(item.swing_id)}]"
                ),
                index,
                item.price,
                color=color,
                fontsize=5.2,
            )

    breaks = [
        item
        for item in frame.structure_breaks
        if visible_low <= item.target_price <= visible_high
        and _candle_index(
            candles,
            item.resolved_at or item.pending_at,
        )
        is not None
    ][-4:]
    for offset, item in enumerate(breaks):
        clock = item.resolved_at or item.pending_at
        index = _candle_index(candles, clock)
        assert index is not None
        color = (
            "#2563eb"
            if item.lifecycle.value == "confirmed"
            else "#dc2626"
            if item.lifecycle.value == "failed"
            else "#64748b"
        )
        axis.scatter(
            [index],
            [item.target_price],
            marker="x",
            s=26,
            color=color,
            linewidths=0.9,
            zorder=6,
        )
        if offset >= max(0, len(breaks) - 2):
            _collision_safe_annotate(
                axis,
                (
                    f"BOS {item.direction.value[0].upper()} "
                    f"{item.lifecycle.value}/{item.scope.value} "
                    f"post={None if item.post_break_state is None else item.post_break_state.value} "
                    f"mss={item.mss_qualified} "
                    f"[{_short_identity(item.bos_id)}] "
                    f"→ {_short_identity(item.target_swing_id)}"
                ),
                index,
                item.target_price,
                color=color,
                fontsize=5.1,
            )

    zones = tuple(frame.support_resistance)[-3:]
    for offset, state in enumerate(zones):
        lower = max(float(state.lower_bound), visible_low)
        upper = min(float(state.upper_bound), visible_high)
        if lower >= upper:
            continue
        start_index = _candle_index(candles, state.confirmed_at)
        start_index = 0 if start_index is None else start_index
        terminal = (
            state.retired_at
            or state.reaccepted_at
            or state.broken_at
        )
        end_index = _candle_index(candles, terminal)
        end_index = len(candles) - 1 if end_index is None else end_index
        color, line_style = _support_resistance_style(
            state.source_kind
        )
        axis.fill_between(
            [start_index - 0.4, end_index + 0.4],
            lower,
            upper,
            color=color,
            alpha=0.035,
            zorder=0,
        )
        axis.hlines(
            state.anchor_price,
            start_index - 0.4,
            end_index + 0.4,
            color=color,
            linewidth=0.55,
            linestyle=line_style,
            alpha=0.78,
            zorder=2,
        )
        _collision_safe_annotate(
            axis,
            (
                f"S/R src={state.source_kind} "
                f"{state.side} {state.lifecycle.value} "
                f"{state.structural_rank}/"
                f"{'protected' if state.is_protected_swing else 'ordinary'} "
                f"vis={state.visibility_strength:.2f} "
                f"react={state.reaction_quality:.2f} "
                f"fresh={state.freshness:.2f} "
                f"dep={state.depletion_risk:.2f} "
                f"[{_short_identity(state.zone_id)}]"
            ),
            min(end_index + 0.35, len(candles) - 1),
            upper,
            color=color,
            fontsize=4.9,
            preferred_side="left",
            priority=offset == len(zones) - 1,
        )

    authoritative_pools = tuple(
        state
        for state in snapshot.observation.liquidity_pool_states
        if state.timeframe is timeframe
    )
    pools = (
        authoritative_pools
        if authoritative_pools
        else tuple(frame.liquidity_pools)
    )[-3:]
    for offset, state in enumerate(pools):
        if not visible_low <= state.midpoint <= visible_high:
            continue
        start_index = _candle_index(candles, state.confirmed_at)
        start_index = 0 if start_index is None else start_index
        terminal = state.resolved_at or state.swept_at
        end_index = _candle_index(candles, terminal)
        end_index = len(candles) - 1 if end_index is None else end_index
        axis.hlines(
            state.midpoint,
            start_index - 0.4,
            end_index + 0.4,
            color="#a21caf",
            linewidth=0.7,
            linestyle=":",
            zorder=2,
        )
        if offset == len(pools) - 1:
            _collision_safe_annotate(
                axis,
                (
                    f"{'EQH' if state.side == 'above' else 'EQL'} "
                    f"{state.lifecycle.value} · {state.touch_count}x "
                    f"[{_short_identity(state.pool_id)}]"
                ),
                min(end_index + 0.35, len(candles) - 1),
                state.midpoint,
                color="#a21caf",
                fontsize=5.0,
                preferred_side="left",
            )
    axis.set_ylim(visible_low, visible_high)


def _support_resistance_style(source_kind: str) -> tuple[str, Any]:
    """Return a stable visual vocabulary for independent S/R sources."""

    return {
        "structural_swing": ("#0f766e", "-"),
        "previous_session": ("#2563eb", "--"),
        "previous_day": ("#d97706", "-."),
        "previous_week": ("#7c3aed", ":"),
        "range_boundary": ("#be123c", (0, (3, 1, 1, 1))),
    }.get(str(source_kind), ("#64748b", "-"))


def _displacement_overlay(
    axis: Any,
    snapshot: EngineSnapshot,
    candles: Sequence[Candle],
) -> None:
    displacement = snapshot.observation.displacement
    if not candles or displacement is None:
        return
    visible = [
        item
        for item in displacement.recent_transitions
        if _candle_index(candles, item.observed_at) is not None
    ][-4:]
    for offset, item in enumerate(visible):
        index = _candle_index(candles, item.observed_at)
        assert index is not None
        price = candles[index].close
        color = "#0891b2" if item.direction.value == "long" else "#c2410c"
        axis.scatter(
            [index],
            [price],
            marker="D",
            s=20,
            color=color,
            zorder=6,
        )
        _collision_safe_annotate(
            axis,
            (
                f"DISP {item.direction.value[0].upper()} "
                f"{item.lifecycle} "
                f"ov={dict(item.state_metrics).get('mean_overlap_ratio', 0.0):.2f}/"
                f"{dict(item.state_metrics).get('max_overlap_ratio', 0.0):.2f} "
                f"clv={dict(item.state_metrics).get('mean_directional_clv', 0.0):.2f} "
                f"[{_short_identity(item.entity_id)}]"
            ),
            index,
            price,
            color=color,
            fontsize=5.1,
            priority=offset == len(visible) - 1,
        )


def _liquidity_inventory_overlay(
    axis: Any,
    snapshot: EngineSnapshot,
    timeframe: Timeframe,
    candles: Sequence[Candle],
) -> None:
    """Show Eye inventory independently from Brain TARGETED overlays."""

    if not candles:
        return
    visible_low, visible_high = axis.get_ylim()
    states = sorted(
        (
            item
            for item in snapshot.observation.liquidity_inventory
            if item.timeframe is timeframe
            and visible_low <= item.price <= visible_high
            and item.confirmed_at <= candles[-1].end
            and (
                item.consumed_at is None
                or item.consumed_at > candles[0].start
            )
        ),
        key=lambda item: (item.confirmed_at, item.item_id),
    )[-4:]
    for offset, item in enumerate(states):
        start = _candle_index(candles, item.confirmed_at)
        start = 0 if start is None else start
        end = _candle_index(candles, item.consumed_at)
        end = len(candles) - 1 if end is None else end
        visible = item.lifecycle is LiquidityInventoryLifecycle.VISIBLE
        color = "#7c3aed" if visible else "#94a3b8"
        axis.hlines(
            item.price,
            start - 0.4,
            end + 0.4,
            color=color,
            linewidth=0.55,
            linestyle="--" if visible else ":",
            alpha=0.75,
            zorder=2,
        )
        if offset == len(states) - 1:
            _collision_safe_annotate(
                axis,
                (
                    f"LIQ {item.kind} {item.lifecycle.value} "
                    f"{item.structural_rank} "
                    f"vis={item.visibility_strength:.2f} "
                    f"[{_short_identity(item.item_id)}]"
                ),
                min(end + 0.35, len(candles) - 1),
                item.price,
                color=color,
                fontsize=4.8,
                preferred_side="left",
            )
    axis.set_ylim(visible_low, visible_high)


def _group5_overlay(
    axis: Any,
    snapshot: EngineSnapshot,
    candles: Sequence[Candle],
    belief: Any,
    focus_entity_id: str | None = None,
) -> None:
    """Mark exact first-pullback, reacceptance, micro-BOS and path order."""

    if not candles or not snapshot.observation.group5_typed_available:
        return
    locations, paths, reacceptances, references = (
        _selected_group5_entities(snapshot, belief, focus_entity_id)
    )
    visible_low, visible_high = axis.get_ylim()
    visible_locations = locations[-2:]
    for offset, item in enumerate(visible_locations):
        if (
            item.first_entered_at is not None
            and item.contact_reference_price is not None
            and visible_low
            <= item.contact_reference_price
            <= visible_high
        ):
            index = _candle_index(candles, item.first_entered_at)
            if index is not None:
                axis.scatter(
                    [index],
                    [item.contact_reference_price],
                    marker="o",
                    s=28,
                    facecolors="none",
                    edgecolors="#0f766e",
                    linewidths=1.0,
                    zorder=7,
                )
                _collision_safe_annotate(
                    axis,
                    (
                        f"1st pullback {item.source_zone_kind} "
                        f"{item.lifecycle.value} "
                        f"[{_short_identity(item.location_id)}]"
                    ),
                    index,
                    item.contact_reference_price,
                    color="#0f766e",
                    fontsize=5.2,
                    priority=offset == len(visible_locations) - 1,
                )

    visible_paths = paths[-2:]
    for path_offset, path in enumerate(visible_paths):
        for step_offset, step in enumerate(path.steps):
            index = _candle_index(candles, step.observed_at)
            if index is None:
                continue
            price = candles[index].close
            axis.scatter(
                [index],
                [price],
                marker=".",
                s=18,
                color="#0369a1",
                zorder=7,
            )
            _collision_safe_annotate(
                axis,
                (
                    f"{step.kind.replace('_', ' ')} "
                    f"[{_short_identity(path.sequence_id)}]"
                ),
                index,
                price,
                color="#0369a1",
                fontsize=4.8,
                priority=(
                    path_offset == len(visible_paths) - 1
                    and step_offset == len(path.steps) - 1
                ),
            )

    visible_reacceptances = reacceptances[-2:]
    for offset, item in enumerate(visible_reacceptances):
        clock = item.held_at or item.reclaimed_at or item.left_at
        index = _candle_index(candles, clock)
        if (
            index is None
            or not visible_low <= item.reference_price <= visible_high
        ):
            continue
        axis.scatter(
            [index],
            [item.reference_price],
            marker="s",
            s=22,
            color="#16a34a",
            zorder=7,
        )
        _collision_safe_annotate(
            axis,
            (
                f"reaccept {item.lifecycle.value} "
                f"[{_short_identity(item.reacceptance_id)}]"
            ),
            index,
            item.reference_price,
            color="#16a34a",
            fontsize=5.1,
            priority=offset == len(visible_reacceptances) - 1,
        )

    bos_by_id = {
        item.bos_id: item
        for item in snapshot.observation.frame(Timeframe.M1).structure_breaks
    }
    visible_references = tuple(
        item for item in references if item.bos_id in bos_by_id
    )[-2:]
    for offset, item in enumerate(visible_references):
        bos = bos_by_id[item.bos_id]
        index = _candle_index(candles, item.resolved_at)
        if (
            index is None
            or not visible_low <= bos.target_price <= visible_high
        ):
            continue
        color = "#15803d" if item.qualified else "#dc2626"
        axis.scatter(
            [index],
            [bos.target_price],
            marker="*",
            s=34,
            color=color,
            zorder=8,
        )
        _collision_safe_annotate(
            axis,
            (
                f"micro BOS {item.outcome} "
                f"[{_short_identity(item.reference_id)}]"
            ),
            index,
            bos.target_price,
            color=color,
            fontsize=5.2,
            priority=offset == len(visible_references) - 1,
        )
    axis.set_ylim(visible_low, visible_high)


def _group3_zone_overlay(
    axis: Any,
    snapshot: EngineSnapshot,
    candles: Sequence[Candle],
) -> None:
    """Draw only causally visible M5 zones without expanding the price axis."""

    if not candles:
        return
    frame = snapshot.observation.frame(Timeframe.M5)
    candidates: list[tuple[str, Any, str]] = []
    candidates.extend(
        (
            (
                "FVG raw"
                if state.qualification is FVGQualification.RAW
                else "FVG linked"
            ),
            state,
            (
                "#94a3b8"
                if state.qualification is FVGQualification.RAW
                else "#0284c7"
                if state.direction.value == "long"
                else "#ea580c"
            ),
        )
        for state in frame.fair_value_gaps
    )
    candidates.extend(
        ("OB", state, "#9333ea")
        for state in frame.order_blocks
    )
    if not candidates:
        return
    visible_low, visible_high = axis.get_ylim()
    def terminal_clock(state: Any) -> pd.Timestamp | None:
        if state.lifecycle in {
            FairValueGapLifecycle.MITIGATED,
            OrderBlockLifecycle.MITIGATED,
        }:
            return state.mitigated_at
        if state.lifecycle is FairValueGapLifecycle.INVALIDATED:
            return state.invalidated_at
        if state.lifecycle is OrderBlockLifecycle.FAILED:
            return state.failed_at
        return None

    visible = sorted(
        (
            item
            for item in candidates
            if (
                item[1].confirmed_at <= candles[-1].end
                and (
                    terminal_clock(item[1]) is None
                    or terminal_clock(item[1]) > candles[0].start
                )
                and item[1].upper_bound >= visible_low
                and item[1].lower_bound <= visible_high
            )
        ),
        key=lambda item: (
            terminal_clock(item[1]) is None,
            item[1].last_updated_at,
            item[1].confirmed_at,
            getattr(
                item[1],
                "fvg_id",
                getattr(item[1], "order_block_id", ""),
            ),
        ),
    )[-8:]
    from matplotlib.patches import Rectangle

    for offset, (label, state, color) in enumerate(visible):
        start_index = next(
            (
                index
                for index, candle in enumerate(candles)
                if candle.start < state.confirmed_at <= candle.end
            ),
            0,
        )
        terminal_at = terminal_clock(state)
        end_index = len(candles) - 1
        if terminal_at is not None:
            end_index = next(
                (
                    index
                    for index, candle in enumerate(candles)
                    if candle.start < terminal_at <= candle.end
                ),
                end_index,
            )
        lower = max(float(state.lower_bound), visible_low)
        upper = min(float(state.upper_bound), visible_high)
        if lower >= upper:
            continue
        axis.add_patch(
            Rectangle(
                (start_index - 0.45, lower),
                max(0.9, end_index - start_index + 0.9),
                upper - lower,
                facecolor=color,
                edgecolor=color,
                linewidth=0.7,
                alpha=0.10,
                zorder=1,
            )
        )
        if label == "OB":
            body_lower = max(float(state.body_lower_bound), visible_low)
            body_upper = min(float(state.body_upper_bound), visible_high)
            if body_lower < body_upper:
                axis.add_patch(
                    Rectangle(
                        (start_index - 0.45, body_lower),
                        max(0.9, end_index - start_index + 0.9),
                        body_upper - body_lower,
                        facecolor="none",
                        edgecolor=color,
                        linewidth=1.0,
                        linestyle="--",
                        zorder=2,
                    )
                )
        midpoint = float(state.midpoint)
        if visible_low <= midpoint <= visible_high:
            axis.hlines(
                midpoint,
                start_index - 0.45,
                end_index + 0.45,
                color=color,
                linewidth=0.5,
                linestyle=":",
                alpha=0.8,
                zorder=2,
            )
        entity_id = getattr(
            state,
            "fvg_id",
            getattr(state, "order_block_id", None),
        )
        _collision_safe_annotate(
            axis,
            (
                f"{label} {state.direction.value[0].upper()} "
                f"{state.lifecycle.value} · {state.age_bars}b "
                + (
                    f"q={state.qualification.value} "
                    if hasattr(state, "qualification")
                    else f"anchors={len(state.anchor_candle_ids)} "
                )
                + f"[{_short_identity(entity_id)}]"
            ),
            min(end_index + 0.42, len(candles) - 1),
            upper,
            color=color,
            fontsize=5.3,
            preferred_side="left",
            priority=offset == len(visible) - 1,
        )
    axis.set_ylim(visible_low, visible_high)


def _group3_text(snapshot: EngineSnapshot) -> str:
    frame = snapshot.observation.frame(Timeframe.M5)
    rows = []
    for label, states in (
        ("FVG", frame.fair_value_gaps),
        ("OB", frame.order_blocks),
    ):
        for state in states[-1:]:
            entity_id = getattr(
                state,
                "fvg_id",
                getattr(state, "order_block_id", "unknown"),
            )
            rows.append(
                f"{label} {state.direction.value[0].upper()} "
                f"{state.lifecycle.value} "
                f"[{state.lower_bound:.2f},{state.upper_bound:.2f}] "
                f"age={state.age_bars}b "
                f"id={_short_identity(entity_id)} "
                f"src={_short_identity(state.source_displacement_id)} "
                + (
                    f"q={state.qualification.value} "
                    f"widthATR={state.width_atr:.2f} "
                    f"fill={state.max_fill_fraction:.2f}"
                    if hasattr(state, "qualification")
                    else (
                        f"scope={state.source_bos_scope.value} "
                        f"mss={state.source_bos_mss_qualified} "
                        f"anchors={len(state.anchor_candle_ids)} "
                        f"body=[{state.body_lower_bound:.2f},"
                        f"{state.body_upper_bound:.2f}]"
                    )
                )
            )
    for label, states in (
        (
            "FVG boundary",
            snapshot.observation.group3_boundary_fvg_transitions,
        ),
        (
            "OB boundary",
            snapshot.observation.group3_boundary_order_block_transitions,
        ),
    ):
        for state in states[-1:]:
            rows.append(
                f"{label} {state.direction.value[0].upper()} "
                f"{state.lifecycle.value} "
                f"[{state.lower_bound:.2f},{state.upper_bound:.2f}] "
                f"reason={state.transition_reason}"
            )
    return "\n".join(rows) if rows else "none"


def _group4_range_overlay(
    axis: Any,
    snapshot: EngineSnapshot,
    candles: Sequence[Candle],
) -> None:
    """Draw typed H1 range geometry without changing the candle scale."""

    if not candles:
        return
    visible_low, visible_high = axis.get_ylim()
    ranges = snapshot.observation.frame(Timeframe.H1).dealing_ranges
    visible = tuple(
        state
        for state in ranges
        if (
            state.formed_at <= candles[-1].end
            and (
                state.broken_at is None
                or state.broken_at > candles[0].start
            )
            and state.upper_bound >= visible_low
            and state.lower_bound <= visible_high
        )
    )[-4:]
    for offset, state in enumerate(visible):
        start_index = next(
            (
                index
                for index, candle in enumerate(candles)
                if candle.start < state.formed_at <= candle.end
            ),
            0,
        )
        terminal_at = state.broken_at
        end_index = len(candles) - 1
        if terminal_at is not None:
            end_index = next(
                (
                    index
                    for index, candle in enumerate(candles)
                    if candle.start < terminal_at <= candle.end
                ),
                end_index,
            )
        color = (
            "#ca8a04"
            if state.lifecycle is DealingRangeLifecycle.MATURE
            else "#a16207"
            if state.lifecycle is DealingRangeLifecycle.FORMING
            else "#78716c"
        )
        linestyle = (
            "-"
            if state.lifecycle is DealingRangeLifecycle.MATURE
            else "--"
        )
        for price in (state.lower_bound, state.upper_bound):
            if visible_low <= price <= visible_high:
                axis.hlines(
                    price,
                    start_index - 0.45,
                    end_index + 0.45,
                    color=color,
                    linewidth=0.8,
                    linestyle=linestyle,
                    alpha=0.9,
                    zorder=2,
                )
        if visible_low <= state.midpoint <= visible_high:
            axis.hlines(
                state.midpoint,
                start_index - 0.45,
                end_index + 0.45,
                color=color,
                linewidth=0.6,
                linestyle=":",
                alpha=0.85,
                zorder=2,
            )
        label_price = min(
            max(state.upper_bound, visible_low),
            visible_high,
        )
        _collision_safe_annotate(
            axis,
            (
                f"typed range {state.lifecycle.value} "
                f"age={state.age_h1_bars}h "
                f"[{_short_identity(state.range_id)}]"
            ),
            min(end_index + 0.42, len(candles) - 1),
            label_price,
            color=color,
            fontsize=5.3,
            preferred_side="left",
            priority=offset == len(visible) - 1,
        )
    axis.set_ylim(visible_low, visible_high)


def _group4_manipulation_overlay(
    axis: Any,
    snapshot: EngineSnapshot,
    candles: Sequence[Candle],
) -> None:
    """Show typed 1m excursion and resolution clocks without future data."""

    if not candles:
        return
    visible_low, visible_high = axis.get_ylim()
    states = tuple(
        state
        for state in snapshot.observation.manipulations
        if (
            candles[0].start <= state.swept_at <= candles[-1].end
            or (
                state.resolved_at is not None
                and candles[0].start
                <= state.resolved_at
                <= candles[-1].end
            )
        )
    )[-6:]
    for offset, state in enumerate(states):
        sweep_index = next(
            (
                index
                for index, candle in enumerate(candles)
                if candle.start < state.swept_at <= candle.end
            ),
            None,
        )
        if (
            sweep_index is not None
            and visible_low <= state.sweep_extreme <= visible_high
        ):
            axis.scatter(
                [sweep_index],
                [state.sweep_extreme],
                marker="x",
                s=30,
                color="#c2410c",
                linewidths=0.9,
                zorder=6,
            )
            _collision_safe_annotate(
                axis,
                (
                    f"sweep {state.source_kind} "
                    f"[{_short_identity(state.manipulation_id)}]"
                ),
                sweep_index,
                state.sweep_extreme,
                color="#c2410c",
                fontsize=5.2,
                priority=offset == len(states) - 1,
            )
        if (
            state.reentry_candidate_at is not None
            and state.reentry_candidate_price is not None
        ):
            candidate_index = _candle_index(
                candles, state.reentry_candidate_at
            )
            if (
                candidate_index is not None
                and visible_low
                <= state.reentry_candidate_price
                <= visible_high
            ):
                axis.scatter(
                    [candidate_index],
                    [state.reentry_candidate_price],
                    marker="s",
                    s=18,
                    facecolors="none",
                    edgecolors="#0284c7",
                    linewidths=0.8,
                    zorder=6,
                )
        if state.reentry_failed_at is not None:
            failed_index = _candle_index(
                candles,
                state.reentry_failed_at,
            )
            if failed_index is not None:
                failed_price = candles[failed_index].close
                if visible_low <= failed_price <= visible_high:
                    axis.scatter(
                        [failed_index],
                        [failed_price],
                        marker="x",
                        s=22,
                        color="#dc2626",
                        linewidths=0.8,
                        zorder=6,
                    )
                    _collision_safe_annotate(
                        axis,
                        (
                            "reentry failed "
                            f"[{_short_identity(state.manipulation_id)}]"
                        ),
                        failed_index,
                        failed_price,
                        color="#dc2626",
                        fontsize=4.8,
                    )
        if state.deadline_elapsed and state.censored_at is not None:
            deadline_index = _candle_index(candles, state.censored_at)
            if deadline_index is not None:
                deadline_price = candles[deadline_index].close
                _collision_safe_annotate(
                    axis,
                    (
                        f"manip deadline run={state.outside_run} "
                        f"hold={state.inside_hold_bars} "
                        f"[{_short_identity(state.manipulation_id)}]"
                    ),
                    deadline_index,
                    deadline_price,
                    color="#64748b",
                    fontsize=4.8,
                )
        if state.resolved_at is None:
            continue
        resolution_index = next(
            (
                index
                for index, candle in enumerate(candles)
                if candle.start < state.resolved_at <= candle.end
            ),
            None,
        )
        resolution_price = (
            state.reentry_price
            if state.lifecycle is ManipulationLifecycle.REACCEPTED
            else next(
                (
                    candle.close
                    for candle in candles
                    if candle.start < state.resolved_at <= candle.end
                ),
                None,
            )
        )
        if (
            resolution_index is not None
            and resolution_price is not None
            and visible_low <= resolution_price <= visible_high
        ):
            axis.scatter(
                [resolution_index],
                [resolution_price],
                marker="o",
                s=18,
                color=(
                    "#16a34a"
                    if state.lifecycle
                    is ManipulationLifecycle.REACCEPTED
                    else "#dc2626"
                ),
                zorder=6,
            )
    axis.set_ylim(visible_low, visible_high)


def _group4_text(
    snapshot: EngineSnapshot,
    focus_entity_id: str | None = None,
) -> str:
    rows = []
    ranges = snapshot.observation.frame(Timeframe.H1).dealing_ranges
    for state in ranges[-1:]:
        value_label = (
            "historical_mid"
            if state.lifecycle.value == "broken"
            else "value"
            if state.mature_at is not None
            else "candidate_mid"
        )
        rows.append(
            f"range {state.lifecycle.value} id={_short_identity(state.range_id)} "
            f"[{state.lower_bound:.2f},{state.upper_bound:.2f}] "
            f"{value_label}={state.value_price:.2f} "
            f"age={state.age_h1_bars}bars "
            f"src={_short_identity(state.lower_source_zone_id)}/"
            f"{_short_identity(state.upper_source_zone_id)}"
        )
    for state in snapshot.observation.manipulations[-1:]:
        resolution = (
            f"resolved={state.resolved_at:%m-%d %H:%M}"
            if state.resolved_at is not None
            else "resolved=pending"
        )
        outcome = (
            f"reentry={state.reentry_price:.2f}"
            if state.reentry_price is not None
            else (
                f"side={state.resolved_side}"
                if state.resolved_side is not None
                else "side=pending"
            )
        )
        rows.append(
            f"manip {state.side} {state.lifecycle.value} "
            f"id={_short_identity(state.manipulation_id)} "
            f"at={state.swept_at:%m-%d %H:%M} "
            f"dur={state.age_1m_bars}b "
            f"outside={state.outside_completed_bars}b "
            f"run={state.outside_run}/{state.outside_run_side} "
            f"hold={state.inside_hold_bars} "
            f"deadline={state.deadline_elapsed} "
            f"{resolution} {outcome} "
            f"src={_short_identity(state.source_id)} "
            f"crossed={len(state.crossed_source_ids)}"
        )
    for state in snapshot.observation.group4_boundary_range_transitions[-1:]:
        rows.append(
            f"range boundary {state.lifecycle.value} "
            f"reason={state.transition_reason}"
        )
    for state in (
        snapshot.observation
        .group4_boundary_manipulation_transitions[-1:]
    ):
        rows.append(
            f"manip boundary censored "
            f"reason={state.transition_reason}"
        )
    if snapshot.observation.group4_ambiguous_sweep_item_ids:
        rows.append(
            "ambiguous dual-side sweep "
            + ",".join(
                _short_identity(value)
                for value in snapshot.observation
                .group4_ambiguous_sweep_item_ids
            )
        )
    if snapshot.observation.group4_atr_unready_sweep_item_ids:
        rows.append(
            "unclassified sweep: prior ATR unavailable "
            + ",".join(
                _short_identity(value)
                for value in snapshot.observation
                .group4_atr_unready_sweep_item_ids
            )
        )
    if focus_entity_id is not None:
        diagnostic = next(
            (
                item
                for item in snapshot.observation.group4_range_funnel
                if (
                    item.maturity_range_id == focus_entity_id
                    and item.observed_at == snapshot.observation.asof
                )
            ),
            None,
        )
        if diagnostic is not None:
            rows.append(
                "range diagnostic "
                f"id={_short_identity(focus_entity_id)} "
                "unmet="
                + (
                    ",".join(diagnostic.unmet_maturity_gates)
                    or "none"
                )
            )
            rows.extend(
                f"  {name}: actual={actual:.3f} "
                f"threshold={threshold:.3f} margin={margin:+.3f}"
                for name, actual, threshold, margin
                in diagnostic.maturity_gates
            )
    return "\n".join(rows) if rows else "none"


def _group5_text(
    snapshot: EngineSnapshot,
    belief: Any,
    focus_entity_id: str | None = None,
) -> str:
    rows: list[str] = []
    locations, paths, reacceptances, micro_bos = (
        _selected_group5_entities(snapshot, belief, focus_entity_id)
    )
    for item in locations[-2:]:
        rows.append(
            f"location {item.lifecycle.value} id={_short_identity(item.location_id)} "
            f"[{item.lower_bound:.2f},{item.upper_bound:.2f}] "
            f"zone={item.source_zone_kind}:{_short_identity(item.source_zone_id)} "
            f"first={item.first_entered_at.strftime('%H:%M') if item.first_entered_at is not None else 'pending'} "
            f"dur={item.state_duration_real_1m_bars}b"
        )
    for item in reacceptances[-2:]:
        rows.append(
            f"reaccept {item.lifecycle.value} "
            f"id={_short_identity(item.reacceptance_id)} "
            f"left={item.left_at:%H:%M} "
            f"reclaim={item.reclaimed_at.strftime('%H:%M') if item.reclaimed_at is not None else 'pending'} "
            f"held={item.held_at.strftime('%H:%M') if item.held_at is not None else 'pending'}"
        )
    for item in micro_bos[-2:]:
        rows.append(
            f"micro BOS {item.outcome} qualified={item.qualified} "
            f"id={_short_identity(item.reference_id)} "
            f"bos={_short_identity(item.bos_id)} at={item.resolved_at:%H:%M}"
        )
    for item in paths[-2:]:
        rows.append(
            f"path {_short_identity(item.sequence_id)} {item.lifecycle.value} "
            + " → ".join(
                f"{step.kind}@{step.observed_at:%H:%M}"
                for step in item.steps[-5:]
            )
        )
    return "\n".join(rows) if rows else "none"


def _level_overlay(
    axis: Any,
    candles: Sequence[Candle],
    levels: Sequence[tuple[str, float, str, str]],
    *,
    direct_labels: bool = False,
) -> None:
    if not candles or not levels:
        return
    low = min(candle.low for candle in candles)
    high = max(candle.high for candle in candles)
    padding = max((high - low) * 0.06, 1e-9)
    axis.set_ylim(low - padding, high + padding)
    visible_low, visible_high = axis.get_ylim()
    for label, price, color, style in levels:
        if visible_low <= price <= visible_high:
            axis.axhline(
                price,
                color=color,
                linestyle=style,
                linewidth=0.8,
                label=label,
            )
            if direct_labels:
                _collision_safe_annotate(
                    axis,
                    f"{label} {price:.2f}",
                    len(candles) - 1,
                    price,
                    color=color,
                    fontsize=5.5,
                    preferred_side="left",
                    priority=True,
                )
        else:
            side = "above" if price > visible_high else "below"
            arrow = "↑" if side == "above" else "↓"
            # Anchor off-screen levels to the visible price boundary so their
            # labels participate in the same collision registry as every
            # structure/event annotation without expanding the candle scale.
            _collision_safe_annotate(
                axis,
                f"{label} {arrow} {price:.2f}",
                len(candles) - 1,
                visible_high if side == "above" else visible_low,
                color=color,
                fontsize=6.0,
                preferred_side="left",
                priority=True,
            )


def _plan_overlay(
    axis: Any,
    plan: Any,
    candles: Sequence[Candle],
    *,
    direct_labels: bool = False,
) -> None:
    if plan is None:
        return
    visible_low, visible_high = axis.get_ylim()
    if (
        plan.entry_zone_lower is not None
        and plan.entry_zone_upper is not None
    ):
        lower = max(float(plan.entry_zone_lower), visible_low)
        upper = min(float(plan.entry_zone_upper), visible_high)
        if lower < upper:
            axis.axhspan(
                lower,
                upper,
                color="#0f766e",
                alpha=0.08,
                zorder=0,
            )
            if direct_labels:
                _collision_safe_annotate(
                    axis,
                    (
                        f"frozen entry zone "
                        f"{_short_identity(plan.entry_location_id or 'legacy')}"
                    ),
                    len(candles) - 1,
                    (lower + upper) / 2.0,
                    color="#0f766e",
                    fontsize=5.5,
                    preferred_side="left",
                    priority=True,
                )
        axis.set_ylim(visible_low, visible_high)
    range_auction = getattr(plan, "range_auction", None)
    if range_auction is not None:
        lower = max(float(range_auction.lower_bound), visible_low)
        upper = min(float(range_auction.upper_bound), visible_high)
        if lower < upper:
            axis.axhspan(
                lower,
                upper,
                color="#7c3aed",
                alpha=0.035,
                zorder=0,
            )
            if direct_labels:
                _collision_safe_annotate(
                    axis,
                    (
                        "FAVR range "
                        f"{_short_identity(range_auction.range_id)} / "
                        "manip "
                        f"{_short_identity(range_auction.manipulation_id)}"
                    ),
                    len(candles) - 1,
                    (lower + upper) / 2.0,
                    color="#7c3aed",
                    fontsize=5.5,
                    preferred_side="left",
                    priority=True,
                )
        axis.set_ylim(visible_low, visible_high)
    levels = [
        ("entry", plan.planned_entry, "#111827", "--"),
        (
            "invalidation "
            f"[{_short_identity(plan.invalidation.source_level_id)}]",
            plan.invalidation.price,
            "#dc2626",
            "--",
        ),
        *[
            (
                f"draw {number} {_short_identity(target.level_id)}",
                target.price,
                "#16a34a",
                "--" if number == 1 else ":",
            )
            for number, target in enumerate(plan.targets[:3], start=1)
        ],
    ]
    if range_auction is not None:
        levels.append(
            (
                "frozen range value "
                f"[{_short_identity(range_auction.range_id)}]",
                range_auction.value_price,
                "#7c3aed",
                ":",
            )
        )
    _level_overlay(
        axis,
        candles,
        levels,
        direct_labels=direct_labels,
    )


def _belief_geometry_overlay(
    axis: Any,
    belief: Any,
    candles: Sequence[Candle],
    *,
    direct_labels: bool = False,
) -> None:
    """Show causal draw/invalidation even before an entry plan is complete."""

    if belief is None or belief.plan is not None:
        return
    levels: list[tuple[str, float, str, str]] = []
    if belief.invalidation is not None:
        levels.append(
            (
                "thesis invalidation "
                f"[{_short_identity(belief.invalidation.source_level_id)}]",
                float(belief.invalidation.price),
                "#dc2626",
                "--",
            )
        )
    levels.extend(
        (
            f"visible draw {number} [{_short_identity(target.level_id)}]",
            float(target.price),
            "#16a34a",
            ":",
        )
        for number, target in enumerate(
            belief.deliverable_targets[:3],
            start=1,
        )
    )
    _level_overlay(
        axis,
        candles,
        levels,
        direct_labels=direct_labels,
    )


def _draw_level_text(
    snapshot: EngineSnapshot,
    draw_id: str,
    fallback: Any | None = None,
) -> str:
    item = next(
        (
            candidate
            for candidate in snapshot.observation.liquidity_inventory
            if candidate.item_id == draw_id
        ),
        None,
    )
    if item is None:
        if fallback is None:
            return f"{draw_id} (inventory provenance unavailable)"
        return (
            f"{draw_id} · {fallback.timeframe.value} {fallback.side} "
            f"@ {fallback.price:.2f} confirmed "
            f"{fallback.confirmed_at:%m-%d %H:%M} "
            "(legacy/non-inventory source)"
        )
    return (
        f"{item.item_id} · {item.kind}/{item.lifecycle.value} "
        f"{item.timeframe.value} {item.side} @ {item.price:.2f} "
        f"confirmed {item.confirmed_at:%m-%d %H:%M} "
        f"src={','.join(_short_identity(value) for value in item.source_ids)}"
    )


def _draw_text(snapshot: EngineSnapshot, plan: Any) -> str:
    draw_id = plan.selected_draw_id or plan.targets[0].level_id
    target = next(
        (item for item in plan.targets if item.level_id == draw_id),
        plan.targets[0],
    )
    text = _draw_level_text(snapshot, draw_id, target)
    selection = getattr(plan, "draw_selection", None)
    if selection is not None:
        text += (
            f" · targeted {selection.selected_at:%m-%d %H:%M} "
            f"because {selection.selection_reason}"
        )
    return text


def _liquidity_route_text(route: Any) -> str:
    """Show the frozen roles in a causal delivery route without inference."""

    def identity(value: Any) -> str:
        return "none" if value is None else str(value)

    def identities(values: Sequence[Any]) -> str:
        return ", ".join(str(value) for value in values) or "none"

    rows = [
        f"FROZEN LIQUIDITY ROUTE {route.route_id}",
        f"  selected {route.selected_at:%Y-%m-%d %H:%M %Z}",
        f"  context draw {identity(route.context_draw_id)}",
        "  intermediate liquidity "
        f"{identities(route.intermediate_liquidity_ids)}",
        "  primary deliverable "
        f"{identity(route.primary_deliverable_target_id)}",
        f"  terminal draw {identity(route.terminal_draw_id)}",
        "  authority barrier "
        f"{identity(getattr(route, 'authority_barrier_id', None))} @ "
        f"{getattr(route, 'authority_barrier_price', None)}",
        f"  path blockers {identities(route.path_blocker_ids)}",
        f"  source paths {identities(route.source_path_ids)}",
    ]
    if getattr(route, "range_context_id", None) is not None:
        rows.extend(
            (
                f"  optional range context {route.range_context_id}",
                f"  range midpoint/value {route.range_midpoint:.2f}",
                "  swept range boundary "
                f"{identity(route.swept_range_boundary_id)}",
                "  opposing range liquidity "
                f"{identity(route.opposing_range_boundary_id)}",
            )
        )
    return "\n".join(rows)


def _partial_geometry_text(snapshot: EngineSnapshot, belief: Any) -> str:
    if belief is None:
        return "No hypothesis-bound structural geometry"
    rows = [
        f"setup {belief.setup_context_id or 'not-started'}",
        f"location {belief.entry_location_id or 'not-reached'}",
        "entry/path not complete — geometry is non-executable",
    ]
    episode_deadline = getattr(belief, "episode_deadline", None)
    if episode_deadline is not None:
        rows.append(
            f"episode deadline {episode_deadline:%Y-%m-%d %H:%M %Z}"
        )
    if belief.invalidation is None:
        rows.append("invalidation not yet causally available")
    else:
        rows.extend(
            (
                f"thesis invalidation {belief.invalidation.price:.2f}",
                f"  source {belief.invalidation.source_level_id} "
                f"at {belief.invalidation.observed_at:%m-%d %H:%M}",
            )
        )
    if belief.deliverable_targets:
        rows.append("visible draw candidates")
        rows.extend(
            "  " + _draw_level_text(snapshot, target.level_id, target)
            for target in belief.deliverable_targets[:3]
        )
    else:
        rows.append("visible draw not yet selected")
    selection = getattr(belief, "draw_selection", None)
    if selection is not None:
        rows.append(
            f"targeted draw {selection.draw_id} at "
            f"{selection.selected_at:%m-%d %H:%M}: "
            f"{selection.selection_reason}"
        )
    route = getattr(belief, "liquidity_route", None)
    if route is not None:
        rows.append(_liquidity_route_text(route))
    return "\n".join(rows)


def _event_timeline(snapshot: EngineSnapshot) -> str:
    retained_typed = tuple(
        event
        for key, timeline
        in snapshot.observation.retained_entity_timelines.items()
        if key.startswith(
            (
                "swing:",
                "structure:",
                "bos:",
                "zone:",
                "pool:",
                "fvg:",
                "order_block:",
                "range:",
                "manipulation:",
                "entry_path:",
            )
        )
        for event in timeline
    )
    by_id = {
        event.event_id: event
        for event in (
            *snapshot.observation.recent_events,
            *retained_typed,
        )
    }
    events = sorted(
        by_id.values(),
        key=lambda item: (
            item.observed_at,
            item.sequence_no,
            item.event_id,
        ),
    )[-8:]
    if not events:
        return "none"
    return "\n".join(
        f"{event.observed_at:%H:%M} {event.timeframe.value:>2s} "
        f"{event.kind.value}"
        f"{'' if event.lifecycle is None else ':' + event.lifecycle} "
        f"id={_short_identity(event.event_id)} "
        f"src={','.join(_short_identity(value) for value in event.source_ids) or '-'} "
        f"age={snapshot.observation.event_ages_minutes.get(event.event_id, 0)}m "
        f"persist={snapshot.observation.event_durations_minutes.get(event.event_id, 0)}m "
        f"str={event.strength:.2f}"
        f"{'' if event.transition_reason is None else ' · ' + event.transition_reason}"
        for event in events
    )


def _sequence_text(
    snapshot: EngineSnapshot,
) -> str:
    belief = _selected_hypothesis(snapshot)
    if belief is None or belief.sequence is None:
        return "none"
    sequence = belief.sequence
    rows = [
        f"setup {sequence.setup_id or 'not-started'}",
        f"protocol {sequence.protocol_version} {sequence.protocol_hash[:10]}",
    ]
    episode_deadline = getattr(belief, "episode_deadline", None)
    if episode_deadline is not None:
        rows.append(
            f"episode deadline {episode_deadline:%Y-%m-%d %H:%M %Z}"
        )
    for step in sequence.steps:
        clock = "" if step.observed_at is None else step.observed_at.strftime("%H:%M")
        rows.append(
            f"{'✓' if step.satisfied else '·'} {step.step_id} "
            f"{step.value:.2f} {clock}"
        )
    return "\n".join(rows)


def _graph_path_text(path: Sequence[str]) -> str:
    if not path:
        return "none"
    return " → ".join(
        value
        if value.isupper() and ":" not in value
        else _short_identity(value)
        for value in path
    )


def _evidence_state_text(values: Mapping[str, Any]) -> str:
    if not values:
        return "none"
    return ", ".join(
        f"{name}={getattr(status, 'value', status)}"
        for name, status in sorted(values.items())
    )


def _temporal_market_reading_text(snapshot: EngineSnapshot) -> str:
    """Render only the decision-time Focus/scene projection.

    The visualizer deliberately does not query the live SceneGraph.  Context
    paths and evidence states are frozen on ``MarketBelief`` and are therefore
    safe to render after the engine has processed later bars.
    """

    belief = snapshot.belief
    focus = getattr(belief, "focus_state", None)
    contexts = dict(getattr(belief, "context_hypotheses", {}))
    global_context = getattr(belief, "global_context", None)
    conflict_evidence = {
        conflict.conflict_id: conflict
        for conflict in (
            ()
            if global_context is None
            else global_context.material_conflicts
        )
    }
    scene_revision = (
        getattr(belief, "scene_revision_id", None)
        or getattr(snapshot.observation, "scene_revision_id", None)
    )
    active = ", ".join(
        timeframe.value
        for timeframe in active_causal_timeframes(snapshot.observation)
    )
    rows = [
        "TEMPORAL MARKET READING",
        f"scales {active}",
        f"scene {scene_revision or 'legacy/unavailable'}",
    ]
    if focus is None:
        rows.extend(
            (
                "\nFOCUS",
                "legacy snapshot — no FocusState",
                "reason/status unknown (not false)",
            )
        )
    else:
        resolution = getattr(
            getattr(focus, "resolution_status", None),
            "value",
            getattr(focus, "resolution_status", "unknown"),
        )
        rows.extend(
            (
                "\nFOCUS",
                "primary "
                + ", ".join(getattr(focus, "primary_timeframes", ())),
                "reason "
                + ", ".join(getattr(focus, "reason_codes", ())),
                "root "
                + _short_identity(
                    getattr(focus, "hypothesis_id", None) or "none"
                ),
                f"question {getattr(focus, 'question', 'unknown')}",
                f"status {resolution}",
                "reselected "
                + str(bool(getattr(focus, "switched", False))).lower(),
            )
        )

    dominant_id = getattr(belief, "dominant_hypothesis_id", None)
    competing_ids = tuple(
        getattr(belief, "competing_hypothesis_ids", ())
    )
    ordered_ids = tuple(
        dict.fromkeys(
            value
            for value in (dominant_id, *competing_ids)
            if value is not None
        )
    )
    rows.append("\nACTIVE COMPETING HYPOTHESES")
    if not ordered_ids:
        rows.append("none retained")
    for identity in ordered_ids[:8]:
        context = contexts.get(identity)
        if context is None:
            rows.append(f"? {_short_identity(identity)} missing projection")
            continue
        marker = "D" if identity == dominant_id else "C"
        rows.append(
            f"[{marker}] {_short_identity(identity)} "
            f"{context.playbook.value}/{context.direction.value} "
            f"{context.sequence_stage}"
        )
        rows.append(
            f"    tf {context.context_timeframe}→"
            f"{context.setup_timeframe}→{context.trigger_timeframe}; "
            f"next={context.next_expected_event or 'none'}"
        )
        rows.append(
            "    draw/target/invalidation "
            f"{_short_identity(context.context_draw_id)} / "
            f"{_short_identity(context.primary_target_id)} / "
            f"{_short_identity(context.invalidation_id)}"
        )
    if len(ordered_ids) > 8:
        rows.append(f"+ {len(ordered_ids) - 8} additional context(s)")

    rows.append("\nCAUSAL SUPPORT PATHS")
    path_rows = 0
    for identity in ordered_ids[:3]:
        context = contexts.get(identity)
        if context is None:
            continue
        for path in context.supporting_graph_paths[:2]:
            rows.append(
                f"+ {_short_identity(identity)} "
                f"{_graph_path_text(path)}"
            )
            path_rows += 1
    if path_rows == 0:
        rows.append("none frozen")

    rows.append("\nMATERIAL CONFLICTS")
    conflict_rows = 0
    for identity in ordered_ids[:3]:
        context = contexts.get(identity)
        if context is None:
            continue
        for conflict_id in context.material_conflict_ids[:3]:
            conflict = conflict_evidence.get(conflict_id)
            if conflict is None:
                rows.append(
                    f"! {_short_identity(identity)} "
                    f"{_short_identity(conflict_id)}"
                )
            else:
                rows.append(
                    f"! {_short_identity(identity)} "
                    f"{conflict.role.value}: "
                    f"{conflict.source_timeframe.value}/"
                    f"{getattr(conflict.source_direction, 'value', 'unknown')} "
                    f"→ {conflict.target_timeframe.value}/"
                    f"{getattr(conflict.target_direction, 'value', 'unknown')} "
                    f"({_short_identity(conflict.event_id)})"
                )
            conflict_rows += 1
    if conflict_rows == 0:
        rows.append("none material")

    rows.append("\nAMBIGUOUS / MISSING EVIDENCE")
    ambiguity_rows = 0
    for identity in ordered_ids[:5]:
        context = contexts.get(identity)
        if context is None:
            continue
        if context.ambiguous_evidence:
            rows.append(
                f"{_short_identity(identity)} ambiguous: "
                f"{_evidence_state_text(context.ambiguous_evidence)}"
            )
            ambiguity_rows += 1
        if context.missing_evidence:
            rows.append(
                f"{_short_identity(identity)} missing: "
                f"{_evidence_state_text(context.missing_evidence)}"
            )
            ambiguity_rows += 1
    conflicts = tuple(getattr(belief, "cross_scale_conflicts", ()))
    unresolved = tuple(getattr(belief, "unresolved_ambiguities", ()))
    if conflicts:
        rows.append(
            "conflicts: "
            + ", ".join(_short_identity(value) for value in conflicts)
        )
        ambiguity_rows += 1
    if unresolved:
        rows.append(
            "unresolved: "
            + ", ".join(_short_identity(value) for value in unresolved)
        )
        ambiguity_rows += 1
    if ambiguity_rows == 0:
        rows.append("none")
    rows.append("UNKNOWN is unresolved evidence, never FALSE")
    return "\n".join(rows)


class DecisionVisualizer:
    PANEL_ORDER = (
        Timeframe.H4,
        Timeframe.H1,
        Timeframe.M15,
        Timeframe.M5,
        Timeframe.M1,
    )
    PANEL_BARS = {
        Timeframe.H4: 20,
        Timeframe.H1: 48,
        Timeframe.M15: 48,
        Timeframe.M5: 36,
        Timeframe.M1: 60,
    }

    def render_observation(
        self,
        observation: Any,
        histories: Mapping[Timeframe, Sequence[Candle]],
        destination: str | Path,
        *,
        case_id: str | None = None,
        case_entity_id: str | None = None,
        scene_graph: Any = None,
    ) -> VisualArtifact:
        """Render one completed-data Eye audit without Brain or actions.

        This entry is deliberately limited to sampled authority cases.  It
        displays the same typed price overlays as the decision view, while
        omitting every belief, utility, risk and execution interpretation.
        """

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        snapshot = _ObservationSnapshot(observation=observation)
        asof = observation.asof
        validated = validate_causal_histories(snapshot, histories)
        active = active_causal_timeframes(observation)
        unsupported = tuple(
            timeframe
            for timeframe in active
            if timeframe not in self.PANEL_BARS
        )
        if unsupported:
            raise ValueError(
                "eye visualizer has no panel for an enabled timeframe"
            )
        panels = {
            timeframe: validated[timeframe][
                -self.PANEL_BARS[timeframe] :
            ]
            for timeframe in active
        }
        figure = plt.figure(
            figsize=(19, max(16, len(active) * 4)),
            dpi=110,
            constrained_layout=True,
        )
        grid = figure.add_gridspec(
            len(active),
            2,
            width_ratios=(3.4, 1.6),
        )
        axes = [
            figure.add_subplot(grid[index, 0])
            for index in range(len(active))
        ]
        info = figure.add_subplot(grid[:, 1])

        for axis, timeframe in zip(axes, active):
            values = panels[timeframe]
            if not values:
                _blank_causal_panel(
                    axis,
                    timeframe,
                    observation.frame(timeframe).cutoff,
                )
                continue
            _candles(axis, values)
            axis._smc_annotation_boxes = [(0.0, 0.69, 0.34, 0.99)]
            axis._smc_annotation_budget = 8
            axis._smc_annotation_count = 0
            axis._smc_annotation_omitted = 0
            _typed_structure_overlay(axis, snapshot, timeframe, values)
            _liquidity_inventory_overlay(
                axis,
                snapshot,
                timeframe,
                values,
            )
            if timeframe is Timeframe.M5:
                _group3_zone_overlay(axis, snapshot, values)
                _displacement_overlay(axis, snapshot, values)
            elif timeframe is Timeframe.H1:
                _group4_range_overlay(axis, snapshot, values)
            elif timeframe is Timeframe.M1:
                _group4_manipulation_overlay(axis, snapshot, values)
                _group5_overlay(
                    axis,
                    snapshot,
                    values,
                    None,
                    case_entity_id,
                )
            _event_markers(axis, snapshot, timeframe, values)
            if values:
                current_price = float(values[-1].close)
                visible_low, visible_high = axis.get_ylim()
                if visible_low <= current_price <= visible_high:
                    axis.axhline(
                        current_price,
                        color="#111827",
                        linewidth=0.65,
                        linestyle="--",
                        alpha=0.75,
                    )
                    _collision_safe_annotate(
                        axis,
                        f"NOW {current_price:.2f}",
                        len(values) - 1,
                        current_price,
                        color="#111827",
                        fontsize=5.4,
                        preferred_side="left",
                        priority=True,
                    )
            omitted = int(
                getattr(axis, "_smc_annotation_omitted", 0)
            )
            if omitted:
                axis.text(
                    0.995,
                    0.015,
                    f"+{omitted} labels omitted; identities remain recorded",
                    ha="right",
                    va="bottom",
                    fontsize=5.0,
                    color="#475569",
                    transform=axis.transAxes,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "#cbd5e1",
                        "alpha": 0.85,
                    },
                    zorder=10,
                )
            axis.axvline(
                len(values) - 0.5,
                color="#111827",
                linewidth=1.0,
            )
            axis.text(
                0.005,
                0.98,
                _metric_text(
                    snapshot,
                    timeframe,
                    None,
                    case_entity_id,
                ),
                ha="left",
                va="top",
                fontsize=6,
                family="monospace",
                transform=axis.transAxes,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#cbd5e1",
                    "alpha": 0.82,
                },
            )
            axis.set_title(
                f"{timeframe.value} · completed through "
                f"{values[-1].end:%Y-%m-%d %H:%M %Z}",
                loc="left",
                fontsize=9,
            )

        graph_nodes = (
            "not projected"
            if scene_graph is None
            else str(len(scene_graph.nodes))
        )
        graph_edges = (
            "not projected"
            if scene_graph is None
            else str(len(scene_graph.edges))
        )
        body = _wrap_panel_text(
            (
                f"EYE CLOCK\n{asof:%Y-%m-%d %H:%M %Z}\n\n"
                f"CASE\n{case_id or 'sampled-eye-audit'}\n\n"
                "AUTHORITY\n"
                "Observer only; no Brain, action, Risk, execution, PnL "
                "or future path.\n\n"
                f"SCENE GRAPH\nrevision "
                f"{observation.scene_revision_id or 'none'}\n"
                f"nodes {graph_nodes}; edges {graph_edges}\n\n"
                f"EVENT MEMORY\n{_event_timeline(snapshot)}\n\n"
                f"5M FVG / ORDER BLOCK\n{_group3_text(snapshot)}\n\n"
                f"H1 RANGE / 1M MANIPULATION\n"
                f"{_group4_text(snapshot, case_entity_id)}\n\n"
                f"EXACT ENTRY / 1M PATH\n"
                f"{_group5_text(snapshot, None, case_entity_id)}\n\n"
                "ANOMALIES\n"
                + (", ".join(observation.anomalies) or "none")
            ),
            width=60,
        )
        info.axis("off")
        info.set_xlim(0.0, 1.0)
        info.set_ylim(0.0, 1.0)
        info.text(
            0.0,
            1.0,
            body,
            ha="left",
            va="top",
            family="monospace",
            fontsize=6.5,
            wrap=True,
            transform=info.transAxes,
            clip_on=True,
        )
        figure.suptitle(
            "CAUSAL MARKET EYE VIEW — COMPLETED DATA ONLY",
            fontsize=13,
            weight="bold",
        )
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, bbox_inches="tight")
        plt.close(figure)
        maximum = max(
            candle.end
            for values in panels.values()
            for candle in values
        )
        if maximum > asof:
            raise AssertionError(
                "saved Eye artifact contains a future candle"
            )
        return VisualArtifact(
            path=destination,
            kind="eye_observation",
            decision_id=(
                case_id
                or f"{observation.symbol}:"
                f"{observation.instrument_id}:"
                f"{observation.asof.isoformat()}"
            ),
            maximum_market_time=maximum,
            hypothesis_key=None,
            setup_id=None,
            entry_location_id=None,
            entry_path_id=(
                case_entity_id
                if any(
                    path.sequence_id == case_entity_id
                    for path in observation.path_sequences
                )
                else None
            ),
        )

    def render_decision(
        self,
        snapshot: EngineSnapshot,
        histories: Mapping[Timeframe, Sequence[Candle]],
        destination: str | Path,
    ) -> VisualArtifact:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        asof = snapshot.observation.asof
        validated_histories = validate_causal_histories(
            snapshot,
            histories,
        )
        active_timeframes = active_causal_timeframes(
            snapshot.observation
        )
        unsupported = tuple(
            timeframe
            for timeframe in active_timeframes
            if timeframe not in self.PANEL_BARS
        )
        if unsupported:
            raise ValueError(
                "decision visualizer has no panel for an enabled timeframe"
            )
        # Preserve the frozen ScaleSpec order instead of silently reordering
        # the user's active registry to a visualizer-owned enum order.
        display_timeframes = active_timeframes
        panels: dict[Timeframe, tuple[Candle, ...]] = {}
        for timeframe in display_timeframes:
            count = self.PANEL_BARS[timeframe]
            panels[timeframe] = validated_histories[timeframe][-count:]

        belief = snapshot.belief
        has_temporal_reading = bool(
            getattr(belief, "focus_state", None) is not None
            or getattr(belief, "context_hypotheses", {})
            or getattr(belief, "scene_revision_id", None)
            or getattr(snapshot.observation, "scene_revision_id", None)
        )
        panel_count = len(display_timeframes)
        figure_height = max(16, panel_count * 4)
        if has_temporal_reading:
            figure = plt.figure(
                figsize=(25, figure_height),
                dpi=110,
                constrained_layout=True,
            )
            grid = figure.add_gridspec(
                panel_count,
                3,
                width_ratios=(3.35, 1.25, 1.65),
            )
            axes = [
                figure.add_subplot(grid[index, 0])
                for index in range(panel_count)
            ]
            reading_info = figure.add_subplot(grid[:, 1])
            info = figure.add_subplot(grid[:, 2])
        else:
            figure = plt.figure(
                figsize=(18, figure_height),
                dpi=110,
                constrained_layout=True,
            )
            grid = figure.add_gridspec(
                panel_count,
                2,
                width_ratios=(3.35, 1.65),
            )
            axes = [
                figure.add_subplot(grid[index, 0])
                for index in range(panel_count)
            ]
            reading_info = None
            info = figure.add_subplot(grid[:, 1])
        selected_belief = _selected_hypothesis(snapshot)
        plan = None if selected_belief is None else selected_belief.plan
        plan_heading, show_plan_overlay = _plan_display_context(
            snapshot,
            selected_belief,
        )
        typed_model = any(
            item.thesis_strength is not None
            for item in snapshot.belief.hypotheses.values()
        )
        focus = getattr(snapshot.belief, "focus_state", None)
        primary_focus = set(
            () if focus is None else focus.primary_timeframes
        )
        for axis, timeframe in zip(axes, display_timeframes):
            values = panels[timeframe]
            focus_label = (
                " · PRIMARY FOCUS"
                if timeframe.value in primary_focus
                else ""
            )
            if timeframe.value in primary_focus:
                for spine in axis.spines.values():
                    spine.set_color("#2563eb")
                    spine.set_linewidth(1.8)
            if not values:
                _blank_causal_panel(
                    axis,
                    timeframe,
                    snapshot.observation.frame(timeframe).cutoff,
                    title_suffix=focus_label,
                )
                continue
            _candles(axis, values)
            # Reserve the metric summary footprint before any event/structure
            # label requests a lane on the same axes.
            axis._smc_annotation_boxes = [(0.0, 0.69, 0.34, 0.99)]
            axis._smc_annotation_budget = 8
            axis._smc_annotation_count = 0
            axis._smc_annotation_omitted = 0
            _typed_structure_overlay(
                axis,
                snapshot,
                timeframe,
                values,
            )
            _liquidity_inventory_overlay(
                axis,
                snapshot,
                timeframe,
                values,
            )
            if timeframe is Timeframe.M5:
                _group3_zone_overlay(
                    axis,
                    snapshot,
                    values,
                )
                _displacement_overlay(axis, snapshot, values)
            elif timeframe is Timeframe.H1:
                _group4_range_overlay(
                    axis,
                    snapshot,
                    values,
                )
            elif timeframe is Timeframe.M1:
                _group4_manipulation_overlay(
                    axis,
                    snapshot,
                    values,
                )
                _group5_overlay(
                    axis,
                    snapshot,
                    values,
                    selected_belief,
                )
            _event_markers(axis, snapshot, timeframe, values)
            axis.set_title(
                (
                    f"{timeframe.value}{focus_label} · completed through "
                    f"{values[-1].end:%Y-%m-%d %H:%M %Z}"
                    if values
                    else f"{timeframe.value}{focus_label}"
                ),
                loc="left",
                fontsize=9,
            )
            if timeframe is Timeframe.H1 and not typed_model:
                frame = snapshot.observation.frame(Timeframe.H1)
                candle_low = min(candle.low for candle in values)
                candle_high = max(candle.high for candle in values)
                range_low = max(
                    frame.metrics["rolling_range_low"],
                    candle_low,
                )
                range_high = min(
                    frame.metrics["rolling_range_high"],
                    candle_high,
                )
                if range_low < range_high:
                    axis.axhspan(
                        range_low,
                        range_high,
                        color="#64748b",
                        alpha=0.06,
                    )
                    axis.text(
                        0.995,
                        0.02,
                        "rolling envelope (descriptive)",
                        ha="right",
                        va="bottom",
                        fontsize=5.2,
                        color="#64748b",
                        transform=axis.transAxes,
                    )
            if show_plan_overlay:
                _plan_overlay(
                    axis,
                    plan,
                    values,
                    direct_labels=timeframe
                    in {Timeframe.M5, Timeframe.M1},
                )
            else:
                _belief_geometry_overlay(
                    axis,
                    selected_belief,
                    values,
                    direct_labels=timeframe
                    in {Timeframe.M5, Timeframe.M1},
                )
            omitted_labels = int(
                getattr(axis, "_smc_annotation_omitted", 0)
            )
            if omitted_labels:
                axis.text(
                    0.995,
                    0.015,
                    (
                        f"+{omitted_labels} labels omitted; "
                        "identities remain in event memory"
                    ),
                    ha="right",
                    va="bottom",
                    fontsize=5.0,
                    color="#475569",
                    transform=axis.transAxes,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "#cbd5e1",
                        "alpha": 0.85,
                    },
                    zorder=10,
                )
            axis.axvline(len(values) - 0.5, color="#111827", linewidth=1.0)
            axis.text(
                0.005,
                0.98,
                _metric_text(snapshot, timeframe, selected_belief),
                ha="left",
                va="top",
                fontsize=6,
                family="monospace",
                transform=axis.transAxes,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#cbd5e1",
                    "alpha": 0.82,
                },
            )
        ranked = snapshot.belief.ranked()
        probabilities = "\n".join(
            (
                f"{item.playbook.value:28s} {item.direction.value:5s} "
                f"thesis={item.thesis_strength:.3f} "
                f"effective={item.effective_probability:.3f} "
                f"readiness={_optional_score(item.entry_readiness)} "
                f"{item.phase.value} "
                f"{max(0, int((asof - item.phase_started_at).total_seconds() // 60))}m"
            )
            if item.thesis_strength is not None
            else (
                f"{item.playbook.value:28s} {item.direction.value:5s} "
                f"legacy_probability={item.probability:.3f}"
                + (
                    ""
                    if item.raw_probability is None
                    else f" raw={item.raw_probability:.3f}"
                )
                + f" {item.phase.value} "
                f"{max(0, int((asof - item.phase_started_at).total_seconds() // 60))}m"
            )
            for item in ranked
        )
        plan_text = _partial_geometry_text(snapshot, selected_belief)
        if plan is not None:
            targets = ", ".join(f"{level.price:.2f}" for level in plan.targets)
            plan_text = (
                f"setup {plan.setup_id or 'legacy/untyped'}\n"
                f"location {plan.entry_location_id or 'legacy/untyped'}\n"
                f"path {plan.entry_path_id or 'legacy/untyped'}\n"
                f"entry {plan.planned_entry:.2f}\n"
                + (
                    ""
                    if plan.entry_zone_lower is None
                    else (
                        f"entry zone [{plan.entry_zone_lower:.2f},"
                        f"{plan.entry_zone_upper:.2f}]\n"
                    )
                )
                + f"invalidation {plan.invalidation.price:.2f}\n"
                f"  source {plan.invalidation.source_level_id}\n"
                f"targets {targets}\n"
                + "\n".join(
                    f"  T{number} {target.timeframe.value} "
                    f"{target.level_id} confirmed {target.confirmed_at:%m-%d %H:%M}"
                    for number, target in enumerate(plan.targets[:3], start=1)
                )
                + "\n"
                + f"primary {plan.primary_target_R:.2f}R\n"
                f"remaining path {plan.remaining_path_R:.2f}R\n"
                f"selected draw {_draw_text(snapshot, plan)}\n"
                f"deadline {plan.deadline:%Y-%m-%d %H:%M %Z}"
            )
            if plan.range_auction is not None:
                context = plan.range_auction
                plan_text += (
                    "\nFAVR frozen range "
                    f"{context.range_id} "
                    f"[{context.lower_bound:.2f},{context.upper_bound:.2f}]"
                    f" value={context.value_price:.2f}"
                    "\n  manipulation "
                    f"{context.manipulation_id} {context.manipulation_side} "
                    f"extreme={context.manipulation_extreme:.2f}"
                    f" swept={context.swept_at:%m-%d %H:%M}"
                    "\n  reentry "
                    f"{context.reentry_price:.2f} candidate "
                    f"{context.reentry_candidate_at:%m-%d %H:%M}; "
                    f"held {context.reentered_at:%m-%d %H:%M}"
                    "\n  opposite liquidity "
                    f"{context.opposite_liquidity_id}"
                )
            if plan.liquidity_route is not None:
                plan_text += "\n" + _liquidity_route_text(
                    plan.liquidity_route
                )
        best_utility_by_action: dict[Action, Any] = {}
        for item in snapshot.decision.utilities:
            prior = best_utility_by_action.get(item.action)
            if prior is None or item.utility > prior.utility:
                best_utility_by_action[item.action] = item
        utilities = "\n".join(
            f"{action.value:8s} "
            + (
                "not evaluated"
                if action not in best_utility_by_action
                else (
                    f"{best_utility_by_action[action].utility:+.3f}R"
                    + (
                        ""
                        if not best_utility_by_action[action].hypothesis_key
                        else " · "
                        + best_utility_by_action[action].hypothesis_key
                    )
                )
            )
            for action in Action
        )
        veto = (
            "none"
            if not snapshot.risk.vetoes
            else ", ".join(value.value for value in snapshot.risk.vetoes)
        )
        sequence_heading = "SELECTED SETUP SEQUENCE"
        evidence_heading = "SELECTED EVIDENCE"
        evidence_text = _format_evidence(snapshot)
        sequence_text = _sequence_text(snapshot)
        if reading_info is not None:
            reading_body = _wrap_panel_text(
                _temporal_market_reading_text(snapshot),
                width=58,
            )
            reading_info.axis("off")
            reading_info.set_xlim(0.0, 1.0)
            reading_info.set_ylim(0.0, 1.0)
            reading_info.text(
                0.0,
                1.0,
                reading_body,
                ha="left",
                va="top",
                family="monospace",
                fontsize=6.2,
                wrap=True,
                transform=reading_info.transAxes,
                clip_on=True,
            )
        info_body = _wrap_panel_text(
            (
                f"DECISION CLOCK\n{asof:%Y-%m-%d %H:%M %Z}\n\n"
                f"MODEL / RISK ACTION\n"
                f"{snapshot.decision.selected_action.value} / "
                f"{snapshot.risk.final_action.value}\n"
                f"advantage {snapshot.decision.advantage:.3f}R\n"
                f"veto: {veto}\n\n"
                f"PLAYBOOK BELIEFS\n{probabilities}\n\n"
                f"{sequence_heading}\n"
                f"{sequence_text}\n\n"
                f"{evidence_heading}\n"
                f"{evidence_text}\n\n"
                f"{plan_heading}\n{plan_text}\n\n"
                f"EVENT MEMORY (ordered / duration)\n{_event_timeline(snapshot)}\n\n"
                f"5M FVG / ORDER BLOCK\n{_group3_text(snapshot)}\n\n"
                f"H1 RANGE / 1M MANIPULATION\n"
                f"{_group4_text(snapshot)}\n\n"
                f"EXACT ENTRY / 1M PATH\n"
                f"{_group5_text(snapshot, selected_belief)}\n\n"
                f"EXECUTION REALITY\n"
                f"source {snapshot.observation.execution.source}\n"
                f"spread {snapshot.observation.execution.spread_points:.2f}, "
                f"cost {snapshot.observation.execution.expected_round_trip_cost_points:.2f}\n"
                f"age {snapshot.observation.execution.data_age_seconds:.1f}s, "
                f"fillability {snapshot.observation.execution.fillability:.2f}\n"
                f"BBO {snapshot.observation.execution.bid} / "
                f"{snapshot.observation.execution.ask}\n"
                f"size {snapshot.observation.execution.bid_size} / "
                f"{snapshot.observation.execution.ask_size}, "
                f"depth imbalance {snapshot.observation.execution.depth_imbalance}\n\n"
                f"ACTION UTILITIES\n{utilities}\n\n"
                f"MODEL WHY\n"
                + "\n".join(snapshot.decision.reasons[:3])
                + "\n\nRISK WHY\n"
                + "\n".join(snapshot.risk.reasons[:3])
            ),
            width=58,
        )
        info.axis("off")
        info.set_xlim(0.0, 1.0)
        info.set_ylim(0.0, 1.0)
        info.text(
            0.0,
            1.0,
            info_body,
            ha="left",
            va="top",
            family="monospace",
            fontsize=6.5,
            wrap=True,
            transform=info.transAxes,
            clip_on=True,
        )
        figure.suptitle(
            "CAUSAL DECISION VIEW — COMPLETED DATA ONLY",
            fontsize=13,
            weight="bold",
        )
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, bbox_inches="tight")
        plt.close(figure)
        maximum = max(
            candle.end for values in panels.values() for candle in values
        )
        if maximum > asof:
            raise AssertionError("saved decision artifact contains a future candle")
        setup_id, entry_location_id, entry_path_id = _belief_identity(
            selected_belief
        )
        return VisualArtifact(
            path=destination,
            kind="decision",
            decision_id=(
                f"{snapshot.observation.symbol}:"
                f"{snapshot.observation.instrument_id}:"
                f"{snapshot.observation.asof.isoformat()}"
            ),
            maximum_market_time=maximum,
            hypothesis_key=(
                None if selected_belief is None else selected_belief.key
            ),
            setup_id=setup_id,
            entry_location_id=entry_location_id,
            entry_path_id=entry_path_id,
        )

    @staticmethod
    def build_index(
        decision_artifacts: Sequence[VisualArtifact],
        destination: str | Path,
    ) -> None:
        rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(item.kind)}</td>"
            f"<td>{html.escape(item.decision_id)}</td>"
            f"<td>{html.escape(item.maximum_market_time.isoformat())}</td>"
            f"<td><a href=\"{html.escape(item.path.name)}\">open</a></td>"
            "</tr>"
            for item in decision_artifacts
        )
        document = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMC causal decision views</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #cbd5e1;padding:.55rem;text-align:left}}
th{{background:#f1f5f9}}code{{font-size:.85rem}}
</style></head><body>
<h1>SMC causal decision views</h1>
<p>Each image contains only completed market data available at its decision.</p>
<table><thead><tr><th>kind</th><th>decision id</th><th>maximum market time</th><th>artifact</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>"""
        Path(destination).write_text(document, encoding="utf-8")


__all__ = [
    "DecisionVisualizer",
    "VisualArtifact",
    "active_causal_timeframes",
    "validate_causal_histories",
]
