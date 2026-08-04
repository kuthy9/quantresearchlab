"""Scale-registry decision views with physically separate future reveals."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
import textwrap
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import pandas as pd

from .decision_trace import (
    active_causal_timeframes,
    decision_packet_sha256,
    read_verified_decision_packet,
    sealed_path_audit_context,
    validate_causal_histories,
    write_frozen_decision_packet,
)
from .model import (
    Action,
    Bar,
    Candle,
    DealingRangeLifecycle,
    EngineSnapshot,
    EventKind,
    FairValueGapLifecycle,
    ManipulationLifecycle,
    OrderBlockLifecycle,
    PlaybookPhase,
    Timeframe,
    to_primitive,
)
from .market_clock import scheduled_gap_kind

if TYPE_CHECKING:
    from .ai_review import PrimitiveProposal
    from .validation import PathTestResult


@dataclass(frozen=True)
class VisualArtifact:
    path: Path
    sha256: str
    kind: str
    decision_hash: str
    maximum_market_time: pd.Timestamp
    hypothesis_key: str | None
    setup_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None


@dataclass(frozen=True)
class RevealPermit:
    decision_hash: str
    decision_asof: pd.Timestamp
    hypothesis_key: str | None
    setup_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None
    permit_hash: str


@dataclass
class SealedVisualAudit:
    """Two-step audit: seal a decision first, then feed later bars and reveal."""

    visualizer: "DecisionVisualizer"
    snapshot: EngineSnapshot
    permit: RevealPermit
    decision_artifact: VisualArtifact
    directory: Path
    ai_proposals: tuple["PrimitiveProposal", ...]
    hypothesis_key: str | None
    decision_packet_path: Path
    decision_packet_sha256: str
    decision_packet_hash: str
    future_1m: list[Candle]

    @classmethod
    def seal(
        cls,
        visualizer: "DecisionVisualizer",
        snapshot: EngineSnapshot,
        histories: Mapping[Timeframe, Sequence[Candle]],
        directory: str | Path,
        *,
        ai_proposals: Sequence["PrimitiveProposal"] = (),
        hypothesis_key: str | None = None,
        previous_snapshot: EngineSnapshot | None = None,
        source_bar: Bar | None = None,
        account_state: Any | None = None,
        belief_position_input: Any | None = None,
        expected_decision_packet_hash: str | None = None,
        expected_decision_packet_sha256: str | None = None,
    ) -> "SealedVisualAudit":
        if hypothesis_key is None:
            raise ValueError(
                "sealed visual audit requires an explicit hypothesis"
            )
        belief = _audit_hypothesis(snapshot, hypothesis_key)
        if (
            belief is None
            or belief.plan is None
            or belief.sequence is None
            or belief.sequence.setup_id is None
        ):
            raise ValueError(
                "sealed visual audit requires a frozen plan and sequence"
            )
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        audit_context = sealed_path_audit_context(
            belief.sequence.setup_id
        )
        packet_path = write_frozen_decision_packet(
            snapshot,
            histories,
            root / "decision_packet.json",
            previous_snapshot,
            hypothesis_key=hypothesis_key,
            audit_context=audit_context,
            source_bar=source_bar,
            account_state=account_state,
            belief_position_input=belief_position_input,
        )
        packet = read_verified_decision_packet(packet_path)
        packet_sha256 = decision_packet_sha256(packet_path)
        if (
            expected_decision_packet_hash is None
        ) != (
            expected_decision_packet_sha256 is None
        ):
            raise ValueError(
                "sealed audit requires both expected packet hashes or neither"
            )
        if (
            expected_decision_packet_hash is not None
            and (
                packet["packet_hash"] != expected_decision_packet_hash
                or packet_sha256 != expected_decision_packet_sha256
            )
        ):
            raise ValueError(
                "sealed audit packet differs from the reviewed candidate"
            )
        if ai_proposals and expected_decision_packet_hash is None:
            raise ValueError(
                "AI proposals require their reviewed candidate packet"
            )
        _validate_ai_proposal_packet(
            ai_proposals,
            packet_hash=packet["packet_hash"],
            packet_sha256=packet_sha256,
        )
        decision = visualizer.render_decision(
            snapshot,
            histories,
            root / "decision.png",
            ai_proposals=ai_proposals,
            audit_hypothesis_key=hypothesis_key,
            audit_context=audit_context,
        )
        permit = visualizer.seal_reveal(
            snapshot,
            hypothesis_key=hypothesis_key,
        )
        expected_identity = (
            permit.hypothesis_key,
            permit.setup_id,
            permit.entry_location_id,
            permit.entry_path_id,
        )
        if (
            (
                decision.hypothesis_key,
                decision.setup_id,
                decision.entry_location_id,
                decision.entry_path_id,
            )
            != expected_identity
            or _decision_packet_identity(packet) != expected_identity
        ):
            raise ValueError(
                "sealed visual artifacts do not share one setup identity"
            )
        return cls(
            visualizer=visualizer,
            snapshot=snapshot,
            permit=permit,
            decision_artifact=decision,
            directory=root,
            ai_proposals=tuple(ai_proposals),
            hypothesis_key=hypothesis_key,
            decision_packet_path=packet_path,
            decision_packet_sha256=packet_sha256,
            decision_packet_hash=packet["packet_hash"],
            future_1m=[],
        )

    def on_bar(self, bar: Bar) -> None:
        if bar.start < self.permit.decision_asof:
            raise ValueError("visual audit cannot receive a pre-decision bar")
        if (bar.symbol, bar.instrument_id) != (
            self.snapshot.observation.symbol,
            self.snapshot.observation.instrument_id,
        ):
            # The sealed setup ends at a contract boundary. A new contract's
            # OHLC belongs to another causal path and cannot enter this buffer.
            return
        if (
            self.future_1m
            and bar.start != self.future_1m[-1].end
            and scheduled_gap_kind(self.future_1m[-1].end, bar.start) is None
        ):
            raise ValueError("visual audit future bars must be contiguous")
        self.future_1m.append(
            Candle(
                timeframe=Timeframe.M1,
                start=bar.start,
                end=bar.end,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                symbol=bar.symbol,
                instrument_id=bar.instrument_id,
                observed_minutes=1,
                expected_minutes=1,
                complete=True,
            )
        )

    def reveal(self, path_result: "PathTestResult") -> tuple[VisualArtifact, Path]:
        packet = read_verified_decision_packet(self.decision_packet_path)
        if (
            decision_packet_sha256(self.decision_packet_path)
            != self.decision_packet_sha256
            or packet["packet_hash"] != self.decision_packet_hash
        ):
            raise ValueError(
                "sealed decision packet changed before future reveal"
            )
        if path_result.decision_hash != self.snapshot.snapshot_hash:
            raise ValueError("path result belongs to another decision")
        if (
            self.hypothesis_key is None
            or path_result.hypothesis_key != self.hypothesis_key
        ):
            raise ValueError("path result belongs to another hypothesis")
        belief = _audit_hypothesis(self.snapshot, self.hypothesis_key)
        if (
            belief is None
            or belief.plan is None
            or belief.sequence is None
            or path_result.setup_id != belief.sequence.setup_id
            or path_result.entry_location_id
            != belief.plan.entry_location_id
            or path_result.entry_path_id != belief.plan.entry_path_id
        ):
            raise ValueError("path result belongs to another causal setup")
        visible = tuple(
            candle
            for candle in self.future_1m
            if candle.end <= path_result.resolved_at
        )
        reveal = self.visualizer.render_reveal(
            self.snapshot,
            self.permit,
            visible,
            self.directory / "future_reveal.png",
            revealed_at=path_result.resolved_at,
            path_result=path_result,
            ai_proposals=self.ai_proposals,
            audit_hypothesis_key=self.hypothesis_key,
        )
        record = self.visualizer.write_audit_record(
            self.snapshot,
            self.permit,
            self.decision_artifact,
            reveal,
            self.directory / "audit.json",
            path_result=path_result,
            ai_proposals=self.ai_proposals,
            audit_hypothesis_key=self.hypothesis_key,
            decision_packet_path=self.decision_packet_path,
            expected_decision_packet_sha256=(
                self.decision_packet_sha256
            ),
            expected_decision_packet_hash=self.decision_packet_hash,
        )
        return reveal, record


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


def _audit_hypothesis(snapshot: EngineSnapshot, key: str | None = None):
    selected_key = (
        key if key is not None else snapshot.decision.best_hypothesis_key
    )
    if selected_key is None:
        return None
    belief = snapshot.belief.hypotheses.get(selected_key)
    if belief is None:
        raise ValueError("visual audit hypothesis identity is absent")
    return belief


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


def _decision_packet_identity(
    packet: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None, str | None]:
    hypothesis_key = packet.get("audit_hypothesis_key")
    belief_payload = (
        packet.get("belief_t", {})
        .get("hypotheses", {})
        .get(hypothesis_key)
        if hypothesis_key is not None
        else None
    )
    sequence_payload = (
        None if belief_payload is None else belief_payload.get("sequence")
    )
    plan_payload = (
        None if belief_payload is None else belief_payload.get("plan")
    )
    return (
        hypothesis_key,
        (
            None
            if sequence_payload is None
            else sequence_payload.get("setup_id")
        ),
        (
            None
            if plan_payload is None
            else plan_payload.get("entry_location_id")
        ),
        (
            None
            if plan_payload is None
            else plan_payload.get("entry_path_id")
        ),
    )


def _validate_ai_proposals(
    snapshot: EngineSnapshot,
    belief: Any,
    proposals: Sequence["PrimitiveProposal"],
) -> None:
    setup_id, entry_location_id, entry_path_id = _belief_identity(
        belief
    )
    expected = (
        snapshot.snapshot_hash,
        None if belief is None else belief.key,
        setup_id,
        entry_location_id,
        entry_path_id,
    )
    if any(
        (
            proposal.decision_hash,
            proposal.hypothesis_key,
            proposal.setup_id,
            proposal.entry_location_id,
            proposal.entry_path_id,
        )
        != expected
        for proposal in proposals
    ):
        raise ValueError(
            "AI primitive proposal belongs to another frozen setup"
        )


def _validate_ai_proposal_packet(
    proposals: Sequence["PrimitiveProposal"],
    *,
    packet_hash: str,
    packet_sha256: str,
) -> None:
    if any(
        proposal.decision_packet_hash != packet_hash
        or proposal.decision_packet_sha256 != packet_sha256
        for proposal in proposals
    ):
        raise ValueError(
            "AI primitive proposal belongs to another decision packet"
        )


def _plan_display_context(
    snapshot: EngineSnapshot,
    belief: Any,
    *,
    explicit_audit_key: str | None,
) -> tuple[str, bool]:
    """Describe plan actionability and whether price panels may show its levels."""

    if belief is None or belief.plan is None:
        return "NO COMPLETE CAUSAL PLAN", False
    if explicit_audit_key is not None:
        return (
            f"SEALED PATH-TEST PLAN — DIAGNOSTIC ONLY ({belief.phase.value})",
            True,
        )
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
    key: str | None = None,
) -> str:
    belief = _audit_hypothesis(snapshot, key)
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
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    if belief is None:
        return (), (), (), ()
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
            f"descriptive compression: {metrics['compression']:+.2f}",
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
    if timeframe is Timeframe.M1 and snapshot.observation.group5_authoritative:
        _, paths, reacceptances, micro_bos_references = (
            _selected_group5_entities(
                snapshot,
                belief,
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
            ("directional displacement", metrics["directional_displacement"]),
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
            ("acceptance", metrics["acceptance_direction"]),
            ("rejection", metrics["rejection_direction"]),
            ("range position", metrics["dealing_range_position"]),
            ("up obstruction ATR", metrics["up_path_obstruction_atr"]),
            ("down obstruction ATR", metrics["down_path_obstruction_atr"]),
        )
    elif timeframe is Timeframe.M5:
        values = (
            ("impulse direction", metrics["impulse_direction"]),
            ("impulse strength", metrics["impulse_strength"]),
            ("first pullback depth", metrics["pullback_depth"]),
            ("pullback completeness", metrics["pullback_completeness"]),
            ("reacceptance", metrics["reacceptance_direction"]),
            ("compression", metrics["compression"]),
        )
    else:
        values = (
            ("acceleration", metrics["acceleration"]),
            ("counter pressure", metrics["counter_pressure"]),
            ("trigger hold", metrics["trigger_hold_direction"]),
            ("trigger age", metrics["trigger_age_bars"]),
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
        EventKind.SUPPORT_RESISTANCE_STATE: "#0f766e",
        EventKind.LIQUIDITY_POOL_STATE: "#a21caf",
        EventKind.LIQUIDITY_SWEEP: "#db2777",
        EventKind.LIQUIDITY_CONSUMED: "#9d174d",
        EventKind.STRUCTURE_BREAK: "#2563eb",
        EventKind.STRUCTURE_BREAK_FAILED: "#94a3b8",
        EventKind.REJECTION: "#ea580c",
        EventKind.IMPULSE: "#0891b2",
        EventKind.REACCEPTANCE: "#16a34a",
        EventKind.COMPRESSION: "#64748b",
        EventKind.TRIGGER_HELD: "#15803d",
        EventKind.TRIGGER_LOST: "#dc2626",
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
                    f"{item.lifecycle.value} "
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
        color = "#0f766e"
        axis.fill_between(
            [start_index - 0.4, end_index + 0.4],
            lower,
            upper,
            color=color,
            alpha=0.035,
            zorder=0,
        )
        if offset == len(zones) - 1:
            _collision_safe_annotate(
                axis,
                (
                    f"{state.side} {state.lifecycle.value} "
                    f"[{_short_identity(state.zone_id)}]"
                ),
                min(end_index + 0.35, len(candles) - 1),
                upper,
                color=color,
                fontsize=4.9,
                preferred_side="left",
            )

    pools = tuple(frame.liquidity_pools)[-3:]
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
                f"[{_short_identity(item.entity_id)}]"
            ),
            index,
            price,
            color=color,
            fontsize=5.1,
            priority=offset == len(visible) - 1,
        )


def _group5_overlay(
    axis: Any,
    snapshot: EngineSnapshot,
    candles: Sequence[Candle],
    belief: Any,
) -> None:
    """Mark exact first-pullback, reacceptance, micro-BOS and path order."""

    if not candles or not snapshot.observation.group5_authoritative:
        return
    locations, paths, reacceptances, references = (
        _selected_group5_entities(snapshot, belief)
    )
    if belief is None:
        return
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
        ("FVG", state, "#0284c7")
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
                f"[{_short_identity(entity_id)}]"
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
                f"src={_short_identity(state.source_displacement_id)}"
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


def _group4_text(snapshot: EngineSnapshot) -> str:
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
            f"{resolution} {outcome} "
            f"src={_short_identity(state.source_id)}"
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
    return "\n".join(rows) if rows else "none"


def _group5_text(snapshot: EngineSnapshot, belief: Any) -> str:
    rows: list[str] = []
    locations, paths, reacceptances, micro_bos = (
        _selected_group5_entities(snapshot, belief)
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
    protected_stop: float | None = None,
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
    if protected_stop is not None:
        levels.append(
            ("protected stop", float(protected_stop), "#ea580c", "-.")
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


def _closed_trade_context(
    context: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not context:
        return None
    value = context.get("closed_trade")
    return value if isinstance(value, Mapping) else None


def _protected_stop_context(
    context: Mapping[str, Any] | None,
) -> float | None:
    if not context or context.get("protected_stop") is None:
        return None
    value = float(context["protected_stop"])
    if value <= 0:
        raise ValueError("visual audit protected stop must be positive")
    return value


def _protected_stop_overlay(
    axis: Any,
    protected_stop: float,
    candles: Sequence[Candle],
    *,
    direct_labels: bool,
) -> None:
    _level_overlay(
        axis,
        candles,
        (("protected stop", float(protected_stop), "#ea580c", "-."),),
        direct_labels=direct_labels,
    )


def _closed_trade_levels(
    trade: Mapping[str, Any],
) -> tuple[tuple[str, float, str, str], ...]:
    raw = (
        ("entry", float(trade["entry_price"]), "#111827", "--"),
        (
            "original invalidation",
            float(trade["original_invalidation"]),
            "#dc2626",
            "--",
        ),
        ("final stop", float(trade["final_stop"]), "#ea580c", ":"),
        ("target", float(trade["target"]), "#16a34a", "--"),
        ("exit", float(trade["exit_price"]), "#7c3aed", "-"),
    )
    grouped: list[list[Any]] = []
    for label, price, color, style in raw:
        existing = next(
            (item for item in grouped if abs(float(item[1]) - price) <= 1e-9),
            None,
        )
        if existing is None:
            grouped.append([label, price, color, style])
        else:
            existing[0] = f"{existing[0]} / {label}"
    return tuple(
        (str(label), float(price), str(color), str(style))
        for label, price, color, style in grouped
    )


def _closed_trade_overlay(
    axis: Any,
    trade: Mapping[str, Any],
    candles: Sequence[Candle],
    *,
    direct_labels: bool,
) -> None:
    _level_overlay(
        axis,
        candles,
        _closed_trade_levels(trade),
        direct_labels=direct_labels,
    )
    if not candles or not direct_labels:
        return
    markers = (
        ("opened", pd.Timestamp(trade["opened_at"]), "#111827", "--"),
        ("closed", pd.Timestamp(trade["closed_at"]), "#7c3aed", "-"),
    )
    for label, timestamp, color, style in markers:
        if timestamp < candles[0].start or timestamp > candles[-1].end:
            continue
        index = next(
            (
                number
                for number, candle in enumerate(candles)
                if candle.start <= timestamp <= candle.end
            ),
            None,
        )
        if index is None:
            continue
        axis.axvline(index, color=color, linestyle=style, linewidth=0.7)
        axis.text(
            index,
            0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=5.5,
            color=color,
            rotation=90,
            transform=axis.get_xaxis_transform(),
        )


def _closed_trade_text(trade: Mapping[str, Any]) -> str:
    return (
        f"{trade['playbook']} / {trade['direction']}\n"
        f"setup {trade.get('setup_id') or 'legacy/untyped'}\n"
        f"location {trade.get('entry_location_id') or 'legacy/untyped'}\n"
        f"path {trade.get('entry_path_id') or 'legacy/untyped'}\n"
        f"decision {pd.Timestamp(trade['decision_time']):%Y-%m-%d %H:%M %Z}\n"
        f"opened {pd.Timestamp(trade['opened_at']):%Y-%m-%d %H:%M %Z}\n"
        f"closed {pd.Timestamp(trade['closed_at']):%Y-%m-%d %H:%M %Z}\n"
        f"entry {float(trade['entry_price']):.2f}\n"
        f"original invalidation {float(trade['original_invalidation']):.2f}\n"
        f"final stop {float(trade['final_stop']):.2f}\n"
        f"target {float(trade['target']):.2f}\n"
        f"exit {float(trade['exit_price']):.2f} ({trade['exit_reason']})\n"
        f"gross {float(trade['gross_R']):+.3f}R, "
        f"cost {float(trade['cost_R']):.3f}R, "
        f"net {float(trade['net_R']):+.3f}R\n"
        f"same-bar ambiguity {bool(trade['ambiguous_same_bar'])}"
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

    return "\n".join(
        (
            f"FROZEN LIQUIDITY ROUTE {route.route_id}",
            f"  selected {route.selected_at:%Y-%m-%d %H:%M %Z}",
            f"  context draw {identity(route.context_draw_id)}",
            "  intermediate liquidity "
            f"{identities(route.intermediate_liquidity_ids)}",
            "  primary deliverable "
            f"{identity(route.primary_deliverable_target_id)}",
            f"  terminal draw {identity(route.terminal_draw_id)}",
            f"  path blockers {identities(route.path_blocker_ids)}",
            f"  source paths {identities(route.source_path_ids)}",
        )
    )


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
    key: str | None = None,
) -> str:
    belief = _audit_hypothesis(snapshot, key)
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


def _review_text(proposals: Sequence["PrimitiveProposal"]) -> str:
    if not proposals:
        return "none"
    rows = [
        f"{proposal.issue.value} ({proposal.confidence:.2f})\n"
        f"→ {proposal.primitive_name} [{proposal.status}]\n"
        f"  definition: v{proposal.formula_version} "
        f"{proposal.definition_hash[:12]}\n"
        f"  origin computation: "
        f"{'evaluable' if proposal.origin_value.evaluable else 'not evaluable'}"
        f" ({proposal.origin_value.reason})\n"
        f"  packet: {proposal.decision_packet_hash[:12]}\n"
        f"  setup: {proposal.hypothesis_key} / "
        f"{proposal.setup_id or 'unstarted'}\n"
        f"  formula: {proposal.formula}\n"
        f"  clock: {proposal.clock_rule}\n"
        f"  path test: {proposal.path_test}"
        for proposal in proposals[:2]
    ]
    if len(proposals) > 2:
        rows.append(
            f"+ {len(proposals) - 2} additional proposal(s) in packet"
        )
    return "\n".join(rows)


def _audit_context_text(context: Mapping[str, Any] | None) -> str:
    if not context:
        return "periodic decision sample"
    rows = []
    for key, value in context.items():
        if key == "closed_trade":
            rows.append("closed trade: frozen geometry shown below")
            continue
        name = str(key).replace("_", " ")
        rendered = json.dumps(
            to_primitive(value),
            ensure_ascii=False,
            sort_keys=True,
        )
        rows.append(f"{name}: {rendered}")
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
        switched_at = getattr(focus, "switched_at", None)
        rows.extend(
            (
                "\nFOCUS",
                "primary "
                + ", ".join(getattr(focus, "primary_timeframes", ())),
                "supplemental "
                + (
                    ", ".join(
                        getattr(focus, "supplemental_timeframes", ())
                    )
                    or "none"
                ),
                "reason "
                + ", ".join(getattr(focus, "reason_codes", ())),
                "trigger "
                + (
                    ", ".join(
                        _short_identity(value)
                        for value in getattr(
                            focus,
                            "trigger_event_ids",
                            (),
                        )
                    )
                    or "none"
                ),
                f"question {getattr(focus, 'question', 'unknown')}",
                f"status {resolution}",
                "switch "
                + (
                    "none"
                    if switched_at is None
                    else switched_at.strftime("%m-%d %H:%M")
                ),
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

    rows.append("\nCAUSAL GRAPH PATHS")
    path_rows = 0
    for identity in ordered_ids[:3]:
        context = contexts.get(identity)
        if context is None:
            continue
        for prefix, paths in (
            ("+", context.supporting_graph_paths),
            ("−", context.contradicting_graph_paths),
        ):
            for path in paths[:2]:
                rows.append(
                    f"{prefix} {_short_identity(identity)} "
                    f"{_graph_path_text(path)}"
                )
                path_rows += 1
    if path_rows == 0:
        rows.append("none frozen")

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

    def render_decision(
        self,
        snapshot: EngineSnapshot,
        histories: Mapping[Timeframe, Sequence[Candle]],
        destination: str | Path,
        *,
        ai_proposals: Sequence["PrimitiveProposal"] = (),
        audit_hypothesis_key: str | None = None,
        audit_context: Mapping[str, Any] | None = None,
        suppress_audit_hypothesis: bool = False,
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
        closed_trade = _closed_trade_context(audit_context)
        protected_stop = _protected_stop_context(audit_context)
        audit_belief = (
            None
            if suppress_audit_hypothesis
            else _audit_hypothesis(snapshot, audit_hypothesis_key)
        )
        _validate_ai_proposals(
            snapshot,
            audit_belief,
            ai_proposals,
        )
        plan = None if audit_belief is None else audit_belief.plan
        if closed_trade is None:
            plan_heading, show_plan_overlay = _plan_display_context(
                snapshot,
                audit_belief,
                explicit_audit_key=audit_hypothesis_key,
            )
        else:
            plan_heading = "FROZEN CLOSED-TRADE GEOMETRY — AUDIT ONLY"
            show_plan_overlay = False
        typed_model = any(
            item.thesis_strength is not None
            for item in snapshot.belief.hypotheses.values()
        )
        focus = getattr(snapshot.belief, "focus_state", None)
        primary_focus = set(
            () if focus is None else focus.primary_timeframes
        )
        supplemental_focus = set(
            () if focus is None else focus.supplemental_timeframes
        )
        for axis, timeframe in zip(axes, display_timeframes):
            values = panels[timeframe]
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
                    audit_belief,
                )
            _event_markers(axis, snapshot, timeframe, values)
            focus_label = (
                " · PRIMARY FOCUS"
                if timeframe.value in primary_focus
                else " · SUPPLEMENTAL FOCUS"
                if timeframe.value in supplemental_focus
                else ""
            )
            if timeframe.value in primary_focus:
                for spine in axis.spines.values():
                    spine.set_color("#2563eb")
                    spine.set_linewidth(1.8)
            elif timeframe.value in supplemental_focus:
                for spine in axis.spines.values():
                    spine.set_color("#d97706")
                    spine.set_linewidth(1.3)
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
                range_low = max(frame.metrics["dealing_range_low"], candle_low)
                range_high = min(frame.metrics["dealing_range_high"], candle_high)
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
                        "legacy rolling range proxy",
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
                    protected_stop=protected_stop,
                    direct_labels=timeframe
                    in {Timeframe.M5, Timeframe.M1},
                )
            else:
                _belief_geometry_overlay(
                    axis,
                    audit_belief,
                    values,
                    direct_labels=timeframe
                    in {Timeframe.M5, Timeframe.M1},
                )
                if protected_stop is not None:
                    _protected_stop_overlay(
                        axis,
                        protected_stop,
                        values,
                        direct_labels=timeframe in {Timeframe.M5, Timeframe.M1},
                    )
            if closed_trade is not None:
                _closed_trade_overlay(
                    axis,
                    closed_trade,
                    values,
                    direct_labels=timeframe in {Timeframe.M5, Timeframe.M1},
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
                        "identities remain in packet/event memory"
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
                _metric_text(snapshot, timeframe, audit_belief),
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
        plan_text = _partial_geometry_text(snapshot, audit_belief)
        if closed_trade is not None:
            plan_text = _closed_trade_text(closed_trade)
        elif plan is not None:
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
                    f"{context.reentry_price:.2f} at "
                    f"{context.reentered_at:%m-%d %H:%M}"
                    "\n  opposite liquidity "
                    f"{context.opposite_liquidity_id}"
                )
            if plan.liquidity_route is not None:
                plan_text += "\n" + _liquidity_route_text(
                    plan.liquidity_route
                )
            if protected_stop is not None:
                plan_text += f"\nprotected stop {protected_stop:.2f}"
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
        sequence_heading = (
            "CURRENT MARKET STATE AT CLOSE (NOT FROZEN THESIS EVIDENCE)"
            if closed_trade is not None
            else "AUDITED SETUP SEQUENCE"
        )
        evidence_heading = (
            "CURRENT EVIDENCE AT CLOSE (NOT FROZEN THESIS EVIDENCE)"
            if closed_trade is not None
            else "AUDITED EVIDENCE"
        )
        evidence_text = (
            (
                "not shown — no close-time hypothesis is substituted "
                "for the original frozen thesis"
                if closed_trade is not None
                else "not shown — no exact audit hypothesis identity "
                "is available at this decision clock"
            )
            if suppress_audit_hypothesis
            else _format_evidence(snapshot, audit_hypothesis_key)
        )
        sequence_text = (
            (
                "not shown — post-outcome view uses frozen trade geometry"
                if closed_trade is not None
                else "not shown — no exact audit hypothesis identity "
                "is available at this decision clock"
            )
            if suppress_audit_hypothesis
            else _sequence_text(snapshot, audit_hypothesis_key)
        )
        clock_heading = (
            "POST-OUTCOME VIEW CLOCK"
            if closed_trade is not None
            else "DECISION CLOCK"
        )
        action_heading = (
            "CURRENT CLOSE-TIME MODEL / RISK ACTION"
            if closed_trade is not None
            else "MODEL / RISK ACTION"
        )
        belief_heading = (
            "CURRENT CLOSE-TIME BELIEFS — NOT ORIGINAL THESIS"
            if closed_trade is not None
            else "PLAYBOOK BELIEFS"
        )
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
                f"{clock_heading}\n{asof:%Y-%m-%d %H:%M %Z}\n\n"
                f"SCENARIO AUDIT\n{_audit_context_text(audit_context)}\n\n"
                f"{action_heading}\n"
                f"{snapshot.decision.selected_action.value} / "
                f"{snapshot.risk.final_action.value}\n"
                f"advantage {snapshot.decision.advantage:.3f}R\n"
                f"veto: {veto}\n\n"
                f"{belief_heading}\n{probabilities}\n\n"
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
                f"{_group5_text(snapshot, audit_belief)}\n\n"
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
                f"AI AUDIT → SEQUENCE PRIMITIVE\n{_review_text(ai_proposals)}\n\n"
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
            (
                "POST-OUTCOME AUDIT VIEW — original-trade outcome is present"
                if closed_trade is not None
                else "CAUSAL DECISION VIEW — future path is not present"
            ),
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
            audit_belief
        )
        return VisualArtifact(
            path=destination,
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            kind="decision",
            decision_hash=snapshot.snapshot_hash,
            maximum_market_time=maximum,
            hypothesis_key=(
                None if audit_belief is None else audit_belief.key
            ),
            setup_id=setup_id,
            entry_location_id=entry_location_id,
            entry_path_id=entry_path_id,
        )

    @staticmethod
    def seal_reveal(
        snapshot: EngineSnapshot,
        hypothesis_key: str | None = None,
    ) -> RevealPermit:
        belief = _audit_hypothesis(snapshot, hypothesis_key)
        setup_id, entry_location_id, entry_path_id = _belief_identity(
            belief
        )
        selected_key = None if belief is None else belief.key
        raw = (
            f"{snapshot.snapshot_hash}|"
            f"{snapshot.observation.asof.isoformat()}|{selected_key}|"
            f"{setup_id}|{entry_location_id}|{entry_path_id}|"
            "future-separate"
        )
        return RevealPermit(
            decision_hash=snapshot.snapshot_hash,
            decision_asof=snapshot.observation.asof,
            hypothesis_key=selected_key,
            setup_id=setup_id,
            entry_location_id=entry_location_id,
            entry_path_id=entry_path_id,
            permit_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )

    def render_reveal(
        self,
        snapshot: EngineSnapshot,
        permit: RevealPermit,
        future_1m: Sequence[Candle],
        destination: str | Path,
        *,
        revealed_at: pd.Timestamp,
        path_result: "PathTestResult | None" = None,
        ai_proposals: Sequence["PrimitiveProposal"] = (),
        audit_hypothesis_key: str | None = None,
    ) -> VisualArtifact:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        revealed_at = pd.Timestamp(revealed_at)
        if revealed_at.tzinfo is None:
            raise ValueError("reveal clock must be timezone aware")
        if permit.decision_hash != snapshot.snapshot_hash:
            raise ValueError("reveal permit is not bound to this decision")
        expected_permit = self.seal_reveal(
            snapshot,
            hypothesis_key=permit.hypothesis_key,
        )
        if permit != expected_permit:
            raise ValueError("reveal permit integrity check failed")
        if (
            audit_hypothesis_key is not None
            and audit_hypothesis_key != permit.hypothesis_key
        ):
            raise ValueError(
                "future reveal hypothesis differs from its permit"
            )
        values = tuple(future_1m)
        if any(
            candle.timeframe is not Timeframe.M1
            or not candle.complete
            or (candle.symbol, candle.instrument_id)
            != (
                snapshot.observation.symbol,
                snapshot.observation.instrument_id,
            )
            for candle in values
        ):
            raise ValueError(
                "future reveal requires completed 1m bars for the sealed contract"
            )
        if any(candle.start < permit.decision_asof for candle in values):
            raise ValueError("reveal path overlaps the decision information set")
        if values and (
            values[0].start != permit.decision_asof
            and scheduled_gap_kind(
                permit.decision_asof,
                values[0].start,
            )
            is None
        ):
            raise ValueError("future reveal begins after an unexplained gap")
        if any(
            right.start != left.end
            and scheduled_gap_kind(left.end, right.start) is None
            for left, right in zip(values[:-1], values[1:])
        ):
            raise ValueError("future reveal path is not causally contiguous")
        path_resolved_at = (
            None
            if path_result is None
            else pd.Timestamp(path_result.resolved_at)
        )
        if not values:
            boundary_outcomes = {"contract_change", "deadline", "right_censored"}
            if (
                path_result is None
                or path_result.outcome not in boundary_outcomes
                or path_resolved_at is None
                or path_resolved_at < permit.decision_asof
                or (
                    path_resolved_at != permit.decision_asof
                    and scheduled_gap_kind(
                        permit.decision_asof,
                        path_resolved_at,
                    )
                    is None
                )
                or path_resolved_at > revealed_at
            ):
                raise ValueError(
                    "empty future reveal requires a terminal decision-boundary result"
                )
        if values and revealed_at < max(candle.end for candle in values):
            raise ValueError("future path cannot be revealed before it has completed")
        if path_result is not None and (
            path_resolved_at < permit.decision_asof
            or path_resolved_at > revealed_at
            or (
                values
                and max(candle.end for candle in values)
                > path_resolved_at
            )
        ):
            raise ValueError(
                "future reveal extends beyond the frozen path resolution"
            )
        if path_result is not None and values:
            last_end = values[-1].end
            boundary_outcomes = {
                "contract_change",
                "deadline",
                "right_censored",
            }
            if (
                last_end != path_resolved_at
                and (
                    path_result.outcome not in boundary_outcomes
                    or scheduled_gap_kind(last_end, path_resolved_at)
                    is None
                )
            ):
                raise ValueError(
                    "future reveal does not reach the frozen path resolution"
                )

        long_path = len(values) > 180
        if long_path:
            figure = plt.figure(
                figsize=(18, 11),
                dpi=110,
                constrained_layout=True,
            )
            grid = figure.add_gridspec(
                3,
                2,
                width_ratios=(3.5, 1.5),
                height_ratios=(1.0, 1.5, 1.5),
            )
            axis = figure.add_subplot(grid[0, 0])
            entry_axis = figure.add_subplot(grid[1, 0])
            resolution_axis = figure.add_subplot(grid[2, 0])
            audit = figure.add_subplot(grid[:, 1])
            axis.plot(
                range(len(values)),
                [candle.close for candle in values],
                color="#0f172a",
                linewidth=0.8,
            )
            axis.grid(
                True,
                color="#dbe4ee",
                linewidth=0.4,
                alpha=0.7,
            )
            ticks = sorted(
                set((0, len(values) // 2, len(values) - 1))
            )
            axis.set_xticks(ticks)
            axis.set_xticklabels(
                [
                    values[index].start.strftime("%m-%d\n%H:%M")
                    for index in ticks
                ],
                fontsize=7,
            )
            entry_clock = (
                None
                if path_result is None
                else getattr(path_result, "entry_touched_at", None)
            )
            entry_index = next(
                (
                    index
                    for index, candle in enumerate(values)
                    if entry_clock is not None
                    and candle.end >= pd.Timestamp(entry_clock)
                ),
                0,
            )
            entry_start = max(0, entry_index - 40)
            entry_end = min(len(values), entry_index + 81)
            resolution_start = max(0, len(values) - 121)
            entry_values = values[entry_start:entry_end]
            resolution_values = values[resolution_start:]
            _candles(entry_axis, entry_values)
            _candles(resolution_axis, resolution_values)
            entry_axis.set_title(
                "ENTRY-TOUCH DETAIL"
                if entry_clock is not None
                else "EARLY PATH DETAIL — entry not touched",
                loc="left",
                fontsize=9,
                weight="bold",
            )
            resolution_axis.set_title(
                "RESOLUTION DETAIL",
                loc="left",
                fontsize=9,
                weight="bold",
            )
            price_axes = (
                (axis, values, 0),
                (entry_axis, entry_values, entry_start),
                (
                    resolution_axis,
                    resolution_values,
                    resolution_start,
                ),
            )
        else:
            figure, (axis, audit) = plt.subplots(
                1,
                2,
                figsize=(17, 6),
                dpi=110,
                constrained_layout=True,
                gridspec_kw={"width_ratios": (3.4, 1.6)},
            )
            _candles(axis, values)
            price_axes = ((axis, values, 0),)
        if not values:
            axis.text(
                0.5,
                0.5,
                "No completed post-decision 1m bar\n"
                "Path terminated at the decision boundary",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=11,
                color="#991b1b",
            )
            axis.set_xticks([])
            axis.set_yticks([])
        audit_belief = _audit_hypothesis(
            snapshot,
            permit.hypothesis_key,
        )
        _validate_ai_proposals(
            snapshot,
            audit_belief,
            ai_proposals,
        )
        plan = None if audit_belief is None else audit_belief.plan
        if plan is not None:
            for price_axis, shown_values, _ in price_axes:
                _plan_overlay(price_axis, plan, shown_values)
        if values:
            entry_clock = (
                None
                if path_result is None
                else getattr(path_result, "entry_touched_at", None)
            )
            entry_global_index = (
                None
                if entry_clock is None
                else next(
                    (
                        index
                        for index, candle in enumerate(values)
                        if candle.end >= pd.Timestamp(entry_clock)
                    ),
                    None,
                )
            )
            resolution_global_index = (
                len(values) - 1
                if (
                    path_resolved_at is not None
                    and values[-1].end == path_resolved_at
                )
                else None
            )
            for price_axis, shown_values, offset in price_axes:
                if not shown_values:
                    continue
                if entry_global_index is not None and (
                    offset
                    <= entry_global_index
                    < offset + len(shown_values)
                ):
                    price_axis.axvline(
                        entry_global_index - offset,
                        color="#2563eb",
                        linestyle=":",
                        linewidth=1.0,
                        label="entry touch",
                    )
                if (
                    resolution_global_index is not None
                    and
                    offset
                    <= resolution_global_index
                    < offset + len(shown_values)
                ):
                    price_axis.axvline(
                        resolution_global_index - offset,
                        color="#991b1b",
                        linestyle="-.",
                        linewidth=1.0,
                        label="resolution",
                    )
            axis.axvline(
                0,
                color="#7c3aed",
                linestyle="--",
                linewidth=0.9,
                label="decision boundary",
            )
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best", fontsize=8)
        audit.axis("off")
        result_text = "No registered path-test result supplied"
        if path_result is not None:
            if path_result.decision_hash != snapshot.snapshot_hash:
                raise ValueError("path result is not bound to the sealed decision")
            expected_setup_id = (
                None
                if audit_belief is None
                or audit_belief.sequence is None
                else audit_belief.sequence.setup_id
            )
            if (
                plan is None
                or audit_belief is None
                or audit_belief.sequence is None
                or expected_setup_id is None
                or path_result.hypothesis_key != audit_belief.key
                or path_result.setup_id != expected_setup_id
                or path_result.entry_location_id != plan.entry_location_id
                or path_result.entry_path_id != plan.entry_path_id
                or path_result.playbook != audit_belief.playbook.value
                or path_result.direction != audit_belief.direction.value
                or pd.Timestamp(path_result.decision_time)
                != snapshot.observation.asof
                or path_result.invalidation_source_id
                != plan.invalidation.source_level_id
                or path_result.target_source_id != plan.targets[0].level_id
                or path_result.protocol_version
                != audit_belief.sequence.protocol_version
                or path_result.protocol_hash
                != audit_belief.sequence.protocol_hash
                or not all(
                    (
                        abs(plan.planned_entry - path_result.entry) <= 1e-9,
                        abs(plan.invalidation.price - path_result.invalidation) <= 1e-9,
                        abs(plan.targets[0].price - path_result.target) <= 1e-9,
                    )
                )
            ):
                raise ValueError("path result does not match the displayed frozen plan")
            result_text = (
                f"outcome {path_result.outcome}\n"
                f"success {path_result.success}\n"
                f"resolved {path_result.resolved_at:%Y-%m-%d %H:%M %Z}\n"
                f"MFE {path_result.mfe_R:.2f}R\n"
                f"MAE {path_result.mae_R:.2f}R\n"
                f"formation {path_result.formation_minutes}m\n"
                f"entry touched {path_result.entry_touched} "
                f"({path_result.time_to_entry_minutes}m)\n"
                f"elapsed {path_result.elapsed_minutes}m\n"
                f"same-bar ambiguity {path_result.ambiguous_same_bar}\n"
                f"protocol {path_result.protocol_hash[:12]}\n"
                f"config {path_result.config_hash[:12]}\n"
                f"code {path_result.code_hash[:12]}"
            )
        audit.text(
            0.0,
            1.0,
            (
                "SEALED PATH TEST\n"
                f"{result_text}\n\n"
                "PRE-REVEAL AI AUDIT\n"
                f"{_review_text(ai_proposals)}\n\n"
                "The AI issue is diagnostic only. Its mapped primitive remains "
                "unvalidated until a separately registered path test passes."
            ),
            ha="left",
            va="top",
            fontsize=8,
            family="monospace",
            wrap=True,
            transform=audit.transAxes,
        )
        axis.set_title(
            f"FUTURE REVEAL — sealed decision {snapshot.snapshot_hash[:12]} "
            f"at {permit.decision_asof:%Y-%m-%d %H:%M %Z}",
            loc="left",
            color="#991b1b",
            weight="bold",
        )
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, bbox_inches="tight")
        plt.close(figure)
        maximum = max(
            (candle.end for candle in values),
            default=permit.decision_asof,
        )
        setup_id, entry_location_id, entry_path_id = _belief_identity(
            audit_belief
        )
        return VisualArtifact(
            path=destination,
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            kind="future_reveal",
            decision_hash=snapshot.snapshot_hash,
            maximum_market_time=maximum,
            hypothesis_key=permit.hypothesis_key,
            setup_id=setup_id,
            entry_location_id=entry_location_id,
            entry_path_id=entry_path_id,
        )

    @staticmethod
    def write_audit_record(
        snapshot: EngineSnapshot,
        permit: RevealPermit,
        decision_artifact: VisualArtifact,
        reveal_artifact: VisualArtifact,
        destination: str | Path,
        *,
        path_result: "PathTestResult | None" = None,
        ai_proposals: Sequence["PrimitiveProposal"] = (),
        audit_hypothesis_key: str | None = None,
        decision_packet_path: str | Path | None = None,
        expected_decision_packet_sha256: str | None = None,
        expected_decision_packet_hash: str | None = None,
    ) -> Path:
        if decision_artifact.kind != "decision":
            raise ValueError("audit record requires a decision artifact")
        if reveal_artifact.kind != "future_reveal":
            raise ValueError("audit record requires a separate reveal artifact")
        if {
            snapshot.snapshot_hash,
            permit.decision_hash,
            decision_artifact.decision_hash,
            reveal_artifact.decision_hash,
        } != {snapshot.snapshot_hash}:
            raise ValueError("audit artifacts do not share one sealed decision")
        expected_identity = (
            permit.hypothesis_key,
            permit.setup_id,
            permit.entry_location_id,
            permit.entry_path_id,
        )
        if (
            (
                decision_artifact.hypothesis_key,
                decision_artifact.setup_id,
                decision_artifact.entry_location_id,
                decision_artifact.entry_path_id,
            )
            != expected_identity
            or (
                reveal_artifact.hypothesis_key,
                reveal_artifact.setup_id,
                reveal_artifact.entry_location_id,
                reveal_artifact.entry_path_id,
            )
            != expected_identity
            or audit_hypothesis_key != permit.hypothesis_key
        ):
            raise ValueError(
                "audit artifacts do not share one frozen setup identity"
            )
        packet_record = None
        if decision_packet_path is not None:
            packet_path = Path(decision_packet_path)
            packet = read_verified_decision_packet(packet_path)
            packet_sha256 = decision_packet_sha256(packet_path)
            packet_identity = _decision_packet_identity(packet)
            if (
                packet.get("decision_hash") != snapshot.snapshot_hash
                or packet.get("future_path", {}).get("included") is not False
                or packet_identity != expected_identity
                or (
                    expected_decision_packet_sha256 is not None
                    and packet_sha256
                    != expected_decision_packet_sha256
                )
                or (
                    expected_decision_packet_hash is not None
                    and packet.get("packet_hash")
                    != expected_decision_packet_hash
                )
            ):
                raise ValueError(
                    "pre-reveal decision packet is not bound to the audit"
                )
            packet_record = {
                "path": str(packet_path),
                "sha256": packet_sha256,
                "packet_hash": packet.get("packet_hash"),
            }
            _validate_ai_proposal_packet(
                ai_proposals,
                packet_hash=packet["packet_hash"],
                packet_sha256=packet_sha256,
            )
        elif ai_proposals:
            raise ValueError(
                "AI proposals require a verified pre-reveal decision packet"
            )
        payload = {
            "decision_hash": snapshot.snapshot_hash,
            "decision_asof": snapshot.observation.asof,
            "permit_hash": permit.permit_hash,
            "decision_artifact": to_primitive(decision_artifact),
            "future_reveal_artifact": to_primitive(reveal_artifact),
            "path_result": (
                None if path_result is None else to_primitive(path_result)
            ),
            "ai_primitive_proposals": [
                to_primitive(proposal) for proposal in ai_proposals
            ],
            "audit_hypothesis_key": audit_hypothesis_key,
            "pre_reveal_decision_packet": packet_record,
            "future_was_separate_from_decision": True,
        }
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(to_primitive(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output

    @staticmethod
    def build_index(
        decision_artifacts: Sequence[VisualArtifact],
        destination: str | Path,
    ) -> None:
        rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(item.kind)}</td>"
            f"<td>{html.escape(item.decision_hash[:16])}</td>"
            f"<td>{html.escape(item.maximum_market_time.isoformat())}</td>"
            f"<td><a href=\"{html.escape(item.path.name)}\">open</a></td>"
            "</tr>"
            for item in decision_artifacts
        )
        document = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMC v2 causal decision review</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #cbd5e1;padding:.55rem;text-align:left}}
th{{background:#f1f5f9}}code{{font-size:.85rem}}
</style></head><body>
<h1>SMC v2 causal decision review</h1>
<p>Decision views and future reveals are separate rows and separate files.</p>
<table><thead><tr><th>kind</th><th>decision hash</th><th>maximum market time</th><th>artifact</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>"""
        Path(destination).write_text(document, encoding="utf-8")


__all__ = [
    "DecisionVisualizer",
    "RevealPermit",
    "SealedVisualAudit",
    "VisualArtifact",
]
