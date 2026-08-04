"""Quota-bound scenario image audits with explicit missing-coverage manifests."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .ai_review import ai_review_identity
from .decision_trace import (
    decision_packet_sha256,
    read_verified_decision_packet,
    write_frozen_decision_packet,
)
from .model import Action, EngineSnapshot, Timeframe, to_primitive
from .visualization import DecisionVisualizer, VisualArtifact


class AuditScenario(str, Enum):
    ENTER = "enter"
    WAIT = "wait"
    ABSTAIN = "abstain"
    VETO = "veto"
    STOP = "stop"
    TARGET = "target"
    PROTECT = "protect"
    HOLD = "hold"
    EXIT = "exit"


@dataclass(frozen=True)
class ScenarioCapture:
    scenario: AuditScenario
    decision_hash: str
    asof: pd.Timestamp
    artifact: VisualArtifact
    context: Mapping[str, Any]
    audit_stage: str
    hypothesis_key: str | None
    setup_id: str | None
    entry_location_id: str | None
    entry_path_id: str | None
    decision_packet: Path
    decision_packet_hash: str
    decision_packet_sha256: str


def classify_audit_scenarios(
    snapshot: EngineSnapshot,
    closed_trades: Sequence[Any] = (),
) -> tuple[tuple[AuditScenario, Mapping[str, Any]], ...]:
    output: list[tuple[AuditScenario, Mapping[str, Any]]] = []
    if snapshot.risk.final_action is Action.ENTER:
        output.append(
            (
                AuditScenario.ENTER,
                {
                    "trigger": "risk_approved_enter",
                    "requested_action": snapshot.risk.requested_action.value,
                    "final_action": snapshot.risk.final_action.value,
                },
            )
        )
    if snapshot.risk.final_action is Action.WAIT:
        output.append(
            (
                AuditScenario.WAIT,
                {
                    "trigger": "risk_retained_wait",
                    "advantage_R": snapshot.decision.advantage,
                },
            )
        )
    if (
        snapshot.risk.final_action is Action.ABSTAIN
        and not (
            snapshot.risk.requested_action is Action.ENTER
            and snapshot.risk.vetoes
        )
    ):
        output.append(
            (
                AuditScenario.ABSTAIN,
                {
                    "trigger": "utility_margin_or_missing_causal_gate",
                    "advantage_R": snapshot.decision.advantage,
                    "model_action": snapshot.decision.selected_action.value,
                },
            )
        )
    if (
        snapshot.risk.requested_action is Action.ENTER
        and snapshot.risk.final_action is not Action.ENTER
        and snapshot.risk.vetoes
    ):
        output.append(
            (
                AuditScenario.VETO,
                {
                    "trigger": "independent_risk_veto",
                    "requested_action": snapshot.risk.requested_action.value,
                    "final_action": snapshot.risk.final_action.value,
                    "vetoes": [item.value for item in snapshot.risk.vetoes],
                },
            )
        )
    if snapshot.risk.final_action is Action.PROTECT:
        output.append(
            (
                AuditScenario.PROTECT,
                {
                    "trigger": "risk_approved_protection",
                    "protected_stop": snapshot.risk.protected_stop,
                },
            )
        )
    if snapshot.risk.final_action is Action.HOLD:
        output.append(
            (
                AuditScenario.HOLD,
                {
                    "trigger": "risk_retained_position_hold",
                    "model_action": snapshot.decision.selected_action.value,
                },
            )
        )
    if snapshot.risk.final_action is Action.EXIT:
        output.append(
            (
                AuditScenario.EXIT,
                {
                    "trigger": "risk_approved_position_exit",
                    "model_action": snapshot.decision.selected_action.value,
                },
            )
        )
    for trade in closed_trades:
        closed_at = pd.Timestamp(trade.closed_at)
        if closed_at > snapshot.observation.asof:
            raise ValueError("scenario audit received a future closed trade")
        reason = str(trade.exit_reason).lower()
        closed_trade = {
            "thesis_hash": trade.thesis_hash,
            "playbook": trade.playbook,
            "direction": trade.direction,
            "setup_id": getattr(trade, "setup_id", None),
            "entry_location_id": getattr(
                trade,
                "entry_location_id",
                None,
            ),
            "entry_path_id": getattr(trade, "entry_path_id", None),
            "decision_time": pd.Timestamp(trade.decision_time),
            "opened_at": pd.Timestamp(trade.opened_at),
            "closed_at": closed_at,
            "entry_price": float(trade.entry_price),
            "original_invalidation": float(trade.original_invalidation),
            "final_stop": float(trade.final_stop),
            "target": float(trade.target),
            "exit_price": float(trade.exit_price),
            "exit_reason": trade.exit_reason,
            "gross_R": float(trade.gross_R),
            "cost_R": float(trade.cost_R),
            "net_R": float(trade.net_R),
            "ambiguous_same_bar": bool(trade.ambiguous_same_bar),
        }
        context = {
            "trigger": "sequential_trade_closed",
            "audit_stage": "post_outcome",
            "future_relative_to_original_thesis_present": True,
            "thesis_hash": trade.thesis_hash,
            "exit_reason": trade.exit_reason,
            "gross_R": trade.gross_R,
            "net_R": trade.net_R,
            "closed_at": closed_at,
            "closed_trade": closed_trade,
        }
        if "target" in reason:
            output.append((AuditScenario.TARGET, context))
        elif "stop" in reason:
            output.append((AuditScenario.STOP, context))
    return tuple(output)


def _position_bound_hypothesis_key(
    snapshot: EngineSnapshot,
    position: Any | None,
) -> str | None:
    if position is None or getattr(position, "setup_id", None) is None:
        return None
    expected = (
        position.setup_id,
        position.entry_location_id,
        position.entry_path_id,
    )
    matches = []
    for hypothesis in snapshot.belief.hypotheses.values():
        plan = hypothesis.plan
        if plan is None:
            continue
        if (
            plan.setup_id,
            plan.entry_location_id,
            plan.entry_path_id,
        ) == expected:
            matches.append(hypothesis.key)
    if len(matches) > 1:
        raise ValueError(
            "position matches multiple visual-audit hypotheses"
        )
    return None if not matches else matches[0]


class ScenarioVisualAuditSampler:
    """Capture distinct causal strata within each preregistered scenario."""

    def __init__(
        self,
        root: str | Path,
        *,
        quota_per_scenario: int = 3,
    ) -> None:
        if quota_per_scenario < 1:
            raise ValueError("scenario audit quota must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.quota_per_scenario = int(quota_per_scenario)
        self._captures: dict[AuditScenario, list[ScenarioCapture]] = {
            scenario: [] for scenario in AuditScenario
        }

    @property
    def captures(self) -> tuple[ScenarioCapture, ...]:
        return tuple(
            capture
            for scenario in AuditScenario
            for capture in self._captures[scenario]
        )

    def observe(
        self,
        snapshot: EngineSnapshot,
        histories: Mapping[Timeframe, Sequence[Any]],
        visualizer: DecisionVisualizer,
        *,
        closed_trades: Sequence[Any] = (),
        ai_proposals: Sequence[Any] = (),
        previous_snapshot: EngineSnapshot | None = None,
        source_bar: Any | None = None,
        account_state: Any | None = None,
        belief_position_input: Any | None = None,
    ) -> tuple[ScenarioCapture, ...]:
        if ai_proposals:
            raise ValueError(
                "AI proposals may only appear in an exact sealed-path audit"
            )
        created: list[ScenarioCapture] = []
        for scenario, raw_context in classify_audit_scenarios(
            snapshot,
            closed_trades,
        ):
            context = dict(raw_context)
            bucket = self._captures[scenario]
            outcome_view = scenario in {
                AuditScenario.STOP,
                AuditScenario.TARGET,
            }
            hypothesis_key = (
                None
                if outcome_view
                else _position_bound_hypothesis_key(
                    snapshot,
                    belief_position_input,
                )
                if scenario
                in {
                    AuditScenario.PROTECT,
                    AuditScenario.HOLD,
                    AuditScenario.EXIT,
                }
                else snapshot.decision.best_hypothesis_key
            )
            closed_identity = context.get("closed_trade", {})
            hypothesis_identity = (
                {
                    "setup_id": None,
                    "entry_location_id": None,
                    "entry_path_id": None,
                }
                if outcome_view or hypothesis_key is None
                else ai_review_identity(snapshot, hypothesis_key)
            )
            setup_id = (
                closed_identity.get("setup_id")
                if outcome_view
                else hypothesis_identity["setup_id"]
            )
            entry_location_id = (
                closed_identity.get("entry_location_id")
                if outcome_view
                else hypothesis_identity["entry_location_id"]
            )
            entry_path_id = (
                closed_identity.get("entry_path_id")
                if outcome_view
                else hypothesis_identity["entry_path_id"]
            )
            audit_stage = (
                "post_outcome"
                if outcome_view
                else "blind_pre_reveal"
            )
            audited_belief = (
                None
                if hypothesis_key is None
                else snapshot.belief.hypotheses.get(hypothesis_key)
            )
            local_hour = snapshot.observation.asof.tz_convert(
                "America/New_York"
            ).hour
            session_bucket = (
                "overnight"
                if local_hour < 8
                else "morning"
                if local_hour < 12
                else "afternoon"
            )
            h4_direction = float(
                snapshot.observation.frame(Timeframe.H4).metrics.get(
                    "structure_direction",
                    0.0,
                )
            )
            regime = (
                "h4_up"
                if h4_direction > 0
                else "h4_down"
                if h4_direction < 0
                else "h4_flat"
            )
            stratum_key = "|".join(
                (
                    (
                        str(closed_identity.get("playbook"))
                        if outcome_view
                        else "none"
                        if audited_belief is None
                        else audited_belief.playbook.value
                    ),
                    (
                        str(closed_identity.get("direction"))
                        if outcome_view
                        else "none"
                        if audited_belief is None
                        else audited_belief.direction.value
                    ),
                    regime,
                    session_bucket,
                )
            )
            if (
                len(bucket) >= self.quota_per_scenario
                or any(
                    item.context.get("stratum_key") == stratum_key
                    for item in bucket
                )
            ):
                continue
            index = len(bucket) + 1
            destination = (
                self.root
                / scenario.value
                / f"{index:02d}-{snapshot.snapshot_hash[:20]}.png"
            )
            suppress_hypothesis = outcome_view or (
                scenario
                in {
                    AuditScenario.PROTECT,
                    AuditScenario.HOLD,
                    AuditScenario.EXIT,
                }
                and hypothesis_key is None
            )
            context.update(
                {
                    "audit_stage": audit_stage,
                    "future_present": outcome_view,
                    "audited_hypothesis_key": hypothesis_key,
                    "audited_setup_id": setup_id,
                    "audited_entry_location_id": entry_location_id,
                    "audited_entry_path_id": entry_path_id,
                    "stratum_key": stratum_key,
                }
            )
            artifact = visualizer.render_decision(
                snapshot,
                histories,
                destination,
                ai_proposals=(),
                audit_context={
                    "scenario": scenario.value,
                    **context,
                },
                audit_hypothesis_key=hypothesis_key,
                suppress_audit_hypothesis=suppress_hypothesis,
            )
            packet = write_frozen_decision_packet(
                snapshot,
                histories,
                destination.with_suffix(".decision_packet.json"),
                previous_snapshot,
                hypothesis_key=hypothesis_key,
                audit_context={
                    "scenario": scenario.value,
                    **context,
                },
                source_bar=source_bar,
                account_state=account_state,
                belief_position_input=belief_position_input,
            )
            verified_packet = read_verified_decision_packet(packet)
            capture = ScenarioCapture(
                scenario=scenario,
                decision_hash=snapshot.snapshot_hash,
                asof=snapshot.observation.asof,
                artifact=artifact,
                context=context,
                audit_stage=audit_stage,
                hypothesis_key=hypothesis_key,
                setup_id=setup_id,
                entry_location_id=entry_location_id,
                entry_path_id=entry_path_id,
                decision_packet=packet,
                decision_packet_hash=verified_packet["packet_hash"],
                decision_packet_sha256=decision_packet_sha256(packet),
            )
            bucket.append(capture)
            created.append(capture)
        return tuple(created)

    def write_manifest(
        self,
        destination: str | Path | None = None,
        *,
        source_bindings: Mapping[str, Any] | None = None,
    ) -> Path:
        output = (
            self.root / "scenario_audit_manifest.json"
            if destination is None
            else Path(destination)
        )
        counts = {
            scenario.value: len(self._captures[scenario])
            for scenario in AuditScenario
        }
        missing = [
            scenario.value
            for scenario in AuditScenario
            if not self._captures[scenario]
        ]
        payload = {
            "format_version": 1,
            "artifact": "systematic_scenario_visual_audit",
            "required_scenarios": [item.value for item in AuditScenario],
            "quota_per_scenario": self.quota_per_scenario,
            "counts": counts,
            "coverage_complete": not missing,
            "coverage_scope": "scenario_presence_only",
            "missing_scenarios": missing,
            "sampling_method": (
                "first_distinct_playbook_direction_h4_regime_session_"
                "stratum_per_scenario"
            ),
            "blind_pre_reveal_images_exclude_future": True,
            "post_outcome_views": sum(
                capture.audit_stage == "post_outcome"
                for capture in self.captures
            ),
            "captures": [
                {
                    "scenario": capture.scenario.value,
                    "decision_hash": capture.decision_hash,
                    "asof": capture.asof,
                    "path": str(capture.artifact.path),
                    "sha256": capture.artifact.sha256,
                    "maximum_market_time": (
                        capture.artifact.maximum_market_time
                    ),
                    "context": capture.context,
                    "audit_stage": capture.audit_stage,
                    "hypothesis_key": capture.hypothesis_key,
                    "setup_id": capture.setup_id,
                    "entry_location_id": capture.entry_location_id,
                    "entry_path_id": capture.entry_path_id,
                    "decision_packet": str(capture.decision_packet),
                    "decision_packet_hash": (
                        capture.decision_packet_hash
                    ),
                    "decision_packet_sha256": (
                        capture.decision_packet_sha256
                    ),
                }
                for capture in self.captures
            ],
            "source_bindings": dict(source_bindings or {}),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                to_primitive(payload),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return output


def visual_audit_code_fingerprint() -> str:
    digest = hashlib.sha256()
    for source in (
        Path(__file__),
        Path(__file__).with_name("visualization.py"),
        Path(__file__).with_name("decision_trace.py"),
        Path(__file__).with_name("ai_review.py"),
        Path(__file__).with_name("path_evidence.py"),
    ):
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "AuditScenario",
    "ScenarioCapture",
    "ScenarioVisualAuditSampler",
    "classify_audit_scenarios",
    "visual_audit_code_fingerprint",
]
