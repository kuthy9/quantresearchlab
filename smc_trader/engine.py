"""Layered one-minute orchestration with no hidden execution side effects."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .calibration import (
    CalibrationError,
    TypedBrainCalibrator,
    model_code_fingerprint,
)
from .causal import CausalMarketReader
from .decision import DecisionConfig, UtilityDecisionLayer
from .model import (
    AccountState,
    Bar,
    EngineSnapshot,
    PositionSnapshot,
    Timeframe,
    content_hash,
)
from .observation import (
    CausalObserver,
    ExecutionRealityInput,
    ObserverConfig,
)
from .playbooks import BrainConfig, PlaybookBrain
from .playbook_registry import load_playbook_registry
from .risk import RiskLimits, StructuralRiskEngine
from .scene_graph import brain_input_contract_hash, parse_scale_specs


_PRIMITIVE_PROTOCOL_FIELDS = (
    "structure_protocol",
    "liquidity_protocol",
    "displacement_protocol",
    "group3_protocol",
    "group4_protocol",
    "group5_protocol",
)


def _configured_primitive_protocol_hashes(
    observer: Mapping[str, Any],
) -> dict[str, str]:
    """Resolve and hash the exact primitive files bound by the model config."""

    hashes: dict[str, str] = {}
    repository = Path(__file__).resolve().parents[1]
    for field in _PRIMITIVE_PROTOCOL_FIELDS:
        value = observer.get(field)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise CalibrationError(
                f"typed calibration requires observer.{field}"
            )
        source = Path(value)
        if not source.is_absolute() and not source.exists():
            source = repository / source
        try:
            raw = source.read_bytes()
        except OSError as error:
            raise CalibrationError(
                f"cannot read typed calibration protocol observer.{field}: {source}"
            ) from error
        hashes[field] = hashlib.sha256(raw).hexdigest()
    return hashes


class ContinuousSMCEngine:
    """Update eyes → brain → decision → risk once per completed 1m bar."""

    def __init__(
        self,
        *,
        reader: CausalMarketReader,
        observer: CausalObserver,
        brain: PlaybookBrain,
        decision: UtilityDecisionLayer,
        risk: StructuralRiskEngine,
    ) -> None:
        self.reader = reader
        self.observer = observer
        self.brain = brain
        self.decision = decision
        self.risk = risk
        self._last_snapshot: EngineSnapshot | None = None

    @classmethod
    def from_config(
        cls,
        path: str | Path = "configs/model.json",
    ) -> "ContinuousSMCEngine":
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = Path(__file__).resolve().parents[1] / source
        payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("model.schema_version must be 1")
        scales_raw = payload.get("scales")
        if not isinstance(scales_raw, list) or not scales_raw:
            raise ValueError("model.scales must register the current causal scale stack")
        scale_specs = parse_scale_specs(scales_raw)
        reader = CausalMarketReader(scale_specs=scale_specs)
        observer_raw = payload.get("observer")
        if not isinstance(observer_raw, Mapping):
            raise ValueError("model.observer must bind all typed primitive protocols")
        primitive_protocol_hashes = _configured_primitive_protocol_hashes(observer_raw)
        minimum = observer_raw.get("minimum_bars", {})
        observer = CausalObserver(
            ObserverConfig(
                atr_period=int(observer_raw.get("atr_period", 14)),
                swing_k=int(observer_raw.get("swing_k", 2)),
                external_liquidity_lookback=int(
                    observer_raw.get("external_liquidity_lookback", 80)
                ),
                memory_events=int(observer_raw.get("memory_events", 512)),
                minimum_bars={
                    timeframe: int(minimum.get(timeframe.value, default))
                    for timeframe, default in {
                        Timeframe.H4: 16,
                        Timeframe.H1: 24,
                        Timeframe.M15: 24,
                        Timeframe.M5: 24,
                        Timeframe.M1: 30,
                    }.items()
                    if any(
                        spec.enabled
                        and spec.native_timeframe is timeframe
                        for spec in scale_specs
                    )
                },
                tick_size=float(payload.get("tick_size", 0.25)),
                point_value=float(payload.get("point_value", 20.0)),
                structure_protocol=observer_raw.get("structure_protocol"),
                liquidity_protocol=observer_raw.get(
                    "liquidity_protocol"
                ),
                displacement_protocol=observer_raw.get(
                    "displacement_protocol"
                ),
                group3_protocol=observer_raw.get("group3_protocol"),
                group4_protocol=observer_raw.get("group4_protocol"),
                group5_protocol=observer_raw.get("group5_protocol"),
                scale_specs=scale_specs,
            )
        )
        registry = load_playbook_registry(
            payload.get("playbook_registry", "configs/playbooks.json")
        )
        calibration_path = payload.get("calibration_artifact")
        calibrator = (
            TypedBrainCalibrator.from_file(
                calibration_path,
                expected_registry_hash=registry.fingerprint,
                expected_code_hash=model_code_fingerprint(),
                expected_primitive_protocol_hashes=primitive_protocol_hashes,
                expected_brain_input_contract_hash=(
                    brain_input_contract_hash(scale_specs)
                ),
            )
            if calibration_path
            else TypedBrainCalibrator.identity()
        )
        brain = PlaybookBrain(
            BrainConfig(
                tick_size=float(payload.get("tick_size", 0.25)),
                minimum_remaining_path_R=float(
                    payload.get("risk", {}).get(
                        "minimum_target_R",
                        1.0,
                    )
                ),
            ),
            registry=registry,
            calibrator=calibrator,
        )
        decision_raw = payload.get("decision", {})
        decision = UtilityDecisionLayer(
            DecisionConfig(
                minimum_utility_advantage=float(
                    decision_raw.get("minimum_utility_advantage", 0.12)
                ),
                uncertainty_penalty=float(
                    decision_raw.get("uncertainty_penalty", 0.35)
                ),
                deadline_penalty_minutes=int(
                    decision_raw.get("deadline_penalty_minutes", 20)
                ),
                maximum_reward_R=float(decision_raw.get("maximum_reward_R", 3.0)),
            )
        )
        risk_raw = payload.get("risk", {})
        risk = StructuralRiskEngine(
            RiskLimits(
                maximum_trade_risk_fraction=float(
                    risk_raw.get("maximum_trade_risk_fraction", 0.01)
                ),
                maximum_total_risk_fraction=float(
                    risk_raw.get("maximum_total_risk_fraction", 0.02)
                ),
                maximum_spread_ticks=float(
                    risk_raw.get("maximum_spread_ticks", 4)
                ),
                maximum_cost_R=float(risk_raw.get("maximum_cost_R", 0.20)),
                minimum_fillability=float(risk_raw.get("minimum_fillability", 0.45)),
                minimum_minutes_to_deadline=int(
                    risk_raw.get("minimum_minutes_to_deadline", 5)
                ),
                minimum_target_R=float(risk_raw.get("minimum_target_R", 1.0)),
                tick_size=float(payload.get("tick_size", 0.25)),
            )
        )
        return cls(
            reader=reader,
            observer=observer,
            brain=brain,
            decision=decision,
            risk=risk,
        )

    @property
    def last_snapshot(self) -> EngineSnapshot | None:
        return self._last_snapshot

    def on_bar(
        self,
        bar: Bar,
        *,
        execution: ExecutionRealityInput | None = None,
        account: AccountState | None = None,
        belief_position: PositionSnapshot | None = None,
        compute_snapshot_hash: bool = True,
    ) -> EngineSnapshot:
        account = account or AccountState(equity=100_000.0)
        update = self.reader.on_bar(bar)
        observation = self.observer.observe(update, execution)
        if {
            "contract_change_history_reset",
            "data_gap_history_reset",
        }.intersection(observation.anomalies):
            self.brain.reset()
        belief = self.brain.update(
            observation,
            position=belief_position if belief_position is not None else account.position,
            scene_graph=self.observer.scene_graph,
            scene_delta=self.observer.last_scene_delta,
        )
        decision = self.decision.decide(observation, belief, account)
        risk = self.risk.review(decision, observation, account)
        snapshot_hash = (
            content_hash(
                {
                    "observation": observation,
                    "belief": belief,
                    "decision": decision,
                    "risk": risk,
                }
            )
            if compute_snapshot_hash
            else ""
        )
        snapshot = EngineSnapshot(
            observation=observation,
            belief=belief,
            decision=decision,
            risk=risk,
            snapshot_hash=snapshot_hash,
        )
        self._last_snapshot = snapshot
        return snapshot

    def histories(self, bars: int = 80):
        return {
            timeframe: self.reader.window(timeframe, bars)
            for timeframe in self.reader.active_timeframes
        }


__all__ = ["ContinuousSMCEngine"]
