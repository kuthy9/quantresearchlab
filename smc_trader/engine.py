"""Layered one-minute orchestration with no hidden execution side effects."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .calibration import (
    TypedBrainCalibrator,
)
from .causal import CausalMarketReader
from .decision import DecisionConfig, UtilityDecisionLayer
from .dol_probability import (
    load_dol_probability_model_artifact,
    load_dol_probability_protocol,
)
from .foundation_registry import (
    FOUNDATION_CANONICAL_IDENTITY,
    FOUNDATION_VERSION,
)
from .model import (
    AccountState,
    Bar,
    EngineSnapshot,
    GlobalMarketContext,
    MarketObservation,
    NeutralEngineSnapshot,
    NeutralMarketState,
    Playbook,
    PositionSnapshot,
    Timeframe,
    to_primitive,
)
from .observation import (
    CausalObserver,
    ExecutionRealityInput,
    ObserverConfig,
)
from .playbooks import (
    BrainConfig,
    PlaybookBrain,
    _NEUTRAL_AUTHORITY_CAPABILITY,
)
from .playbook_registry import load_playbook_registry
from .signal_policy import (
    load_dol_calibration_artifact,
    load_outcome_model_artifact,
    load_path_likelihood_artifact,
    load_signal_artifact_pins,
    load_signal_policy_protocol,
)
from .risk import RiskLimits, StructuralRiskEngine
from .scene_graph import (
    build_neutral_market_state,
    build_open_market_theses,
    parse_scale_specs,
    update_global_market_context,
)
from .semantics import load_semantic_selection


_REQUIRED_PRIMITIVE_PROTOCOLS = (
    "structure_protocol",
    "liquidity_protocol",
    "displacement_protocol",
    "group3_protocol",
    "group4_protocol",
    "group5_protocol",
)
_LIVE_READINESS_TOKEN = object()
RUNTIME_ACTION_POLICY_SCHEMA_VERSION = 2
NEUTRAL_ENGINE_CHECKPOINT_SCHEMA_VERSION = 5
MODEL_SCHEMA_VERSION = 2
ACTION_PIPELINE_SCHEMA_VERSION = 1
LEGACY_ACTION_PIPELINE_MODE = "legacy_decision_risk_compat"


def normalize_action_disabled_playbooks(
    values: Iterable[Playbook | str] = (),
) -> tuple[Playbook, ...]:
    """Return one deterministic, duplicate-free runtime action deny-list."""

    normalized: list[Playbook] = []
    for value in values:
        if isinstance(value, Playbook):
            playbook = value
        elif type(value) is str:
            try:
                playbook = Playbook(value)
            except ValueError as exc:
                raise ValueError(
                    f"unknown action-disabled playbook: {value!r}"
                ) from exc
        else:
            raise TypeError(
                "action-disabled playbooks must be Playbook values or strings"
            )
        if playbook in normalized:
            raise ValueError(
                f"duplicate action-disabled playbook: {playbook.value}"
            )
        normalized.append(playbook)
    return tuple(sorted(normalized, key=lambda item: item.value))


class RuntimeActionBeliefView:
    """Decision-only view that denies new entries by playbook.

    The Engine snapshot retains the original ``MarketBelief``.  Lifecycle,
    semantic, calibration, Shadow and case-library consumers therefore see the
    unmodified Brain result.  Existing positions resolve against the raw
    position-candidate interface so disabling a playbook never abandons its
    already-frozen HOLD/PROTECT/EXIT management.
    """

    __slots__ = ("_belief", "_disabled_new_entries")

    def __init__(
        self,
        belief: Any,
        disabled_new_entries: Iterable[Playbook | str],
    ) -> None:
        self._belief = belief
        self._disabled_new_entries = frozenset(
            normalize_action_disabled_playbooks(disabled_new_entries)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._belief, name)

    def action_candidate_items(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (candidate_id, hypothesis)
            for candidate_id, hypothesis in self._belief.action_candidate_items()
            if hypothesis.playbook not in self._disabled_new_entries
        )

    def owns_actionable_entry_episode(
        self,
        candidate_id: str,
        hypothesis: Any,
    ) -> bool:
        return bool(
            hypothesis.playbook not in self._disabled_new_entries
            and self._belief.owns_actionable_entry_episode(
                candidate_id,
                hypothesis,
            )
        )

    def position_candidate_items(self) -> tuple[tuple[str, Any], ...]:
        return tuple(self._belief.position_candidate_items())

    def resolve_hypothesis(self, identity: str | None) -> Any:
        return self._belief.resolve_hypothesis(identity)


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
        runtime_mode: str,
        action_pipeline_mode: str,
        action_disabled_playbooks: Iterable[Playbook | str] = (),
        _readiness_token: object | None = None,
    ) -> None:
        if runtime_mode not in {"development", "live"}:
            raise ValueError(
                "runtime_mode must be development or live"
            )
        if (
            runtime_mode == "live"
            and _readiness_token is not _LIVE_READINESS_TOKEN
        ):
            raise RuntimeError(
                "live engine must be constructed through the "
                "from_config readiness gate"
            )
        self.reader = reader
        self.observer = observer
        self.brain = brain
        self.decision = decision
        self.risk = risk
        self.runtime_mode = runtime_mode
        if action_pipeline_mode != LEGACY_ACTION_PIPELINE_MODE:
            raise ValueError("unsupported action pipeline mode")
        self._action_pipeline_mode = action_pipeline_mode
        self.action_disabled_playbooks = normalize_action_disabled_playbooks(
            action_disabled_playbooks
        )
        self._last_snapshot: EngineSnapshot | NeutralEngineSnapshot | None = None
        self._last_belief_position: PositionSnapshot | None = None
        self._neutral_market_state: NeutralMarketState | None = None
        # Exact model bytes used to construct this runtime.  Shadow parity
        # binds this identity so two engines with coincidentally equal early
        # outputs cannot be mistaken for the same registered runtime.
        self._model_config_sha256: str | None = None
        self._foundation_version: str | None = None
        self._foundation_registry_identity: str | None = None
        self._neutral_checkpoint_schema_version = (
            NEUTRAL_ENGINE_CHECKPOINT_SCHEMA_VERSION
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_neutral_checkpoint_schema_version"] = (
            NEUTRAL_ENGINE_CHECKPOINT_SCHEMA_VERSION
        )
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError(
                "checkpoint neutral market state schema changed"
            )
        last_snapshot = state.get("_last_snapshot")
        neutral_market_state = state.get("_neutral_market_state")
        foundation_version = state.get("_foundation_version")
        foundation_identity = state.get("_foundation_registry_identity")
        action_pipeline_mode = state.get("_action_pipeline_mode")
        if (
            state.get("_neutral_checkpoint_schema_version")
            != NEUTRAL_ENGINE_CHECKPOINT_SCHEMA_VERSION
            or "_neutral_market_state" not in state
            or "_last_snapshot" not in state
            or "_foundation_version" not in state
            or "_foundation_registry_identity" not in state
            or "_action_pipeline_mode" not in state
            or action_pipeline_mode != LEGACY_ACTION_PIPELINE_MODE
            or (foundation_version is None) != (foundation_identity is None)
            or foundation_version not in {None, FOUNDATION_VERSION}
            or foundation_identity not in {
                None,
                FOUNDATION_CANONICAL_IDENTITY,
            }
            or (
                neutral_market_state is not None
                and not isinstance(neutral_market_state, NeutralMarketState)
            )
            or not isinstance(
                last_snapshot,
                (EngineSnapshot, NeutralEngineSnapshot, type(None)),
            )
            or (
                last_snapshot is not None
                and last_snapshot.neutral_market_state
                != neutral_market_state
            )
            or (
                neutral_market_state is not None
                and last_snapshot is not None
                and last_snapshot.observation.asof
                != neutral_market_state.asof
            )
            or (
                isinstance(last_snapshot, EngineSnapshot)
                and neutral_market_state is not None
                and (
                    last_snapshot.belief.global_context is None
                    or last_snapshot.belief.global_context.open_market_theses
                    != neutral_market_state.open_market_theses
                )
            )
        ):
            raise ValueError(
                "checkpoint neutral market state schema changed"
            )
        self.__dict__.update(state)

    @classmethod
    def from_config(
        cls,
        path: str | Path = "configs/model.json",
        *,
        runtime_mode: str,
        action_disabled_playbooks: Iterable[Playbook | str] = (),
    ) -> "ContinuousSMCEngine":
        if runtime_mode not in {"development", "live"}:
            raise ValueError(
                "runtime_mode must be development or live"
            )
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = Path(__file__).resolve().parents[1] / source
        raw_config = source.read_bytes()
        payload: dict[str, Any] = json.loads(raw_config)
        if payload.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"model.schema_version must be {MODEL_SCHEMA_VERSION}"
            )
        semantic_selection = load_semantic_selection(
            payload.get("semantic_selection"),
            root=Path(__file__).resolve().parents[1],
        )
        action_pipeline = payload.get("action_pipeline")
        if (
            not isinstance(action_pipeline, Mapping)
            or action_pipeline.get("schema_version")
            != ACTION_PIPELINE_SCHEMA_VERSION
            or set(action_pipeline) != {"schema_version", "mode"}
        ):
            raise ValueError(
                "model.action_pipeline schema is missing or unsupported"
            )
        action_pipeline_mode = action_pipeline.get("mode")
        if action_pipeline_mode != LEGACY_ACTION_PIPELINE_MODE:
            raise ValueError(
                "model.action_pipeline.mode must select the explicit legacy "
                "Decision/Risk compatibility authority"
            )
        scales_raw = payload.get("scales")
        if not isinstance(scales_raw, list) or not scales_raw:
            raise ValueError("model.scales must register the current causal scale stack")
        scale_specs = parse_scale_specs(scales_raw)
        tick_size = float(payload.get("tick_size", 0.25))
        reader = CausalMarketReader(
            scale_specs=scale_specs,
            tick_size=tick_size,
        )
        observer_raw = payload.get("observer")
        if not isinstance(observer_raw, Mapping):
            raise ValueError("model.observer must bind all typed primitive protocols")
        missing_protocols = [
            field
            for field in _REQUIRED_PRIMITIVE_PROTOCOLS
            if not isinstance(observer_raw.get(field), (str, Path))
            or not str(observer_raw.get(field)).strip()
        ]
        if missing_protocols:
            raise ValueError(
                "model.observer must bind typed primitive protocols: "
                + ", ".join(missing_protocols)
            )
        foundation_registry = semantic_selection.foundation_registry
        minimum = observer_raw.get("minimum_bars", {})
        observer = CausalObserver(
            ObserverConfig(
                atr_period=int(observer_raw.get("atr_period", 14)),
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
                tick_size=tick_size,
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
                semantic_registry=str(
                    semantic_selection.atomic_registry.source_path
                ),
                scale_specs=scale_specs,
                canonical_foundation_enabled=True,
            ),
            semantic_registry=semantic_selection.atomic_registry,
        )
        registry = load_playbook_registry(
            payload.get("playbook_registry", "configs/playbooks.json")
        )
        group5_protocol = observer.group5_protocol
        if group5_protocol is None:
            raise ValueError("current engine requires typed Group 5 state")
        favr_parked = "parked" in registry.for_playbook(
            Playbook.FAILED_AUCTION_VALUE_RETURN
        ).status
        if group5_protocol.favr_enabled == favr_parked:
            raise ValueError(
                "Group 5 favr_enabled must agree with the FAVR "
                "development restriction"
            )
        if runtime_mode == "live":
            readiness = payload.get("release_readiness")
            checks = {
                "active_model_natural_authority_validated": bool(
                    isinstance(readiness, Mapping)
                    and readiness.get(
                        "active_model_natural_authority_validated"
                    )
                    is True
                ),
                "economic_validation_complete": bool(
                    isinstance(readiness, Mapping)
                    and readiness.get("economic_validation_complete")
                    is True
                ),
                "rolling_oof_complete": bool(
                    isinstance(readiness, Mapping)
                    and readiness.get("rolling_oof_complete") is True
                ),
                "mbo_stability_validated": bool(
                    isinstance(readiness, Mapping)
                    and readiness.get("mbo_stability_validated") is True
                ),
                "live_execution_allowed": bool(
                    isinstance(readiness, Mapping)
                    and readiness.get("live_execution_allowed") is True
                ),
                "group5_dfp_lsr_input_authority_validated": (
                    group5_protocol.dfp_lsr_input_authority_validated
                ),
            }
            missing = tuple(
                name for name, ready in checks.items() if not ready
            )
            if missing:
                raise RuntimeError(
                    "live execution readiness is incomplete: "
                    + ", ".join(missing)
                )
        calibration_path = payload.get("calibration_artifact")
        calibrator = (
            TypedBrainCalibrator.from_file(
                calibration_path,
                expected_registry_hash=registry.fingerprint,
            )
            if calibration_path
            else TypedBrainCalibrator.identity()
        )
        if runtime_mode == "live" and not calibrator.is_ready:
            raise RuntimeError(
                "live execution readiness is incomplete: "
                "typed_brain_calibration_ready"
            )
        path_hypotheses_raw = payload.get("path_hypotheses")
        if not isinstance(path_hypotheses_raw, Mapping):
            raise ValueError(
                "model.path_hypotheses must bind the shadow path protocol"
            )
        path_protocol_path = path_hypotheses_raw.get("protocol")
        expected_path_fingerprint = path_hypotheses_raw.get(
            "path_protocol_fingerprint"
        )
        expected_dol_fingerprint = path_hypotheses_raw.get(
            "dol_protocol_fingerprint"
        )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                path_protocol_path,
                expected_path_fingerprint,
                expected_dol_fingerprint,
            )
        ):
            raise ValueError(
                "model.path_hypotheses requires protocol and exact fingerprints"
            )
        dol_probability_raw = payload.get("dol_probability")
        if (
            not isinstance(dol_probability_raw, Mapping)
            or set(dol_probability_raw)
            != {
                "protocol",
                "protocol_fingerprint",
                "model_artifact",
                "model_artifact_fingerprint",
            }
            or not isinstance(dol_probability_raw.get("protocol"), str)
            or not str(dol_probability_raw.get("protocol")).strip()
            or not isinstance(
                dol_probability_raw.get("protocol_fingerprint"),
                str,
            )
            or len(dol_probability_raw["protocol_fingerprint"]) != 64
        ):
            raise ValueError(
                "model.dol_probability must bind the exact shadow protocol"
            )
        dol_model_path = dol_probability_raw.get("model_artifact")
        dol_model_fingerprint = dol_probability_raw.get(
            "model_artifact_fingerprint"
        )
        if (dol_model_path is None) != (dol_model_fingerprint is None) or (
            dol_model_path is not None
            and (
                not isinstance(dol_model_path, str)
                or not dol_model_path.strip()
                or not isinstance(dol_model_fingerprint, str)
                or len(dol_model_fingerprint) != 64
            )
        ):
            raise ValueError(
                "model.dol_probability artifact path and fingerprint must "
                "be admitted together"
            )
        dol_probability_protocol = load_dol_probability_protocol(
            dol_probability_raw["protocol"]
        )
        dol_probability_model_artifact = (
            None
            if dol_model_path is None
            else load_dol_probability_model_artifact(
                dol_model_path,
                protocol=dol_probability_protocol,
                expected_fingerprint=dol_model_fingerprint,
            )
        )
        signal_policy_raw = payload.get("signal_policy")
        if (
            not isinstance(signal_policy_raw, Mapping)
            or set(signal_policy_raw)
            != {
                "protocol",
                "protocol_fingerprint",
                "path_likelihood_artifact",
                "dol_calibration_artifact",
                "outcome_model_artifact",
                "artifact_pins",
            }
            or not isinstance(signal_policy_raw.get("protocol"), str)
            or not str(signal_policy_raw.get("protocol")).strip()
            or not isinstance(
                signal_policy_raw.get("protocol_fingerprint"),
                str,
            )
            or len(signal_policy_raw["protocol_fingerprint"]) != 64
        ):
            raise ValueError(
                "model.signal_policy must bind the exact fail-closed shadow "
                "protocol"
            )
        signal_artifact_names = (
            "path_likelihood_artifact",
            "dol_calibration_artifact",
            "outcome_model_artifact",
            "artifact_pins",
        )
        supplied_signal_artifacts = tuple(
            signal_policy_raw.get(name) is not None
            for name in signal_artifact_names
        )
        if any(supplied_signal_artifacts) and not all(supplied_signal_artifacts):
            raise ValueError(
                "model.signal_policy artifacts must be admitted as one exact set"
            )
        path_likelihood_artifact = None
        dol_calibration_artifact = None
        outcome_model_artifact = None
        signal_artifact_pins = None
        if all(supplied_signal_artifacts):
            pins_binding = signal_policy_raw["artifact_pins"]
            if (
                not isinstance(pins_binding, Mapping)
                or set(pins_binding) != {"path", "admission_id"}
                or any(
                    not isinstance(pins_binding.get(name), str)
                    or not pins_binding[name].strip()
                    for name in ("path", "admission_id")
                )
                or any(
                    not isinstance(signal_policy_raw[name], str)
                    or not signal_policy_raw[name].strip()
                    for name in signal_artifact_names[:-1]
                )
                or dol_probability_model_artifact is None
            ):
                raise ValueError(
                    "model.signal_policy artifact set or external pins are invalid"
                )
            signal_artifact_pins = load_signal_artifact_pins(
                pins_binding["path"],
                expected_admission_id=pins_binding["admission_id"],
            )
            path_likelihood_artifact = load_path_likelihood_artifact(
                signal_policy_raw["path_likelihood_artifact"],
                expected_artifact_id=(
                    signal_artifact_pins.path_likelihood_artifact_id
                ),
            )
            dol_calibration_artifact = load_dol_calibration_artifact(
                signal_policy_raw["dol_calibration_artifact"],
                expected_artifact_id=(
                    signal_artifact_pins.dol_calibration_artifact_id
                ),
            )
            outcome_model_artifact = load_outcome_model_artifact(
                signal_policy_raw["outcome_model_artifact"],
                expected_artifact_id=(
                    signal_artifact_pins.outcome_model_artifact_id
                ),
            )
        # Loading the protocol here provides an early exact-fingerprint check;
        # PlaybookBrain repeats the binding at its ownership boundary.
        signal_policy_protocol = load_signal_policy_protocol(
            signal_policy_raw["protocol"]
        )
        if (
            signal_policy_protocol.fingerprint
            != signal_policy_raw["protocol_fingerprint"]
        ):
            raise ValueError("model Signal Policy fingerprint is stale")
        brain = PlaybookBrain(
            BrainConfig(
                tick_size=tick_size,
                minimum_remaining_path_R=float(
                    payload.get("risk", {}).get(
                        "minimum_target_R",
                        1.0,
                    )
                ),
            ),
            registry=registry,
            calibrator=calibrator,
            path_hypotheses_protocol=path_protocol_path,
            expected_path_protocol_fingerprint=expected_path_fingerprint,
            expected_dol_protocol_fingerprint=expected_dol_fingerprint,
            dol_probability_protocol=dol_probability_raw["protocol"],
            expected_dol_probability_protocol_fingerprint=(
                dol_probability_raw["protocol_fingerprint"]
            ),
            dol_probability_model_artifact=(
                dol_probability_model_artifact
            ),
            signal_policy_protocol=signal_policy_raw["protocol"],
            expected_signal_policy_fingerprint=(
                signal_policy_raw["protocol_fingerprint"]
            ),
            path_likelihood_artifact=path_likelihood_artifact,
            dol_calibration_artifact=dol_calibration_artifact,
            outcome_model_artifact=outcome_model_artifact,
            signal_artifact_pins=signal_artifact_pins,
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
            ),
            calibration_ready=calibrator.is_ready,
            calibration_version=calibrator.version,
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
                tick_size=tick_size,
            )
        )
        engine = cls(
            reader=reader,
            observer=observer,
            brain=brain,
            decision=decision,
            risk=risk,
            runtime_mode=runtime_mode,
            action_pipeline_mode=action_pipeline_mode,
            action_disabled_playbooks=action_disabled_playbooks,
            _readiness_token=_LIVE_READINESS_TOKEN,
        )
        engine._model_config_sha256 = hashlib.sha256(raw_config).hexdigest()
        engine._foundation_version = foundation_registry.foundation_version
        engine._foundation_registry_identity = foundation_registry.identity
        return engine

    @property
    def model_config_sha256(self) -> str | None:
        return getattr(self, "_model_config_sha256", None)

    @property
    def foundation_version(self) -> str | None:
        return getattr(self, "_foundation_version", None)

    @property
    def foundation_registry_identity(self) -> str | None:
        return getattr(self, "_foundation_registry_identity", None)

    @property
    def last_snapshot(self) -> EngineSnapshot | NeutralEngineSnapshot | None:
        return self._last_snapshot

    @property
    def neutral_market_state(self) -> NeutralMarketState | None:
        return self._neutral_market_state

    @property
    def action_pipeline_mode(self) -> str:
        return self._action_pipeline_mode

    @property
    def runtime_action_policy_identity(self) -> Mapping[str, Any]:
        return {
            "schema_version": RUNTIME_ACTION_POLICY_SCHEMA_VERSION,
            "action_pipeline_mode": self.action_pipeline_mode,
            "scope": "new_entry_action_candidates_only",
            "disabled_new_entry_playbooks": [
                playbook.value for playbook in self.action_disabled_playbooks
            ],
            "decision_belief_projection": "action_filtered",
            "engine_snapshot_belief_projection": "raw",
            "position_management_projection": "raw",
            "trade_intent_projection": "disabled_in_legacy_compat",
        }

    def on_bar(
        self,
        bar: Bar,
        *,
        execution: ExecutionRealityInput | None = None,
        account: AccountState | None = None,
        belief_position: PositionSnapshot | None = None,
    ) -> EngineSnapshot:
        if isinstance(self._last_snapshot, NeutralEngineSnapshot):
            raise RuntimeError(
                "Engine entry mode cannot change after neutral-only input"
            )
        account = account or AccountState(equity=100_000.0)
        observation = self._observe_bar(bar, execution=execution)
        if {
            "contract_change_history_reset",
            "data_gap_history_reset",
        }.intersection(observation.anomalies):
            self.brain.reset()
        resolved_belief_position = (
            belief_position
            if belief_position is not None
            else account.position
        )
        _, neutral_market_state = self._project_neutral(
            observation
        )
        scene_graph = self.observer.scene_graph
        scene_delta = self.observer.last_scene_delta
        if neutral_market_state is not None:
            belief = self.brain.update(
                observation,
                position=resolved_belief_position,
                scene_graph=scene_graph,
                scene_delta=scene_delta,
                _precomputed_neutral_state=neutral_market_state,
                _neutral_authority_capability=(
                    _NEUTRAL_AUTHORITY_CAPABILITY
                ),
            )
        else:
            belief = self.brain.update(
                observation,
                position=resolved_belief_position,
                scene_graph=scene_graph,
                scene_delta=scene_delta,
            )
        if belief.trade_intents:
            raise RuntimeError(
                "legacy Decision/Risk compatibility mode rejects non-zero "
                "TradeIntent authority"
            )
        decision_belief = (
            RuntimeActionBeliefView(
                belief,
                self.action_disabled_playbooks,
            )
            if self.action_disabled_playbooks
            else belief
        )
        decision = self.decision.decide(observation, decision_belief, account)
        risk = self.risk.review(decision, observation, account)
        snapshot = EngineSnapshot(
            observation=observation,
            belief=belief,
            decision=decision,
            risk=risk,
            neutral_market_state=neutral_market_state,
        )
        self._last_snapshot = snapshot
        self._last_belief_position = resolved_belief_position
        self._neutral_market_state = neutral_market_state
        return snapshot

    def on_bar_neutral_input(
        self,
        bar: Bar,
        *,
        execution: ExecutionRealityInput | None = None,
    ) -> NeutralEngineSnapshot:
        """Advance only Eye and the neutral market projection for one bar."""

        if isinstance(self._last_snapshot, EngineSnapshot):
            raise RuntimeError(
                "Engine entry mode cannot change after full evaluation"
            )
        if not self.observer.config.project_scene_graph:
            raise RuntimeError(
                "neutral-only Engine requires Scene Graph projection"
            )
        observation = self._observe_bar(bar, execution=execution)
        _, neutral_market_state = self._project_neutral(observation)
        if neutral_market_state is None:
            raise RuntimeError(
                "neutral-only Engine did not produce a neutral state"
            )
        snapshot = NeutralEngineSnapshot(
            observation=observation,
            neutral_market_state=neutral_market_state,
        )
        self._last_snapshot = snapshot
        self._neutral_market_state = neutral_market_state
        return snapshot

    def _observe_bar(
        self,
        bar: Bar,
        *,
        execution: ExecutionRealityInput | None,
    ) -> MarketObservation:
        update = self.reader.on_bar(bar)
        return self.observer.observe(update, execution)

    def _project_neutral(
        self,
        observation: MarketObservation,
    ) -> tuple[GlobalMarketContext | None, NeutralMarketState | None]:
        scene_graph = self.observer.scene_graph
        scene_delta = self.observer.last_scene_delta
        if scene_graph is None or scene_delta is None:
            return None, None
        raw_global_context = update_global_market_context(
            (
                None
                if self._neutral_market_state is None
                else self._neutral_market_state.global_context
            ),
            observation,
            scene_delta,
            scene_graph,
        )
        previous_neutral_theses = (
            ()
            if self._neutral_market_state is None
            else self._neutral_market_state.open_market_theses
        )
        neutral_global_context = replace(
            raw_global_context,
            open_market_theses=build_open_market_theses(
                previous_neutral_theses,
                observation,
                scene_delta,
                scene_graph,
                raw_global_context,
            ),
        )
        neutral_market_state = build_neutral_market_state(
            self._neutral_market_state,
            observation,
            neutral_global_context,
        )
        return neutral_global_context, neutral_market_state

    def compact_scene_graph_runtime(self) -> Mapping[str, Any]:
        """Compact only cold Scene Graph history after snapshot consumption.

        The unified replay runner owns when this opt-in operation is legal.
        Engine merely supplies the exact current belief/position identities
        that the graph cannot infer from Eye state alone.
        """

        snapshot = self._last_snapshot
        if snapshot is None:
            raise RuntimeError(
                "scene graph compaction requires a completed Engine snapshot"
            )

        identities: set[str] = set()

        def collect(value: Any) -> None:
            if isinstance(value, str):
                if value:
                    identities.add(value)
                return
            if isinstance(value, Mapping):
                for key, item in value.items():
                    collect(key)
                    collect(item)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        if isinstance(snapshot, EngineSnapshot):
            belief = snapshot.belief
            collect(to_primitive(belief.thesis_candidates))
            # Dormant episodes and childless Context Theses have no action
            # authority, but their frozen identities are still causal owners
            # for a later terminal delta.  Compaction must not turn that future
            # terminal into an unresolvable bounded-view disappearance.
            collect(to_primitive(belief.retained_episode_candidates))
            collect(to_primitive(belief.position_management_candidates))
            collect(to_primitive(belief.context_theses))
            collect(to_primitive(belief.entry_episodes))
            if belief.focus_state is not None:
                collect(to_primitive(belief.focus_state))
            context = belief.global_context
            if context is not None:
                collect(to_primitive(context.open_market_theses))
                collect(to_primitive(context.authority_stack))

                # Do not protect the complete market-wide obstruction, draw
                # or conflict inventories.  They can contain thousands of
                # historical identities and made opt-in compaction effectively
                # unbounded.  Candidate/open-thesis payloads above contribute
                # only conflicts they actually reference; retain the endpoints
                # of those conflicts so their explanation remains resolvable.
                referenced_conflict_ids = set(belief.cross_scale_conflicts)
                referenced_conflict_ids.update(
                    conflict_id
                    for thesis in context.open_market_theses
                    for conflict_id in thesis.conflict_ids
                )
                for conflict in context.material_conflicts:
                    if conflict.conflict_id in referenced_conflict_ids:
                        collect(to_primitive(conflict))
            if self._last_belief_position is not None:
                collect(to_primitive(self._last_belief_position))
        # For a neutral snapshot, compact_runtime_history receives the current
        # Observation below and protects its materialized state directly.  Do
        # not also turn every string in that bounded view into a graph root.
        neutral = snapshot.neutral_market_state
        if neutral is not None:
            collect(to_primitive(neutral.market_episodes))
            collect(to_primitive(neutral.open_market_theses))
        return self.observer.scene_graph.compact_runtime_history(
            snapshot.observation,
            protected_source_ids=tuple(sorted(identities)),
        )

    def histories(self, bars: int = 80):
        return {
            timeframe: self.reader.window(timeframe, bars)
            for timeframe in self.reader.active_timeframes
        }


__all__ = [
    "ContinuousSMCEngine",
    "NEUTRAL_ENGINE_CHECKPOINT_SCHEMA_VERSION",
    "RUNTIME_ACTION_POLICY_SCHEMA_VERSION",
    "RuntimeActionBeliefView",
    "normalize_action_disabled_playbooks",
]
