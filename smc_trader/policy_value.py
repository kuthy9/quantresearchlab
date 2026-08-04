"""Preregistered managed-policy value calibration for v2.1 enter variants."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration import CalibrationError, model_code_fingerprint
from .decision import DecisionConfig, UtilityDecisionLayer
from .engine import ContinuousSMCEngine
from .model import (
    AccountState,
    Action,
    ActionUtility,
    Decision,
    MarketBelief,
    MarketObservation,
    Playbook,
    PlaybookPhase,
)
from .playbook_registry import load_playbook_registry


@dataclass(frozen=True)
class PolicyValuePoint:
    structural_score_R: float
    managed_gross_R: float
    episodes: int


@dataclass(frozen=True)
class PlaybookPolicyValueMap:
    playbook: Playbook
    episodes: int
    direction_episodes: Mapping[str, int]
    points: tuple[PolicyValuePoint, ...]

    def supports(self, structural_score_R: float) -> bool:
        if not math.isfinite(float(structural_score_R)):
            return False
        return (
            self.points[0].structural_score_R
            <= float(structural_score_R)
            <= self.points[-1].structural_score_R
        )

    def apply(self, structural_score_R: float) -> float:
        if not math.isfinite(float(structural_score_R)):
            raise CalibrationError("managed-policy structural score is not finite")
        if not self.supports(structural_score_R):
            raise CalibrationError(
                "managed-policy structural score is outside fitted support"
            )
        x = np.asarray(
            [point.structural_score_R for point in self.points],
            dtype=float,
        )
        y = np.asarray(
            [point.managed_gross_R for point in self.points],
            dtype=float,
        )
        return float(
            np.interp(
                float(structural_score_R),
                x,
                y,
                left=y[0],
                right=y[-1],
            )
        )


@dataclass(frozen=True)
class ManagedPolicyValueCalibrator:
    version: str
    fingerprint: str
    policy_protocol_hash: str
    registry_hash: str
    managed_policy_code_hash: str
    managed_policy_pipeline_hash: str
    policy_base_config_hash: str
    maps: Mapping[Playbook, PlaybookPolicyValueMap]
    status: str

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_policy_protocol_hash: str,
        expected_registry_hash: str,
        expected_code_hash: str,
        expected_pipeline_hash: str,
        expected_policy_base_config_hash: str,
    ) -> "ManagedPolicyValueCalibrator":
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = Path(__file__).resolve().parents[1] / source
        raw = source.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise CalibrationError("managed-policy artifact root must be an object")
        if str(payload.get("status", "")) != "ready":
            raise CalibrationError("managed-policy artifact is not ready")
        calibration_version = str(
            payload.get("calibration_version", "")
        ).strip()
        if not calibration_version:
            raise CalibrationError(
                "managed-policy artifact has no calibration version"
            )
        bindings = {
            "policy_value_protocol_hash": expected_policy_protocol_hash,
            "playbook_registry_hash": expected_registry_hash,
            "managed_policy_code_hash": expected_code_hash,
            "managed_policy_pipeline_hash": expected_pipeline_hash,
            "policy_base_config_hash": expected_policy_base_config_hash,
        }
        for field, expected in bindings.items():
            actual = str(payload.get(field, ""))
            if actual != str(expected):
                raise CalibrationError(
                    f"managed-policy artifact {field} is stale"
                )
        raw_maps = payload.get("playbooks")
        if not isinstance(raw_maps, Mapping):
            raise CalibrationError("managed-policy artifact has no playbook maps")
        maps: dict[Playbook, PlaybookPolicyValueMap] = {}
        for playbook in Playbook:
            value = raw_maps.get(playbook.value)
            if not isinstance(value, Mapping):
                raise CalibrationError(
                    f"managed-policy artifact omits {playbook.value}"
                )
            map_status = str(value.get("status", ""))
            if map_status == "unavailable":
                continue
            if map_status != "ready":
                raise CalibrationError(
                    f"{playbook.value} managed value has an invalid status"
                )
            raw_points = value.get("points")
            if not isinstance(raw_points, list) or len(raw_points) < 2:
                raise CalibrationError(
                    f"{playbook.value} managed value requires two points"
                )
            points = tuple(
                PolicyValuePoint(
                    structural_score_R=float(point["structural_score_R"]),
                    managed_gross_R=float(point["managed_gross_R"]),
                    episodes=int(point["episodes"]),
                )
                for point in raw_points
            )
            if any(
                not all(
                    math.isfinite(number)
                    for number in (
                        point.structural_score_R,
                        point.managed_gross_R,
                    )
                )
                or point.episodes <= 0
                for point in points
            ):
                raise CalibrationError(
                    f"{playbook.value} managed value has invalid points"
                )
            if any(
                right.structural_score_R <= left.structural_score_R
                or right.managed_gross_R < left.managed_gross_R
                for left, right in zip(points[:-1], points[1:])
            ):
                raise CalibrationError(
                    f"{playbook.value} managed value is not monotone"
                )
            direction_episodes = value.get("direction_episodes", {})
            if not isinstance(direction_episodes, Mapping):
                raise CalibrationError(
                    f"{playbook.value} direction episodes are invalid"
                )
            episodes = int(value.get("episodes", 0))
            parsed_direction_episodes = {
                str(key): int(count)
                for key, count in direction_episodes.items()
            }
            if (
                episodes <= 0
                or sum(point.episodes for point in points) != episodes
                or sum(parsed_direction_episodes.values()) != episodes
                or any(count < 0 for count in parsed_direction_episodes.values())
            ):
                raise CalibrationError(
                    f"{playbook.value} managed episode counts are inconsistent"
                )
            maps[playbook] = PlaybookPolicyValueMap(
                playbook=playbook,
                episodes=episodes,
                direction_episodes=parsed_direction_episodes,
                points=points,
            )
        return cls(
            version=calibration_version,
            fingerprint=hashlib.sha256(raw).hexdigest(),
            policy_protocol_hash=str(
                payload.get("policy_value_protocol_hash", "")
            ),
            registry_hash=str(payload.get("playbook_registry_hash", "")),
            managed_policy_code_hash=str(
                payload.get("managed_policy_code_hash", "")
            ),
            managed_policy_pipeline_hash=str(
                payload.get("managed_policy_pipeline_hash", "")
            ),
            policy_base_config_hash=str(
                payload.get("policy_base_config_hash", "")
            ),
            maps=maps,
            status="ready",
        )

    def available(self, playbook: Playbook) -> bool:
        return playbook in self.maps

    def supports(
        self,
        playbook: Playbook,
        structural_score_R: float,
    ) -> bool:
        mapping = self.maps.get(playbook)
        return (
            mapping is not None
            and mapping.supports(structural_score_R)
        )

    def apply(self, playbook: Playbook, structural_score_R: float) -> float:
        mapping = self.maps.get(playbook)
        if mapping is None:
            raise CalibrationError(
                f"managed-policy value is unavailable for {playbook.value}"
            )
        return mapping.apply(structural_score_R)


def monotone_policy_value_points(
    structural_scores_R: Sequence[float],
    managed_gross_returns_R: Sequence[float],
    *,
    bins: int = 8,
    minimum_bin_episodes: int = 15,
    prior_mean_R: float = 0.0,
    prior_weight: int = 20,
) -> tuple[PolicyValuePoint, ...]:
    """Fit a fixed-bin monotone gross-value map without threshold search."""

    x = np.asarray(structural_scores_R, dtype=float)
    y = np.asarray(managed_gross_returns_R, dtype=float)
    if len(x) != len(y) or len(x) == 0:
        raise CalibrationError("managed-policy inputs must be non-empty and aligned")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise CalibrationError("managed-policy inputs contain non-finite values")
    if not math.isfinite(float(prior_mean_R)) or int(prior_weight) < 0:
        raise CalibrationError("managed-policy smoothing prior is invalid")
    order = np.argsort(x, kind="stable")
    groups = [
        values
        for values in np.array_split(order, min(int(bins), len(order)))
        if len(values) >= int(minimum_bin_episodes)
    ]
    if len(groups) < 2:
        raise CalibrationError("too few populated managed-policy value bins")
    blocks = [
        {
            "x": float(np.mean(x[index])),
            "y": float(
                (
                    np.sum(y[index])
                    + float(prior_mean_R) * int(prior_weight)
                )
                / (len(index) + int(prior_weight))
            ),
            "n": int(len(index)),
        }
        for index in groups
    ]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index]["y"] <= blocks[index + 1]["y"]:
            index += 1
            continue
        left = blocks[index]
        right = blocks[index + 1]
        count = left["n"] + right["n"]
        blocks[index : index + 2] = [
            {
                "x": (
                    left["x"] * left["n"] + right["x"] * right["n"]
                )
                / count,
                "y": (
                    left["y"] * left["n"] + right["y"] * right["n"]
                )
                / count,
                "n": count,
            }
        ]
        index = max(0, index - 1)
    if len(blocks) < 2:
        raise CalibrationError(
            "managed-policy calibration collapsed to one level"
        )
    if any(
        right["x"] <= left["x"]
        for left, right in zip(blocks[:-1], blocks[1:])
    ):
        raise CalibrationError(
            "managed-policy structural scores have no distinct levels"
        )
    return tuple(
        PolicyValuePoint(
            structural_score_R=float(block["x"]),
            managed_gross_R=float(block["y"]),
            episodes=int(block["n"]),
        )
        for block in blocks
    )


def path_structural_score(utility: ActionUtility) -> float:
    """Return the cost-free structural score frozen by the v2.1 protocol."""

    if utility.action is not Action.ENTER:
        raise ValueError("structural score requires an enter utility")
    required = (
        "expected_gross_R",
        "uncertainty",
        "deadline",
        "phase_readiness",
    )
    missing = [name for name in required if name not in utility.components]
    if missing:
        raise ValueError(
            f"enter utility omits structural components: {missing}"
        )
    return float(sum(float(utility.components[name]) for name in required))


class ManagedUtilityDecisionLayer(UtilityDecisionLayer):
    """Replace path-implied enter value with calibrated managed gross value."""

    def __init__(
        self,
        calibrator: ManagedPolicyValueCalibrator,
        config: DecisionConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.calibrator = calibrator

    def _managed_flat_utilities(
        self,
        observation: MarketObservation,
        belief: MarketBelief,
        *,
        eligible_hypothesis_key: str,
    ) -> list[ActionUtility]:
        base = super()._flat_utilities(observation, belief)
        output: list[ActionUtility] = []
        for utility in base:
            if utility.action is not Action.ENTER:
                output.append(utility)
                continue
            if utility.hypothesis_key is None:
                raise ValueError("enter utility lacks a hypothesis")
            hypothesis = belief.hypotheses[utility.hypothesis_key]
            score = path_structural_score(utility)
            selected_by_behavior_policy = (
                utility.hypothesis_key == eligible_hypothesis_key
            )
            available = self.calibrator.available(hypothesis.playbook)
            supported = self.calibrator.supports(
                hypothesis.playbook,
                score,
            )
            executable = hypothesis.phase is PlaybookPhase.EXECUTABLE
            if (
                selected_by_behavior_policy
                and available
                and supported
                and executable
            ):
                managed = self.calibrator.apply(
                    hypothesis.playbook,
                    score,
                )
                enter = (
                    managed
                    + float(utility.components["cost_R"])
                    + float(utility.components["fillability"])
                )
                reason = (
                    f"managed-policy {self.calibrator.version}; "
                    f"{utility.reason}"
                )
            else:
                managed = -1.0
                enter = -1.0
                reason = (
                    "enter variant was not selected by the frozen behavior "
                    "policy, managed value is unavailable/outside support, "
                    "or phase is not executable; "
                    f"{utility.reason}"
                )
            output.append(
                ActionUtility(
                    action=Action.ENTER,
                    utility=float(enter),
                    components={
                        "path_structural_score_R": score,
                        "managed_gross_R": float(managed),
                        "cost_R": float(utility.components["cost_R"]),
                        "fillability": float(
                            utility.components["fillability"]
                        ),
                        "managed_value_available": 1.0 if available else -1.0,
                        "managed_value_in_support": 1.0 if supported else -1.0,
                        "phase_executable": 1.0 if executable else -1.0,
                        "selected_by_behavior_policy": (
                            1.0 if selected_by_behavior_policy else -1.0
                        ),
                    },
                    hypothesis_key=utility.hypothesis_key,
                    reason=reason,
                )
            )
        return output

    def decide(
        self,
        observation: MarketObservation,
        belief: MarketBelief,
        account: AccountState | None = None,
    ) -> Decision:
        account = account or AccountState(equity=100_000.0)
        behavior_layer = UtilityDecisionLayer(self.config)
        behavior_decision = behavior_layer.decide(
            observation,
            belief,
            account,
        )
        if (
            account.position is not None
            or behavior_decision.selected_action is not Action.ENTER
            or behavior_decision.best_hypothesis_key is None
        ):
            return behavior_decision

        utilities = self._managed_flat_utilities(
            observation,
            belief,
            eligible_hypothesis_key=behavior_decision.best_hypothesis_key,
        )
        ranked = sorted(
            utilities,
            key=lambda item: item.utility,
            reverse=True,
        )
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        behavior_enter = next(
            utility
            for utility in ranked
            if (
                utility.action is Action.ENTER
                and utility.hypothesis_key
                == behavior_decision.best_hypothesis_key
            )
        )
        advantage = (
            float(best.utility - second.utility)
            if second is not None
            else 0.0
        )
        selected = best.action
        reasons = [best.reason]
        if best is not behavior_enter:
            reasons.insert(
                0,
                "frozen behavior-policy enter was conservatively filtered: "
                f"managed net utility={behavior_enter.utility:.3f}R; "
                f"{behavior_enter.reason}",
            )
        if (
            selected is not Action.ABSTAIN
            and advantage < self.config.minimum_utility_advantage
        ):
            selected = Action.ABSTAIN
            reasons.insert(
                0,
                f"best action advantage {advantage:.3f}R is below "
                f"{self.config.minimum_utility_advantage:.3f}R",
            )
        if any(
            name.startswith("warmup_")
            for name in observation.anomalies
        ):
            selected = Action.ABSTAIN
            reasons.insert(
                0,
                "multitimeframe observer is still warming up",
            )
        plan = None
        if best.hypothesis_key is not None:
            hypothesis = belief.hypotheses.get(best.hypothesis_key)
            plan = None if hypothesis is None else hypothesis.plan
        return Decision(
            asof=observation.asof,
            selected_action=selected,
            utilities=tuple(ranked),
            best_hypothesis_key=best.hypothesis_key,
            advantage=advantage,
            reasons=tuple(reasons),
            plan=plan,
        )


def managed_policy_code_fingerprint() -> str:
    """Bind v2.1 value logic to the complete legacy engine and this module."""

    source = Path(__file__)
    digest = hashlib.sha256()
    digest.update(b"legacy_model_code_hash\0")
    digest.update(model_code_fingerprint().encode("ascii"))
    digest.update(b"\0policy_value.py\0")
    digest.update(source.read_bytes())
    return digest.hexdigest()


def managed_policy_pipeline_fingerprint() -> str:
    """Bind the offline episode and fitting pipeline used by the artifact."""

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    digest.update(b"managed_policy_code_hash\0")
    digest.update(managed_policy_code_fingerprint().encode("ascii"))
    for relative in (
        "smc_trader/calibration_replay.py",
        "scripts/run_managed_policy_calibration.py",
        "scripts/build_managed_policy_episodes.py",
        "scripts/fit_managed_policy_value.py",
    ):
        digest.update(b"\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def build_v2_1_engine(
    config_path: str | Path = "configs/model_v2_1.json",
) -> ContinuousSMCEngine:
    """Build v2.1 without mutating the frozen v2 engine factory."""

    config_source = Path(config_path)
    payload: dict[str, Any] = json.loads(
        config_source.read_text(encoding="utf-8")
    )
    if not str(payload.get("version", "")).startswith("2.1."):
        raise ValueError("managed-policy engine requires a v2.1 config")
    artifact_path = payload.get("managed_policy_artifact")
    protocol_path = payload.get(
        "policy_value_protocol",
        "configs/policy_value_protocol_v2_1.json",
    )
    base_config_path = payload.get("policy_base_config")
    expected_base_hash = str(payload.get("policy_base_config_hash", ""))
    if not artifact_path or not base_config_path or not expected_base_hash:
        raise ValueError("v2.1 config lacks managed-policy artifact bindings")
    base_config_source = Path(base_config_path)
    actual_base_hash = hashlib.sha256(base_config_source.read_bytes()).hexdigest()
    if actual_base_hash != expected_base_hash:
        raise CalibrationError("v2.1 policy base config hash is stale")
    registry = load_playbook_registry(
        payload.get("playbook_registry", "configs/playbooks_v2.json")
    )
    base_payload = json.loads(base_config_source.read_text(encoding="utf-8"))
    base_registry = load_playbook_registry(
        base_payload.get("playbook_registry", "configs/playbooks_v2.json")
    )
    if base_registry.fingerprint != registry.fingerprint:
        raise CalibrationError(
            "v2.1 runtime and policy-base playbook registries differ"
        )
    protocol_source = Path(protocol_path)
    protocol_hash = hashlib.sha256(protocol_source.read_bytes()).hexdigest()
    calibrator = ManagedPolicyValueCalibrator.from_file(
        artifact_path,
        expected_policy_protocol_hash=protocol_hash,
        expected_registry_hash=registry.fingerprint,
        expected_code_hash=managed_policy_code_fingerprint(),
        expected_pipeline_hash=managed_policy_pipeline_fingerprint(),
        expected_policy_base_config_hash=expected_base_hash,
    )
    engine = ContinuousSMCEngine.from_config(base_config_source)
    engine.decision = ManagedUtilityDecisionLayer(
        calibrator,
        engine.decision.config,
    )
    return engine


__all__ = [
    "ManagedPolicyValueCalibrator",
    "ManagedUtilityDecisionLayer",
    "PlaybookPolicyValueMap",
    "PolicyValuePoint",
    "build_v2_1_engine",
    "managed_policy_code_fingerprint",
    "managed_policy_pipeline_fingerprint",
    "monotone_policy_value_points",
    "path_structural_score",
]
