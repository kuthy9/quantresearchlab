from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from brain.core.calibration import (
    CalibrationError,
    TYPED_CALIBRATION_DIMENSIONS,
    TypedBrainCalibrator,
)
from shares.core.engine import (
    ContinuousSMCEngine,
    LEGACY_ACTION_PIPELINE_MODE,
    normalize_action_disabled_playbooks,
)
from shares.core.model import Playbook
from brain.core.playbook_registry import load_playbook_registry


DFP = Playbook.DISPLACEMENT_FIRST_PULLBACK
LSR = Playbook.LIQUIDITY_SWEEP_REVERSAL
FAVR = Playbook.FAILED_AUCTION_VALUE_RETURN
REGISTRY_HASH = "a" * 64


def _dimension_map(offset: float = 0.0) -> dict[str, object]:
    return {
        "episodes": 100,
        "points": [
            {
                "raw_value": 0.0,
                "calibrated_value": offset,
                "episodes": 50,
            },
            {
                "raw_value": 1.0,
                "calibrated_value": 1.0,
                "episodes": 50,
            },
        ],
    }


def _artifact() -> dict[str, object]:
    return {
        "calibration_version": "typed-test",
        "status": "ready",
        "playbook_registry_hash": REGISTRY_HASH,
        "playbooks": {
            DFP.value: {
                "status": "active",
                "dimensions": {
                    dimension: _dimension_map(0.2)
                    for dimension in TYPED_CALIBRATION_DIMENSIONS
                },
            },
            LSR.value: {
                "status": "active",
                "dimensions": {
                    dimension: _dimension_map(0.1)
                    for dimension in TYPED_CALIBRATION_DIMENSIONS
                },
            },
            FAVR.value: {
                "status": "parked_missing_natural_authority",
                "dimensions": {},
            },
        },
    }


def _write_artifact(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "typed_calibration.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _load(tmp_path: Path, payload: dict[str, object]) -> TypedBrainCalibrator:
    return TypedBrainCalibrator.from_file(
        _write_artifact(tmp_path, payload),
        expected_registry_hash=REGISTRY_HASH,
    )


def test_typed_calibrator_maps_active_dimensions_and_passes_sequence(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, _artifact())
    calibrator = TypedBrainCalibrator.from_file(
        path,
        expected_registry_hash=REGISTRY_HASH,
    )

    assert calibrator.version == "typed-test"
    assert calibrator.is_ready
    assert calibrator.apply(DFP, "thesis_strength", 0.5) == pytest.approx(0.6)
    assert calibrator.apply(LSR, "uncertainty", 0.5) == pytest.approx(0.55)
    assert calibrator.apply(DFP, "sequence_progress", 0.37) == 0.37
    with pytest.raises(CalibrationError, match="parked"):
        calibrator.apply(FAVR, "thesis_strength", 0.5)


def test_typed_identity_returns_raw_for_every_dimension_and_playbook() -> None:
    calibrator = TypedBrainCalibrator.identity()
    assert not calibrator.is_ready
    for playbook in Playbook:
        for dimension in (*TYPED_CALIBRATION_DIMENSIONS, "sequence_progress"):
            assert calibrator.apply(playbook, dimension, 0.43) == 0.43


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("status", "draft"), "only a ready"),
        (
            lambda value: value.__setitem__("playbook_registry_hash", "c" * 64),
            "registry hash is stale",
        ),
        (
            lambda value: value["playbooks"][DFP.value]["dimensions"].pop(
                "delivery_quality"
            ),
            "exactly the five",
        ),
        (
            lambda value: value["playbooks"][DFP.value]["dimensions"][
                "thesis_strength"
            ]["points"][1].__setitem__("calibrated_value", 0.1),
            "not monotone",
        ),
        (
            lambda value: value["playbooks"][FAVR.value].__setitem__(
                "status", "active"
            ),
            "must be parked",
        ),
    ],
)
def test_typed_artifact_fails_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = copy.deepcopy(_artifact())
    mutation(payload)
    with pytest.raises(CalibrationError, match=message):
        _load(tmp_path, payload)


def test_engine_accepts_current_typed_config_and_rejects_incomplete_current_config(
    tmp_path: Path,
) -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )

    assert isinstance(engine.brain.calibrator, TypedBrainCalibrator)
    assert engine.runtime_mode == "development"
    assert not engine.brain.calibrator.is_ready
    assert not engine.decision.calibration_ready
    assert engine.decision.calibration_version == "identity-unvalidated"
    assert engine.observer.config.eye_authority_mode is False
    assert engine.observer.config.project_scene_graph is True
    assert engine.observer.semantic_registry.identity == (
        "f92b24c86bf942defc88de4edb7be16cc2a30dd64fde3b4432657780648b1f0c"
    )
    with pytest.raises(TypeError, match="runtime_mode"):
        ContinuousSMCEngine.from_config("configs/model.json")
    with pytest.raises(RuntimeError, match="readiness gate"):
        ContinuousSMCEngine(
            reader=engine.reader,
            observer=engine.observer,
            brain=engine.brain,
            decision=engine.decision,
            risk=engine.risk,
            runtime_mode="live",
            action_pipeline_mode=LEGACY_ACTION_PIPELINE_MODE,
        )
    incomplete = tmp_path / "model.json"
    incomplete.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="model.schema_version must be 4"):
        ContinuousSMCEngine.from_config(
            incomplete,
            runtime_mode="development",
        )
    payload = json.loads(Path("configs/model.json").read_text(encoding="utf-8"))
    payload["observer"].pop("interaction_protocol")
    incomplete.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="interaction_protocol"):
        ContinuousSMCEngine.from_config(
            incomplete,
            runtime_mode="development",
        )

    payload = json.loads(Path("configs/model.json").read_text(encoding="utf-8"))
    legacy_protocol = payload["observer"].pop("interaction_protocol")
    payload["observer"]["group5_protocol"] = legacy_protocol
    incomplete.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="no longer accepts group5_protocol"):
        ContinuousSMCEngine.from_config(
            incomplete,
            runtime_mode="development",
        )

    payload = json.loads(Path("configs/model.json").read_text(encoding="utf-8"))
    payload["observer"]["group5_protocol"] = "configs/conflicting-entry.json"
    incomplete.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="no longer accepts group5_protocol"):
        ContinuousSMCEngine.from_config(
            incomplete,
            runtime_mode="development",
        )

    payload = json.loads(Path("configs/model.json").read_text(encoding="utf-8"))
    payload["action_pipeline"]["mode"] = "trade_intent_fsm"
    incomplete.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy Decision/Risk compatibility"):
        ContinuousSMCEngine.from_config(
            incomplete,
            runtime_mode="development",
        )

    for field, value, message in (
        (
            "atomic_registry",
            None,
            "semantic_selection fields differ",
        ),
        (
            "atomic_semantics_version",
            "smc_semantics_v2.0",
            "does not implement the selected atomic semantics",
        ),
        (
            "foundation_registry_identity",
            "0" * 64,
            "canonical identity mismatch",
        ),
        (
            "foundation_registry",
            "semantics/missing_foundation.json",
            "must be a direct regular file",
        ),
    ):
        payload = json.loads(
            Path("configs/model.json").read_text(encoding="utf-8")
        )
        if value is None:
            payload["semantic_selection"].pop(field)
        else:
            payload["semantic_selection"][field] = value
        incomplete.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            ContinuousSMCEngine.from_config(
                incomplete,
                runtime_mode="development",
            )


def test_engine_routes_existing_optional_eye_projection_flags(
    tmp_path: Path,
) -> None:
    payload = json.loads(Path("configs/model.json").read_text(encoding="utf-8"))
    payload["observer"].update(
        project_scene_graph=False,
        materialize_event_view=False,
        persist_state_projections=False,
    )
    configured = tmp_path / "model.json"
    configured.write_text(json.dumps(payload), encoding="utf-8")

    engine = ContinuousSMCEngine.from_config(
        configured,
        runtime_mode="development",
    )

    assert engine.observer.config.project_scene_graph is False
    assert engine.observer.config.materialize_event_view is False
    assert engine.observer.config.persist_state_projections is False

    payload["observer"]["project_scene_graph"] = True
    configured.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Scene Graph disabled"):
        ContinuousSMCEngine.from_config(
            configured,
            runtime_mode="development",
        )

    payload["observer"]["project_scene_graph"] = "false"
    configured.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="projection flag must be boolean"):
        ContinuousSMCEngine.from_config(
            configured,
            runtime_mode="development",
        )


def test_engine_runtime_action_policy_is_explicit_deterministic_and_fail_closed() -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
        action_disabled_playbooks=(LSR,),
    )

    assert engine.action_disabled_playbooks == (LSR,)
    assert engine.runtime_action_policy_identity == {
        "schema_version": 2,
        "action_pipeline_mode": LEGACY_ACTION_PIPELINE_MODE,
        "scope": "new_entry_action_candidates_only",
        "disabled_new_entry_playbooks": [LSR.value],
        "decision_belief_projection": "action_filtered",
        "engine_snapshot_belief_projection": "raw",
        "position_management_projection": "raw",
        "trade_intent_projection": "disabled_in_legacy_compat",
    }
    assert normalize_action_disabled_playbooks((LSR.value, DFP)) == (
        DFP,
        LSR,
    )
    with pytest.raises(ValueError, match="duplicate action-disabled"):
        normalize_action_disabled_playbooks((LSR, LSR.value))
    with pytest.raises(ValueError, match="unknown action-disabled"):
        normalize_action_disabled_playbooks(("not_a_playbook",))


def _stub_bar() -> SimpleNamespace:
    """A bar stub carrying only the clock the Engine reads off it."""

    return SimpleNamespace(end=pd.Timestamp("2024-06-03 09:30", tz="America/New_York"))


def test_engine_snapshot_retains_raw_belief_while_decision_gets_policy_view() -> None:
    lsr = SimpleNamespace(playbook=LSR)
    dfp = SimpleNamespace(playbook=DFP)
    raw_belief = SimpleNamespace(
        trade_intents=(),
        action_candidate_items=lambda: (
            ("candidate:lsr", lsr),
            ("candidate:dfp", dfp),
        )
    )
    observation = SimpleNamespace(anomalies=())
    captured: dict[str, object] = {}

    def decide(_observation, belief, _account):
        captured["belief"] = belief
        return SimpleNamespace()

    engine = ContinuousSMCEngine(
        reader=SimpleNamespace(on_bar=lambda _bar: SimpleNamespace()),
        observer=SimpleNamespace(
            observe=lambda _update, _execution: observation,
            config=SimpleNamespace(tick_size=0.25, point_value=20.0),
            scene_graph=SimpleNamespace(),
            last_scene_delta=None,
        ),
        brain=SimpleNamespace(
            update=lambda _observation, **_kwargs: raw_belief,
            project_shadow_trade_intents=lambda *_args: pytest.fail(
                "legacy authority must not invoke TradeIntent projection"
            ),
        ),
        decision=SimpleNamespace(decide=decide),
        risk=SimpleNamespace(review=lambda *_args: SimpleNamespace()),
        runtime_mode="development",
        action_pipeline_mode=LEGACY_ACTION_PIPELINE_MODE,
        action_disabled_playbooks=(LSR,),
    )

    snapshot = engine.on_bar(_stub_bar())

    assert snapshot.belief is raw_belief
    decision_belief = captured["belief"]
    assert decision_belief is not raw_belief
    assert decision_belief.action_candidate_items() == (
        ("candidate:dfp", dfp),
    )
    assert raw_belief.action_candidate_items() == (
        ("candidate:lsr", lsr),
        ("candidate:dfp", dfp),
    )

    engine.brain.update = lambda _observation, **_kwargs: SimpleNamespace(
        trade_intents=(object(),),
        action_candidate_items=lambda: (),
    )
    engine.decision.decide = lambda *_args: pytest.fail(
        "Decision must not run after non-zero TradeIntent rejection"
    )
    with pytest.raises(RuntimeError, match="rejects non-zero TradeIntent"):
        engine.on_bar(_stub_bar())


def test_engine_live_mode_has_one_fail_closed_release_gate(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="live execution readiness"):
        ContinuousSMCEngine.from_config(
            "configs/model.json",
            runtime_mode="live",
        )

    group5 = json.loads(
        Path("configs/primitives_entry.json").read_text(encoding="utf-8")
    )
    group5_path = tmp_path / "primitives_entry.json"
    group5_path.write_text(json.dumps(group5), encoding="utf-8")

    model = json.loads(
        Path("configs/model.json").read_text(encoding="utf-8")
    )
    model["observer"]["interaction_protocol"] = str(group5_path)
    model["release_readiness"] = {
        "active_model_natural_authority_validated": True,
        "economic_validation_complete": True,
        "rolling_oof_complete": True,
        "mbo_stability_validated": True,
        "live_execution_allowed": True,
    }
    model_path = tmp_path / "model-live.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")

    with pytest.raises(RuntimeError, match="typed_brain_calibration_ready"):
        ContinuousSMCEngine.from_config(
            model_path,
            runtime_mode="live",
        )

    registry = load_playbook_registry(model["playbook_registry"])
    artifact = _artifact()
    artifact["playbook_registry_hash"] = registry.fingerprint
    artifact_path = _write_artifact(tmp_path, artifact)
    model["calibration_artifact"] = str(artifact_path)
    model_path.write_text(json.dumps(model), encoding="utf-8")

    engine = ContinuousSMCEngine.from_config(model_path, runtime_mode="live")
    assert isinstance(engine, ContinuousSMCEngine)
    assert engine.runtime_mode == "live"
    assert engine.brain.calibrator.is_ready
    assert engine.decision.calibration_ready
    assert (
        engine.decision.calibration_version
        == engine.brain.calibrator.version
    )

    for field in (
        "active_model_natural_authority_validated",
        "economic_validation_complete",
        "rolling_oof_complete",
        "mbo_stability_validated",
        "live_execution_allowed",
    ):
        blocked = json.loads(json.dumps(model))
        blocked["release_readiness"][field] = False
        blocked_path = tmp_path / f"model-live-missing-{field}.json"
        blocked_path.write_text(json.dumps(blocked), encoding="utf-8")
        with pytest.raises(RuntimeError, match=field):
            ContinuousSMCEngine.from_config(
                blocked_path,
                runtime_mode="live",
            )

    with pytest.raises(ValueError, match="runtime_mode"):
        ContinuousSMCEngine.from_config(
            "configs/model.json",
            runtime_mode="paper",
        )
