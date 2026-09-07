from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from brain.core.dol_probability import (
    DOLCandidatePathSupport,
    DOLProbabilityArtifactError,
    DOLProbabilityModelArtifact,
    DOLProbabilityProtocolError,
    NO_TARGET_BEFORE_HORIZON,
    load_dol_probability_model_artifact,
    load_dol_probability_protocol,
    marginalize_dol_probabilities,
)
from brain.core.dol_ranking import (
    DOLCandidateFact,
    DOLDirection,
    DOLObstructionFact,
    DOLObstructionViewFact,
    load_dol_ranking_protocol,
    rank_dol_candidates,
)
from brain.core.path_belief import (
    PathKind,
    create_path_competition_set,
    load_path_belief_protocol,
    restore_path_competition_set,
)


T0 = pd.Timestamp("2026-08-21 09:30:00", tz="America/New_York")
EXPIRY = pd.Timestamp("2026-08-21 16:00:00", tz="America/New_York")


def _path_protocol():
    return load_path_belief_protocol("brain/configs/path_hypotheses.json")


def _ranking_protocol():
    return load_dol_ranking_protocol("brain/configs/path_hypotheses.json")


def _probability_protocol():
    return load_dol_probability_protocol("brain/configs/dol_probability.json")


def _path_state():
    return create_path_competition_set(
        _path_protocol(),
        instrument_id="NQ:front",
        market_epoch_id="epoch:nq:2026-08-21",
        authority_structure_id="h1-structure:123",
        horizon_id="ny-session:2026-08-21",
        formed_at=T0,
        common_expires_at=EXPIRY,
    )


def _candidate(
    candidate_id: str,
    *,
    price: float,
    path: PathKind = PathKind.CONTINUATION,
    side: str = "above",
    source_ids: tuple[str, ...] | None = None,
    strength: float = 0.5,
) -> DOLCandidateFact:
    return DOLCandidateFact(
        candidate_id=candidate_id,
        timeframe="1H",
        side=side,
        target_price=price,
        source_kind="candidate_liquidity_level",
        source_ids=(f"event:{candidate_id}",) if source_ids is None else source_ids,
        structural_rank="external",
        strength=strength,
        age_real_completed_bars=4,
        path=path,
    )


def _obstacle(
    obstruction_id: str,
    *,
    lower: float,
    upper: float | None = None,
    hard: bool = True,
    source_ids: tuple[str, ...] | None = None,
) -> DOLObstructionFact:
    return DOLObstructionFact(
        obstruction_id=obstruction_id,
        lower_bound=lower,
        upper_bound=lower if upper is None else upper,
        hard=hard,
        source_kind="test_obstruction",
        source_ids=(f"event:{obstruction_id}",) if source_ids is None else source_ids,
    )


def _view(
    direction: DOLDirection = DOLDirection.LONG,
    *obstacles: DOLObstructionFact,
) -> DOLObstructionViewFact:
    return DOLObstructionViewFact(
        direction=direction,
        hard_barriers=tuple(item for item in obstacles if item.hard),
        soft_frictions=tuple(item for item in obstacles if not item.hard),
    )


_DEFAULT_ARTIFACT = object()


def _run(
    candidates,
    *,
    view: DOLObstructionViewFact | None = None,
    path_state=None,
    model_artifact=_DEFAULT_ARTIFACT,
):
    resolved_artifact = (
        _model_artifact()
        if model_artifact is _DEFAULT_ARTIFACT
        else model_artifact
    )
    return marginalize_dol_probabilities(
        _probability_protocol(),
        ranking_protocol=_ranking_protocol(),
        direction=DOLDirection.LONG,
        current_price=100.0,
        external_draw_candidates=candidates,
        obstruction_view=_view() if view is None else view,
        path_state=_path_state() if path_state is None else path_state,
        model_artifact=resolved_artifact,
    )


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_payload(protocol, *, no_target_weight: float = 2.0):
    path_state = _path_state()
    parameters = protocol.development_model.payload()
    parameters["no_target_log_weight"] = {
        path.value: no_target_weight for path in PathKind
    }
    semantic = {
        "schema_version": 2,
        "artifact_id": "dol-artifact:test-v1",
        "model_version": protocol.model_version,
        "protocol_fingerprint": protocol.fingerprint,
        "ranking_protocol_fingerprint": protocol.ranking_protocol_fingerprint,
        "source_path_protocol_fingerprint": path_state.protocol_fingerprint,
        "source_path_model_version": path_state.model_version,
        "fit_status": "fitted",
        "admission_status": "admitted",
        "calibration_status": "fitted_admitted",
        "authority": "shadow_only",
        "action_authority": False,
        "parameters": parameters,
    }
    return {**semantic, "artifact_fingerprint": _canonical_hash(semantic)}


def _model_artifact():
    protocol = _probability_protocol()
    payload = _artifact_payload(protocol, no_target_weight=0.0)
    return DOLProbabilityModelArtifact(
        schema_version=payload["schema_version"],
        artifact_id=payload["artifact_id"],
        model_version=payload["model_version"],
        protocol_fingerprint=payload["protocol_fingerprint"],
        ranking_protocol_fingerprint=payload[
            "ranking_protocol_fingerprint"
        ],
        source_path_protocol_fingerprint=payload[
            "source_path_protocol_fingerprint"
        ],
        source_path_model_version=payload["source_path_model_version"],
        fit_status=payload["fit_status"],
        admission_status=payload["admission_status"],
        calibration_status=payload["calibration_status"],
        authority=payload["authority"],
        action_authority=False,
        parameters=protocol.development_model,
        fingerprint=payload["artifact_fingerprint"],
    )


def test_unfitted_protocol_cannot_manufacture_a_probability_result() -> None:
    protocol = _probability_protocol()

    assert protocol.status == "development_unvalidated"
    assert protocol.calibration_status == "not_fitted_not_admitted"
    assert protocol.authority == "shadow_only"
    assert protocol.action_authority is False
    with pytest.raises(
        DOLProbabilityArtifactError,
        match="requires a fitted/admitted model artifact",
    ):
        _run((_candidate("draw", price=110.0),), model_artifact=None)


def test_default_protocol_path_resolves_outside_repository_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _probability_protocol()
    monkeypatch.chdir(tmp_path)

    loaded = load_dol_probability_protocol()

    assert loaded == expected


def test_no_target_is_explicit_and_multi_path_candidate_is_marginalized() -> None:
    candidate = _candidate("multi-path", price=110.0)
    support = DOLCandidatePathSupport(
        candidate=candidate,
        supported_paths=(PathKind.REVERSAL, PathKind.CONTINUATION),
    )

    result = _run((support,))

    assert result.no_target_outcome == NO_TARGET_BEFORE_HORIZON
    assert len(result.ranked_candidates) == 1
    ranked = result.ranked_candidates[0]
    assert ranked.supported_paths == (
        PathKind.CONTINUATION,
        PathKind.REVERSAL,
    )
    assert tuple(item.path for item in ranked.path_contributions) == (
        PathKind.CONTINUATION,
        PathKind.REVERSAL,
    )
    assert ranked.probability == pytest.approx(
        math.fsum(
            item.path_probability * item.conditional_probability
            for item in ranked.path_contributions
        )
    )
    no_target_by_path = {
        item.path: item for item in result.no_target_path_contributions
    }
    assert set(no_target_by_path) == set(PathKind)
    for path in (
        PathKind.DEEPER_RETRACEMENT,
        PathKind.BALANCE,
        PathKind.FAILED_BREAKOUT,
        PathKind.RESIDUAL_UNKNOWN,
    ):
        assert no_target_by_path[path].conditional_probability == 1.0
    assert math.fsum(
        [
            *(item.probability for item in result.ranked_candidates),
            result.no_target_probability,
        ]
    ) == pytest.approx(1.0)


def test_path_conditionals_normalize_with_overlapping_candidate_support() -> None:
    candidates = (
        DOLCandidatePathSupport(
            _candidate("a", price=105.0, strength=0.3),
            (PathKind.CONTINUATION, PathKind.REVERSAL),
        ),
        DOLCandidatePathSupport(
            _candidate("b", price=110.0, path=PathKind.REVERSAL, strength=0.8),
            (PathKind.REVERSAL,),
        ),
    )

    result = _run(candidates)
    contributions = [
        contribution
        for candidate in result.ranked_candidates
        for contribution in candidate.path_contributions
    ] + list(result.no_target_path_contributions)
    for path in PathKind:
        path_contributions = [item for item in contributions if item.path is path]
        assert math.fsum(
            item.conditional_probability for item in path_contributions
        ) == pytest.approx(1.0)
        assert math.fsum(
            item.marginal_probability for item in path_contributions
        ) == pytest.approx(path_contributions[0].path_probability)


def test_obstacle_filtering_is_exactly_reused_from_existing_ranking() -> None:
    candidate = _candidate(
        "draw",
        price=110.0,
        source_ids=("event:draw",),
    )
    obstacles = (
        _obstacle("hard", lower=103.0),
        _obstacle("soft", lower=105.0, hard=False),
        _obstacle("outside", lower=111.0),
        _obstacle("self", lower=104.0, source_ids=("event:draw",)),
        _obstacle("target", lower=110.0),
    )
    view = _view(DOLDirection.LONG, *reversed(obstacles))
    legacy = rank_dol_candidates(
        _ranking_protocol(),
        direction=DOLDirection.LONG,
        current_price=100.0,
        external_draw_candidates=(candidate,),
        obstruction_view=view,
        path_state=_path_state(),
    ).ranked_candidates[0]

    probability = _run((candidate,), view=view).ranked_candidates[0]

    assert probability.base_conditional_log_weight == legacy.conditional_log_weight
    assert probability.feature_values == legacy.feature_values
    assert probability.hard_obstacle_ids == legacy.hard_obstacle_ids
    assert probability.soft_obstacle_ids == legacy.soft_obstacle_ids
    assert probability.excluded_obstacles == legacy.excluded_obstacles


def test_empty_inventory_is_exact_no_target_distribution() -> None:
    result = _run(())

    assert result.ranked_candidates == ()
    assert result.excluded_candidates == ()
    assert result.no_target_probability == pytest.approx(1.0)
    assert all(
        item.conditional_probability == 1.0
        for item in result.no_target_path_contributions
    )


def test_input_order_checkpoint_and_obstacle_order_do_not_change_replay() -> None:
    state = _path_state()
    restored = restore_path_competition_set(
        state.state_dict(),
        protocol=_path_protocol(),
    )
    candidates = (
        DOLCandidatePathSupport(
            _candidate("b", price=110.0),
            (PathKind.REVERSAL, PathKind.CONTINUATION),
        ),
        _candidate("a", price=105.0),
    )
    obstacles = (
        _obstacle("hard", lower=103.0),
        _obstacle("soft", lower=104.0, hard=False),
    )

    first = _run(
        candidates,
        view=_view(DOLDirection.LONG, *obstacles),
        path_state=state,
    )
    replay = _run(
        tuple(reversed(candidates)),
        view=_view(DOLDirection.LONG, *reversed(obstacles)),
        path_state=restored,
    )

    assert replay == first
    assert replay.probability_id == first.probability_id


def test_invalid_and_nonfinite_inputs_fail_closed() -> None:
    candidate = _candidate("draw", price=110.0)
    with pytest.raises(ValueError, match="path support"):
        DOLCandidatePathSupport(
            candidate,
            (PathKind.CONTINUATION, PathKind.RESIDUAL_UNKNOWN),
        )
    with pytest.raises(ValueError, match="identities must be unique"):
        _run((candidate, candidate))
    with pytest.raises(ValueError, match="finite number"):
        marginalize_dol_probabilities(
            _probability_protocol(),
            ranking_protocol=_ranking_protocol(),
            direction=DOLDirection.LONG,
            current_price=float("nan"),
            external_draw_candidates=(),
            obstruction_view=_view(),
            path_state=_path_state(),
        )


def test_ranking_and_artifact_fingerprints_fail_closed(tmp_path: Path) -> None:
    protocol = _probability_protocol()
    wrong_ranking = replace(_ranking_protocol(), fingerprint="0" * 64)
    with pytest.raises(DOLProbabilityProtocolError, match="fingerprint"):
        marginalize_dol_probabilities(
            protocol,
            ranking_protocol=wrong_ranking,
            direction=DOLDirection.LONG,
            current_price=100.0,
            external_draw_candidates=(),
            obstruction_view=_view(),
            path_state=_path_state(),
        )

    payload = _artifact_payload(protocol)
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DOLProbabilityArtifactError, match="binding"):
        load_dol_probability_model_artifact(
            artifact_path,
            protocol=protocol,
            expected_fingerprint="f" * 64,
        )
    artifact = load_dol_probability_model_artifact(
        artifact_path,
        protocol=protocol,
        expected_fingerprint=payload["artifact_fingerprint"],
    )
    fitted = _run((_candidate("draw", price=110.0),), model_artifact=artifact)
    assert fitted.calibration_status == "fitted_admitted"
    assert fitted.model_source == "fitted_admitted_artifact"
    assert fitted.model_fingerprint == payload["artifact_fingerprint"]
    assert fitted.status == "development_unvalidated"
    assert fitted.authority == "shadow_only"
    assert fitted.action_authority is False

    payload["parameters"]["no_target_log_weight"]["continuation"] = 99.0
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DOLProbabilityArtifactError, match="artifact boundary"):
        load_dol_probability_model_artifact(
            artifact_path,
            protocol=protocol,
            expected_fingerprint=payload["artifact_fingerprint"],
        )


def test_rehashed_artifact_with_wrong_protocol_binding_is_rejected(
    tmp_path: Path,
) -> None:
    protocol = _probability_protocol()
    payload = _artifact_payload(protocol)
    payload["protocol_fingerprint"] = "0" * 64
    semantic = {
        key: value
        for key, value in payload.items()
        if key != "artifact_fingerprint"
    }
    payload["artifact_fingerprint"] = _canonical_hash(semantic)
    artifact_path = tmp_path / "wrong-binding.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DOLProbabilityArtifactError, match="binding"):
        load_dol_probability_model_artifact(
            artifact_path,
            protocol=protocol,
            expected_fingerprint=payload["artifact_fingerprint"],
        )


def test_fitted_artifact_is_bound_to_one_path_protocol_and_model() -> None:
    foreign_protocol = replace(
        _path_protocol(),
        model_version="foreign-path-model-v1",
        fingerprint="e" * 64,
    )
    foreign_state = create_path_competition_set(
        foreign_protocol,
        instrument_id="NQ:front",
        market_epoch_id="epoch:nq:foreign",
        authority_structure_id="h1-structure:foreign",
        horizon_id="ny-session:foreign",
        formed_at=T0,
        common_expires_at=EXPIRY,
    )

    with pytest.raises(DOLProbabilityArtifactError, match="binding is invalid"):
        _run(
            (_candidate("draw", price=110.0),),
            path_state=foreign_state,
            model_artifact=_model_artifact(),
        )
