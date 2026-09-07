from __future__ import annotations

import math

import pandas as pd
import pytest

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
    reduce_path_competition_set,
    restore_path_competition_set,
)


T0 = pd.Timestamp("2026-08-21 09:30:00", tz="America/New_York")
EXPIRY = pd.Timestamp("2026-08-21 16:00:00", tz="America/New_York")


def _path_protocol():
    return load_path_belief_protocol("brain/configs/path_hypotheses.json")


def _dol_protocol():
    return load_dol_ranking_protocol("brain/configs/path_hypotheses.json")


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
    side: str = "above",
    path: PathKind = PathKind.CONTINUATION,
    source_ids: tuple[str, ...] | None = None,
    strength: float = 0.5,
    age_bars: int = 4,
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
        age_real_completed_bars=age_bars,
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
    direction: DOLDirection,
    *obstacles: DOLObstructionFact,
) -> DOLObstructionViewFact:
    return DOLObstructionViewFact(
        direction=direction,
        hard_barriers=tuple(item for item in obstacles if item.hard),
        soft_frictions=tuple(item for item in obstacles if not item.hard),
    )


def test_dol_protocol_is_explicitly_unvalidated_shadow_only() -> None:
    protocol = _dol_protocol()

    assert protocol.status == "development_unvalidated"
    assert protocol.authority == "shadow_only"
    assert "not_calibrated" in protocol.probability_interpretation
    assert "not_calibrated" in protocol.joint_quality_interpretation
    assert protocol.fingerprint


@pytest.mark.parametrize(
    (
        "direction",
        "expected_id",
        "wrong_side_id",
        "not_ahead_id",
        "at_current_id",
    ),
    [
        (
            DOLDirection.LONG,
            "above",
            "below",
            "above-behind",
            "at-current-above",
        ),
        (
            DOLDirection.SHORT,
            "below",
            "above",
            "below-behind",
            "at-current-below",
        ),
    ],
)
def test_direction_and_strict_forward_target_filter(
    direction: DOLDirection,
    expected_id: str,
    wrong_side_id: str,
    not_ahead_id: str,
    at_current_id: str,
) -> None:
    candidates = (
        _candidate("above", price=110.0),
        _candidate("below", price=90.0, side="below", path=PathKind.REVERSAL),
        _candidate("at-current-above", price=100.0),
        _candidate(
            "at-current-below",
            price=100.0,
            side="below",
            path=PathKind.REVERSAL,
        ),
        _candidate("above-behind", price=90.0),
        _candidate(
            "below-behind",
            price=110.0,
            side="below",
            path=PathKind.REVERSAL,
        ),
    )

    result = rank_dol_candidates(
        _dol_protocol(),
        direction=direction,
        current_price=100.0,
        external_draw_candidates=candidates,
        obstruction_view=_view(direction),
        path_state=_path_state(),
    )

    assert tuple(item.candidate_id for item in result.ranked_candidates) == (
        expected_id,
    )
    exclusions = dict(result.excluded_candidates)
    assert exclusions[wrong_side_id] == "side_opposes_requested_direction"
    assert exclusions[not_ahead_id] == "target_not_strictly_ahead"
    assert exclusions[at_current_id] == "target_not_strictly_ahead"


def test_path_obstacles_use_open_interval_and_exclude_target_self() -> None:
    candidate = _candidate(
        "draw:a",
        price=110.0,
        source_ids=("event:draw:a", "ob:reverse-source"),
    )
    obstacles = (
        _obstacle("at-current", lower=100.0),
        _obstacle("at-target", lower=110.0),
        _obstacle("beyond-target", lower=111.0),
        _obstacle("hard-far", lower=106.0),
        _obstacle("hard-near", lower=102.0),
        _obstacle("soft-mid", lower=104.0, hard=False),
        _obstacle("draw:a", lower=103.0),
        _obstacle(
            "shared-source",
            lower=103.5,
            source_ids=("event:draw:a",),
        ),
        _obstacle(
            "source-has-candidate",
            lower=104.5,
            source_ids=("draw:a",),
        ),
        _obstacle("ob:reverse-source", lower=105.0),
        _obstacle("co-located", lower=109.5, upper=110.5),
    )

    result = rank_dol_candidates(
        _dol_protocol(),
        direction=DOLDirection.LONG,
        current_price=100.0,
        external_draw_candidates=(candidate,),
        obstruction_view=_view(DOLDirection.LONG, *reversed(obstacles)),
        path_state=_path_state(),
    )

    ranked = result.ranked_candidates[0]
    assert ranked.hard_obstacle_ids == ("hard-near", "hard-far")
    assert ranked.soft_obstacle_ids == ("soft-mid",)
    excluded = dict(ranked.excluded_obstacles)
    assert excluded["at-current"] == "outside_strict_path_interval"
    assert excluded["beyond-target"] == "outside_strict_path_interval"
    assert excluded["at-target"] == "target_self_colocated"
    assert excluded["draw:a"] == "target_self_identity"
    assert excluded["shared-source"] == "target_self_source"
    assert excluded["source-has-candidate"] == "target_self_source"
    assert excluded["ob:reverse-source"] == "target_self_source"
    assert excluded["co-located"] == "target_self_colocated"


def test_short_obstacles_are_ordered_by_directional_first_contact() -> None:
    result = rank_dol_candidates(
        _dol_protocol(),
        direction=DOLDirection.SHORT,
        current_price=100.0,
        external_draw_candidates=(
            _candidate(
                "ssl",
                price=90.0,
                side="below",
                path=PathKind.REVERSAL,
            ),
        ),
        obstruction_view=_view(
            DOLDirection.SHORT,
            _obstacle("far", lower=93.0, upper=94.0),
            _obstacle("near", lower=96.0, upper=97.0),
            _obstacle("soft", lower=94.5, upper=95.0, hard=False),
        ),
        path_state=_path_state(),
    )

    ranked = result.ranked_candidates[0]
    assert ranked.hard_obstacle_ids == ("near", "far")
    assert ranked.soft_obstacle_ids == ("soft",)


def test_ties_are_stable_and_softmax_probabilities_sum_to_one() -> None:
    candidates = (
        _candidate("draw:b", price=110.0),
        _candidate("draw:a", price=110.0),
    )

    result = rank_dol_candidates(
        _dol_protocol(),
        direction=DOLDirection.LONG,
        current_price=100.0,
        external_draw_candidates=candidates,
        obstruction_view=_view(DOLDirection.LONG),
        path_state=_path_state(),
    )

    assert tuple(item.candidate_id for item in result.ranked_candidates) == (
        "draw:a",
        "draw:b",
    )
    probabilities = tuple(
        item.normalized_diagnostic_weight
        for item in result.ranked_candidates
    )
    assert probabilities == pytest.approx((0.5, 0.5))
    assert math.fsum(probabilities) == pytest.approx(1.0)
    assert all(
        item.diagnostic_joint_quality
        == pytest.approx(item.path_probability * 0.5)
        for item in result.ranked_candidates
    )


def test_output_references_exact_path_hypothesis_and_probability() -> None:
    path_protocol = _path_protocol()
    state = _path_state()
    contribution = path_protocol.make_contribution(
        competition_set_id=state.competition_set_id,
        rule_id="acceptance_continuation",
        source_event_ids=("event:h1-authority",),
        known_at=T0 + pd.Timedelta(minutes=1),
    )
    updated, _ = reduce_path_competition_set(
        path_protocol,
        state,
        asof=T0 + pd.Timedelta(minutes=1),
        contributions=(contribution,),
        real_completed_bar=True,
    )

    result = rank_dol_candidates(
        _dol_protocol(),
        direction=DOLDirection.LONG,
        current_price=100.0,
        external_draw_candidates=(
            _candidate("continuation", price=110.0),
            _candidate(
                "reversal",
                price=110.0,
                path=PathKind.REVERSAL,
            ),
        ),
        obstruction_view=_view(DOLDirection.LONG),
        path_state=updated,
    )

    by_id = {item.candidate_id: item for item in result.ranked_candidates}
    continuation = updated.member(PathKind.CONTINUATION)
    reversal = updated.member(PathKind.REVERSAL)
    assert by_id["continuation"].path_hypothesis_id == continuation.hypothesis_id
    assert by_id["continuation"].path_probability == continuation.probability
    assert by_id["reversal"].path_hypothesis_id == reversal.hypothesis_id
    assert by_id["reversal"].path_probability == reversal.probability
    assert result.ranked_candidates[0].candidate_id == "continuation"
    assert result.probability_interpretation.endswith("not_calibrated_posterior")


def test_ranking_is_deterministic_across_input_order_and_path_checkpoint() -> None:
    protocol = _dol_protocol()
    path_protocol = _path_protocol()
    state = _path_state()
    candidates = (
        _candidate("draw:c", price=115.0, strength=0.9, age_bars=1),
        _candidate("draw:a", price=105.0, strength=0.3, age_bars=10),
        _candidate("draw:b", price=110.0, strength=0.6, age_bars=5),
    )
    obstacles = (
        _obstacle("hard", lower=103.0),
        _obstacle("soft", lower=104.0, hard=False),
    )
    first = rank_dol_candidates(
        protocol,
        direction=DOLDirection.LONG,
        current_price=100.0,
        external_draw_candidates=candidates,
        obstruction_view=_view(DOLDirection.LONG, *obstacles),
        path_state=state,
    )
    restored = restore_path_competition_set(
        state.state_dict(),
        protocol=path_protocol,
    )
    replay = rank_dol_candidates(
        protocol,
        direction=DOLDirection.LONG,
        current_price=100.0,
        external_draw_candidates=tuple(reversed(candidates)),
        obstruction_view=_view(DOLDirection.LONG, *reversed(obstacles)),
        path_state=restored,
    )

    assert replay == first
    assert replay.ranking_id == first.ranking_id
    assert math.fsum(
        item.normalized_diagnostic_weight
        for item in replay.ranked_candidates
    ) == pytest.approx(1.0)
