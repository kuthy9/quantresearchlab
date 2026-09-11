"""The local conditional hypothesis Brain: geometry, contract, association, forecast."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brain.core.belief_updater import evaluate, normalize_log_weights
from brain.core.forecast import ForecastError, ForecastInput, HypothesisForecaster
from brain.core.hypothesis_pool import (
    HypothesisPool,
    HypothesisPoolError,
    PoolConfig,
    _assign,
    information_gap,
)
from brain.core.hypothesis_proposer import (
    FEATURE_DIM,
    FEATURE_NAMES,
    ForecastIndex,
    HypothesisProposer,
    HypothesisProposerError,
    ProposerConfig,
    load_hypothesis_protocol,
    protocol_fingerprint,
)
from brain.core.trajectory import (
    RealizedPath,
    TrajectoryError,
    detrended_shape,
    direction_vector,
    path_attributes,
    shape_matrix,
)
from brain.research.churn_diagnostics import (
    association_distance_profile,
    cloud_drift,
    cluster_jitter,
    summarize_churn,
)
from brain.research.cluster_study import centroid_reproduction, eta_squared, medoids
from brain.research.design_study import (
    paired_verdict,
    prototype_geometry,
    raw_representation,
    retrieval_skill,
    skill_profile,
    two_channel_representation,
)
from brain.research.forecast_index import (
    ATTRIBUTE_NAMES,
    build_index,
    fit_principal_basis,
)
from contract.brain.forecast import (
    DIRECTION_DIM,
    DIRECTION_FEATURE_NAMES,
    MAX_CLOUD_NODES,
    MAX_LIVE_HYPOTHESES,
    REPRESENTATION_DIM,
    SHAPE_COMPONENT_COUNT,
    TRAJECTORY_CURVE_LENGTH,
    BeliefUncertainty,
    ConditionalCloud,
    Hypothesis,
    HypothesisStatus,
    LifecycleOperation,
    MarketBeliefState,
    PathAttributes,
    TrajectoryNode,
    belief_revision_id,
    mode_ambiguity,
    node_identity,
    retrieval_confidence,
    support_overlap,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "brain" / "configs" / "hypothesis_protocol.json"
ASOF = pd.Timestamp("2022-01-03 10:00:00-05:00")


def _attrs(**overrides) -> PathAttributes:
    base = dict(
        r_5=0.1, r_15=0.3, r_30=0.6, r_60=1.0,
        mfe_0_15=0.4, mfe_15_30=0.3, mfe_30_60=0.5,
        mae_0_15=-0.2, mae_15_30=-0.1, mae_30_60=-0.05,
        time_to_mfe=0.9, time_to_mae=0.2, path_efficiency=0.5,
        rv_30=1.0, rv_60=1.5,
    )
    base.update(overrides)
    return PathAttributes(**base)


def _ramp(level: float) -> tuple[float, ...]:
    """A straight ramp to ``level`` over the horizon."""

    return tuple(
        level * (step + 1) / TRAJECTORY_CURVE_LENGTH
        for step in range(TRAJECTORY_CURVE_LENGTH)
    )


def _node(
    level: float,
    *,
    mass: float = 0.3,
    spread: float = 0.5,
    members: tuple[int, ...] | None = None,
    component_spread: float = 1.0,
) -> TrajectoryNode:
    """A node at ``level``, supported by a named set of historical samples.

    ``members`` defaults to a block derived from the level, so two calls with
    the same level rest on the same history and two different levels do not.
    Identity in the pool is decided by these sets, so a test that does not name
    them is not testing what the pool actually does.
    """

    curve = _ramp(level)
    if members is None:
        base = int(round(level * 1000))
        members = tuple(range(base, base + 40))
    return TrajectoryNode(
        node_id=node_identity(curve),
        curve=curve,
        components=tuple(float(level) for _ in range(REPRESENTATION_DIM)),
        dispersion=tuple(spread for _ in range(TRAJECTORY_CURVE_LENGTH)),
        mass=mass,
        member_count=len(members),
        attributes=_attrs(r_60=level),
        member_ids=members,
        component_spread=component_spread,
    )


def _cloud(nodes, *, asof=ASOF, neighbours=200) -> ConditionalCloud:
    # component_scale of one makes the pool's relative gate read as an absolute
    # distance, so these tests can state gates in the units the nodes use.
    assigned = sum(n.member_count for n in nodes)
    return ConditionalCloud(
        asof=asof,
        neighbour_count=max(neighbours, assigned),
        assigned_count=assigned,
        cluster_count=6,
        nodes=tuple(nodes),
        component_scale=1.0,
    )


# -- trajectory geometry ------------------------------------------------------


def test_a_curve_is_unreadable_until_the_horizon_elapses():
    path = RealizedPath(anchor_price=100.0, anchor_atr=2.0)
    assert path.curve() == ()
    for _ in range(30):
        path = path.extend(close=101.0, high=101.5, low=99.5)
    assert len(path.curve()) == 30
    assert path.curve()[-1] == pytest.approx(0.5)
    with pytest.raises(TrajectoryError, match="60"):
        path.full_curve()


def test_excursions_are_incremental_not_nested():
    """A later window reports only what it added beyond the running extreme."""

    highs = [1.0] * 30 + [3.0] * 30
    lows = [-1.0] * 60
    attributes = path_attributes(curve=_ramp(2.0), highs=highs, lows=lows)
    assert attributes.mfe_0_15 == pytest.approx(1.0)
    assert attributes.mfe_15_30 == pytest.approx(0.0)
    assert attributes.mfe_30_60 == pytest.approx(2.0)
    assert attributes.mae_0_15 == pytest.approx(-1.0)
    assert attributes.mae_15_30 == pytest.approx(0.0)
    assert attributes.mae_30_60 == pytest.approx(0.0)


def test_time_to_extreme_distinguishes_early_from_late_tops():
    lows = [-0.5] * 60
    early = path_attributes(curve=_ramp(1.0), highs=[3.0] + [1.0] * 59, lows=lows)
    late = path_attributes(curve=_ramp(1.0), highs=[1.0] * 59 + [3.0], lows=lows)
    assert early.time_to_mfe < 0.1
    assert late.time_to_mfe == pytest.approx(1.0)


def test_path_efficiency_separates_a_clean_run_from_churn():
    straight = path_attributes(
        curve=_ramp(6.0), highs=[6.0] * 60, lows=[0.0] * 60
    ).path_efficiency
    zigzag = tuple(1.0 if step % 2 == 0 else 0.0 for step in range(TRAJECTORY_CURVE_LENGTH))
    churn = path_attributes(curve=zigzag, highs=[1.0] * 60, lows=[0.0] * 60).path_efficiency
    assert straight == pytest.approx(1.0)
    assert churn < 0.1


def test_realized_volatility_is_an_attribute_only():
    """It is reported, and it appears nowhere in the identity contract."""

    quiet = path_attributes(curve=_ramp(2.0), highs=[2.0] * 60, lows=[0.0] * 60)
    assert quiet.rv_60 > 0.0
    assert "rv_60" in ATTRIBUTE_NAMES
    node = _node(2.0)
    assert len(node.curve) == TRAJECTORY_CURVE_LENGTH
    assert len(node.components) == REPRESENTATION_DIM


def test_attribute_names_match_the_contract_ordering():
    assert ATTRIBUTE_NAMES == tuple(_attrs().as_mapping())
    assert len(ATTRIBUTE_NAMES) == 15


def test_a_favorable_excursion_may_not_be_negative():
    with pytest.raises(ValueError, match="excursions"):
        _attrs(mfe_0_15=-1.0)
    with pytest.raises(ValueError, match="excursions"):
        _attrs(mae_0_15=1.0)


# -- contract -----------------------------------------------------------------


def test_a_belief_may_not_hold_more_than_three_hypotheses():
    hypotheses = tuple(
        Hypothesis(
            hypothesis_id=f"h{i}", node_id=f"n{i}", spawned_at=ASOF, asof=ASOF,
            age_bars=0, prior_log_weight=-1.0, evidence_log_weight=0.0,
            probability=0.25, expected_curve=_ramp(1.0), realized_divergence=0.0,
            association_distance=0.0, attributes=_attrs(),
        )
        for i in range(4)
    )
    with pytest.raises(ValueError, match="at most"):
        MarketBeliefState(
            asof=ASOF, hypotheses=hypotheses, residual_probability=0.0,
            uncertainty=BeliefUncertainty(0.5, 0.5, 0.5), revision_id="r",
        )


def test_probabilities_and_residual_must_sum_to_one():
    hypothesis = Hypothesis(
        hypothesis_id="h", node_id="n", spawned_at=ASOF, asof=ASOF, age_bars=0,
        prior_log_weight=-1.0, evidence_log_weight=0.0, probability=0.6,
        expected_curve=_ramp(1.0), realized_divergence=0.0,
        association_distance=0.0, attributes=_attrs(),
    )
    with pytest.raises(ValueError, match="sum to one"):
        MarketBeliefState(
            asof=ASOF, hypotheses=(hypothesis,), residual_probability=0.6,
            uncertainty=BeliefUncertainty(0.5, 0.5, 0.5), revision_id="r",
        )


def test_an_empty_belief_carries_a_residual_of_one():
    state = MarketBeliefState(
        asof=ASOF, hypotheses=(), residual_probability=1.0,
        uncertainty=BeliefUncertainty(1.0, 0.0, 1.0), revision_id="r",
    )
    assert state.leading is None
    assert state.probability_of("anything") == 0.0


def test_a_belief_may_never_claim_action_authority():
    with pytest.raises(ValueError, match="no action authority"):
        MarketBeliefState(
            asof=ASOF, hypotheses=(), residual_probability=1.0,
            uncertainty=BeliefUncertainty(1.0, 0.0, 1.0), revision_id="r",
            action_authority_ready=True,
        )


def test_a_cloud_may_surface_more_futures_than_the_pool_has_slots():
    """Extraction breadth and working-set size are different limits.

    A SPLIT is made of a node the pool has no slot for yet, so capping the cloud
    at the number of slots would make it unreachable by construction.
    """

    cloud = _cloud([_node(float(i), mass=0.2) for i in range(4)])
    assert len(cloud.nodes) == 4
    with pytest.raises(ValueError, match="at most"):
        _cloud([_node(float(i) / 2, mass=0.1) for i in range(MAX_CLOUD_NODES + 1)])


def test_the_pool_still_keeps_at_most_three_of_them():
    pool = HypothesisPool(config=PoolConfig(association_max_distance_scale=0.1))
    advance = _advance(
        pool, minute=1, close=100.0, nodes=[_node(float(i), mass=0.2) for i in range(4)]
    )
    assert len(advance.hypotheses) <= MAX_LIVE_HYPOTHESES


def test_node_masses_cannot_exceed_the_whole_neighbourhood():
    with pytest.raises(ValueError, match="exceed"):
        _cloud([_node(1.0, mass=0.6), _node(-1.0, mass=0.6)])


def test_residual_mass_is_what_no_node_covers():
    cloud = _cloud([_node(1.0, mass=0.3), _node(-1.0, mass=0.25)])
    assert cloud.covered_mass == pytest.approx(0.55)
    assert cloud.residual_mass == pytest.approx(0.45)


def test_node_identity_is_content_addressed():
    assert node_identity(_ramp(1.0)) == node_identity(_ramp(1.0))
    assert node_identity(_ramp(1.0)) != node_identity(_ramp(1.5))


def test_revision_id_is_deterministic_and_content_addressed():
    kwargs = dict(
        asof=ASOF, hypotheses=(), residual_probability=1.0,
        index_fingerprint="idx", protocol_fingerprint="proto",
    )
    assert belief_revision_id(**kwargs) == belief_revision_id(**kwargs)
    assert belief_revision_id(**{**kwargs, "index_fingerprint": "other"}) != (
        belief_revision_id(**kwargs)
    )


# -- uncertainty --------------------------------------------------------------


def test_an_empty_pool_is_maximally_uncertain_not_maximally_confident():
    """Nothing named, nothing covered, and the reading must say so."""

    empty = BeliefUncertainty(
        mode_ambiguity=0.0, representation_coverage=0.0, retrieval_confidence=0.0
    )
    assert empty.combined == pytest.approx(2.0 / 3.0)
    assert not empty.well_supported


def test_ambiguity_separates_agreeing_claims_from_opposed_ones():
    """Identical entropy, opposite meanings — which is why it is its own number."""

    agreeing = ((1.0,) * REPRESENTATION_DIM, (1.01,) * REPRESENTATION_DIM)
    opposed = ((5.0,) * REPRESENTATION_DIM, (-5.0,) * REPRESENTATION_DIM)
    assert mode_ambiguity(agreeing, (0.5, 0.5), scale=1.0) < 0.1
    assert mode_ambiguity(opposed, (0.5, 0.5), scale=1.0) > 0.9


def test_ambiguity_is_weighted_by_how_much_each_claim_is_believed():
    """A negligible outlier far away is not the same as two claims pulling apart."""

    components = ((1.0,) * REPRESENTATION_DIM, (-9.0,) * REPRESENTATION_DIM)
    contested = mode_ambiguity(components, (0.5, 0.5), scale=1.0)
    lopsided = mode_ambiguity(components, (0.98, 0.02), scale=1.0)
    assert lopsided < contested


def test_a_single_claim_cannot_disagree_with_itself():
    single = (((1.0,) * REPRESENTATION_DIM),)
    assert mode_ambiguity(single, (1.0,), scale=1.0) == 0.0
    assert mode_ambiguity((), (), scale=1.0) == 0.0


def test_retrieval_confidence_needs_both_enough_cases_and_close_ones():
    """A full complement of remote analogues is not precedent."""

    close = dict(mean_distance=0.1, target_count=200, distance_scale=1.0)
    assert retrieval_confidence(neighbour_count=200, **close) > 0.9
    # Enough cases, but all of them far away.
    assert retrieval_confidence(
        neighbour_count=200, mean_distance=20.0, target_count=200, distance_scale=1.0
    ) < 0.1
    # Close cases, but only twelve of them — the example the design calls out.
    assert retrieval_confidence(
        neighbour_count=12, mean_distance=0.1, target_count=200, distance_scale=1.0
    ) < 0.1


def test_the_three_uncertainty_components_stay_separate():
    """Each says a different thing, so none can stand in for another."""

    uncertainty = BeliefUncertainty(
        mode_ambiguity=0.3, representation_coverage=0.8, retrieval_confidence=0.6
    )
    assert uncertainty.combined == pytest.approx((0.3 + 0.2 + 0.4) / 3.0)
    assert uncertainty.well_supported
    thin = BeliefUncertainty(
        mode_ambiguity=0.0, representation_coverage=1.0, retrieval_confidence=0.1
    )
    # Sharp and fully covered, and still not to be trusted: no precedent.
    assert not thin.well_supported
    with pytest.raises(ValueError):
        BeliefUncertainty(
            mode_ambiguity=1.4, representation_coverage=0.0, retrieval_confidence=0.0
        )


# -- belief updater -----------------------------------------------------------


def test_the_matching_curve_scores_above_the_opposing_one():
    spread = tuple(0.5 for _ in range(TRAJECTORY_CURVE_LENGTH))
    path = RealizedPath(anchor_price=100.0, anchor_atr=1.0)
    for step in range(1, 21):
        path = path.extend(close=100.0 + step * 0.05, high=100.0 + step * 0.06, low=100.0)
    up = evaluate(_ramp(3.0), spread, path)
    down = evaluate(_ramp(-3.0), spread, path)
    assert up.evidence_log_weight > down.evidence_log_weight
    assert up.divergence < down.divergence


def test_evidence_is_recomputed_not_accumulated():
    curve, spread = _ramp(1.0), tuple(0.5 for _ in range(TRAJECTORY_CURVE_LENGTH))
    path = RealizedPath(anchor_price=100.0, anchor_atr=1.0)
    for _ in range(7):
        path = path.extend(close=100.5, high=100.6, low=99.9)
    assert evaluate(curve, spread, path).evidence_log_weight == (
        evaluate(curve, spread, path).evidence_log_weight
    )


def test_the_updater_scores_only_the_elapsed_points():
    curve, spread = _ramp(1.0), tuple(0.5 for _ in range(TRAJECTORY_CURVE_LENGTH))
    path = RealizedPath(anchor_price=100.0, anchor_atr=1.0)
    for _ in range(3):
        path = path.extend(close=100.0, high=100.1, low=99.9)
    assert evaluate(curve, spread, path).decided_points == 3


def test_a_wide_claim_is_harder_to_falsify_than_a_tight_one():
    curve = _ramp(1.0)
    path = RealizedPath(anchor_price=100.0, anchor_atr=1.0)
    for _ in range(10):
        path = path.extend(close=104.0, high=104.0, low=104.0)
    tight = evaluate(curve, tuple(0.3 for _ in range(60)), path).divergence
    wide = evaluate(curve, tuple(3.0 for _ in range(60)), path).divergence
    assert wide < tight


def test_normalization_keeps_the_residual_as_a_competing_term():
    probabilities, residual = normalize_log_weights((0.0, 0.0), residual_log_weight=0.0)
    assert residual == pytest.approx(1 / 3)
    assert sum(probabilities) + residual == pytest.approx(1.0)


# -- association --------------------------------------------------------------


def test_assignment_is_globally_optimal_not_greedy():
    """Greedy nearest-first would take (0,0) and strand the better total."""

    matched, _, _ = _assign(np.array([[0.9, 1.0], [1.0, 5.0]]), gate=10.0)
    assert matched == {0: 1, 1: 0}


def test_a_pair_beyond_the_gate_is_not_a_match():
    matched, live, nodes = _assign(np.array([[9.0]]), gate=1.0)
    assert matched == {}
    assert live == {0} and nodes == {0}


def _advance(pool, *, minute, close, nodes):
    asof = ASOF + pd.Timedelta(minutes=minute)
    return pool.advance(
        asof=asof, close=close, high=close + 0.5, low=close - 0.5, atr=1.0,
        cloud=_cloud(nodes, asof=asof),
    )


def test_a_matched_node_keeps_the_hypothesis_identity_and_ages_it():
    pool = HypothesisPool()
    first = _advance(pool, minute=1, close=100.0, nodes=[_node(1.0, mass=0.5)])
    identity = first.hypotheses[0].hypothesis_id
    second = _advance(pool, minute=2, close=100.1, nodes=[_node(1.0, mass=0.5)])
    assert second.hypotheses[0].hypothesis_id == identity
    assert second.hypotheses[0].age_bars == 1


def test_a_node_beyond_the_gate_spawns_rather_than_inheriting():
    pool = HypothesisPool(config=PoolConfig(association_max_distance_scale=0.5))
    first = _advance(pool, minute=1, close=100.0, nodes=[_node(1.0, mass=0.5)])
    identity = first.hypotheses[0].hypothesis_id
    second = _advance(pool, minute=2, close=100.1, nodes=[_node(9.0, mass=0.5)])
    assert second.hypotheses[0].hypothesis_id != identity
    assert {r.operation for r in second.records} >= {
        LifecycleOperation.SPAWN,
        LifecycleOperation.RETIRE,
    }


def test_a_split_needs_the_support_to_divide_not_just_a_neighbour_to_appear():
    """H1's three hundred samples separate into two groups — that is a split."""

    pool = HypothesisPool()
    parent = tuple(range(300))
    _advance(pool, minute=1, close=100.0, nodes=[_node(1.0, mass=0.5, members=parent)])
    advance = _advance(
        pool,
        minute=2,
        close=100.1,
        nodes=[
            _node(1.0, mass=0.3, members=parent[:170]),
            _node(1.2, mass=0.3, members=parent[170:]),
        ],
    )
    split = [r for r in advance.records if r.operation is LifecycleOperation.SPLIT]
    assert split, [r.operation.value for r in advance.records]
    assert split[0].support_overlap == pytest.approx(130 / 300)
    child = next(h for h in advance.hypotheses if h.hypothesis_id in split[0].hypothesis_ids)
    assert child.lineage  # the child knows which claim it came out of


def test_a_nearby_node_on_unrelated_history_spawns_rather_than_splitting():
    """Proximity is not inheritance: nothing of the live claim carried over."""

    pool = HypothesisPool()
    _advance(
        pool, minute=1, close=100.0, nodes=[_node(1.0, mass=0.5, members=tuple(range(300)))]
    )
    advance = _advance(
        pool,
        minute=2,
        close=100.1,
        nodes=[
            _node(1.0, mass=0.3, members=tuple(range(300))),
            _node(1.2, mass=0.3, members=tuple(range(9000, 9130))),
        ],
    )
    operations = {r.operation for r in advance.records}
    assert LifecycleOperation.SPAWN in operations
    assert LifecycleOperation.SPLIT not in operations


def test_a_match_whose_support_was_replaced_is_not_an_update():
    """Same coordinates, different history — a different claim in the same coat."""

    pool = HypothesisPool()
    first = _advance(
        pool, minute=1, close=100.0, nodes=[_node(1.0, mass=0.5, members=tuple(range(300)))]
    )
    identity = first.hypotheses[0].hypothesis_id
    advance = _advance(
        pool,
        minute=2,
        close=100.1,
        nodes=[_node(1.0, mass=0.5, members=tuple(range(9000, 9300)))],
    )
    assert advance.hypotheses[0].hypothesis_id != identity
    retire = [r for r in advance.records if r.operation is LifecycleOperation.RETIRE]
    assert any("replaced" in r.reason for r in retire)


def test_two_claims_whose_supports_converge_on_one_node_merge():
    """The exact dual of a split: the information gap between them closed."""

    pool = HypothesisPool()
    left, right = tuple(range(100)), tuple(range(100, 200))
    _advance(
        pool,
        minute=1,
        close=100.0,
        nodes=[
            _node(1.0, mass=0.3, members=left),
            _node(1.15, mass=0.3, members=right),
        ],
    )
    advance = _advance(
        pool, minute=2, close=100.1, nodes=[_node(1.07, mass=0.6, members=left + right)]
    )
    merged = [r for r in advance.records if r.operation is LifecycleOperation.MERGE]
    assert merged, [r.operation.value for r in advance.records]
    assert len(advance.hypotheses) == 1
    assert advance.hypotheses[0].lineage


def test_two_claims_that_still_say_different_things_do_not_merge():
    """Converged support is not enough: the claims must be indistinguishable."""

    pool = HypothesisPool(config=PoolConfig(association_max_distance_scale=3.0))
    left, right = tuple(range(100)), tuple(range(100, 200))
    _advance(
        pool,
        minute=1,
        close=100.0,
        nodes=[
            _node(-2.0, mass=0.3, members=left),
            _node(2.0, mass=0.3, members=right),
        ],
    )
    advance = _advance(
        pool, minute=2, close=100.1, nodes=[_node(2.0, mass=0.6, members=left + right)]
    )
    assert not [r for r in advance.records if r.operation is LifecycleOperation.MERGE]


def test_information_gap_reads_separation_against_the_spread_it_is_measured_in():
    """Two claims a hair apart inside a wide band are one claim written twice."""

    tight = information_gap(_node(1.0, spread=0.05), _node(1.5, spread=0.05))
    wide = information_gap(_node(1.0, spread=5.0), _node(1.5, spread=5.0))
    assert wide < tight
    assert information_gap(_node(1.0), _node(1.0)) == pytest.approx(0.0)


def test_a_node_at_the_same_centre_with_a_different_spread_is_not_the_same_claim():
    """A tight knot and a diffuse ring share a centroid and claim different things."""

    pool = HypothesisPool()
    members = tuple(range(300))
    first = _advance(
        pool,
        minute=1,
        close=100.0,
        nodes=[_node(1.0, mass=0.5, members=members, component_spread=0.1)],
    )
    identity = first.hypotheses[0].hypothesis_id
    advance = _advance(
        pool,
        minute=2,
        close=100.1,
        nodes=[_node(1.0, mass=0.5, members=members, component_spread=4.0)],
    )
    assert advance.hypotheses[0].hypothesis_id != identity


def test_a_matched_update_records_what_it_matched_on():
    pool = HypothesisPool()
    members = tuple(range(300))
    _advance(pool, minute=1, close=100.0, nodes=[_node(1.0, mass=0.5, members=members)])
    advance = _advance(
        pool, minute=2, close=100.1, nodes=[_node(1.0, mass=0.5, members=members[:200])]
    )
    update = next(
        r
        for r in advance.records
        if r.operation is LifecycleOperation.UPDATE and "matched" in r.reason
    )
    assert update.support_overlap == pytest.approx(200 / 300)
    assert advance.hypotheses[0].support_overlap == pytest.approx(200 / 300)


def test_an_empty_cloud_retires_everything_and_publishes_a_full_residual():
    pool = HypothesisPool()
    _advance(pool, minute=1, close=100.0, nodes=[_node(1.0, mass=0.5)])
    advance = _advance(pool, minute=2, close=100.1, nodes=[])
    assert advance.hypotheses == ()
    assert advance.residual_probability == 1.0
    assert any(r.operation is LifecycleOperation.RETIRE for r in advance.records)


def test_the_pool_refuses_an_out_of_order_clock():
    pool = HypothesisPool()
    _advance(pool, minute=5, close=100.0, nodes=[_node(1.0)])
    with pytest.raises(HypothesisPoolError, match="out-of-order"):
        _advance(pool, minute=4, close=100.0, nodes=[_node(1.0)])


def test_the_pool_refuses_a_cloud_from_another_clock():
    with pytest.raises(HypothesisPoolError, match="different clock"):
        HypothesisPool().advance(
            asof=ASOF, close=100.0, high=100.5, low=99.5, atr=1.0,
            cloud=_cloud([_node(1.0)], asof=ASOF + pd.Timedelta(minutes=1)),
        )


def test_a_hypothesis_is_not_falsified_by_its_first_minute():
    pool = HypothesisPool(
        config=PoolConfig(falsification_divergence=0.01, falsification_minimum_age_bars=5)
    )
    _advance(pool, minute=1, close=100.0, nodes=[_node(1.0, mass=0.6)])
    advance = _advance(pool, minute=2, close=80.0, nodes=[_node(1.0, mass=0.6)])
    assert not [
        r
        for r in advance.records
        if r.operation is LifecycleOperation.RETIRE and "falsified" in r.reason
    ]


def test_a_falsified_hypothesis_retires():
    pool = HypothesisPool(
        config=PoolConfig(falsification_divergence=0.5, falsification_minimum_age_bars=2)
    )
    _advance(pool, minute=1, close=100.0, nodes=[_node(3.0, mass=0.6)])
    operations: list[LifecycleOperation] = []
    for minute in range(2, 12):
        advance = _advance(
            pool, minute=minute, close=100.0 - minute * 3.0, nodes=[_node(3.0, mass=0.6)]
        )
        operations.extend(r.operation for r in advance.records)
    assert LifecycleOperation.RETIRE in operations


def test_no_sequence_of_association_outcomes_can_break_the_pool_invariants():
    """The bound, the probability sum and one-node-per-hypothesis, under noise."""

    import random

    rng = random.Random(31)
    pool = HypothesisPool(config=PoolConfig(association_max_distance_scale=2.0))
    price, largest = 100.0, 0
    exercised: set[str] = set()
    for minute in range(1, 900):
        price += rng.gauss(0, 0.4)
        count = rng.randint(0, MAX_LIVE_HYPOTHESES)
        # Levels are drawn densely enough that some pairs fall inside the
        # association gate and some outside, so every outcome is reachable.
        levels = rng.sample(
            [-3.0, -1.5, -0.6, -0.5, 0.5, 0.6, 1.5, 1.6, 3.0], k=count
        )
        # Support windows are drawn from one shared pool and slide with the
        # level, so adjacent claims share history and distant ones do not —
        # which is what makes every lifecycle outcome reachable at all.
        nodes = [
            _node(
                level,
                mass=0.9 / max(1, count),
                members=tuple(range(int((level + 3.0) * 20), int((level + 3.0) * 20) + 40)),
            )
            for level in levels
        ]
        advance = _advance(pool, minute=minute, close=price, nodes=nodes)
        largest = max(largest, len(advance.hypotheses), len(pool.members))
        exercised |= {r.operation.value for r in advance.records}
        total = (
            sum(h.probability for h in advance.hypotheses) + advance.residual_probability
        )
        assert total == pytest.approx(1.0, abs=1e-9)
        assert len({h.node_id for h in advance.hypotheses}) == len(advance.hypotheses)
    assert largest <= MAX_LIVE_HYPOTHESES
    assert exercised == {op.value for op in LifecycleOperation}


# -- retrieval and extraction -------------------------------------------------


def _index(rows: int = 400, seed: int = 5) -> ForecastIndex:
    """A synthetic index whose contexts split cleanly into two future regimes."""

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, FEATURE_DIM))
    features[rows // 2 :] += 6.0
    up = np.array([_ramp(3.0) for _ in range(rows // 2)])
    down = np.array([_ramp(-3.0) for _ in range(rows - rows // 2)])
    curves = np.vstack([up, down]) + rng.normal(
        scale=0.05, size=(rows, TRAJECTORY_CURVE_LENGTH)
    )
    closes = 100.0 + curves
    index, _ = build_index(
        features=features,
        anchor_prices=np.full(rows, 100.0),
        anchor_atrs=np.full(rows, 1.0),
        future_closes=closes,
        future_highs=closes + 0.1,
        future_lows=closes - 0.1,
    )
    return index


def _proposer(**overrides) -> HypothesisProposer:
    config = dict(neighbours=60, minimum_neighbours=10, max_cluster_count=3)
    config.update(overrides)
    return HypothesisProposer(index=_index(), config=ProposerConfig(**config))


def test_retrieval_surfaces_what_followed_the_nearest_contexts():
    cloud = _proposer().propose(np.zeros(FEATURE_DIM), asof=ASOF)
    assert cloud.nodes
    assert cloud.nodes[0].terminal_return > 0


def test_retrieval_follows_the_context_when_it_moves():
    cloud = _proposer().propose(np.full(FEATURE_DIM, 6.0), asof=ASOF)
    assert cloud.nodes[0].terminal_return < 0


def test_too_few_neighbours_yields_a_cloud_with_no_nodes():
    proposer = HypothesisProposer(
        index=_index(),
        config=ProposerConfig(neighbours=5, minimum_neighbours=5, max_cluster_count=3),
    )
    proposer.config = ProposerConfig(
        neighbours=5, minimum_neighbours=5, max_cluster_count=3
    )
    # Force the shortfall: ask for more assigned neighbours than exist.
    starved = HypothesisProposer(
        index=proposer.index,
        config=ProposerConfig(neighbours=30, minimum_neighbours=30, max_cluster_count=3),
    )
    starved._reference = starved._reference[:10]
    cloud = starved.propose(np.zeros(FEATURE_DIM), asof=ASOF)
    assert cloud.nodes == ()
    assert cloud.residual_mass == 1.0


def test_a_nan_context_component_does_not_poison_retrieval():
    context = np.zeros(FEATURE_DIM)
    context[3] = math.nan
    assert _proposer().propose(context, asof=ASOF).nodes


def test_extraction_is_deterministic_over_the_same_cloud():
    proposer = _proposer(neighbours=80, max_cluster_count=4)
    first = proposer.propose(np.zeros(FEATURE_DIM), asof=ASOF)
    second = proposer.propose(np.zeros(FEATURE_DIM), asof=ASOF)
    assert [n.node_id for n in first.nodes] == [n.node_id for n in second.nodes]


def test_a_node_is_a_real_observed_curve_not_an_average():
    proposer = _proposer()
    cloud = proposer.propose(np.zeros(FEATURE_DIM), asof=ASOF)
    reference = proposer.index.reference_curves
    for node in cloud.nodes:
        assert np.isclose(reference, np.asarray(node.curve)).all(axis=1).any()


def test_a_misshapen_index_is_refused():
    index = _index()
    with pytest.raises(HypothesisProposerError, match="shape"):
        ForecastIndex(
            fingerprint="f",
            feature_center=np.zeros(FEATURE_DIM),
            feature_scale=np.ones(FEATURE_DIM),
            reference_features=index.reference_features,
            reference_curves=index.reference_curves[:, :10],
            reference_scores=index.reference_scores,
            reference_attributes=index.reference_attributes,
            attribute_names=index.attribute_names,
            principal_mean=index.principal_mean,
            principal_components=index.principal_components,
            direction_centre=index.direction_centre,
            direction_spread=index.direction_spread,
            shape_centre=index.shape_centre,
            shape_spread=index.shape_spread,
            direction_weight=index.direction_weight,
            component_scale=index.component_scale,
            context_scale=index.context_scale,
        )


def test_the_feature_vector_is_fixed_width_and_uniquely_named():
    assert len(FEATURE_NAMES) == FEATURE_DIM
    assert len(set(FEATURE_NAMES)) == FEATURE_DIM


# -- principal basis and study surfaces ---------------------------------------


def test_detrending_erases_where_the_path_ended():
    """Ramps to wildly different levels all have the same shape: none at all."""

    rng = np.random.default_rng(2)
    levels = rng.normal(size=(300, 1)) * 5.0
    ramps = levels * np.linspace(1 / 60, 1.0, TRAJECTORY_CURVE_LENGTH)
    assert np.abs(shape_matrix(ramps)).max() < 1e-9
    assert max(abs(v) for v in detrended_shape(_ramp(9.0))) < 1e-9


def test_the_shape_basis_captures_a_common_bow_whatever_the_endpoint():
    """The part raw-curve PCA crushed: two paths that end together, shaped apart."""

    rng = np.random.default_rng(2)
    steps = np.arange(1, TRAJECTORY_CURVE_LENGTH + 1)
    bow = np.sin(np.pi * steps / TRAJECTORY_CURVE_LENGTH)
    levels = rng.normal(size=(300, 1)) * 5.0
    amplitude = rng.normal(size=(300, 1))
    curves = levels * (steps / TRAJECTORY_CURVE_LENGTH) + amplitude * bow
    basis = fit_principal_basis(
        shape_matrix(curves + rng.normal(scale=0.01, size=curves.shape))
    )
    assert basis.explained_variance_ratio[0] > 0.9
    assert basis.components.shape == (SHAPE_COMPONENT_COUNT, TRAJECTORY_CURVE_LENGTH)


def test_the_two_channels_answer_different_questions():
    """Same destination, opposite journeys — Direction agrees, Shape does not."""

    steps = np.arange(1, TRAJECTORY_CURVE_LENGTH + 1) / TRAJECTORY_CURVE_LENGTH
    bow = np.sin(np.pi * steps)
    dipped = tuple(2.0 * steps - 3.0 * bow)
    rallied = tuple(2.0 * steps + 3.0 * bow)
    highs = tuple(max(v, 0.0) + 0.1 for v in dipped)
    lows = tuple(min(v, 0.0) - 0.1 for v in dipped)
    left = path_attributes(curve=dipped, highs=highs, lows=lows)
    right = path_attributes(
        curve=rallied,
        highs=tuple(max(v, 0.0) + 0.1 for v in rallied),
        lows=tuple(min(v, 0.0) - 0.1 for v in rallied),
    )
    assert left.r_60 == pytest.approx(right.r_60)
    shape_distance = float(
        np.linalg.norm(np.array(detrended_shape(dipped)) - np.array(detrended_shape(rallied)))
    )
    assert shape_distance > 5.0
    assert len(direction_vector(left)) == DIRECTION_DIM
    assert DIRECTION_FEATURE_NAMES[0] == "r_5"


def test_medoids_are_real_rows_not_averages():
    rng = np.random.default_rng(3)
    scores = np.vstack([rng.normal(-5, 0.2, (50, 2)), rng.normal(5, 0.2, (50, 2))])
    labels = np.array([0] * 50 + [1] * 50)
    picked = medoids(scores, labels)
    assert picked.size == 2
    assert set(picked.tolist()) <= set(range(100))


def test_eta_squared_is_one_when_the_partition_explains_everything():
    outcome = np.array([1.0] * 20 + [5.0] * 20)
    labels = np.array([0] * 20 + [1] * 20)
    assert eta_squared(outcome, labels) == pytest.approx(1.0)


def test_centroid_reproduction_rewards_shapes_that_come_back():
    reference = np.array([_ramp(3.0), _ramp(-3.0)])
    same = np.array([_ramp(3.02), _ramp(-2.98)])
    different = np.array([_ramp(0.0), _ramp(0.1)])
    assert centroid_reproduction(reference, same, gate=0.5)["matched_fraction"] == 1.0
    assert centroid_reproduction(reference, different, gate=0.001)["matched_fraction"] == 0.0


# -- the adaptive local cut ---------------------------------------------------


def test_one_blob_is_one_mode_not_a_decorative_split():
    """A cloud with no structure must not be sliced into halves."""

    rng = np.random.default_rng(4)
    blob = rng.normal(size=(150, REPRESENTATION_DIM)) * 0.4
    labels, chosen, separation = _proposer().local_cut(blob)
    assert chosen == 1
    assert set(labels) == {0}
    assert separation == 0.0


def test_two_dense_regions_are_cut_into_two():
    rng = np.random.default_rng(4)
    left = rng.normal(size=(80, REPRESENTATION_DIM)) * 0.2
    right = left + 12.0
    labels, chosen, separation = _proposer().local_cut(np.vstack([left, right]))
    assert chosen == 2
    assert separation > 0.5
    assert len(set(labels[:80])) == 1 and len(set(labels[80:])) == 1


def test_a_cloud_that_keeps_spreading_takes_a_finer_cut():
    """Three separated groups are three, not the two a fixed k would allow."""

    rng = np.random.default_rng(4)
    groups = [
        rng.normal(size=(60, REPRESENTATION_DIM)) * 0.2 + offset
        for offset in (0.0, 15.0, 30.0)
    ]
    _, chosen, _ = _proposer().local_cut(np.vstack(groups))
    assert chosen == 3


def test_the_separation_floor_is_what_decides_between_one_and_many():
    rng = np.random.default_rng(4)
    blob = rng.normal(size=(150, REPRESENTATION_DIM)) * 0.4
    permissive = HypothesisProposer(
        index=_index(),
        config=ProposerConfig(neighbours=60, minimum_neighbours=10, separation_floor=0.0),
    )
    _, chosen, _ = permissive.local_cut(blob)
    assert chosen > 1


def test_a_cloud_carries_its_support_and_how_far_the_neighbours_were():
    cloud = _proposer().propose(np.zeros(FEATURE_DIM), asof=ASOF)
    assert cloud.mean_neighbour_distance > 0.0
    for node in cloud.nodes:
        assert len(node.member_ids) == node.member_count
        assert node.component_spread >= 0.0
    assert len({i for node in cloud.nodes for i in node.member_ids}) == sum(
        node.member_count for node in cloud.nodes
    )


# -- churn diagnostics --------------------------------------------------------


def test_jitter_is_near_perfect_on_a_cleanly_separated_cloud():
    rng = np.random.default_rng(7)
    scores = np.vstack([rng.normal(-8, 0.2, (60, 3)), rng.normal(8, 0.2, (60, 3))])
    agreement, shift = cluster_jitter(scores, cluster_count=2)
    assert agreement > 0.95
    assert shift < 0.5


def test_jitter_exposes_an_unstable_cut_of_a_single_blob():
    rng = np.random.default_rng(8)
    agreement, _ = cluster_jitter(rng.normal(0, 1, (120, 3)), cluster_count=6)
    assert agreement < 0.95


def test_cloud_drift_reads_neighbourhood_turnover():
    assert cloud_drift(np.arange(10), np.arange(10)) == 0.0
    assert cloud_drift(np.arange(10), np.arange(10, 20)) == 1.0
    assert 0.0 < cloud_drift(np.arange(10), np.arange(5, 15)) < 1.0


def test_churn_summary_flags_an_operation_that_fires_on_a_still_cloud():
    frame = pd.DataFrame(
        [
            {"operation": "split", "jitter_ari": 0.2, "cloud_drift": 0.02,
             "association_distance": 0.4},
            {"operation": "spawn", "jitter_ari": 0.98, "cloud_drift": 0.9,
             "association_distance": 3.0},
        ]
    )
    summary = summarize_churn(frame).set_index("operation")
    assert (
        summary.loc["split", "artefact_suspicion"]
        > summary.loc["spawn", "artefact_suspicion"]
    )


def test_the_association_gate_can_be_read_off_the_distances():
    profile = association_distance_profile([0.1, 0.2, 0.3, 0.4, 5.0])
    assert list(profile.percentile) == [5, 10, 25, 50, 75, 90, 95, 99]
    assert profile.association_distance.is_monotonic_increasing


# -- the four design measurements ---------------------------------------------


def test_retrieval_skill_finds_skill_where_skill_exists():
    """A context that determines the future must beat a random draw."""

    rng = np.random.default_rng(6)
    steps = np.arange(1, TRAJECTORY_CURVE_LENGTH + 1) / TRAJECTORY_CURVE_LENGTH
    reference = np.array(
        [
            (1.0 if row % 2 else -1.0) * steps * 3.0
            + rng.normal(scale=0.05, size=TRAJECTORY_CURVE_LENGTH)
            for row in range(400)
        ]
    )
    realized = np.array([steps * 3.0 for _ in range(40)])
    # Every retrieval returns only the rising half, which is what happened.
    neighbourhoods = [np.arange(1, 400, 2) for _ in range(40)]
    frame = retrieval_skill(
        neighbourhoods=neighbourhoods,
        reference_curves=reference,
        realized_curves=realized,
    )
    assert frame["conditional_rmse"].mean() < frame["random_rmse"].mean()
    verdict = paired_verdict(frame, left="conditional_rmse", right="random_rmse")
    assert verdict["significant"]


def test_retrieval_skill_reports_none_when_the_context_says_nothing():
    """The test must be able to fail, or it is not a test."""

    rng = np.random.default_rng(6)
    reference = rng.normal(size=(400, TRAJECTORY_CURVE_LENGTH))
    realized = rng.normal(size=(40, TRAJECTORY_CURVE_LENGTH))
    neighbourhoods = [rng.choice(400, size=50, replace=False) for _ in range(40)]
    frame = retrieval_skill(
        neighbourhoods=neighbourhoods,
        reference_curves=reference,
        realized_curves=realized,
    )
    assert not paired_verdict(frame, left="conditional_rmse", right="random_rmse")[
        "significant"
    ]


def test_skill_profile_separates_no_information_from_over_confidence():
    """RMSE alone cannot; correlation and the optimal scaling can."""

    rng = np.random.default_rng(8)
    steps = np.arange(1, TRAJECTORY_CURVE_LENGTH + 1) / TRAJECTORY_CURVE_LENGTH
    # The context's first component decides the sign of the future.
    features = rng.normal(size=(600, FEATURE_DIM))
    curves = np.array(
        [
            np.sign(features[row, 0]) * steps * 3.0
            + rng.normal(scale=0.3, size=TRAJECTORY_CURVE_LENGTH)
            for row in range(600)
        ]
    )
    informative = skill_profile(
        pool_features=features[:400],
        pool_curves=curves[:400],
        scored_features=features[400:],
        scored_curves=curves[400:],
        neighbours=40,
        stride=1,
    )
    assert informative["correlation"] > 0.4
    assert informative["optimal_alpha"] > 0.5
    assert informative["sign_agreement"] > 0.65

    # The same machinery on a context that decides nothing must report nothing.
    noise = rng.normal(size=(600, TRAJECTORY_CURVE_LENGTH))
    empty = skill_profile(
        pool_features=features[:400],
        pool_curves=noise[:400],
        scored_features=features[400:],
        scored_curves=noise[400:],
        neighbours=40,
        stride=1,
    )
    assert abs(empty["correlation"]) < 0.2
    assert empty["optimal_alpha"] < 0.5
    # The gap between the two is what the measurement is for.
    assert informative["correlation"] > abs(empty["correlation"]) + 0.3


def test_the_two_representations_are_told_apart_by_their_prototypes():
    """Raw PCA concentrates variance on one axis; the split design does not."""

    rng = np.random.default_rng(6)
    steps = np.arange(1, TRAJECTORY_CURVE_LENGTH + 1) / TRAJECTORY_CURVE_LENGTH
    bow = np.sin(np.pi * steps)
    curves = np.array(
        [
            rng.normal(0, 2.0) * steps
            + rng.normal(0, 1.0) * bow
            + rng.normal(scale=0.05, size=TRAJECTORY_CURVE_LENGTH)
            for _ in range(400)
        ]
    )
    attributes = [
        path_attributes(
            curve=tuple(row),
            highs=tuple(max(v, 0.0) + 0.1 for v in row),
            lows=tuple(min(v, 0.0) - 0.1 for v in row),
        )
        for row in curves
    ]
    raw = raw_representation(curves, attributes)
    split = two_channel_representation(curves, attributes)
    assert raw.leading_share > split.leading_share
    geometry = prototype_geometry(split.scores, curves, cluster_count=3)
    assert float(geometry["shape_span"].iloc[0]) > 0.0


def test_the_protocol_fingerprint_is_the_bytes_a_reader_would_see(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    first = protocol_fingerprint(path)
    path.write_text('{"a": 1} ', encoding="utf-8")
    assert protocol_fingerprint(path) != first


# -- protocol and the published surface ---------------------------------------


def test_the_shipped_protocol_is_shadow_only():
    protocol = load_hypothesis_protocol(PROTOCOL_PATH)
    assert protocol["authority"] == "shadow_only"
    assert protocol["action_authority_ready"] is False
    assert protocol["trajectory"]["curve_length_minutes"] == TRAJECTORY_CURVE_LENGTH
    assert protocol["trajectory"]["shape_components"] == SHAPE_COMPONENT_COUNT
    assert protocol["trajectory"]["direction_dim"] == DIRECTION_DIM
    assert protocol["trajectory"]["representation_dim"] == REPRESENTATION_DIM


def test_the_model_binding_cites_the_protocol_it_actually_ships():
    """A protocol edit that forgets the binding leaves the model pointing at a
    file that no longer exists in the form it claims."""

    model = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
    binding = model["hypothesis_protocol"]
    assert binding["protocol"] == "brain/configs/hypothesis_protocol.json"
    assert binding["hypothesis_protocol_fingerprint"] == protocol_fingerprint(
        PROTOCOL_PATH
    )


def test_a_protocol_claiming_authority_is_refused(tmp_path):
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["action_authority_ready"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HypothesisProposerError, match="action authority"):
        load_hypothesis_protocol(path)


def _forecaster() -> HypothesisForecaster:
    return HypothesisForecaster(proposer=_proposer(), protocol_fingerprint="proto")


def test_the_forecaster_publishes_one_belief_per_clock():
    state = _forecaster().observe(
        ForecastInput(
            asof=ASOF, close=100.0, high=100.5, low=99.5,
            context=np.zeros(FEATURE_DIM), atr=1.0,
        )
    )
    assert isinstance(state, MarketBeliefState)
    assert state.authority == "shadow_only"
    assert state.action_authority_ready is False
    assert state.cloud is not None
    assert all(h.status is HypothesisStatus.ACTIVE for h in state.hypotheses)


def test_the_forecaster_is_deterministic_across_identical_replays():
    contexts = [np.full(FEATURE_DIM, value / 10.0) for value in range(30)]

    def run() -> list[str]:
        forecaster = _forecaster()
        return [
            forecaster.observe(
                ForecastInput(
                    asof=ASOF + pd.Timedelta(minutes=minute),
                    close=100.0 + minute * 0.1,
                    high=100.6 + minute * 0.1,
                    low=99.4 + minute * 0.1,
                    context=context,
                    atr=1.0,
                )
            ).revision_id
            for minute, context in enumerate(contexts)
        ]

    assert run() == run()


def test_the_forecaster_refuses_a_clock_with_neither_atr_nor_snapshot():
    with pytest.raises(ForecastError, match="Eye snapshot"):
        _forecaster().observe(
            ForecastInput(
                asof=ASOF, close=100.0, high=100.5, low=99.5,
                context=np.zeros(FEATURE_DIM),
            )
        )


def test_the_forecaster_refuses_a_snapshot_that_reports_no_atr():
    """Every curve is in ATR units, so a missing ATR is not a defaultable zero."""

    from contract.market import Timeframe

    class _Quality:
        atr = None

    class _State:
        quality = _Quality()

    class _Snapshot:
        timeframe_states = {Timeframe.M1: _State()}

    with pytest.raises(ForecastError, match="ATR"):
        _forecaster().observe(
            ForecastInput(
                asof=ASOF, close=100.0, high=100.5, low=99.5,
                context=np.zeros(FEATURE_DIM), snapshot=_Snapshot(),
            )
        )


def test_the_association_gate_scales_with_the_principal_basis():
    """A fixed distance would mean something different in every volatility regime."""

    from brain.core.hypothesis_pool import PoolConfig as _Config

    members = tuple(range(300))
    nodes = [_node(1.0, mass=0.5, members=members)]
    wide = ConditionalCloud(
        asof=ASOF + pd.Timedelta(minutes=2), neighbour_count=300,
        assigned_count=300, cluster_count=6,
        nodes=tuple([_node(3.0, mass=0.5, members=members)]), component_scale=20.0,
    )
    pool = HypothesisPool(config=_Config(association_max_distance_scale=0.5))
    first = _advance(pool, minute=1, close=100.0, nodes=nodes)
    identity = first.hypotheses[0].hypothesis_id
    # The node moved 2 * sqrt(18) ~= 8.49 away; against a scale of 20 the gate
    # is 10.0, so it is still the same claim — and its support never changed.
    advance = pool.advance(
        asof=wide.asof, close=100.1, high=100.6, low=99.6, atr=1.0, cloud=wide
    )
    assert advance.hypotheses[0].hypothesis_id == identity


def test_a_cloud_must_carry_a_positive_component_scale():
    with pytest.raises(ValueError, match="component_scale"):
        ConditionalCloud(
            asof=ASOF, neighbour_count=10, assigned_count=0, cluster_count=0,
            nodes=(), component_scale=0.0,
        )
