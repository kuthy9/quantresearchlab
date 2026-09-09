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
)
from brain.core.hypothesis_proposer import (
    FEATURE_DIM,
    FEATURE_NAMES,
    ForecastIndex,
    HypothesisProposer,
    HypothesisProposerError,
    ProposerConfig,
    load_hypothesis_protocol,
)
from brain.core.trajectory import RealizedPath, TrajectoryError, path_attributes
from brain.research.churn_diagnostics import (
    association_distance_profile,
    cloud_drift,
    cluster_jitter,
    summarize_churn,
)
from brain.research.cluster_study import centroid_reproduction, eta_squared, medoids
from brain.research.forecast_index import (
    ATTRIBUTE_NAMES,
    build_index,
    fit_principal_basis,
)
from contract.brain.forecast import (
    MAX_CLOUD_NODES,
    MAX_LIVE_HYPOTHESES,
    PRINCIPAL_COMPONENT_COUNT,
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
    distribution_ambiguity,
    entropy_uncertainty,
    node_identity,
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


def _node(level: float, *, mass: float = 0.3, spread: float = 0.5) -> TrajectoryNode:
    curve = _ramp(level)
    return TrajectoryNode(
        node_id=node_identity(curve),
        curve=curve,
        components=tuple(float(level) for _ in range(PRINCIPAL_COMPONENT_COUNT)),
        dispersion=tuple(spread for _ in range(TRAJECTORY_CURVE_LENGTH)),
        mass=mass,
        member_count=40,
        attributes=_attrs(r_60=level),
    )


def _cloud(nodes, *, asof=ASOF, neighbours=200) -> ConditionalCloud:
    # component_scale of one makes the pool's relative gate read as an absolute
    # distance, so these tests can state gates in the units the nodes use.
    return ConditionalCloud(
        asof=asof,
        neighbour_count=neighbours,
        assigned_count=sum(n.member_count for n in nodes),
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
    assert len(node.components) == PRINCIPAL_COMPONENT_COUNT


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
    """The residual is "some future I am not naming", not one named outcome."""

    assert entropy_uncertainty((), 1.0) == pytest.approx(1.0)
    assert entropy_uncertainty((0.97,), 0.03) < 0.2
    assert entropy_uncertainty((0.25, 0.25, 0.25), 0.25) == pytest.approx(1.0)


def test_ambiguity_separates_agreeing_claims_from_opposed_ones():
    """Identical entropy, opposite meanings — which is why it is its own number."""

    agreeing = ((1.0,) * PRINCIPAL_COMPONENT_COUNT, (1.01,) * PRINCIPAL_COMPONENT_COUNT)
    opposed = ((5.0,) * PRINCIPAL_COMPONENT_COUNT, (-5.0,) * PRINCIPAL_COMPONENT_COUNT)
    assert distribution_ambiguity(agreeing, scale=1.0) < 0.1
    assert distribution_ambiguity(opposed, scale=1.0) > 0.9


def test_a_single_claim_cannot_disagree_with_itself():
    single = (((1.0,) * PRINCIPAL_COMPONENT_COUNT),)
    assert distribution_ambiguity(single, scale=1.0) == 0.0
    assert distribution_ambiguity((), scale=1.0) == 0.0


def test_the_three_uncertainty_components_stay_separate():
    uncertainty = BeliefUncertainty(entropy=0.9, distribution_ambiguity=0.1, coverage=0.5)
    assert uncertainty.combined == pytest.approx(0.5)
    with pytest.raises(ValueError):
        BeliefUncertainty(entropy=1.4, distribution_ambiguity=0.0, coverage=0.0)


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


def test_a_second_node_next_to_a_live_claim_is_a_split():
    pool = HypothesisPool(config=PoolConfig(association_max_distance_scale=3.0))
    _advance(pool, minute=1, close=100.0, nodes=[_node(1.0, mass=0.5)])
    advance = _advance(
        pool, minute=2, close=100.1, nodes=[_node(1.0, mass=0.3), _node(2.0, mass=0.3)]
    )
    assert any(r.operation is LifecycleOperation.SPLIT for r in advance.records)


def test_two_live_claims_collapsing_onto_one_node_is_a_merge():
    pool = HypothesisPool(config=PoolConfig(association_max_distance_scale=3.0))
    _advance(
        pool, minute=1, close=100.0, nodes=[_node(1.0, mass=0.3), _node(2.0, mass=0.3)]
    )
    advance = _advance(pool, minute=2, close=100.1, nodes=[_node(1.5, mass=0.6)])
    assert any(r.operation is LifecycleOperation.MERGE for r in advance.records)
    assert len(advance.hypotheses) == 1


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
        nodes = [_node(level, mass=0.9 / max(1, count)) for level in levels]
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
    config = dict(neighbours=60, minimum_neighbours=10, cluster_count=3)
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
        config=ProposerConfig(neighbours=5, minimum_neighbours=5, cluster_count=3),
    )
    proposer.config = ProposerConfig(
        neighbours=5, minimum_neighbours=5, cluster_count=3
    )
    # Force the shortfall: ask for more assigned neighbours than exist.
    starved = HypothesisProposer(
        index=proposer.index,
        config=ProposerConfig(neighbours=30, minimum_neighbours=30, cluster_count=3),
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
    proposer = _proposer(neighbours=80, cluster_count=4)
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
            component_scale=index.component_scale,
        )


def test_the_feature_vector_is_fixed_width_and_uniquely_named():
    assert len(FEATURE_NAMES) == FEATURE_DIM
    assert len(set(FEATURE_NAMES)) == FEATURE_DIM


# -- principal basis and study surfaces ---------------------------------------


def test_the_principal_basis_captures_most_of_a_low_rank_curve_set():
    rng = np.random.default_rng(2)
    levels = rng.normal(size=(300, 1))
    curves = levels * np.linspace(0, 1, TRAJECTORY_CURVE_LENGTH)
    basis = fit_principal_basis(curves + rng.normal(scale=0.01, size=curves.shape))
    assert basis.explained_variance_ratio[0] > 0.95
    assert basis.components.shape == (PRINCIPAL_COMPONENT_COUNT, TRAJECTORY_CURVE_LENGTH)


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


# -- protocol and the published surface ---------------------------------------


def test_the_shipped_protocol_is_shadow_only():
    protocol = load_hypothesis_protocol(PROTOCOL_PATH)
    assert protocol["authority"] == "shadow_only"
    assert protocol["action_authority_ready"] is False
    assert protocol["trajectory"]["curve_length_minutes"] == TRAJECTORY_CURVE_LENGTH
    assert protocol["trajectory"]["principal_components"] == PRINCIPAL_COMPONENT_COUNT


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

    nodes = [_node(1.0, mass=0.5)]
    wide = ConditionalCloud(
        asof=ASOF + pd.Timedelta(minutes=2), neighbour_count=200,
        assigned_count=40, cluster_count=6,
        nodes=tuple([_node(3.0, mass=0.5)]), component_scale=10.0,
    )
    pool = HypothesisPool(config=_Config(association_max_distance_scale=0.5))
    first = _advance(pool, minute=1, close=100.0, nodes=nodes)
    identity = first.hypotheses[0].hypothesis_id
    # The node moved 2 * sqrt(5) ~= 4.47 away; against a scale of 10 the gate is
    # 5.0, so it is still the same claim.
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
