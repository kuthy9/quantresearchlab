"""The naturally discovered hypothesis Brain: contract, updater, pool, forecast."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brain.core.belief_updater import (
    BeliefUpdaterConfig,
    RealizedPath,
    evaluate,
    expected_return_at,
    normalize_log_weights,
)
from brain.core.forecast import ForecastError, ForecastInput, HypothesisForecaster
from brain.core.hypothesis_pool import HypothesisPool, HypothesisPoolError, PoolConfig
from brain.core.hypothesis_proposer import (
    FEATURE_DIM,
    FEATURE_NAMES,
    HypothesisProposer,
    ProposerConfig,
    load_hypothesis_protocol,
)
from contract.brain.forecast import (
    MAX_LIVE_HYPOTHESES,
    TRAJECTORY_COMPONENTS,
    TRAJECTORY_DIM,
    Hypothesis,
    HypothesisProposal,
    HypothesisStatus,
    LifecycleOperation,
    MarketBeliefState,
    ModeLibrary,
    TrajectoryMode,
    belief_revision_id,
    normalized_entropy,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "brain" / "configs" / "hypothesis_protocol.json"
ASOF = pd.Timestamp("2022-01-03 10:00:00-05:00")


def _mode(mode_id: str, level: float, *, spread: float = 1.0, **kwargs) -> TrajectoryMode:
    return TrajectoryMode(
        mode_id=mode_id,
        medoid=tuple(level for _ in range(TRAJECTORY_DIM)),
        dispersion=tuple(spread for _ in range(TRAJECTORY_DIM)),
        support=50,
        **kwargs,
    )


def _library(modes: tuple[TrajectoryMode, ...]) -> ModeLibrary:
    return ModeLibrary(
        library_id="test_library",
        fingerprint="f" * 64,
        fitted_at=ASOF,
        algorithm="test",
        modes=modes,
        feature_names=TRAJECTORY_COMPONENTS,
        observation_count=1000,
        noise_count=100,
    )


def _flat_library() -> ModeLibrary:
    return _library((_mode("up", 1.0), _mode("down", -1.0), _mode("flat", 0.0)))


def _hierarchical_library() -> ModeLibrary:
    return _library(
        (
            _mode("child_a", 1.0, parent_mode_id="parent"),
            _mode("child_b", 1.05, parent_mode_id="parent"),
            _mode("parent", 1.02, child_mode_ids=("child_a", "child_b")),
            _mode("other", -2.0),
        )
    )


# -- contract ----------------------------------------------------------------


def test_trajectory_contract_is_fourteen_named_components():
    assert TRAJECTORY_DIM == 14
    assert len(set(TRAJECTORY_COMPONENTS)) == TRAJECTORY_DIM


def test_a_belief_may_not_hold_more_than_three_hypotheses():
    hypotheses = tuple(
        Hypothesis(
            hypothesis_id=f"h{i}",
            mode_id=f"m{i}",
            spawned_at=ASOF,
            asof=ASOF,
            age_bars=0,
            prior_log_weight=-1.0,
            evidence_log_weight=0.0,
            probability=0.25,
            expected_trajectory=tuple(0.0 for _ in range(TRAJECTORY_DIM)),
            realized_divergence=0.0,
        )
        for i in range(4)
    )
    with pytest.raises(ValueError, match="at most"):
        MarketBeliefState(
            asof=ASOF,
            hypotheses=hypotheses,
            residual_probability=0.0,
            uncertainty=0.5,
            revision_id="r",
        )


def test_probabilities_and_residual_must_sum_to_one():
    hypothesis = Hypothesis(
        hypothesis_id="h",
        mode_id="m",
        spawned_at=ASOF,
        asof=ASOF,
        age_bars=0,
        prior_log_weight=-1.0,
        evidence_log_weight=0.0,
        probability=0.6,
        expected_trajectory=tuple(0.0 for _ in range(TRAJECTORY_DIM)),
        realized_divergence=0.0,
    )
    with pytest.raises(ValueError, match="sum to one"):
        MarketBeliefState(
            asof=ASOF,
            hypotheses=(hypothesis,),
            residual_probability=0.6,
            uncertainty=0.5,
            revision_id="r",
        )


def test_an_empty_belief_carries_a_residual_of_one():
    state = MarketBeliefState(
        asof=ASOF, hypotheses=(), residual_probability=1.0, uncertainty=0.0, revision_id="r"
    )
    assert state.leading is None
    assert state.probability_of("anything") == 0.0


def test_a_belief_may_never_claim_action_authority():
    with pytest.raises(ValueError, match="no action authority"):
        MarketBeliefState(
            asof=ASOF,
            hypotheses=(),
            residual_probability=1.0,
            uncertainty=0.0,
            revision_id="r",
            action_authority_ready=True,
        )


def test_mode_dispersion_must_be_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        TrajectoryMode(
            mode_id="m",
            medoid=tuple(0.0 for _ in range(TRAJECTORY_DIM)),
            dispersion=tuple(0.0 for _ in range(TRAJECTORY_DIM)),
            support=10,
        )


def test_a_library_rejects_a_mode_citing_an_unknown_parent():
    with pytest.raises(ValueError, match="unknown parent"):
        _library((_mode("a", 1.0, parent_mode_id="ghost"),))


def test_normalized_entropy_spans_certainty_to_a_flat_spread():
    assert normalized_entropy((1.0,)) == 0.0
    assert normalized_entropy((0.25, 0.25, 0.25, 0.25)) == pytest.approx(1.0)


def test_revision_id_is_deterministic_and_content_addressed():
    first = belief_revision_id(
        asof=ASOF,
        hypotheses=(),
        residual_probability=1.0,
        mode_library_fingerprint="lib",
        protocol_fingerprint="proto",
    )
    same = belief_revision_id(
        asof=ASOF,
        hypotheses=(),
        residual_probability=1.0,
        mode_library_fingerprint="lib",
        protocol_fingerprint="proto",
    )
    different = belief_revision_id(
        asof=ASOF,
        hypotheses=(),
        residual_probability=1.0,
        mode_library_fingerprint="other",
        protocol_fingerprint="proto",
    )
    assert first == same
    assert first != different


# -- belief updater ----------------------------------------------------------


def test_a_component_is_unreadable_until_its_horizon_elapses():
    path = RealizedPath(anchor_price=100.0, anchor_atr=2.0)
    assert path.realized("r_5") is None
    for _ in range(5):
        path = path.extend(close=101.0, high=101.5, low=99.5)
    assert path.realized("r_5") == pytest.approx(0.5)
    assert path.realized("r_10") is None


def test_excursions_read_the_extreme_not_the_close():
    path = RealizedPath(anchor_price=100.0, anchor_atr=1.0)
    for high in (101.0, 104.0, 100.5):
        path = path.extend(close=100.0, high=high, low=98.0)
    for _ in range(12):
        path = path.extend(close=100.0, high=100.1, low=99.9)
    assert path.realized("mfe_15") == pytest.approx(4.0)
    assert path.realized("mae_15") == pytest.approx(-2.0)


def test_the_matching_mode_scores_above_the_opposing_one():
    up, down = _mode("up", 1.0, spread=0.5), _mode("down", -1.0, spread=0.5)
    path = RealizedPath(anchor_price=100.0, anchor_atr=1.0)
    for step in range(1, 21):
        path = path.extend(close=100.0 + step * 0.05, high=100.0 + step * 0.06, low=100.0)
    assert evaluate(up, path).evidence_log_weight > evaluate(down, path).evidence_log_weight
    assert evaluate(up, path).divergence < evaluate(down, path).divergence


def test_evidence_is_recomputed_not_accumulated():
    """Scoring the same path twice returns the same weight, never a doubled one."""

    mode = _mode("m", 0.5)
    path = RealizedPath(anchor_price=100.0, anchor_atr=1.0)
    for _ in range(7):
        path = path.extend(close=100.5, high=100.6, low=99.9)
    first, second = evaluate(mode, path), evaluate(mode, path)
    assert first.evidence_log_weight == second.evidence_log_weight


def test_expected_return_interpolates_between_horizon_knots():
    mode = TrajectoryMode(
        mode_id="m",
        medoid=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0) + tuple(0.0 for _ in range(8)),
        dispersion=tuple(1.0 for _ in range(TRAJECTORY_DIM)),
        support=10,
    )
    assert expected_return_at(mode, 0) == 0.0
    assert expected_return_at(mode, 5) == pytest.approx(1.0)
    assert expected_return_at(mode, 60) == pytest.approx(5.0)
    assert 1.0 < expected_return_at(mode, 7) < 2.0


def test_normalization_keeps_the_residual_as_a_competing_term():
    probabilities, residual = normalize_log_weights((0.0, 0.0), residual_log_weight=0.0)
    assert residual == pytest.approx(1 / 3)
    assert sum(probabilities) + residual == pytest.approx(1.0)


# -- pool lifecycle ----------------------------------------------------------


def _advance(pool: HypothesisPool, *, minute: int, close: float, proposals=()):
    return pool.advance(
        asof=ASOF + pd.Timedelta(minutes=minute),
        close=close,
        high=close + 0.5,
        low=close - 0.5,
        atr=1.0,
        proposals=proposals,
    )


def test_the_pool_spawns_from_proposals_and_stays_bounded():
    pool = HypothesisPool(library=_flat_library())
    proposals = (
        HypothesisProposal(mode_id="up", prior=0.4, neighbour_count=80, neighbour_distance=1.0),
        HypothesisProposal(mode_id="down", prior=0.3, neighbour_count=60, neighbour_distance=1.2),
        HypothesisProposal(mode_id="flat", prior=0.2, neighbour_count=40, neighbour_distance=1.5),
    )
    advance = _advance(pool, minute=1, close=100.0, proposals=proposals)
    assert len(advance.hypotheses) <= MAX_LIVE_HYPOTHESES
    assert {r.operation for r in advance.records} >= {LifecycleOperation.SPAWN}
    assert sum(h.probability for h in advance.hypotheses) + advance.residual_probability == (
        pytest.approx(1.0)
    )


def test_a_weak_proposal_does_not_spawn():
    pool = HypothesisPool(library=_flat_library(), config=PoolConfig(spawn_minimum_prior=0.5))
    advance = _advance(
        pool,
        minute=1,
        close=100.0,
        proposals=(
            HypothesisProposal(mode_id="up", prior=0.2, neighbour_count=10, neighbour_distance=1.0),
        ),
    )
    assert advance.hypotheses == ()
    assert advance.residual_probability == 1.0


def test_the_pool_refuses_an_out_of_order_clock():
    pool = HypothesisPool(library=_flat_library())
    _advance(pool, minute=5, close=100.0)
    with pytest.raises(HypothesisPoolError, match="out-of-order"):
        _advance(pool, minute=4, close=100.0)


def test_a_hypothesis_retires_when_its_horizon_elapses():
    pool = HypothesisPool(
        library=_flat_library(), config=PoolConfig(retire_maximum_age_bars=3)
    )
    proposals = (
        HypothesisProposal(mode_id="flat", prior=0.9, neighbour_count=90, neighbour_distance=0.1),
    )
    _advance(pool, minute=1, close=100.0, proposals=proposals)
    operations: list[LifecycleOperation] = []
    for minute in range(2, 8):
        advance = _advance(pool, minute=minute, close=100.0, proposals=())
        operations.extend(r.operation for r in advance.records)
    assert LifecycleOperation.RETIRE in operations


def test_a_falsified_hypothesis_retires():
    pool = HypothesisPool(
        library=_flat_library(),
        config=PoolConfig(falsification_divergence=0.5, retire_maximum_age_bars=60),
    )
    _advance(
        pool,
        minute=1,
        close=100.0,
        proposals=(
            HypothesisProposal(mode_id="up", prior=0.9, neighbour_count=90, neighbour_distance=0.1),
        ),
    )
    operations: list[LifecycleOperation] = []
    # The mode claims a rise of one ATR; the tape collapses instead.
    for minute in range(2, 10):
        advance = _advance(pool, minute=minute, close=100.0 - minute * 2.0)
        operations.extend(r.operation for r in advance.records)
    assert LifecycleOperation.RETIRE in operations
    assert pool.members == ()


def test_converged_siblings_merge_into_their_parent():
    pool = HypothesisPool(
        library=_hierarchical_library(), config=PoolConfig(merge_maximum_distance=1.0)
    )
    proposals = (
        HypothesisProposal(mode_id="child_a", prior=0.4, neighbour_count=40, neighbour_distance=1.0),
        HypothesisProposal(mode_id="child_b", prior=0.4, neighbour_count=40, neighbour_distance=1.0),
    )
    _advance(pool, minute=1, close=100.0, proposals=proposals)
    advance = _advance(pool, minute=2, close=100.5)
    merged = [r for r in advance.records if r.operation is LifecycleOperation.MERGE]
    assert merged, "two siblings claiming near-identical futures must merge"
    assert {m.mode_id for m in pool.members} == {"parent"}


def test_an_undecided_parent_splits_into_its_two_children():
    pool = HypothesisPool(
        library=_hierarchical_library(),
        config=PoolConfig(
            split_minimum_age_bars=2, split_maximum_imbalance=1.0, merge_maximum_distance=0.0
        ),
    )
    _advance(
        pool,
        minute=1,
        close=100.0,
        proposals=(
            HypothesisProposal(
                mode_id="parent", prior=0.9, neighbour_count=90, neighbour_distance=0.1
            ),
        ),
    )
    operations: list[LifecycleOperation] = []
    for minute in range(2, 8):
        advance = _advance(pool, minute=minute, close=100.0 + minute * 0.1)
        operations.extend(r.operation for r in advance.records)
    assert LifecycleOperation.SPLIT in operations


def test_the_residual_is_never_argued_away():
    pool = HypothesisPool(library=_flat_library(), config=PoolConfig(residual_floor=0.1))
    advance = _advance(
        pool,
        minute=1,
        close=100.0,
        proposals=(
            HypothesisProposal(
                mode_id="up", prior=0.999, neighbour_count=999, neighbour_distance=0.0
            ),
        ),
    )
    assert advance.residual_probability > 0.0


# -- proposer ----------------------------------------------------------------


def _proposer(library: ModeLibrary, *, config: ProposerConfig | None = None):
    rng = np.random.default_rng(7)
    rows = 400
    features = rng.normal(size=(rows, FEATURE_DIM))
    # The first half of the reference set lives near the origin and realized
    # "up"; the second half is displaced and realized "down".
    features[rows // 2 :] += 6.0
    modes = ["up"] * (rows // 2) + ["down"] * (rows - rows // 2)
    return HypothesisProposer(
        library=library,
        reference_features=features,
        reference_modes=modes,
        center=np.zeros(FEATURE_DIM),
        scale=np.ones(FEATURE_DIM),
        config=config or ProposerConfig(neighbours=50, minimum_neighbours=10),
    )


def test_the_proposer_returns_what_followed_the_nearest_contexts():
    proposer = _proposer(_flat_library())
    proposals = proposer.propose(np.zeros(FEATURE_DIM))
    assert proposals
    assert proposals[0].mode_id == "up"
    assert sum(p.prior for p in proposals) <= 1.0 + 1e-12


def test_the_proposer_follows_the_context_when_it_moves():
    proposer = _proposer(_flat_library())
    proposals = proposer.propose(np.full(FEATURE_DIM, 6.0))
    assert proposals[0].mode_id == "down"


def test_the_proposer_says_nothing_when_too_few_neighbours_are_assigned():
    proposer = _proposer(
        _flat_library(), config=ProposerConfig(neighbours=20, minimum_neighbours=20)
    )
    proposer._modes = tuple([None] * len(proposer._modes))
    assert proposer.propose(np.zeros(FEATURE_DIM)) == ()


def test_a_nan_context_component_does_not_poison_retrieval():
    proposer = _proposer(_flat_library())
    context = np.zeros(FEATURE_DIM)
    context[3] = math.nan
    assert proposer.propose(context)


def test_the_feature_vector_is_fixed_width_and_uniquely_named():
    assert len(FEATURE_NAMES) == FEATURE_DIM
    assert len(set(FEATURE_NAMES)) == FEATURE_DIM


# -- protocol and forecast ---------------------------------------------------


def test_the_shipped_protocol_is_shadow_only():
    protocol = load_hypothesis_protocol(PROTOCOL_PATH)
    assert protocol["authority"] == "shadow_only"
    assert protocol["action_authority_ready"] is False
    assert protocol["trajectory"]["components"] == list(TRAJECTORY_COMPONENTS)


def test_a_protocol_claiming_authority_is_refused(tmp_path):
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["action_authority_ready"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="action authority"):
        load_hypothesis_protocol(path)


def _forecaster(library: ModeLibrary) -> HypothesisForecaster:
    return HypothesisForecaster(
        proposer=_proposer(library),
        library=library,
        protocol_fingerprint="proto",
        pool_config=PoolConfig(),
    )


def test_the_forecaster_publishes_one_belief_per_clock():
    library = _flat_library()
    forecaster = _forecaster(library)
    state = forecaster.observe(
        ForecastInput(
            asof=ASOF,
            close=100.0,
            high=100.5,
            low=99.5,
            context=np.zeros(FEATURE_DIM),
            atr=1.0,
        )
    )
    assert isinstance(state, MarketBeliefState)
    assert state.authority == "shadow_only"
    assert state.action_authority_ready is False
    assert state.mode_library_fingerprint == library.fingerprint
    assert all(h.status is HypothesisStatus.ACTIVE for h in state.hypotheses)


def test_the_forecaster_is_deterministic_across_identical_replays():
    library = _flat_library()
    contexts = [np.full(FEATURE_DIM, value / 10.0) for value in range(30)]

    def run() -> list[str]:
        forecaster = _forecaster(library)
        ids = []
        for minute, context in enumerate(contexts):
            state = forecaster.observe(
                ForecastInput(
                    asof=ASOF + pd.Timedelta(minutes=minute),
                    close=100.0 + minute * 0.1,
                    high=100.6 + minute * 0.1,
                    low=99.4 + minute * 0.1,
                    context=context,
                    atr=1.0,
                )
            )
            ids.append(state.revision_id)
        return ids

    assert run() == run()


def test_the_forecaster_refuses_a_clock_with_neither_atr_nor_snapshot():
    forecaster = _forecaster(_flat_library())
    with pytest.raises(ForecastError, match="Eye snapshot"):
        forecaster.observe(
            ForecastInput(
                asof=ASOF, close=100.0, high=100.5, low=99.5, context=np.zeros(FEATURE_DIM)
            )
        )


def test_the_forecaster_refuses_a_snapshot_that_reports_no_atr():
    """The library is fitted in ATR units, so a missing ATR is not a defaultable
    zero — it means this clock cannot be scored at all."""

    from contract.market import Timeframe

    class _Quality:
        atr = None

    class _State:
        quality = _Quality()

    class _Snapshot:
        timeframe_states = {Timeframe.M1: _State()}

    forecaster = _forecaster(_flat_library())
    with pytest.raises(ForecastError, match="ATR"):
        forecaster.observe(
            ForecastInput(
                asof=ASOF,
                close=100.0,
                high=100.5,
                low=99.5,
                context=np.zeros(FEATURE_DIM),
                snapshot=_Snapshot(),
            )
        )


def test_the_forecaster_refuses_a_library_its_proposer_does_not_share():
    with pytest.raises(ForecastError, match="share one mode library"):
        HypothesisForecaster(
            proposer=_proposer(_flat_library()),
            library=_flat_library(),
            protocol_fingerprint="proto",
        )


# -- hierarchy: what makes SPLIT and MERGE reachable --------------------------


def _split_context_proposer(library: ModeLibrary) -> HypothesisProposer:
    """A reference set whose neighbours divide evenly between two sibling leaves."""

    rng = np.random.default_rng(3)
    rows = 400
    features = rng.normal(scale=0.5, size=(rows, FEATURE_DIM))
    modes = ["child_a" if index % 2 == 0 else "child_b" for index in range(rows)]
    return HypothesisProposer(
        library=library,
        reference_features=features,
        reference_modes=modes,
        center=np.zeros(FEATURE_DIM),
        scale=np.ones(FEATURE_DIM),
        config=ProposerConfig(neighbours=60, minimum_neighbours=10, minimum_prior=0.6),
    )


def test_an_ambiguous_neighbourhood_proposes_the_shared_ancestor():
    """Neither leaf clears the threshold alone, so the honest claim is coarser."""

    proposer = _split_context_proposer(_hierarchical_library())
    proposals = proposer.propose(np.zeros(FEATURE_DIM))
    assert [p.mode_id for p in proposals] == ["parent"]
    assert proposals[0].prior == pytest.approx(1.0)


def test_a_clear_neighbourhood_still_proposes_the_leaf():
    library = _hierarchical_library()
    rng = np.random.default_rng(4)
    features = rng.normal(scale=0.5, size=(200, FEATURE_DIM))
    proposer = HypothesisProposer(
        library=library,
        reference_features=features,
        reference_modes=["child_a"] * 200,
        center=np.zeros(FEATURE_DIM),
        scale=np.ones(FEATURE_DIM),
        config=ProposerConfig(neighbours=50, minimum_neighbours=10, minimum_prior=0.6),
    )
    assert [p.mode_id for p in proposer.propose(np.zeros(FEATURE_DIM))] == ["child_a"]


def test_frontier_nodes_never_double_count_the_same_neighbours():
    proposer = _split_context_proposer(_hierarchical_library())
    proposals = proposer.propose(np.zeros(FEATURE_DIM))
    assert sum(p.prior for p in proposals) <= 1.0 + 1e-12


def test_merge_uses_the_lowest_common_ancestor_not_only_direct_siblings():
    """Two leaves under different parents still converge onto a shared ancestor."""

    library = _library(
        (
            _mode("leaf_a", 1.00, parent_mode_id="branch_left"),
            _mode("leaf_b", 1.01, parent_mode_id="branch_left"),
            _mode("leaf_c", 1.02, parent_mode_id="branch_right"),
            _mode("branch_left", 1.0, parent_mode_id="root", child_mode_ids=("leaf_a", "leaf_b")),
            _mode("branch_right", 1.02, parent_mode_id="root", child_mode_ids=("leaf_c",)),
            _mode("root", 1.01, child_mode_ids=("branch_left", "branch_right")),
        )
    )
    pool = HypothesisPool(library=library, config=PoolConfig(merge_maximum_distance=1.0))
    _advance(
        pool,
        minute=1,
        close=100.0,
        proposals=(
            HypothesisProposal(mode_id="leaf_a", prior=0.4, neighbour_count=40, neighbour_distance=1.0),
            HypothesisProposal(mode_id="leaf_c", prior=0.4, neighbour_count=40, neighbour_distance=1.0),
        ),
    )
    advance = _advance(pool, minute=2, close=100.4)
    assert any(r.operation is LifecycleOperation.MERGE for r in advance.records)
    assert {m.mode_id for m in pool.members} == {"root"}


def test_a_hypothesis_is_not_falsified_by_its_first_minute():
    """One minute of tape cannot refute a claim about the next hour."""

    pool = HypothesisPool(
        library=_flat_library(),
        config=PoolConfig(falsification_divergence=0.01, falsification_minimum_age_bars=5),
    )
    _advance(
        pool,
        minute=1,
        close=100.0,
        proposals=(
            HypothesisProposal(mode_id="up", prior=0.9, neighbour_count=90, neighbour_distance=0.1),
        ),
    )
    advance = _advance(pool, minute=2, close=80.0)
    falsified = [
        r
        for r in advance.records
        if r.operation is LifecycleOperation.RETIRE and "falsified" in r.reason
    ]
    assert not falsified, "a one-bar-old hypothesis must survive its first divergence"


def _binary_library() -> ModeLibrary:
    """Four leaves under two parents under a root — SPLIT and MERGE have room."""

    return _library(
        (
            _mode("leaf_aa", 0.5, spread=0.8, parent_mode_id="pair_a"),
            _mode("leaf_ab", 0.7, spread=0.8, parent_mode_id="pair_a"),
            _mode("leaf_ba", -0.5, spread=0.8, parent_mode_id="pair_b"),
            _mode("leaf_bb", -0.7, spread=0.8, parent_mode_id="pair_b"),
            _mode("pair_a", 0.6, spread=0.8, parent_mode_id="root",
                  child_mode_ids=("leaf_aa", "leaf_ab")),
            _mode("pair_b", -0.6, spread=0.8, parent_mode_id="root",
                  child_mode_ids=("leaf_ba", "leaf_bb")),
            _mode("root", 0.0, spread=0.8, child_mode_ids=("pair_a", "pair_b")),
        )
    )


def test_no_sequence_of_lifecycle_operations_can_break_the_pool_invariants():
    """Randomized proposals over a random walk, checked every clock.

    The bound, the probability sum and one-mode-per-hypothesis are the three
    things a downstream consumer relies on unconditionally, so they are checked
    against arbitrary lifecycle sequences rather than hand-picked ones.
    """

    import random

    library = _binary_library()
    rng = random.Random(99)
    pool = HypothesisPool(
        library=library,
        config=PoolConfig(
            split_minimum_age_bars=3,
            split_maximum_imbalance=0.9,
            merge_maximum_distance=1.5,
            falsification_minimum_age_bars=4,
        ),
    )
    mode_ids = [mode.mode_id for mode in library.modes]
    price = 100.0
    largest = 0
    exercised: set[str] = set()
    for minute in range(1, 1200):
        price += rng.gauss(0, 0.4)
        proposals = tuple(
            HypothesisProposal(
                mode_id=mode_id,
                prior=rng.uniform(0.1, 0.5),
                neighbour_count=20,
                neighbour_distance=1.0,
            )
            for mode_id in rng.sample(mode_ids, k=rng.randint(0, 4))
        )
        advance = pool.advance(
            asof=ASOF + pd.Timedelta(minutes=minute),
            close=price,
            high=price + 0.5,
            low=price - 0.5,
            atr=1.0,
            proposals=proposals,
        )
        largest = max(largest, len(advance.hypotheses), len(pool.members))
        exercised |= {record.operation.value for record in advance.records}
        total = sum(h.probability for h in advance.hypotheses) + advance.residual_probability
        assert total == pytest.approx(1.0, abs=1e-9)
        assert len({h.mode_id for h in advance.hypotheses}) == len(advance.hypotheses)
    assert largest <= MAX_LIVE_HYPOTHESES
    assert exercised == {op.value for op in LifecycleOperation}


# -- vocabulary pins ----------------------------------------------------------
#
# The context vector one-hot encodes three of the Eye's vocabularies. Guessing
# their members silently degrades every affected feature to a constant, which no
# other test would catch, so each is pinned against its source of truth here.


def test_direction_encoding_matches_the_eye_vocabulary():
    from contract.market import Direction

    from brain.core.hypothesis_proposer import _direction

    assert {member.name for member in Direction} == {"LONG", "SHORT"}
    assert _direction(Direction.LONG) == 1.0
    assert _direction(Direction.SHORT) == -1.0
    assert _direction(None) == 0.0


def test_delivery_phase_encoding_covers_every_registered_phase():
    from eyes.core.market_state import DeliveryPhase

    from brain.core.hypothesis_proposer import DELIVERY_PHASES

    registered = {member.value for member in DeliveryPhase}
    assert registered <= set(DELIVERY_PHASES)
    assert DELIVERY_PHASES[-1] == "other"
    assert set(DELIVERY_PHASES) - registered == {"other"}


def test_session_phase_encoding_covers_every_registered_phase():
    import pandas as pd_

    from eyes.core.market_state import session_name_phase

    from brain.core.hypothesis_proposer import SESSION_PHASES

    clocks = pd_.date_range(
        "2022-01-03 00:00", periods=24 * 60, freq="1min", tz="America/New_York"
    )
    emitted = {session_name_phase(clock)[1] for clock in clocks}
    assert emitted <= set(SESSION_PHASES)
    assert SESSION_PHASES[-1] == "other"
    assert set(SESSION_PHASES) - emitted == {"other"}
