"""The Eye-to-Brain link, end to end and on the real wiring.

Every other Brain test hands the forecaster a stub snapshot or a precomputed
context vector. This file drives the registered Eye from ``configs/model.json``
the way the Brain's own data path builds it, publishes real ``MarketSnapshot``
objects, and checks that the live path (``ForecastInput.snapshot``) and the
dataset path (``observation_features`` on the same snapshot) publish the same
belief. The retrieval index is synthetic: no fitted index ships with the
repository, and this file is about the wiring, not the forecast's skill.
"""
from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from brain.core.belief_updater import BeliefUpdaterConfig
from brain.core.forecast import ForecastInput, HypothesisForecaster
from brain.core.hypothesis_pool import PoolConfig
from brain.core.hypothesis_proposer import (
    FEATURE_DIM,
    FEATURE_NAMES,
    HypothesisProposer,
    ProposerConfig,
    load_hypothesis_protocol,
    observation_features,
    protocol_fingerprint,
)
from brain.research.forecast_index import ForecastIndex, build_index
from brain.research.trajectory_dataset import CONTEXT_LOOKBACK_MINUTES, build_eye
from contract.brain.forecast import TRAJECTORY_CURVE_LENGTH, MarketBeliefState
from contract.market import Timeframe
from eyes.core.market_state import MarketSnapshot
from shares.tests.helpers import session_bars


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "configs" / "model.json"
PROTOCOL_PATH = ROOT / "brain" / "configs" / "hypothesis_protocol.json"
# How many of the warm clocks at the end of the session each test replays.
REPLAY_CLOCKS = 40
# The context components ``observation_features`` documents as NaN until the
# Eye has formed their referent (see ``_ratio`` and the range/displacement
# reads in brain/core/hypothesis_proposer.py).
_OPTIONAL_REFERENT = re.compile(
    r"^(?:(?:4H|1H|15m|5m|1m)_(?:dist_\w+_atr|range_location|range_width_atr|displacement_score)"
    r"|rel_child_location_mean|rel_parent_invalidation_min_atr"
    r"|session_dist_prior_day_(?:high|low)_atr)$"
)


@pytest.fixture(scope="module")
def eye_clocks() -> list[tuple[object, MarketSnapshot, list[float]]]:
    """Every clock of one synthetic session at which the Eye published a
    snapshot with a warm one-minute ATR, with the close history the Brain's
    dataset path would have handed over beside it."""

    reader, observer = build_eye(MODEL_PATH, root=ROOT)
    history: list[float] = []
    clocks = []
    for bar in session_bars(1):
        observation = observer.observe(reader.on_bar(bar))
        history.append(float(bar.close))
        snapshot = observation.market_snapshot
        if snapshot is None or len(history) <= CONTEXT_LOOKBACK_MINUTES:
            continue
        m1 = snapshot.timeframe_states.get(Timeframe.M1)
        atr = getattr(getattr(m1, "quality", None), "atr", None)
        if atr is None or float(atr) <= 0.0:
            continue
        clocks.append((bar, snapshot, history[-(CONTEXT_LOOKBACK_MINUTES + 1) :]))
    assert clocks, "the Eye never published a warm snapshot over one session"
    return clocks


def _synthetic_index(rows: int = 400, seed: int = 11) -> ForecastIndex:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, FEATURE_DIM))
    features[rows // 2 :] += 6.0
    ramp = np.arange(1, TRAJECTORY_CURVE_LENGTH + 1) / TRAJECTORY_CURVE_LENGTH
    curves = np.vstack(
        [np.tile(3.0 * ramp, (rows // 2, 1)), np.tile(-3.0 * ramp, (rows - rows // 2, 1))]
    ) + rng.normal(scale=0.05, size=(rows, TRAJECTORY_CURVE_LENGTH))
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


def _forecaster() -> HypothesisForecaster:
    """Built exactly as ``brain/scripts/replay_hypothesis_belief.py`` builds it
    from the shipped protocol, with the retrieval index swapped for a synthetic one."""

    protocol = load_hypothesis_protocol(PROTOCOL_PATH)
    return HypothesisForecaster(
        proposer=HypothesisProposer(
            index=_synthetic_index(), config=ProposerConfig.from_protocol(protocol)
        ),
        protocol_fingerprint=protocol_fingerprint(PROTOCOL_PATH),
        pool_config=PoolConfig.from_protocol(protocol),
        updater_config=BeliefUpdaterConfig.from_protocol(protocol),
    )


def test_a_real_eye_snapshot_reads_into_the_full_context_vector(eye_clocks) -> None:
    bar, snapshot, closes = eye_clocks[-1]
    assert type(snapshot) is MarketSnapshot

    context = observation_features(
        snapshot, closes=closes, bar_high_low=(float(bar.high), float(bar.low))
    )

    assert len(context) == FEATURE_DIM == len(FEATURE_NAMES)
    # NaN is the documented reading for a component whose referent the Eye has
    # not formed yet -- a protected swing, a dealing range, an unswept pool, a
    # scored displacement, a prior-day level. Every other component is a
    # direction, a phase, a count or an ATR the Eye always publishes, and a NaN
    # there would mean the Brain is reading a field the Eye no longer fills.
    nan_components = {
        name for name, value in zip(FEATURE_NAMES, context) if math.isnan(value)
    }
    unexpected = {
        name for name in nan_components if not _OPTIONAL_REFERENT.match(name)
    }
    assert not unexpected, f"Eye fields the Brain reads but the Eye left unset: {sorted(unexpected)}"


def test_the_live_snapshot_path_publishes_the_belief_the_dataset_path_would(
    eye_clocks,
) -> None:
    """``ForecastInput.snapshot`` is the live entry; ``observation_features`` plus an
    explicit ATR is what the trajectory dataset and the replay scripts feed. Both
    must reach the same published belief, revision for revision."""

    live = _forecaster()
    dataset = _forecaster()

    for bar, snapshot, closes in eye_clocks[-REPLAY_CLOCKS:]:
        common = dict(
            asof=snapshot.asof,
            close=float(bar.close),
            high=float(bar.high),
            low=float(bar.low),
        )
        from_snapshot = live.observe(
            ForecastInput(snapshot=snapshot, closes=closes, **common)
        )
        from_features = dataset.observe(
            ForecastInput(
                context=observation_features(
                    snapshot,
                    closes=closes,
                    bar_high_low=(float(bar.high), float(bar.low)),
                ),
                atr=float(snapshot.timeframe_states[Timeframe.M1].quality.atr),
                **common,
            )
        )

        assert isinstance(from_snapshot, MarketBeliefState)
        # The Eye stamps a snapshot at the minute it became known, the close of
        # the bar that produced it; the belief carries that clock unchanged.
        assert snapshot.asof == bar.start + pd.Timedelta(minutes=1)
        assert from_snapshot.asof == snapshot.asof
        assert from_snapshot.protocol_fingerprint == protocol_fingerprint(PROTOCOL_PATH)
        assert from_snapshot.revision_id == from_features.revision_id
        assert from_snapshot.residual_probability == from_features.residual_probability
        assert [h.hypothesis_id for h in from_snapshot.hypotheses] == [
            h.hypothesis_id for h in from_features.hypotheses
        ]
