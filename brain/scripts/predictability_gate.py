#!/usr/bin/env python3
"""TEMPORARY. Does X_t predict Y_t at all? Delete this once it has answered.

The hypothesis lifecycle was built, measured and found to publish nothing
tradeable: the conditional future cloud retrieved by ``X_t`` was no closer to
the realized path than a random slice of history. That result was produced by
one particular machine — k-NN retrieval, a two-channel representation, local
clustering — so it cannot distinguish "there is nothing to predict" from "that
machine cannot find it".

This script removes the machine. It asks the flat supervised question directly:
given features at ``t``, can any of three models beat the climatological mean on
seven targets, out of sample, under a protocol that makes leakage impossible?

If the answer is no across every feature group and every model, the problem is
the data, not the Brain's architecture, and no amount of redesign downstream
will help. If the answer is yes anywhere, that cell is where the Brain should be
rebuilt.

**Feature groups**

``raw``      OHLCV alone: return ladder, realized volatility, range position,
             volume behaviour, session clock. No Eye required, so this group
             runs on any window without an Eye pass.
``eye``      The Eye's published state: the 142 structural components of
             ``observation_features`` — every timeframe block, the cross-scale
             relations and the session block — with the eight tape-derived
             price components removed, so the group is genuinely disjoint from
             ``raw``.
``raw+eye``  Both, concatenated.

**Targets**  ``r_15`` ``r_30`` ``r_60`` (ATR-normalized cumulative return),
``mfe_60`` ``mae_60`` (excursion extremes), their decomposition ``range_60``
``asymmetry_60`` ``time_to_touch`` (magnitude, side, and first arrival), and
``shape_pc1`` ``shape_pc2`` (the detrended path shape). Every target is measured
in ATR units and every one is computed from the same sixty completed minutes.

**Baselines**  ``climatology`` (the training mean), ``time_of_day`` (the
training mean per minute of the session) and ``volatility_only`` (ridge on the
tape's ATR and realized-volatility ladder, always the same reference whatever
group is under test). A feature set has to beat the last of these, not the
first, to have shown anything.

**Hyperparameters are selected under the same purge.** ``RidgeCV``'s
leave-one-out is not a test on minute data — the row left out has near-twins
on either side — and it chose α = 0.1 on 142 collinear Eye features, with
coefficients of ±40 that cancelled in training and not out of sample (R² of
−2.3). The ridge penalty is now chosen by blocked, purged cross-validation
inside the training window, and LightGBM early-stops on a purged tail of it.

**Protocol** Train → purge → embargo → holdout, rolled forward over several
folds. The purge drops every training row whose sixty-minute label window
reaches into the embargo or the holdout; the embargo then holds a further gap
open. Both are at least the label horizon, which is what makes it impossible
for a training row's future to overlap a holdout row's past.

This is a research surface. It reads the future by construction and its output
is ``shadow_only``; it grants no research, empirical or trading authority.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from brain.core.trajectory import curve_matrix, shape_matrix  # noqa: E402
from shares.core.io import load_ohlcv  # noqa: E402
from brain.research.design_study import path_attribute_rows  # noqa: E402
from brain.scripts._windows import EXCHANGE_TZ, load_dataset  # noqa: E402
from contract.brain.forecast import TRAJECTORY_CURVE_LENGTH  # noqa: E402

HORIZON = TRAJECTORY_CURVE_LENGTH
TARGETS = (
    "r_15", "r_30", "r_60",
    "mfe_60", "mae_60",
    # The excursion pair decomposed. ``mfe_60`` and ``mae_60`` were the only
    # targets anything predicted, and they are not independent: both grow with
    # volatility. Splitting them into how far the path travelled in total and
    # which side it favoured separates the part that is magnitude from the part
    # that is direction wearing magnitude's clothes.
    "range_60", "asymmetry_60", "time_to_touch",
    "shape_pc1", "shape_pc2",
)

# How far price must move, in ATR, before "it got there" is a fact rather than
# noise. ``time_to_touch`` is the fraction of the horizon spent waiting for it.
TOUCH_THRESHOLD_ATR = 1.0

# The eight tape-derived components of the Eye's context vector. They are
# computed from recent closes rather than from published structure, so leaving
# them in the "eye" group would make it overlap the "raw" group and blur exactly
# the comparison this script exists to make. Pinned by name, and checked at
# run time against the proposer's vocabulary so a rename cannot silently put
# them back.
TAPE_FEATURE_NAMES: frozenset[str] = frozenset(
    {
        "past_r_5_atr", "past_r_15_atr", "past_r_30_atr", "past_r_60_atr",
        "past_rv_30_atr", "past_rv_60_atr", "bar_range_atr", "atr_ratio_m1_h1",
    }
)


class GateError(RuntimeError):
    """The gate refuses to report a number it cannot compute honestly."""


def tape_dataset(
    source: Path, *, start: str, end: str, atr_period: int
) -> dict:
    """Build the same dataset shape from OHLCV alone, with no Eye pass.

    The ``raw`` feature group needs no published Eye state, so it can be scored
    over any window immediately instead of waiting on an Eye run. ATR is Wilder's
    over ``atr_period`` completed minutes, which is the same definition the Eye
    uses for its one-minute quality scale; nothing else here touches Eye code.

    Rows are dropped when their sixty-minute future would run past the end of
    the data or across a session gap, so no target is ever assembled from bars
    the market did not actually trade consecutively.
    """

    frame = load_ohlcv(source, start=start, end=end).frame
    closes = frame["close"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    previous = np.concatenate(([closes[0]], closes[:-1]))
    true_range = np.maximum(
        highs - lows, np.maximum(np.abs(highs - previous), np.abs(lows - previous))
    )
    atr = (
        pd.Series(true_range)
        .ewm(alpha=1.0 / atr_period, adjust=False, min_periods=atr_period)
        .mean()
        .to_numpy(dtype=float)
    )

    rows = len(frame)
    usable = np.arange(rows - HORIZON)
    # A future window must be sixty consecutive traded minutes.
    stamps = frame.index
    contiguous = (
        stamps[usable + HORIZON] - stamps[usable]
    ) == pd.Timedelta(minutes=HORIZON)
    healthy = np.isfinite(atr[usable]) & (atr[usable] > 0.0)
    keep = usable[contiguous & healthy]
    if keep.size == 0:
        raise GateError("no bar in this window has a complete sixty-minute future")

    offsets = np.arange(1, HORIZON + 1)
    window = keep[:, None] + offsets[None, :]
    return {
        "index": stamps[keep],
        "features": np.zeros((keep.size, 0)),
        "prices": np.column_stack(
            [closes[keep], highs[keep], lows[keep], atr[keep]]
        ),
        "future_closes": closes[window],
        "future_highs": highs[window],
        "future_lows": lows[window],
        "volumes": frame["volume"].to_numpy(dtype=float)[keep]
        if "volume" in frame
        else np.ones(keep.size),
    }


@dataclass(frozen=True)
class Fold:
    """One train → purge → embargo → holdout arrangement, named by session."""

    index: int
    train: np.ndarray
    holdout: np.ndarray
    train_sessions: tuple[str, str]
    holdout_sessions: tuple[str, str]

    def describe(self) -> str:
        return (
            f"fold {self.index}: train {self.train.size:5d} rows "
            f"({self.train_sessions[0]}..{self.train_sessions[1]})  "
            f"holdout {self.holdout.size:5d} rows "
            f"({self.holdout_sessions[0]}..{self.holdout_sessions[1]})"
        )


def session_labels(index: pd.DatetimeIndex) -> np.ndarray:
    """The exchange session each clock belongs to, using the 18:00 boundary."""

    local = index.tz_convert(EXCHANGE_TZ) + pd.Timedelta(hours=6)
    return np.array([str(value) for value in local.date])


def build_folds(
    index: pd.DatetimeIndex,
    *,
    train_sessions: int,
    holdout_sessions: int,
    embargo_minutes: int,
    step_sessions: int,
    start_session: str | None,
) -> list[Fold]:
    """Roll a purged, embargoed split forward across the available sessions.

    Purging is done on the clock, not on the session boundary: a training row is
    dropped when ``t + horizon`` reaches the first embargoed minute, which is the
    exact condition under which its label would share minutes with the holdout's
    past. The embargo then holds a further gap open on top of that, so no
    holdout row's own history touches a training row's label either.
    """

    if embargo_minutes < HORIZON:
        raise GateError(
            f"an embargo shorter than the {HORIZON}-minute label horizon cannot "
            "prevent overlap"
        )
    labels = session_labels(index)
    ordered = list(dict.fromkeys(labels))
    if start_session is not None:
        if start_session not in ordered:
            later = [s for s in ordered if s >= start_session]
            if not later:
                raise GateError(f"no session on or after {start_session}")
            start_session = later[0]
        ordered = ordered[ordered.index(start_session):]
    span = train_sessions + holdout_sessions
    if len(ordered) < span:
        raise GateError(
            f"{len(ordered)} sessions available, {span} needed for one fold"
        )

    folds: list[Fold] = []
    for number, offset in enumerate(range(0, len(ordered) - span + 1, step_sessions)):
        block = ordered[offset : offset + span]
        train_names = set(block[:train_sessions])
        holdout_names = set(block[train_sessions:])
        train = np.flatnonzero(np.isin(labels, list(train_names)))
        holdout = np.flatnonzero(np.isin(labels, list(holdout_names)))
        if train.size == 0 or holdout.size == 0:
            continue

        embargo_start = index[holdout[0]] - pd.Timedelta(minutes=embargo_minutes)
        # Purge: a training row whose label window reaches the embargo is gone.
        keep = index[train] + pd.Timedelta(minutes=HORIZON) <= embargo_start
        train = train[keep]
        if train.size == 0:
            continue
        folds.append(
            Fold(
                index=number,
                train=train,
                holdout=holdout,
                train_sessions=(block[0], block[train_sessions - 1]),
                holdout_sessions=(block[train_sessions], block[-1]),
            )
        )
    if not folds:
        raise GateError("no fold survived purging")
    return folds


def raw_features(
    *, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    volumes: np.ndarray, atrs: np.ndarray, index: pd.DatetimeIndex,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Features from the tape alone, all normalized by the anchor ATR.

    Everything here is causal: each column reads only bars at or before ``t``.
    """

    columns: dict[str, np.ndarray] = {}
    series = pd.Series(closes)
    for span in (1, 5, 15, 30, 60, 120, 240):
        columns[f"ret_{span}"] = (
            (series - series.shift(span)).to_numpy(dtype=float) / atrs
        )
    for span in (15, 30, 60, 240):
        step = series.diff().to_numpy(dtype=float)
        squared = pd.Series(step**2).rolling(span).sum().to_numpy(dtype=float)
        columns[f"rv_{span}"] = np.sqrt(np.maximum(squared, 0.0)) / atrs
    for span in (30, 60, 240):
        top = pd.Series(highs).rolling(span).max().to_numpy(dtype=float)
        bottom = pd.Series(lows).rolling(span).min().to_numpy(dtype=float)
        width = np.maximum(top - bottom, 1e-9)
        columns[f"range_pos_{span}"] = (closes - bottom) / width
        columns[f"range_width_{span}"] = width / atrs
    volume = pd.Series(volumes)
    for span in (15, 60, 240):
        mean = volume.rolling(span).mean().to_numpy(dtype=float)
        columns[f"vol_rel_{span}"] = volumes / np.where(mean > 0, mean, np.nan)
    columns["atr"] = atrs
    local = index.tz_convert(EXCHANGE_TZ)
    minute = local.hour * 60 + local.minute
    columns["tod_sin"] = np.sin(2 * np.pi * minute / 1440)
    columns["tod_cos"] = np.cos(2 * np.pi * minute / 1440)
    columns["dow"] = local.dayofweek.to_numpy(dtype=float)

    names = tuple(columns)
    matrix = np.column_stack([columns[name] for name in names])
    return matrix, names


def build_targets(
    *, prices: np.ndarray, future_closes: np.ndarray,
    future_highs: np.ndarray, future_lows: np.ndarray,
    train_rows: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """The seven targets, with the shape basis fitted on training rows only.

    Fitting the shape basis on the whole window would leak the holdout's own
    geometry into the definition of its target, which is the subtlest way this
    kind of study goes wrong.
    """

    from sklearn.decomposition import PCA

    curves = curve_matrix(
        anchor_prices=prices[:, 0], anchor_atrs=prices[:, 3],
        future_closes=future_closes,
    )
    attributes = path_attribute_rows(
        anchor_prices=prices[:, 0], anchor_atrs=prices[:, 3],
        future_closes=future_closes, future_highs=future_highs,
        future_lows=future_lows,
    )
    shapes = shape_matrix(curves)
    basis = PCA(n_components=2, svd_solver="full", random_state=0)
    basis.fit(shapes[train_rows])
    scores = basis.transform(shapes)

    anchor = prices[:, 0].reshape(-1, 1)
    scale = prices[:, 3].reshape(-1, 1)
    mfe = (future_highs[:, :HORIZON].max(axis=1).reshape(-1, 1) - anchor) / scale
    mae = (future_lows[:, :HORIZON].min(axis=1).reshape(-1, 1) - anchor) / scale
    up = np.maximum(mfe[:, 0], 0.0)
    down = np.maximum(-mae[:, 0], 0.0)
    span = up + down
    # A path that never moved has no asymmetry to report, and reporting one
    # would be dividing noise by noise.
    asymmetry = np.divide(up - down, span, out=np.zeros_like(span), where=span > 1e-9)

    # First arrival: the first minute either side is reached, as a fraction of
    # the horizon. A path that reaches neither waited the whole hour.
    highs = (future_highs[:, :HORIZON] - anchor) / scale
    lows = (future_lows[:, :HORIZON] - anchor) / scale
    touched = (highs >= TOUCH_THRESHOLD_ATR) | (lows <= -TOUCH_THRESHOLD_ATR)
    ever = touched.any(axis=1)
    first = np.where(ever, touched.argmax(axis=1) + 1, HORIZON)
    columns = {
        "r_15": curves[:, 14],
        "r_30": curves[:, 29],
        "r_60": curves[:, 59],
        "mfe_60": up,
        "mae_60": np.minimum(mae[:, 0], 0.0),
        "range_60": span,
        "asymmetry_60": asymmetry,
        "time_to_touch": first / float(HORIZON),
        "shape_pc1": scores[:, 0],
        "shape_pc2": scores[:, 1],
    }
    return np.column_stack([columns[name] for name in TARGETS]), TARGETS


def fit_predict(
    model: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    train_tod: np.ndarray | None = None,
    test_tod: np.ndarray | None = None,
) -> np.ndarray:
    """The baselines a real signal has to beat, then the two candidate fits.

    ``climatology`` is the weakest honest baseline — predict the training mean.
    ``time_of_day`` adds the one structure everyone knows is there: the session
    has a shape, and volatility at the open is not volatility at lunch.
    ``volatility_only`` is the baseline that matters: excursions grow with
    volatility and volatility is persistent, so a model of the current ATR and
    recent realized volatility already predicts how far price can travel. A rich
    feature set earns its place only by beating *that*, not by beating the mean.
    """

    if model == "climatology":
        return np.full(test_x.shape[0], float(train_y.mean()))
    if model == "time_of_day":
        if train_tod is None or test_tod is None:
            raise GateError("the time-of-day baseline needs the session clock")
        overall = float(train_y.mean())
        table = np.full(int(train_tod.max()) + 1, overall)
        order = np.argsort(train_tod, kind="stable")
        buckets, starts = np.unique(train_tod[order], return_index=True)
        sums = np.add.reduceat(train_y[order], starts)
        counts = np.diff(np.append(starts, train_tod.size))
        # A bucket the training window barely saw falls back to the overall
        # mean rather than to whatever those few rows happened to do.
        healthy = counts >= 20
        table[buckets[healthy]] = sums[healthy] / counts[healthy]
        return table[np.clip(test_tod, 0, table.size - 1)]
    if model == "volatility_only":
        from sklearn.linear_model import RidgeCV

        estimator = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0))
        estimator.fit(train_x, train_y)
        return estimator.predict(test_x)
    if model == "ridge":
        from sklearn.linear_model import Ridge

        alpha = select_ridge_alpha(train_x, train_y)
        estimator = Ridge(alpha=alpha).fit(train_x, train_y)
        return estimator.predict(test_x)
    if model == "lightgbm":
        import lightgbm as lgb

        # Early-stop on a purged tail of the training window: the last fifth
        # of it, with a horizon-wide gap before it, so the stopping rule never
        # sees a row whose label overlaps what it was fitted on.
        cut = int(train_x.shape[0] * 0.8)
        gap = HORIZON
        fit_x, fit_y = train_x[: cut - gap], train_y[: cut - gap]
        tail_x, tail_y = train_x[cut:], train_y[cut:]
        estimator = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=15,
            min_child_samples=500, subsample=0.7, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=10.0, random_state=0, verbose=-1,
        )
        estimator.fit(
            fit_x, fit_y,
            eval_set=[(tail_x, tail_y)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        return estimator.predict(test_x)
    raise GateError(f"unknown model {model}")


def select_ridge_alpha(train_x: np.ndarray, train_y: np.ndarray) -> float:
    """Choose the ridge penalty by *blocked, purged* cross-validation.

    ``RidgeCV`` selects by leave-one-out, and on minute data that is not a
    test at all: the row left out has near-identical neighbours on both sides,
    so the procedure rewards fitting the noise and chose α = 0.1 on 142
    collinear features, with coefficients of ±40 that cancelled in training and
    did not out of sample. Here the training window is cut into contiguous
    blocks, a horizon-wide gap is purged at each boundary, and the penalty that
    generalizes across blocks wins.
    """

    from sklearn.linear_model import Ridge

    rows = train_x.shape[0]
    blocks = 5
    edges = np.linspace(0, rows, blocks + 1, dtype=int)
    alphas = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
    scores = {alpha: [] for alpha in alphas}
    for b in range(blocks):
        lo, hi = edges[b], edges[b + 1]
        test = np.arange(lo, hi)
        left = np.arange(0, max(0, lo - HORIZON))
        right = np.arange(min(rows, hi + HORIZON), rows)
        train = np.concatenate([left, right])
        if train.size < 100 or test.size < 100:
            continue
        base = ((train_y[test] - train_y[train].mean()) ** 2).mean()
        for alpha in alphas:
            fitted = Ridge(alpha=alpha).fit(train_x[train], train_y[train])
            err = ((train_y[test] - fitted.predict(train_x[test])) ** 2).mean()
            scores[alpha].append(1.0 - err / base if base > 0 else 0.0)
    means = {alpha: float(np.mean(v)) for alpha, v in scores.items() if v}
    if not means:
        return alphas[-1]
    return max(means, key=means.get)


def _fit_reference(
    model: str, train_ref: np.ndarray, train_y: np.ndarray, test_ref: np.ndarray, **extra
) -> np.ndarray:
    """A baseline fitted on the fixed reference matrix, whatever group is under test."""

    return fit_predict(model, train_ref, train_y, test_ref, **extra)


def standardize_pair(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score both halves on the *training* statistics, imputing to the centre."""

    centre = np.nanmean(train, axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    spread = np.nanstd(train, axis=0)
    spread = np.where(np.isfinite(spread) & (spread > 1e-12), spread, 1.0)
    out = []
    for block in (train, test):
        z = (block - centre) / spread
        out.append(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0))
    return out[0], out[1]


def block_bootstrap_interval(
    errors_model: np.ndarray, errors_baseline: np.ndarray, *, draws: int = 2000
) -> tuple[float, float, float]:
    """Interval on the mean squared-error difference, resampled in blocks.

    Blocks of one horizon, because two rows less than sixty minutes apart share
    most of their label and an i.i.d. interval would be far too narrow.
    """

    difference = errors_model - errors_baseline
    block = HORIZON
    blocks = max(1, difference.size // block)
    rng = np.random.default_rng(17)
    means = np.empty(draws)
    for draw in range(draws):
        picked = rng.integers(0, blocks, size=blocks)
        means[draw] = np.concatenate(
            [difference[i * block : (i + 1) * block] for i in picked]
        ).mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(difference.mean()), float(low), float(high)


def evaluate(
    *, folds: list[Fold], features: np.ndarray, targets: np.ndarray,
    group: str, models: tuple[str, ...], names: tuple[str, ...],
    tod: np.ndarray, challenger: str, reference: np.ndarray,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], list[np.ndarray]]]:
    """Every (fold, target, model) cell, scored against the climatology.

    The per-row squared-error differences are kept alongside the summary so the
    interval can be taken over the folds *pooled*. Asking each fold to clear
    significance on its own is a different and much harsher question than asking
    whether the improvement is real, and condition 1 is the second one.
    """

    rows = []
    pooled: dict[tuple[str, str, str], list[np.ndarray]] = {}
    # The volatility baseline is a fixed reference, not a property of the
    # group under test: it is always fitted on the tape's own volatility
    # columns, so every group is measured against the same simple story.
    for fold in folds:
        train_x, test_x = standardize_pair(
            features[fold.train], features[fold.holdout]
        )
        train_ref, test_ref = standardize_pair(
            reference[fold.train], reference[fold.holdout]
        )
        extra = dict(train_tod=tod[fold.train], test_tod=tod[fold.holdout])
        for position, target in enumerate(TARGETS):
            train_y = targets[fold.train, position]
            test_y = targets[fold.holdout, position]
            if not np.isfinite(train_y).all() or not np.isfinite(test_y).all():
                raise GateError(f"target {target} carries a non-finite value")
            baseline = fit_predict("climatology", train_x, train_y, test_x)
            baseline_errors = (test_y - baseline) ** 2
            challenge_errors = None
            if challenger != "climatology":
                challenge = _fit_reference(
                    challenger, train_ref, train_y, test_ref, **extra
                )
                challenge_errors = (test_y - challenge) ** 2
            for model in models:
                if model == "climatology":
                    continue
                if model in ("volatility_only",):
                    predicted = _fit_reference(model, train_ref, train_y, test_ref, **extra)
                else:
                    predicted = fit_predict(model, train_x, train_y, test_x, **extra)
                errors = (test_y - predicted) ** 2
                mean, low, high = block_bootstrap_interval(errors, baseline_errors)
                pooled.setdefault((group, target, model), []).append(
                    errors - baseline_errors
                )
                denominator = float((predicted * predicted).sum())
                rows.append(
                    {
                        "group": group,
                        "fold": fold.index,
                        "holdout_from": fold.holdout_sessions[0],
                        "target": target,
                        "model": model,
                        # Out-of-sample R^2 against the climatology: positive
                        # means the model beat "predict the training mean".
                        "oos_r2": 1.0 - errors.mean() / baseline_errors.mean(),
                        "mse_difference": mean,
                        "ci_low": low,
                        "ci_high": high,
                        # The least-squares optimal rescaling of the prediction.
                        # Near zero means the best use of it is to ignore it.
                        "alpha": float((test_y * predicted).sum() / denominator)
                        if denominator > 0
                        else 0.0,
                        "corr": float(np.corrcoef(predicted, test_y)[0, 1])
                        if predicted.std() > 0
                        else 0.0,
                        # Beating the mean is table stakes. This is the number
                        # that says whether the feature set earned its keep
                        # against the simple volatility-persistence story.
                        "r2_vs_challenger": (
                            1.0 - errors.mean() / challenge_errors.mean()
                            if challenge_errors is not None
                            and challenge_errors.mean() > 0
                            and model != challenger
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows), pooled


def pooled_interval(differences: list[np.ndarray], *, draws: int = 2000) -> tuple[float, float, float]:
    """One interval over every fold's rows, resampled in whole blocks.

    Folds are resampled independently and then concatenated, so a single fold
    cannot carry the interval on its own and the block structure inside each
    fold is preserved.
    """

    rng = np.random.default_rng(23)
    block = HORIZON
    means = np.empty(draws)
    for draw in range(draws):
        parts = []
        for series in differences:
            blocks = max(1, series.size // block)
            picked = rng.integers(0, blocks, size=blocks)
            parts.append(
                np.concatenate([series[i * block : (i + 1) * block] for i in picked])
            )
        means[draw] = np.concatenate(parts).mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(np.concatenate(differences).mean()), float(low), float(high)


def verdict(
    frame: pd.DataFrame, pooled: dict[tuple[str, str, str], list[np.ndarray]]
) -> pd.DataFrame:
    """Apply the five conditions to every (group, target, model) cell.

    No fixed correlation threshold appears anywhere. A cell passes only when the
    model beat the baseline out of sample *and* all five hold at once.
    """

    rows = []
    for (group, target, model), block in frame.groupby(
        ["group", "target", "model"], sort=False
    ):
        beats = block["oos_r2"].mean() > 0.0
        # 1. the interval supports the improvement: better means lower error.
        # Taken over the folds pooled, which is the question "is the
        # improvement real"; the per-fold count is reported beside it because it
        # is the stricter reading and the gap between the two is informative.
        mean_difference, low, high = pooled_interval(pooled[(group, target, model)])
        significant = bool(high < 0.0)
        folds_significant = int((block["ci_high"] < 0.0).sum())
        # 2. the direction is the same across the rolling windows.
        consistent = bool((block["oos_r2"] > 0).mean() >= 0.8)
        # 3. dropping the best fold must not flip the conclusion.
        without_best = block.sort_values("oos_r2").iloc[:-1]["oos_r2"].mean()
        robust = bool(without_best > 0.0) if len(block) > 1 else False
        # 4. the prediction is worth using at its own scale.
        scaled = bool(block["alpha"].abs().mean() > 0.2)
        rows.append(
            {
                "group": group, "target": target, "model": model,
                "folds": len(block),
                "oos_r2": block["oos_r2"].mean(),
                "oos_r2_worst": block["oos_r2"].min(),
                "pooled_ci_low": low,
                "pooled_ci_high": high,
                "folds_signif": folds_significant,
                "alpha": block["alpha"].mean(),
                "corr": block["corr"].mean(),
                "r2_vs_chal": block["r2_vs_challenger"].mean(),
                "beats_baseline": beats,
                "ci_supports": significant,
                "consistent": consistent,
                "not_one_month": robust,
                "alpha_not_zero": scaled,
                "PASS": bool(beats and significant and consistent and robust and scaled),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="outputs/hypothesis_v3/dataset.npz")
    parser.add_argument(
        "--source",
        default="data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet",
        help="OHLCV parquet, used for the raw group's volume and for --no-eye runs",
    )
    parser.add_argument("--train-sessions", type=int, default=60)
    parser.add_argument("--holdout-sessions", type=int, default=20)
    parser.add_argument("--step-sessions", type=int, default=20)
    parser.add_argument("--start-session", default="2022-01-03")
    parser.add_argument(
        "--embargo-minutes",
        type=int,
        default=HORIZON,
        help=(
            "gap held open between the purged training window and the holdout. "
            f"Never below the {HORIZON}-minute label horizon."
        ),
    )
    parser.add_argument(
        "--groups", default="raw,eye,raw+eye",
        help="comma-separated feature groups; 'eye' needs an Eye-built dataset",
    )
    parser.add_argument(
        "--models", default="time_of_day,volatility_only,ridge,lightgbm"
    )
    parser.add_argument(
        "--challenger",
        default="volatility_only",
        help=(
            "the baseline a rich feature set must beat to have earned its "
            "place. Beating the climatological mean is table stakes."
        ),
    )
    parser.add_argument(
        "--from-tape",
        action="store_true",
        help=(
            "build the window straight from OHLCV instead of an Eye dataset. "
            "Only the 'raw' group is available this way, and it needs no Eye "
            "pass, so the core question can be answered without one."
        ),
    )
    parser.add_argument("--tape-start", default="2021-12-01")
    parser.add_argument("--tape-end", default="2023-01-01")
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    groups = tuple(name.strip() for name in args.groups.split(",") if name.strip())
    models = tuple(name.strip() for name in args.models.split(",") if name.strip())

    if args.from_tape:
        if groups != ("raw",):
            raise GateError("--from-tape carries no Eye state; use --groups raw")
        data = tape_dataset(
            ROOT / args.source,
            start=args.tape_start,
            end=args.tape_end,
            atr_period=args.atr_period,
        )
        volumes = data["volumes"]
    else:
        data = load_dataset(ROOT / args.dataset)
        frame = pd.read_parquet(ROOT / args.source, columns=["volume"])
        volumes = (
            frame["volume"].reindex(data["index"]).to_numpy(dtype=float)
            if "volume" in frame
            else np.ones(len(data["index"]))
        )
    index = data["index"]
    prices = data["prices"]

    folds = build_folds(
        index,
        train_sessions=args.train_sessions,
        holdout_sessions=args.holdout_sessions,
        embargo_minutes=args.embargo_minutes,
        step_sessions=args.step_sessions,
        start_session=args.start_session,
    )
    print(f"{len(index)} observation points, {len(set(session_labels(index)))} sessions")
    for fold in folds:
        print("  " + fold.describe())
    print(
        f"\npurge: a training row is dropped when t+{HORIZON}min reaches the "
        f"embargo; embargo: {args.embargo_minutes} min held open before each "
        "holdout. No training label can share a minute with a holdout row's past."
    )

    raw, raw_names = raw_features(
        closes=prices[:, 0], highs=prices[:, 1], lows=prices[:, 2],
        volumes=volumes, atrs=prices[:, 3], index=index,
    )
    available = {"raw": (raw, raw_names)}
    if data["features"].shape[1]:
        from brain.core.hypothesis_proposer import FEATURE_NAMES

        missing = TAPE_FEATURE_NAMES - set(FEATURE_NAMES)
        if missing:
            raise GateError(
                f"tape features not in the Eye vocabulary: {sorted(missing)}"
            )
        keep = [
            position
            for position, name in enumerate(FEATURE_NAMES)
            if name not in TAPE_FEATURE_NAMES
        ]
        eye = data["features"][:, keep]
        names = tuple(FEATURE_NAMES[i] for i in keep)
        available["eye"] = (eye, names)
        available["raw+eye"] = (np.hstack([raw, eye]), raw_names + names)
        print(
            f"\nfeature groups: raw {raw.shape[1]}, eye {eye.shape[1]}, "
            f"raw+eye {raw.shape[1] + eye.shape[1]}"
        )
    else:
        print(f"\nfeature groups: raw {raw.shape[1]} (no Eye state in this window)")

    targets, _ = build_targets(
        prices=prices,
        future_closes=data["future_closes"],
        future_highs=data["future_highs"],
        future_lows=data["future_lows"],
        train_rows=folds[0].train,
    )

    local = index.tz_convert(EXCHANGE_TZ)
    tod_bucket = (local.hour * 60 + local.minute).to_numpy(dtype=int)
    # The volatility-persistence reference: the anchor ATR and the realized
    # volatility ladder from the tape, and nothing else.
    reference = raw[:, [i for i, n in enumerate(raw_names) if n == "atr" or n.startswith("rv_")]]

    results = []
    pooled: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for group in groups:
        if group not in available:
            raise GateError(f"unknown feature group {group}")
        matrix, _ = available[group]
        print(f"\nfitting {group} …", flush=True)
        block, block_pooled = evaluate(
            folds=folds, features=matrix, targets=targets,
            group=group, models=models, names=available[group][1],
            tod=tod_bucket, challenger=args.challenger, reference=reference,
        )
        results.append(block)
        pooled.update(block_pooled)
    frame = pd.concat(results, ignore_index=True)

    pd.set_option("display.width", 240)
    print("\n=== every (group, target, model) cell, averaged over folds ===")
    table = verdict(frame, pooled)
    print(
        table.to_string(index=False, float_format=lambda v: f"{v:8.4f}")
    )

    passed = table[table["PASS"]]
    print("\n=== the five conditions ===")
    print(
        "  beats_baseline   mean OOS R^2 against climatology is positive\n"
        "  ci_supports      the block-bootstrap interval on the error difference,\n"
        "                   taken over every fold pooled, excludes zero\n"
        "                   (folds_signif reports how many folds clear it alone)\n"
        "  consistent       the sign holds on at least 80% of the rolling windows\n"
        "  not_one_month    dropping the single best fold does not flip it\n"
        "  alpha_not_zero   the optimal rescaling of the prediction is not ~0"
    )
    if passed.empty:
        print(
            "\nNO CELL PASSES. On this data, with these features, targets and "
            "models, X_t does not predict Y_t out of sample. The failure is in "
            "the data or the feature set, not in the Brain's architecture."
        )
    else:
        print(f"\n{len(passed)} CELL(S) PASS:")
        print(passed.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print(f"\n=== step 2: can rich features beat '{args.challenger}'? ===")
    challenge = table[table["model"] != args.challenger].copy()
    baseline_row = table[table["model"] == args.challenger]
    if not baseline_row.empty:
        print(
            baseline_row[["group", "target", "oos_r2", "corr", "PASS"]]
            .rename(columns={"oos_r2": "baseline_r2"})
            .to_string(index=False, float_format=lambda v: f"{v:8.4f}")
        )
    print(f"\n  and what each candidate adds on top of it (r2_vs_chal):")
    print(
        challenge[["group", "target", "model", "oos_r2", "r2_vs_chal", "corr"]]
        .to_string(index=False, float_format=lambda v: f"{v:8.4f}")
    )
    print(
        "\n  r2_vs_chal above zero means the feature set beat the volatility "
        "story; at or below zero the extra features bought nothing and the\n"
        "  predictable part was volatility persistence all along."
    )

    if len(groups) > 1:
        print("\n=== step 4: what EYE adds on top of RAW (delta OOS R^2) ===")
        wide = table.pivot_table(
            index=["target", "model"], columns="group", values="oos_r2"
        )
        if "raw" in wide.columns:
            for column in ("eye", "raw+eye"):
                if column in wide.columns:
                    wide[f"delta_{column}"] = wide[column] - wide["raw"]
        print(wide.to_string(float_format=lambda v: f"{v:8.4f}"))
        print(
            "\n  delta_raw+eye is the incremental value of the Eye's published "
            "structure. A positive delta with a sign that holds across targets "
            "is evidence;\n  alternating signs of similar size are not."
        )

    print("\n=== ablation: what each group adds, by mean OOS R^2 ===")
    pivot = table.pivot_table(
        index=["target", "model"], columns="group", values="oos_r2"
    )
    print(pivot.to_string(float_format=lambda v: f"{v:8.4f}"))
    print(
        "\n  An interpretable ablation is one where raw+eye is at least as good "
        "as its parts and the eye group's contribution has a sign that holds "
        "across targets. Noise looks like alternating signs of similar size."
    )

    if args.output:
        destination = ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination.with_suffix(".folds.csv"), index=False)
        table.to_csv(destination.with_suffix(".verdict.csv"), index=False)
        print(f"\nwrote {destination.with_suffix('.folds.csv')}")
        print(f"wrote {destination.with_suffix('.verdict.csv')}")

    raise SystemExit(0 if not passed.empty else 1)


if __name__ == "__main__":
    main()
