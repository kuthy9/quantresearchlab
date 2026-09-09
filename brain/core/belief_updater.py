"""Minute-by-minute scoring of how each live hypothesis is holding up.

A hypothesis claims a sixty-minute shape.  Every completed bar decides a little
more of that claim, and this module measures the gap between what the mode said
and what the tape did.

The evidence weight is **recomputed in full on every clock**, never accumulated
across clocks.  Accumulating would count the same realized minute once per
subsequent bar; recomputing keeps the score a pure function of the path so far,
which also makes a replay bit-for-bit reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

from contract.brain.forecast import (
    TRAJECTORY_COMPONENTS,
    TRAJECTORY_COMPONENT_HORIZON,
    TRAJECTORY_DIM,
    TrajectoryMode,
)

# The cumulative-return components, in horizon order.  The expected return at an
# arbitrary age is linearly interpolated between these knots, which is what lets
# the updater say something on every bar rather than only at six milestones.
_RETURN_KNOTS: tuple[tuple[int, int], ...] = tuple(
    (TRAJECTORY_COMPONENT_HORIZON[i], i)
    for i, name in enumerate(TRAJECTORY_COMPONENTS)
    if name.startswith("r_")
)
_MAX_HORIZON = max(TRAJECTORY_COMPONENT_HORIZON)


class BeliefUpdateError(RuntimeError):
    """The updater refuses to score a path it cannot interpret."""


@dataclass(frozen=True)
class BeliefUpdaterConfig:
    """How forgiving the likelihood is, and how fast old evidence fades."""

    likelihood_scale_floor: float = 0.25
    log_likelihood_clip: float = 4.0
    evidence_decay_per_bar: float = 0.98

    def __post_init__(self) -> None:
        if self.likelihood_scale_floor <= 0.0:
            raise ValueError("likelihood_scale_floor must be positive")
        if self.log_likelihood_clip <= 0.0:
            raise ValueError("log_likelihood_clip must be positive")
        if not 0.0 < self.evidence_decay_per_bar <= 1.0:
            raise ValueError("evidence_decay_per_bar must lie in (0, 1]")

    @classmethod
    def from_protocol(cls, payload: Mapping[str, Any]) -> "BeliefUpdaterConfig":
        section = payload.get("belief_updater", {})
        return cls(
            likelihood_scale_floor=float(section["likelihood_scale_floor"]),
            log_likelihood_clip=float(section["log_likelihood_clip"]),
            evidence_decay_per_bar=float(section["evidence_decay_per_bar"]),
        )


@dataclass(frozen=True)
class RealizedPath:
    """What actually happened since a hypothesis was spawned.

    Prices are stored raw and normalized on read, so one path can be scored
    against several modes without re-deriving anything.  ``anchor_atr`` is the
    one-minute ATR at spawn: the same scale the mode library was fitted in.
    """

    anchor_price: float
    anchor_atr: float
    closes: tuple[float, ...] = ()
    highs: tuple[float, ...] = ()
    lows: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.anchor_price):
            raise ValueError("anchor_price must be finite")
        if not math.isfinite(self.anchor_atr) or self.anchor_atr <= 0.0:
            raise ValueError("anchor_atr must be finite and positive")
        if not (len(self.closes) == len(self.highs) == len(self.lows)):
            raise ValueError("realized close/high/low series must be the same length")

    @property
    def age(self) -> int:
        """Completed bars observed since the anchor."""

        return len(self.closes)

    def extend(self, *, close: float, high: float, low: float) -> "RealizedPath":
        return replace(
            self,
            closes=self.closes + (float(close),),
            highs=self.highs + (float(high),),
            lows=self.lows + (float(low),),
        )

    def realized(self, name: str) -> float | None:
        """The realized value of one trajectory component, if it is decided yet."""

        index = TRAJECTORY_COMPONENTS.index(name)
        horizon = TRAJECTORY_COMPONENT_HORIZON[index]
        if self.age < horizon:
            return None
        atr = self.anchor_atr
        if name.startswith("r_"):
            return (self.closes[horizon - 1] - self.anchor_price) / atr
        if name.startswith("mfe_"):
            return (max(self.highs[:horizon]) - self.anchor_price) / atr
        if name.startswith("mae_"):
            return (min(self.lows[:horizon]) - self.anchor_price) / atr
        if name.startswith("rv_"):
            series = (self.anchor_price,) + self.closes[:horizon]
            diffs = [series[i + 1] - series[i] for i in range(horizon)]
            return math.sqrt(sum(value * value for value in diffs)) / atr
        raise BeliefUpdateError(f"unknown trajectory component {name!r}")

    def current_return(self) -> float | None:
        """ATR-normalized cumulative return at the current age."""

        if not self.closes:
            return None
        return (self.closes[-1] - self.anchor_price) / self.anchor_atr


def expected_return_at(mode: TrajectoryMode, age: int) -> float:
    """The mode's expected cumulative return at an arbitrary age.

    Linear interpolation between the six return knots, flat before the first
    and after the last.  This is a reading convenience, not a claim that the
    path between knots was straight.
    """

    if age <= 0:
        return 0.0
    knots = _RETURN_KNOTS
    first_h, first_i = knots[0]
    if age <= first_h:
        # Between the anchor (return 0 at age 0) and the first knot.
        return mode.medoid[first_i] * (age / first_h)
    last_h, last_i = knots[-1]
    if age >= last_h:
        return mode.medoid[last_i]
    for (low_h, low_i), (high_h, high_i) in zip(knots, knots[1:]):
        if low_h <= age <= high_h:
            span = high_h - low_h
            weight = (age - low_h) / span if span else 0.0
            return mode.medoid[low_i] * (1.0 - weight) + mode.medoid[high_i] * weight
    raise BeliefUpdateError(f"age {age} falls outside the interpolation knots")


def expected_dispersion_at(mode: TrajectoryMode, age: int) -> float:
    """The mode's return dispersion at an arbitrary age, interpolated likewise."""

    if age <= 0:
        return mode.dispersion[_RETURN_KNOTS[0][1]]
    knots = _RETURN_KNOTS
    first_h, first_i = knots[0]
    if age <= first_h:
        return mode.dispersion[first_i]
    last_h, last_i = knots[-1]
    if age >= last_h:
        return mode.dispersion[last_i]
    for (low_h, low_i), (high_h, high_i) in zip(knots, knots[1:]):
        if low_h <= age <= high_h:
            span = high_h - low_h
            weight = (age - low_h) / span if span else 0.0
            return (
                mode.dispersion[low_i] * (1.0 - weight)
                + mode.dispersion[high_i] * weight
            )
    raise BeliefUpdateError(f"age {age} falls outside the interpolation knots")


@dataclass(frozen=True)
class PathEvidence:
    """One hypothesis's standing against its realized path on this clock."""

    evidence_log_weight: float
    divergence: float
    decided_components: int
    age: int


def evaluate(
    mode: TrajectoryMode,
    path: RealizedPath,
    config: BeliefUpdaterConfig | None = None,
) -> PathEvidence:
    """Score one mode against one realized path, from scratch.

    Two sources contribute: every trajectory component whose horizon has already
    elapsed, and the interpolated cumulative return at the current age.  The
    second is what makes the score move on bars that decide no component.
    """

    config = config or BeliefUpdaterConfig()
    if not isinstance(mode, TrajectoryMode):
        raise BeliefUpdateError("evaluate() needs a TrajectoryMode")
    if not isinstance(path, RealizedPath):
        raise BeliefUpdateError("evaluate() needs a RealizedPath")

    squares: list[float] = []
    decided = 0
    for index in range(TRAJECTORY_DIM):
        name = TRAJECTORY_COMPONENTS[index]
        realized = path.realized(name)
        if realized is None:
            continue
        decided += 1
        scale = max(mode.dispersion[index], config.likelihood_scale_floor)
        squares.append(((realized - mode.medoid[index]) / scale) ** 2)

    current = path.current_return()
    if current is not None and path.age < _MAX_HORIZON:
        scale = max(
            expected_dispersion_at(mode, path.age), config.likelihood_scale_floor
        )
        squares.append(((current - expected_return_at(mode, path.age)) / scale) ** 2)

    if not squares:
        return PathEvidence(
            evidence_log_weight=0.0, divergence=0.0, decided_components=0, age=path.age
        )

    # Mean rather than sum: otherwise an older hypothesis is penalized simply
    # for having been scored on more components than a younger rival.
    mean_square = sum(squares) / len(squares)
    log_weight = -0.5 * mean_square * (config.evidence_decay_per_bar ** path.age)
    log_weight = max(-config.log_likelihood_clip, min(config.log_likelihood_clip, log_weight))
    return PathEvidence(
        evidence_log_weight=float(log_weight),
        divergence=float(math.sqrt(mean_square)),
        decided_components=decided,
        age=path.age,
    )


def normalize_log_weights(
    log_weights: tuple[float, ...],
    *,
    residual_log_weight: float,
) -> tuple[tuple[float, ...], float]:
    """Log-sum-exp normalization over the hypotheses plus the residual.

    The residual competes as an ordinary term, so it can never be squeezed to
    zero by confident-looking hypotheses; it is the standing weight of "none of
    these".  Returns the hypothesis probabilities and the residual probability.
    """

    terms = tuple(log_weights) + (residual_log_weight,)
    if any(not math.isfinite(value) for value in terms):
        raise BeliefUpdateError("cannot normalize non-finite log weights")
    ceiling = max(terms)
    exponentials = [math.exp(value - ceiling) for value in terms]
    total = sum(exponentials)
    if total <= 0.0:
        raise BeliefUpdateError("log-weight normalization collapsed to zero mass")
    probabilities = tuple(value / total for value in exponentials[:-1])
    residual = exponentials[-1] / total
    return probabilities, residual


__all__ = [
    "BeliefUpdateError",
    "BeliefUpdaterConfig",
    "PathEvidence",
    "RealizedPath",
    "evaluate",
    "expected_dispersion_at",
    "expected_return_at",
    "normalize_log_weights",
]
