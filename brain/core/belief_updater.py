"""Minute-by-minute scoring of how each live hypothesis is holding up.

A hypothesis claims a sixty-point curve.  Every completed bar decides one more
point of it, and this module measures the gap between the claimed curve and the
realized one over exactly the bars that have elapsed.

The evidence weight is **recomputed in full on every clock**, never accumulated.
Accumulating would count the same realized minute once per subsequent bar;
recomputing keeps the score a pure function of the path so far, which is also
what makes a replay reproduce every ``revision_id``.

Comparing curves point-for-point replaced an earlier scheme that interpolated
between six horizon knots.  The claimed curve now has a value at every minute,
so there is nothing left to interpolate and no bar on which the score cannot
move.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from contract.brain.forecast import TRAJECTORY_CURVE_LENGTH

from .trajectory import RealizedPath


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
class PathEvidence:
    """One hypothesis's standing against its realized path on this clock."""

    evidence_log_weight: float
    divergence: float
    decided_points: int
    age: int


def evaluate(
    expected_curve: Sequence[float],
    dispersion: Sequence[float],
    path: RealizedPath,
    config: BeliefUpdaterConfig | None = None,
) -> PathEvidence:
    """Score one claimed curve against one realized path, from scratch.

    Only the elapsed points count.  ``dispersion`` is the within-node spread at
    each point, which is what makes a wide, uncertain claim hard to falsify and
    a tight one easy — the right asymmetry.
    """

    config = config or BeliefUpdaterConfig()
    if not isinstance(path, RealizedPath):
        raise BeliefUpdateError("evaluate() needs a RealizedPath")
    curve = tuple(float(v) for v in expected_curve)
    spread = tuple(float(v) for v in dispersion)
    if len(curve) != TRAJECTORY_CURVE_LENGTH or len(spread) != TRAJECTORY_CURVE_LENGTH:
        raise BeliefUpdateError(
            f"expected curve and dispersion must be {TRAJECTORY_CURVE_LENGTH} points"
        )

    realized = path.curve()[:TRAJECTORY_CURVE_LENGTH]
    if not realized:
        return PathEvidence(
            evidence_log_weight=0.0, divergence=0.0, decided_points=0, age=path.age
        )

    squares = [
        ((value - curve[index]) / max(spread[index], config.likelihood_scale_floor)) ** 2
        for index, value in enumerate(realized)
    ]
    # Mean rather than sum: otherwise an older hypothesis is penalized simply
    # for having been scored on more points than a younger rival.
    mean_square = sum(squares) / len(squares)
    log_weight = -0.5 * mean_square * (config.evidence_decay_per_bar ** path.age)
    log_weight = max(
        -config.log_likelihood_clip, min(config.log_likelihood_clip, log_weight)
    )
    return PathEvidence(
        evidence_log_weight=float(log_weight),
        divergence=float(math.sqrt(mean_square)),
        decided_points=len(realized),
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
    "normalize_log_weights",
]
