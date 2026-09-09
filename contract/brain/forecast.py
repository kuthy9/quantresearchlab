"""Brain forecast: the naturally discovered hypothesis contract.

The Brain no longer names a frozen taxonomy of market paths.  It fits a library
of *trajectory modes* from history — each mode is one way the next sixty minutes
actually unfolded, in ATR units — and at every completed clock it keeps at most
three of them alive as competing hypotheses.

``MarketBeliefState`` is what the Brain publishes each minute.  Its hypotheses
never account for the whole future: ``residual_probability`` is the standing
admission that the live modes may all be wrong, and it is never normalized away.

Everything here is shadow-only.  A forecast carries no action authority, and
``MarketBeliefState`` refuses to be constructed claiming otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import pandas as pd

from contract.market.primitives import aware_timestamp, content_hash

FORECAST_SCHEMA_VERSION = 1

# The initial trajectory vector.  Sixty raw price points cannot be clustered
# directly, so a trajectory is summarized by six ATR-normalized returns, three
# favorable/adverse excursion pairs and two realized-volatility terms.  The
# order is part of the contract: a mode library's medoid is read positionally.
TRAJECTORY_COMPONENTS: tuple[str, ...] = (
    "r_1",
    "r_5",
    "r_10",
    "r_15",
    "r_30",
    "r_60",
    "mfe_15",
    "mae_15",
    "mfe_30",
    "mae_30",
    "mfe_60",
    "mae_60",
    "rv_30",
    "rv_60",
)
TRAJECTORY_DIM = len(TRAJECTORY_COMPONENTS)

# The horizon each component observes, used to decide which components are
# already decided at a given hypothesis age.
TRAJECTORY_COMPONENT_HORIZON: tuple[int, ...] = (
    1,
    5,
    10,
    15,
    30,
    60,
    15,
    15,
    30,
    30,
    60,
    60,
    30,
    60,
)

# The Brain keeps at most this many live hypotheses.  This is a runtime working
# set, not a claim that only three futures exist: the mode library may hold any
# number of modes.
MAX_LIVE_HYPOTHESES = 3

FORECAST_AUTHORITY = "shadow_only"
FORECAST_PROTOCOL_STATUS = "development_unvalidated"


class HypothesisStatus(str, Enum):
    """Whether a hypothesis is still competing for the next sixty minutes."""

    ACTIVE = "active"
    RETIRED = "retired"


class LifecycleOperation(str, Enum):
    """The five things the pool may do to its hypotheses on one clock."""

    SPAWN = "spawn"
    UPDATE = "update"
    SPLIT = "split"
    MERGE = "merge"
    RETIRE = "retire"


def _finite(value: object, *, name: str) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _trajectory(values: object, *, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise TypeError(f"{name} must be a sequence of floats")
    vector = tuple(_finite(item, name=name) for item in values)  # type: ignore[union-attr]
    if len(vector) != TRAJECTORY_DIM:
        raise ValueError(
            f"{name} must carry {TRAJECTORY_DIM} components, got {len(vector)}"
        )
    return vector


def _identity(*parts: object) -> str:
    return content_hash([str(part) for part in parts])


@dataclass(frozen=True)
class TrajectoryMode:
    """One naturally occurring sixty-minute shape, plus how tight it is.

    ``medoid`` is a real observed trajectory, never a centroid average: an
    average of two opposite futures is a third future that never happened.
    ``dispersion`` is the per-component spread inside the mode and is what the
    belief updater uses as its likelihood scale.
    """

    mode_id: str
    medoid: tuple[float, ...]
    dispersion: tuple[float, ...]
    support: int
    parent_mode_id: str | None = None
    child_mode_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.mode_id:
            raise ValueError("mode_id is required")
        object.__setattr__(self, "medoid", _trajectory(self.medoid, name="medoid"))
        dispersion = _trajectory(self.dispersion, name="dispersion")
        if any(value <= 0.0 for value in dispersion):
            raise ValueError("mode dispersion must be strictly positive")
        object.__setattr__(self, "dispersion", dispersion)
        if int(self.support) <= 0:
            raise ValueError("a mode must be supported by at least one observation")
        object.__setattr__(self, "support", int(self.support))
        children = tuple(str(item) for item in self.child_mode_ids)
        if len(set(children)) != len(children):
            raise ValueError("child_mode_ids must be unique")
        if self.mode_id in children or self.mode_id == self.parent_mode_id:
            raise ValueError("a mode may not be its own parent or child")
        object.__setattr__(self, "child_mode_ids", children)

    def component(self, name: str) -> float:
        """Read one named medoid component."""

        return self.medoid[TRAJECTORY_COMPONENTS.index(name)]


@dataclass(frozen=True)
class ModeLibrary:
    """The fitted set of modes, sealed by the fingerprint of what produced it."""

    library_id: str
    fingerprint: str
    fitted_at: pd.Timestamp
    algorithm: str
    modes: tuple[TrajectoryMode, ...]
    feature_names: tuple[str, ...]
    observation_count: int
    noise_count: int
    schema_version: int = FORECAST_SCHEMA_VERSION
    authority: str = FORECAST_AUTHORITY
    protocol_status: str = FORECAST_PROTOCOL_STATUS

    def __post_init__(self) -> None:
        if not self.library_id or not self.fingerprint:
            raise ValueError("a mode library must carry an identity and a fingerprint")
        object.__setattr__(
            self, "fitted_at", aware_timestamp(self.fitted_at, name="fitted_at")
        )
        modes = tuple(self.modes)
        if not modes:
            raise ValueError("a mode library must hold at least one mode")
        if any(not isinstance(mode, TrajectoryMode) for mode in modes):
            raise TypeError("modes must be TrajectoryMode instances")
        ids = [mode.mode_id for mode in modes]
        if len(set(ids)) != len(ids):
            raise ValueError("mode ids must be unique inside a library")
        known = set(ids)
        for mode in modes:
            if mode.parent_mode_id is not None and mode.parent_mode_id not in known:
                raise ValueError(f"mode {mode.mode_id} cites an unknown parent")
            missing = [child for child in mode.child_mode_ids if child not in known]
            if missing:
                raise ValueError(f"mode {mode.mode_id} cites unknown children {missing}")
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "feature_names", tuple(str(n) for n in self.feature_names))
        if int(self.observation_count) < len(modes):
            raise ValueError("a library cannot hold more modes than observations")
        object.__setattr__(self, "observation_count", int(self.observation_count))
        if int(self.noise_count) < 0:
            raise ValueError("noise_count cannot be negative")
        object.__setattr__(self, "noise_count", int(self.noise_count))
        if self.authority != FORECAST_AUTHORITY:
            raise ValueError("a mode library is shadow-only")
        if self.protocol_status != FORECAST_PROTOCOL_STATUS:
            raise ValueError("a mode library is development-unvalidated")

    def mode(self, mode_id: str) -> TrajectoryMode:
        for candidate in self.modes:
            if candidate.mode_id == mode_id:
                return candidate
        raise KeyError(mode_id)

    @property
    def mode_ids(self) -> tuple[str, ...]:
        return tuple(mode.mode_id for mode in self.modes)


@dataclass(frozen=True)
class HypothesisProposal:
    """One context-conditioned candidate, before the pool decides anything.

    ``prior`` is the empirical frequency with which the retrieved historical
    neighbours of the current context went on to realize this mode.
    """

    mode_id: str
    prior: float
    neighbour_count: int
    neighbour_distance: float

    def __post_init__(self) -> None:
        if not self.mode_id:
            raise ValueError("mode_id is required")
        prior = _finite(self.prior, name="prior")
        if not 0.0 <= prior <= 1.0:
            raise ValueError("a proposal prior must be a probability")
        object.__setattr__(self, "prior", prior)
        if int(self.neighbour_count) < 0:
            raise ValueError("neighbour_count cannot be negative")
        object.__setattr__(self, "neighbour_count", int(self.neighbour_count))
        distance = _finite(self.neighbour_distance, name="neighbour_distance")
        if distance < 0.0:
            raise ValueError("neighbour_distance cannot be negative")
        object.__setattr__(self, "neighbour_distance", distance)


@dataclass(frozen=True)
class Hypothesis:
    """One live claim about the next sixty minutes, and how it is holding up."""

    hypothesis_id: str
    mode_id: str
    spawned_at: pd.Timestamp
    asof: pd.Timestamp
    age_bars: int
    prior_log_weight: float
    evidence_log_weight: float
    probability: float
    expected_trajectory: tuple[float, ...]
    realized_divergence: float
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not self.mode_id:
            raise ValueError("a hypothesis needs an identity and a mode")
        object.__setattr__(
            self, "spawned_at", aware_timestamp(self.spawned_at, name="spawned_at")
        )
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="asof"))
        if self.asof < self.spawned_at:
            raise ValueError("a hypothesis cannot be observed before it was spawned")
        age = int(self.age_bars)
        if age < 0:
            raise ValueError("age_bars cannot be negative")
        object.__setattr__(self, "age_bars", age)
        object.__setattr__(
            self, "prior_log_weight", _finite(self.prior_log_weight, name="prior_log_weight")
        )
        object.__setattr__(
            self,
            "evidence_log_weight",
            _finite(self.evidence_log_weight, name="evidence_log_weight"),
        )
        probability = _finite(self.probability, name="probability")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(
            self,
            "expected_trajectory",
            _trajectory(self.expected_trajectory, name="expected_trajectory"),
        )
        divergence = _finite(self.realized_divergence, name="realized_divergence")
        if divergence < 0.0:
            raise ValueError("realized_divergence cannot be negative")
        object.__setattr__(self, "realized_divergence", divergence)
        if not isinstance(self.status, HypothesisStatus):
            raise TypeError("status must be a HypothesisStatus")
        lineage = tuple(str(item) for item in self.lineage)
        if self.hypothesis_id in lineage:
            raise ValueError("a hypothesis may not be its own ancestor")
        object.__setattr__(self, "lineage", lineage)

    @property
    def log_weight(self) -> float:
        """The unnormalized log score the pool ranks and renormalizes."""

        return self.prior_log_weight + self.evidence_log_weight


@dataclass(frozen=True)
class LifecycleRecord:
    """One SPAWN/UPDATE/SPLIT/MERGE/RETIRE the pool performed on this clock."""

    asof: pd.Timestamp
    operation: LifecycleOperation
    hypothesis_ids: tuple[str, ...]
    mode_ids: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="asof"))
        if not isinstance(self.operation, LifecycleOperation):
            raise TypeError("operation must be a LifecycleOperation")
        object.__setattr__(
            self, "hypothesis_ids", tuple(str(item) for item in self.hypothesis_ids)
        )
        object.__setattr__(self, "mode_ids", tuple(str(item) for item in self.mode_ids))
        if not self.hypothesis_ids:
            raise ValueError("a lifecycle record must name at least one hypothesis")
        if not self.reason:
            raise ValueError("a lifecycle record must state why it happened")


@dataclass(frozen=True)
class MarketBeliefState:
    """The Brain's published per-clock forecast.

    ``residual_probability`` is the weight of "none of the live modes".  It is
    never renormalized away, so ``sum(probabilities) + residual == 1`` exactly,
    and a clock with no live hypothesis publishes a residual of one rather than
    an empty, falsely confident belief.
    """

    asof: pd.Timestamp
    hypotheses: tuple[Hypothesis, ...]
    residual_probability: float
    uncertainty: float
    revision_id: str
    lifecycle_records: tuple[LifecycleRecord, ...] = ()
    mode_library_fingerprint: str = ""
    protocol_fingerprint: str = ""
    schema_version: int = FORECAST_SCHEMA_VERSION
    authority: str = FORECAST_AUTHORITY
    protocol_status: str = FORECAST_PROTOCOL_STATUS
    action_authority_ready: bool = False

    # Probabilities are accumulated in log space and renormalized, so the sum
    # is exact only up to floating-point rounding.
    PROBABILITY_TOLERANCE = 1e-9

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="asof"))
        hypotheses = tuple(self.hypotheses)
        if any(not isinstance(item, Hypothesis) for item in hypotheses):
            raise TypeError("hypotheses must be Hypothesis instances")
        if len(hypotheses) > MAX_LIVE_HYPOTHESES:
            raise ValueError(
                f"the Brain keeps at most {MAX_LIVE_HYPOTHESES} live hypotheses, "
                f"got {len(hypotheses)}"
            )
        ids = [item.hypothesis_id for item in hypotheses]
        if len(set(ids)) != len(ids):
            raise ValueError("hypothesis ids must be unique inside one belief")
        modes = [item.mode_id for item in hypotheses]
        if len(set(modes)) != len(modes):
            raise ValueError("one mode may back at most one live hypothesis")
        if any(item.asof != self.asof for item in hypotheses):
            raise ValueError("every live hypothesis must be observed on this clock")
        if any(item.status is not HypothesisStatus.ACTIVE for item in hypotheses):
            raise ValueError("a published belief carries only active hypotheses")
        object.__setattr__(self, "hypotheses", hypotheses)

        residual = _finite(self.residual_probability, name="residual_probability")
        if not 0.0 <= residual <= 1.0:
            raise ValueError("residual_probability must be a probability")
        total = sum(item.probability for item in hypotheses) + residual
        if abs(total - 1.0) > self.PROBABILITY_TOLERANCE:
            raise ValueError(
                "hypothesis probabilities and the residual must sum to one, "
                f"got {total!r}"
            )
        if not hypotheses and residual != 1.0:
            raise ValueError("an empty belief must carry a residual of one")
        object.__setattr__(self, "residual_probability", residual)

        uncertainty = _finite(self.uncertainty, name="uncertainty")
        if not 0.0 <= uncertainty <= 1.0:
            raise ValueError("uncertainty must lie in [0, 1]")
        object.__setattr__(self, "uncertainty", uncertainty)

        if not self.revision_id:
            raise ValueError("revision_id is required")
        records = tuple(self.lifecycle_records)
        if any(not isinstance(item, LifecycleRecord) for item in records):
            raise TypeError("lifecycle_records must be LifecycleRecord instances")
        if any(item.asof != self.asof for item in records):
            raise ValueError("lifecycle records must belong to this clock")
        object.__setattr__(self, "lifecycle_records", records)

        if self.authority != FORECAST_AUTHORITY:
            raise ValueError("a published belief is shadow-only")
        if self.protocol_status != FORECAST_PROTOCOL_STATUS:
            raise ValueError("a published belief is development-unvalidated")
        if self.action_authority_ready:
            raise ValueError("forecast output carries no action authority")

    @property
    def leading(self) -> Hypothesis | None:
        """The most probable live hypothesis, or ``None`` when the pool is empty."""

        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda item: (item.probability, item.hypothesis_id))

    def probability_of(self, mode_id: str) -> float:
        for item in self.hypotheses:
            if item.mode_id == mode_id:
                return item.probability
        return 0.0


def belief_revision_id(
    *,
    asof: pd.Timestamp,
    hypotheses: tuple[Hypothesis, ...],
    residual_probability: float,
    mode_library_fingerprint: str,
    protocol_fingerprint: str,
) -> str:
    """Deterministic identity for one published belief.

    Two runs over the same bars, library and protocol produce the same id, which
    is what makes a replay auditable.
    """

    return _identity(
        FORECAST_SCHEMA_VERSION,
        aware_timestamp(asof, name="asof").isoformat(),
        mode_library_fingerprint,
        protocol_fingerprint,
        f"{float(residual_probability):.12f}",
        *[
            f"{item.hypothesis_id}:{item.mode_id}:{item.age_bars}:"
            f"{item.probability:.12f}:{item.log_weight:.12f}"
            for item in sorted(hypotheses, key=lambda h: h.hypothesis_id)
        ],
    )


def normalized_entropy(probabilities: tuple[float, ...]) -> float:
    """Shannon entropy over the given outcomes, scaled by the widest the Brain
    can be (``MAX_LIVE_HYPOTHESES + 1`` outcomes).

    This is the raw measure. For a published belief use ``belief_uncertainty``,
    which reads the residual correctly.
    """

    weights = [float(value) for value in probabilities if float(value) > 0.0]
    if not weights:
        return 0.0
    entropy = -sum(value * math.log(value) for value in weights)
    ceiling = math.log(MAX_LIVE_HYPOTHESES + 1)
    if ceiling <= 0.0:
        return 0.0
    return min(1.0, max(0.0, entropy / ceiling))


def belief_uncertainty(
    probabilities: tuple[float, ...], residual_probability: float
) -> float:
    """How little the Brain can commit on this clock, in [0, 1].

    The residual is not one outcome — it is "some mode I am not naming", and
    treating it as a single alternative would make total ignorance look like
    certainty: an empty pool carries a residual of one, whose entropy as a lone
    outcome is zero. So the residual mass is spread across every slot the Brain
    is not currently using, which is the most conservative reading available.

    An empty pool therefore scores 1.0, and a pool with one near-certain
    hypothesis scores near 0.0.
    """

    live = tuple(float(value) for value in probabilities)
    residual = float(residual_probability)
    unnamed = MAX_LIVE_HYPOTHESES + 1 - len(live)
    if unnamed <= 0:
        return normalized_entropy(live + (residual,))
    return normalized_entropy(live + tuple(residual / unnamed for _ in range(unnamed)))


__all__ = [
    "FORECAST_AUTHORITY",
    "FORECAST_PROTOCOL_STATUS",
    "FORECAST_SCHEMA_VERSION",
    "Hypothesis",
    "HypothesisProposal",
    "HypothesisStatus",
    "LifecycleOperation",
    "LifecycleRecord",
    "MAX_LIVE_HYPOTHESES",
    "MarketBeliefState",
    "ModeLibrary",
    "TRAJECTORY_COMPONENTS",
    "TRAJECTORY_COMPONENT_HORIZON",
    "TRAJECTORY_DIM",
    "TrajectoryMode",
    "belief_revision_id",
    "belief_uncertainty",
    "normalized_entropy",
]
