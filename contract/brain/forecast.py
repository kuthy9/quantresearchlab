"""Brain forecast: the local conditional future contract.

The Brain does not hold a taxonomy of market paths, and it no longer holds a
global library of them either.  At every completed clock it asks a narrower
question: *given a context like this one, what did the next sixty minutes
actually do?*  It retrieves the nearest historical contexts, reads their futures
as a **conditional future cloud**, and extracts at most three representative
trajectory nodes carrying meaningful probability mass.

Identity is path geometry.  A trajectory is the ATR-normalized cumulative
return curve over the next sixty minutes; that curve, projected onto a globally
fitted principal basis, is what decides whether two futures are the same claim.
Realized volatility is carried as an attribute and never as an identity
dimension — two paths that arrive in the same place by the same shape are the
same claim regardless of how noisily they got there.

Nothing here accounts for the whole future.  ``residual_probability`` is the
share of the conditional cloud that no live node covers, and it is never
normalized away.

Everything is shadow-only.  A forecast carries no action authority, and
``MarketBeliefState`` refuses to be constructed claiming otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import pandas as pd

from contract.market.primitives import aware_timestamp, content_hash

FORECAST_SCHEMA_VERSION = 2

# One trajectory is the ATR-normalized cumulative return at each of the next
# sixty completed minutes: r_1 .. r_60. The whole curve is the identity input;
# no hand-picked subset of it is.
TRAJECTORY_CURVE_LENGTH = 60

# The principal basis the curve is projected onto. Five components is where the
# curve's shape is captured without the basis starting to fit single paths.
PRINCIPAL_COMPONENT_COUNT = 5

# The Brain keeps at most this many live hypotheses. A working-set bound, not a
# claim that only three futures exist.
MAX_LIVE_HYPOTHESES = 3

# Horizons the path attributes are read at. These describe a trajectory; they
# do not identify it.
ATTRIBUTE_RETURN_HORIZONS: tuple[int, ...] = (5, 15, 30, 60)
ATTRIBUTE_EXCURSION_WINDOWS: tuple[tuple[int, int], ...] = ((0, 15), (15, 30), (30, 60))
ATTRIBUTE_VOLATILITY_HORIZONS: tuple[int, ...] = (30, 60)

FORECAST_AUTHORITY = "shadow_only"
FORECAST_PROTOCOL_STATUS = "development_unvalidated"


class HypothesisStatus(str, Enum):
    """Whether a hypothesis is still competing for the next sixty minutes."""

    ACTIVE = "active"
    RETIRED = "retired"


class LifecycleOperation(str, Enum):
    """The five things one clock's association may do to the working set."""

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


def _curve(values: object, *, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise TypeError(f"{name} must be a sequence of floats")
    vector = tuple(_finite(item, name=name) for item in values)  # type: ignore[union-attr]
    if len(vector) != TRAJECTORY_CURVE_LENGTH:
        raise ValueError(
            f"{name} must carry {TRAJECTORY_CURVE_LENGTH} points, got {len(vector)}"
        )
    return vector


def _unit(value: object, *, name: str) -> float:
    number = _finite(value, name=name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


@dataclass(frozen=True)
class PathAttributes:
    """What a trajectory looks like, once it is already identified.

    Excursions are **incremental**: ``mfe_15_30`` is how much further the path
    ran beyond its first-fifteen-minute high, not the high over the first thirty.
    Nested cumulative excursions restate the same extreme three times and make
    the later windows nearly collinear with the earlier ones.
    """

    r_5: float
    r_15: float
    r_30: float
    r_60: float
    mfe_0_15: float
    mfe_15_30: float
    mfe_30_60: float
    mae_0_15: float
    mae_15_30: float
    mae_30_60: float
    # Fraction of the horizon elapsed when the extreme was set. A rally that
    # tops out at minute five is a different animal from one that tops at
    # minute fifty-five, even with an identical close.
    time_to_mfe: float
    time_to_mae: float
    # Net displacement over distance travelled, in [0, 1]. Near one is a clean
    # directional run; near zero is churn that ended where it started.
    path_efficiency: float
    # Attributes, never identity dimensions.
    rv_30: float
    rv_60: float

    def __post_init__(self) -> None:
        for name in (
            "r_5", "r_15", "r_30", "r_60",
            "mfe_0_15", "mfe_15_30", "mfe_30_60",
            "mae_0_15", "mae_15_30", "mae_30_60",
            "rv_30", "rv_60",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        for name in ("time_to_mfe", "time_to_mae", "path_efficiency"):
            object.__setattr__(self, name, _unit(getattr(self, name), name=name))
        if any(
            getattr(self, name) < 0.0
            for name in ("mfe_0_15", "mfe_15_30", "mfe_30_60", "rv_30", "rv_60")
        ):
            raise ValueError("favorable excursions and volatilities cannot be negative")
        if any(
            getattr(self, name) > 0.0
            for name in ("mae_0_15", "mae_15_30", "mae_30_60")
        ):
            raise ValueError("adverse excursions cannot be positive")

    def as_mapping(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in (
                "r_5", "r_15", "r_30", "r_60",
                "mfe_0_15", "mfe_15_30", "mfe_30_60",
                "mae_0_15", "mae_15_30", "mae_30_60",
                "time_to_mfe", "time_to_mae", "path_efficiency",
                "rv_30", "rv_60",
            )
        }


@dataclass(frozen=True)
class TrajectoryNode:
    """One representative future extracted from this clock's conditional cloud.

    ``curve`` is a medoid — a real observed trajectory — never a centroid
    average, because the average of two opposite futures is a third future that
    never happened.  ``mass`` is the share of the retrieved neighbourhood that
    fell into this node, which is what makes it a probability rather than a
    shape someone liked the look of.
    """

    node_id: str
    curve: tuple[float, ...]
    components: tuple[float, ...]
    dispersion: tuple[float, ...]
    mass: float
    member_count: int
    attributes: PathAttributes

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id is required")
        object.__setattr__(self, "curve", _curve(self.curve, name="curve"))
        object.__setattr__(self, "dispersion", _curve(self.dispersion, name="dispersion"))
        if any(value <= 0.0 for value in self.dispersion):
            raise ValueError("node dispersion must be strictly positive")
        components = tuple(_finite(v, name="components") for v in self.components)
        if len(components) != PRINCIPAL_COMPONENT_COUNT:
            raise ValueError(
                f"components must carry {PRINCIPAL_COMPONENT_COUNT} values, "
                f"got {len(components)}"
            )
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "mass", _unit(self.mass, name="mass"))
        if int(self.member_count) <= 0:
            raise ValueError("a node must be supported by at least one neighbour")
        object.__setattr__(self, "member_count", int(self.member_count))
        if not isinstance(self.attributes, PathAttributes):
            raise TypeError("attributes must be PathAttributes")

    @property
    def terminal_return(self) -> float:
        """Where the node's curve ends, in ATR units."""

        return self.curve[-1]


@dataclass(frozen=True)
class ConditionalCloud:
    """The retrieved futures for one clock, and what was extracted from them.

    ``residual_mass`` is the share of the neighbourhood that no kept node
    covers.  It is measured, not floored into existence: a cloud that three
    nodes genuinely explain reports a small residual, and one that they do not
    reports a large one.
    """

    asof: pd.Timestamp
    neighbour_count: int
    assigned_count: int
    cluster_count: int
    nodes: tuple[TrajectoryNode, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="asof"))
        nodes = tuple(self.nodes)
        if any(not isinstance(node, TrajectoryNode) for node in nodes):
            raise TypeError("nodes must be TrajectoryNode instances")
        if len(nodes) > MAX_LIVE_HYPOTHESES:
            raise ValueError(
                f"a cloud may surface at most {MAX_LIVE_HYPOTHESES} nodes, "
                f"got {len(nodes)}"
            )
        ids = [node.node_id for node in nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("node ids must be unique inside one cloud")
        total = sum(node.mass for node in nodes)
        if total > 1.0 + 1e-9:
            raise ValueError("node masses cannot exceed the whole neighbourhood")
        object.__setattr__(self, "nodes", nodes)
        for name in ("neighbour_count", "assigned_count", "cluster_count"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        if self.assigned_count > self.neighbour_count:
            raise ValueError("more neighbours were assigned than were retrieved")

    @property
    def covered_mass(self) -> float:
        return sum(node.mass for node in self.nodes)

    @property
    def residual_mass(self) -> float:
        return max(0.0, 1.0 - self.covered_mass)


@dataclass(frozen=True)
class BeliefUncertainty:
    """Three separate things that "uncertain" can mean, kept separate.

    ``entropy`` — how evenly the probability is spread over what is named.
    ``distribution_ambiguity`` — how far apart the named claims are from each
    other. Three tightly agreeing hypotheses and three wildly opposed ones can
    carry identical entropy and mean completely different things.
    ``coverage`` — how much of the conditional cloud nothing named covers at all.

    ``combined`` is their mean, offered as a single sortable number. The three
    components are the authoritative reading; the mean is a convenience and
    claims no principled aggregation.
    """

    entropy: float
    distribution_ambiguity: float
    coverage: float

    def __post_init__(self) -> None:
        for name in ("entropy", "distribution_ambiguity", "coverage"):
            object.__setattr__(self, name, _unit(getattr(self, name), name=name))

    @property
    def combined(self) -> float:
        return (self.entropy + self.distribution_ambiguity + self.coverage) / 3.0


@dataclass(frozen=True)
class Hypothesis:
    """One live claim about the next sixty minutes, and how it is holding up.

    A hypothesis persists across clocks by *association*: each clock's freshly
    extracted nodes are matched against the live set, and a matched hypothesis
    keeps its identity, its age and the path it has been judged against.
    ``association_distance`` is how well it matched on this clock, and is the
    raw material for telling a real change from clustering jitter.
    """

    hypothesis_id: str
    node_id: str
    spawned_at: pd.Timestamp
    asof: pd.Timestamp
    age_bars: int
    prior_log_weight: float
    evidence_log_weight: float
    probability: float
    expected_curve: tuple[float, ...]
    realized_divergence: float
    association_distance: float
    attributes: PathAttributes
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not self.node_id:
            raise ValueError("a hypothesis needs an identity and a node")
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
        for name in ("prior_log_weight", "evidence_log_weight"):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        object.__setattr__(self, "probability", _unit(self.probability, name="probability"))
        object.__setattr__(
            self, "expected_curve", _curve(self.expected_curve, name="expected_curve")
        )
        for name in ("realized_divergence", "association_distance"):
            value = _finite(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        if not isinstance(self.attributes, PathAttributes):
            raise TypeError("attributes must be PathAttributes")
        if not isinstance(self.status, HypothesisStatus):
            raise TypeError("status must be a HypothesisStatus")
        lineage = tuple(str(item) for item in self.lineage)
        if self.hypothesis_id in lineage:
            raise ValueError("a hypothesis may not be its own ancestor")
        object.__setattr__(self, "lineage", lineage)

    @property
    def log_weight(self) -> float:
        return self.prior_log_weight + self.evidence_log_weight

    @property
    def terminal_return(self) -> float:
        return self.expected_curve[-1]


@dataclass(frozen=True)
class LifecycleRecord:
    """One SPAWN/UPDATE/SPLIT/MERGE/RETIRE the association produced."""

    asof: pd.Timestamp
    operation: LifecycleOperation
    hypothesis_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    reason: str
    association_distance: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="asof"))
        if not isinstance(self.operation, LifecycleOperation):
            raise TypeError("operation must be a LifecycleOperation")
        object.__setattr__(
            self, "hypothesis_ids", tuple(str(item) for item in self.hypothesis_ids)
        )
        object.__setattr__(self, "node_ids", tuple(str(item) for item in self.node_ids))
        if not self.hypothesis_ids:
            raise ValueError("a lifecycle record must name at least one hypothesis")
        if not self.reason:
            raise ValueError("a lifecycle record must state why it happened")
        distance = _finite(self.association_distance, name="association_distance")
        if distance < 0.0:
            raise ValueError("association_distance cannot be negative")
        object.__setattr__(self, "association_distance", distance)


@dataclass(frozen=True)
class MarketBeliefState:
    """The Brain's published per-clock forecast."""

    asof: pd.Timestamp
    hypotheses: tuple[Hypothesis, ...]
    residual_probability: float
    uncertainty: BeliefUncertainty
    revision_id: str
    cloud: ConditionalCloud | None = None
    lifecycle_records: tuple[LifecycleRecord, ...] = ()
    index_fingerprint: str = ""
    protocol_fingerprint: str = ""
    schema_version: int = FORECAST_SCHEMA_VERSION
    authority: str = FORECAST_AUTHORITY
    protocol_status: str = FORECAST_PROTOCOL_STATUS
    action_authority_ready: bool = False

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
        nodes = [item.node_id for item in hypotheses]
        if len(set(nodes)) != len(nodes):
            raise ValueError("one node may back at most one live hypothesis")
        if any(item.asof != self.asof for item in hypotheses):
            raise ValueError("every live hypothesis must be observed on this clock")
        if any(item.status is not HypothesisStatus.ACTIVE for item in hypotheses):
            raise ValueError("a published belief carries only active hypotheses")
        object.__setattr__(self, "hypotheses", hypotheses)

        residual = _unit(self.residual_probability, name="residual_probability")
        total = sum(item.probability for item in hypotheses) + residual
        if abs(total - 1.0) > self.PROBABILITY_TOLERANCE:
            raise ValueError(
                "hypothesis probabilities and the residual must sum to one, "
                f"got {total!r}"
            )
        if not hypotheses and residual != 1.0:
            raise ValueError("an empty belief must carry a residual of one")
        object.__setattr__(self, "residual_probability", residual)

        if not isinstance(self.uncertainty, BeliefUncertainty):
            raise TypeError("uncertainty must be a BeliefUncertainty")
        if not self.revision_id:
            raise ValueError("revision_id is required")
        if self.cloud is not None:
            if not isinstance(self.cloud, ConditionalCloud):
                raise TypeError("cloud must be a ConditionalCloud")
            if self.cloud.asof != self.asof:
                raise ValueError("the cloud must belong to this clock")
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
        if not self.hypotheses:
            return None
        return max(
            self.hypotheses, key=lambda item: (item.probability, item.hypothesis_id)
        )

    def probability_of(self, node_id: str) -> float:
        for item in self.hypotheses:
            if item.node_id == node_id:
                return item.probability
        return 0.0


def node_identity(curve: tuple[float, ...]) -> str:
    """Content-addressed identity for one representative curve.

    Two clocks that surface the same observed trajectory name it the same way,
    which is what lets association fall back on exact identity when the geometry
    has not moved at all.
    """

    return content_hash([round(float(value), 8) for value in curve])[:32]


def belief_revision_id(
    *,
    asof: pd.Timestamp,
    hypotheses: tuple[Hypothesis, ...],
    residual_probability: float,
    index_fingerprint: str,
    protocol_fingerprint: str,
) -> str:
    """Deterministic identity for one published belief."""

    return content_hash(
        [
            str(FORECAST_SCHEMA_VERSION),
            aware_timestamp(asof, name="asof").isoformat(),
            index_fingerprint,
            protocol_fingerprint,
            f"{float(residual_probability):.12f}",
            *[
                f"{item.hypothesis_id}:{item.node_id}:{item.age_bars}:"
                f"{item.probability:.12f}:{item.log_weight:.12f}"
                for item in sorted(hypotheses, key=lambda h: h.hypothesis_id)
            ],
        ]
    )


def normalized_entropy(probabilities: tuple[float, ...]) -> float:
    """Shannon entropy scaled by the widest the Brain can be."""

    weights = [float(value) for value in probabilities if float(value) > 0.0]
    if not weights:
        return 0.0
    entropy = -sum(value * math.log(value) for value in weights)
    ceiling = math.log(MAX_LIVE_HYPOTHESES + 1)
    if ceiling <= 0.0:
        return 0.0
    return min(1.0, max(0.0, entropy / ceiling))


def entropy_uncertainty(
    probabilities: tuple[float, ...], residual_probability: float
) -> float:
    """Entropy over the named claims plus the residual, read correctly.

    The residual is "some future I am not naming", not one named outcome.  As a
    lone outcome its entropy is zero, so an empty pool — total ignorance — would
    score as perfect confidence.  Spreading the residual across the slots the
    Brain is not using is the most conservative reading available, and makes an
    empty pool score one.
    """

    live = tuple(float(value) for value in probabilities)
    residual = float(residual_probability)
    unnamed = MAX_LIVE_HYPOTHESES + 1 - len(live)
    if unnamed <= 0:
        return normalized_entropy(live + (residual,))
    return normalized_entropy(live + tuple(residual / unnamed for _ in range(unnamed)))


def distribution_ambiguity(
    components: tuple[tuple[float, ...], ...], *, scale: float
) -> float:
    """How far apart the named claims are from one another, in [0, 1].

    Mean pairwise distance in the principal basis, saturated against ``scale``
    (the basis's own spread) so the number stays comparable across windows.
    Fewer than two claims cannot disagree, and score zero.
    """

    if len(components) < 2:
        return 0.0
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("ambiguity scale must be finite and positive")
    distances: list[float] = []
    for index, left in enumerate(components):
        for right in components[index + 1 :]:
            distances.append(
                math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
            )
    mean = sum(distances) / len(distances)
    return min(1.0, max(0.0, mean / (mean + scale)))


__all__ = [
    "ATTRIBUTE_EXCURSION_WINDOWS",
    "ATTRIBUTE_RETURN_HORIZONS",
    "ATTRIBUTE_VOLATILITY_HORIZONS",
    "BeliefUncertainty",
    "ConditionalCloud",
    "FORECAST_AUTHORITY",
    "FORECAST_PROTOCOL_STATUS",
    "FORECAST_SCHEMA_VERSION",
    "Hypothesis",
    "HypothesisStatus",
    "LifecycleOperation",
    "LifecycleRecord",
    "MAX_LIVE_HYPOTHESES",
    "MarketBeliefState",
    "PRINCIPAL_COMPONENT_COUNT",
    "PathAttributes",
    "TRAJECTORY_CURVE_LENGTH",
    "TrajectoryNode",
    "belief_revision_id",
    "distribution_ambiguity",
    "entropy_uncertainty",
    "node_identity",
    "normalized_entropy",
]
