"""Brain forecast: the local conditional future contract.

The Brain does not hold a taxonomy of market paths, and it no longer holds a
global library of them either.  At every completed clock it asks a narrower
question: *given a context like this one, what did the next sixty minutes
actually do?*  It retrieves the nearest historical contexts, reads their futures
as a **conditional future cloud**, cuts that cloud into as many pieces as it
actually has, and keeps at most three of the resulting representative
trajectory nodes live.

Identity is path geometry, and it is carried on **two separate channels**.

Fitting the raw cumulative-return curve directly does not work: its first
principal component absorbed 80.7% of the variance, so distance was decided
almost entirely by where the path ended and the representation collapsed into a
quantization of direction. "Fell, came back, rallied" and "rallied straight"
became the same claim, and the difference between them is the informative part.

So a trajectory is described by:

* **Direction** — where it went and how far: the return ladder, the incremental
  excursions, when each extreme was set, and path efficiency.
* **Shape** — what form it took getting there: the curve with the straight line
  to its endpoint removed and scaled to unit RMS, projected onto a globally
  fitted principal basis. Zero at both ends by construction, so it carries no
  destination information at all.

Realized volatility is an attribute and appears on neither channel.

A node also carries **which historical observation points support it**. Nodes
are re-clustered every clock and carry no identity of their own, so the pool
decides whether a claim persisted by how those sets were inherited rather than
by how far a centroid moved: the same coordinates can be produced by completely
different history, and that is a different assertion.

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

# The principal basis the *detrended shape* is projected onto. The raw curve is
# no longer projected at all; see the module docstring for why.
SHAPE_COMPONENT_COUNT = 5

# A path that is already straight has no shape to normalize. Below this RMS the
# detrended residual is rounding noise, and scaling it to unit RMS would amplify
# that noise into a spurious identity.
SHAPE_SCALE_FLOOR = 1e-9

# The Direction channel, in order. Magnitude and the timing of magnitude both
# live here, which is what frees the Shape channel to carry nothing but form.
DIRECTION_FEATURE_NAMES: tuple[str, ...] = (
    "r_5",
    "r_15",
    "r_30",
    "r_60",
    "mfe_0_15",
    "mfe_15_30",
    "mfe_30_60",
    "mae_0_15",
    "mae_15_30",
    "mae_30_60",
    "time_to_mfe",
    "time_to_mae",
    "path_efficiency",
)
DIRECTION_DIM = len(DIRECTION_FEATURE_NAMES)

# One representation vector is the standardized Direction channel followed by
# the shape components.
REPRESENTATION_DIM = DIRECTION_DIM + SHAPE_COMPONENT_COUNT

# The Brain keeps at most this many live hypotheses. A working-set bound, not a
# claim that only three futures exist.
MAX_LIVE_HYPOTHESES = 3

# How many representative futures one clock's cloud may surface. Deliberately
# larger than the working set: extraction breadth and working-set size are
# different limits, and collapsing them makes SPLIT unreachable — a split is
# made of a node the pool has no slot for yet, which cannot exist when the cloud
# is capped at the number of slots.
MAX_CLOUD_NODES = 8

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

    ``components`` is the cluster's **centroid**, not the medoid's coordinates,
    and the split is deliberate. A medoid is a discrete choice, so a small shift
    in the cloud can jump it to a different historical curve even when the
    cluster itself barely moved; associating on that measured a median
    consecutive-clock distance of 3.37 against the centroid's 1.94. The medoid
    is what gets published and checked; the centroid is what gets matched.
    """

    node_id: str
    curve: tuple[float, ...]
    components: tuple[float, ...]
    dispersion: tuple[float, ...]
    mass: float
    member_count: int
    attributes: PathAttributes
    # Which historical observation points support this node. Identity across
    # clocks is decided by how these sets are inherited, not by how far the
    # centroid moved: two nodes can sit in the same place while resting on
    # completely different history, and that is not the same claim.
    member_ids: tuple[int, ...] = ()
    # Spread of the members around the centroid in representation space. Two
    # clouds can share a centroid and be nothing alike — one tight, one a
    # diffuse ring — so a match has to see this too.
    component_spread: float = 0.0

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id is required")
        object.__setattr__(self, "curve", _curve(self.curve, name="curve"))
        object.__setattr__(self, "dispersion", _curve(self.dispersion, name="dispersion"))
        if any(value <= 0.0 for value in self.dispersion):
            raise ValueError("node dispersion must be strictly positive")
        components = tuple(_finite(v, name="components") for v in self.components)
        if len(components) != REPRESENTATION_DIM:
            raise ValueError(
                f"components must carry {REPRESENTATION_DIM} values, "
                f"got {len(components)}"
            )
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "mass", _unit(self.mass, name="mass"))
        if int(self.member_count) <= 0:
            raise ValueError("a node must be supported by at least one neighbour")
        object.__setattr__(self, "member_count", int(self.member_count))
        if not isinstance(self.attributes, PathAttributes):
            raise TypeError("attributes must be PathAttributes")
        ids = tuple(int(value) for value in self.member_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("member_ids must be unique")
        if ids and len(ids) != self.member_count:
            raise ValueError("member_ids and member_count disagree")
        object.__setattr__(self, "member_ids", ids)
        spread = _finite(self.component_spread, name="component_spread")
        if spread < 0.0:
            raise ValueError("component_spread cannot be negative")
        object.__setattr__(self, "component_spread", spread)

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
    # The typical distance between two unrelated futures in this representation.
    # Association gates and ambiguity are expressed as fractions of it, so a
    # threshold stays meaningful across windows fitted in different volatility
    # regimes — and so a distance is compared against a distance.
    component_scale: float = 1.0
    # How far the retrieved neighbours actually were. A full complement of
    # remote analogues is not precedent, and without this the published belief
    # would have no way to say so.
    mean_neighbour_distance: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="asof"))
        scale = _finite(self.component_scale, name="component_scale")
        if scale <= 0.0:
            raise ValueError("component_scale must be positive")
        object.__setattr__(self, "component_scale", scale)
        nodes = tuple(self.nodes)
        if any(not isinstance(node, TrajectoryNode) for node in nodes):
            raise TypeError("nodes must be TrajectoryNode instances")
        if len(nodes) > MAX_CLOUD_NODES:
            raise ValueError(
                f"a cloud may surface at most {MAX_CLOUD_NODES} nodes, "
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
        distance = _finite(self.mean_neighbour_distance, name="mean_neighbour_distance")
        if distance < 0.0:
            raise ValueError("mean_neighbour_distance cannot be negative")
        object.__setattr__(self, "mean_neighbour_distance", distance)

    @property
    def covered_mass(self) -> float:
        return sum(node.mass for node in self.nodes)

    @property
    def residual_mass(self) -> float:
        return max(0.0, 1.0 - self.covered_mass)


@dataclass(frozen=True)
class BeliefUncertainty:
    """Three different things that "uncertain" can mean, kept separate.

    ``mode_ambiguity`` — within the future that *is* covered, how much the live
    hypotheses disagree with each other. Three tightly agreeing claims and three
    wildly opposed ones can carry identical probability spreads and mean
    opposite things.

    ``representation_coverage`` — how much of the local conditional cloud the
    live pool actually explains. This is about the pool's reach, not its
    confidence.

    ``retrieval_confidence`` — whether the present state has enough close
    historical precedent to be talking about at all. A belief can read
    ``H1 = 0.82`` with a residual of 0.05 and still be worthless if it rests on
    twelve distant neighbours; without this term nothing in the output would say
    so. High means well-supported.

    ``combined`` is a convenience scalar, defined so that *low* retrieval
    confidence raises it: being unsupported is a form of not knowing.
    """

    mode_ambiguity: float
    representation_coverage: float
    retrieval_confidence: float

    def __post_init__(self) -> None:
        for name in (
            "mode_ambiguity",
            "representation_coverage",
            "retrieval_confidence",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name=name))

    @property
    def combined(self) -> float:
        """One sortable number; the three components remain authoritative."""

        return (
            self.mode_ambiguity
            + (1.0 - self.representation_coverage)
            + (1.0 - self.retrieval_confidence)
        ) / 3.0

    @property
    def well_supported(self) -> bool:
        """Whether the present state has precedent worth reasoning from."""

        return self.retrieval_confidence >= 0.5


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
    # The weighted share of last clock's supporting samples this claim still
    # rests on. One means the same history; near zero means the geometry
    # survived but the evidence under it was replaced. A claim spawned on this
    # clock reports zero because it inherited nothing — read it together with
    # ``age_bars``, which is zero there and positive for a survivor.
    support_overlap: float = 1.0

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
        object.__setattr__(
            self, "support_overlap", _unit(self.support_overlap, name="support_overlap")
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
    # What the operation was actually decided on. ``support_overlap`` is the
    # weighted share of historical samples carried across; ``dispersion_shift``
    # is how much the cloud's spread around that claim changed. A record that
    # names only a distance cannot distinguish "the same claim, updated" from
    # "a different claim that happens to sit in the same place".
    support_overlap: float = 0.0
    dispersion_shift: float = 0.0

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
        object.__setattr__(
            self, "support_overlap", _unit(self.support_overlap, name="support_overlap")
        )
        shift = _finite(self.dispersion_shift, name="dispersion_shift")
        object.__setattr__(self, "dispersion_shift", shift)


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


def support_overlap(
    left: Sequence[int], right: Sequence[int]
) -> float:
    """How much of one claim's historical support the other still rests on.

    This is the Jaccard index of the two supporting sample sets. It is what
    separates "the same hypothesis, updated" from "a hypothesis that happens to
    sit where the old one did, resting on completely different history" — two
    situations that centroid distance alone reports identically.

    Either side being empty means the question cannot be answered, and the
    answer is zero rather than a defaulted one.
    """

    first = set(int(value) for value in left)
    second = set(int(value) for value in right)
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


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


def mode_ambiguity(
    components: tuple[tuple[float, ...], ...],
    probabilities: tuple[float, ...],
    *,
    scale: float,
) -> float:
    """How much the covered future's claims disagree, in [0, 1].

    This is the expected distance between two futures drawn independently from
    the published claims, saturated against ``scale`` so it stays comparable
    across windows. It is *not* a mean over pairs: normalizing by the pair
    weights would cancel the probabilities out entirely whenever there are only
    two claims, which is exactly the case the weighting exists for. Two
    dominant claims pulling apart is a contested future; one dominant claim and
    a negligible outlier far away is not.

    Mass the claims do not cover contributes nothing here — how much of the
    cloud goes unspoken for is representation coverage's question, not this one.

    Fewer than two claims cannot disagree, and score zero.
    """

    if len(components) < 2:
        return 0.0
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("ambiguity scale must be finite and positive")
    weights = tuple(float(value) for value in probabilities)
    if len(weights) != len(components):
        raise ValueError("one probability is required per claim")
    expected = 0.0
    for index, left in enumerate(components):
        for offset, right in enumerate(components[index + 1 :], start=index + 1):
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
            expected += 2.0 * weights[index] * weights[offset] * distance
    if expected <= 0.0:
        return 0.0
    return min(1.0, max(0.0, expected / (expected + scale)))


def retrieval_confidence(
    *,
    neighbour_count: int,
    mean_distance: float,
    target_count: int,
    distance_scale: float,
) -> float:
    """Whether the present state has enough close precedent to reason from.

    Two independent ways to have none: too few neighbours, or neighbours that
    are nominally the nearest but still far away. Both are needed — a full
    complement of remote analogues is no better supported than a handful of
    close ones — so the two terms multiply rather than average.
    """

    if target_count < 1:
        raise ValueError("target_count must be positive")
    if not math.isfinite(distance_scale) or distance_scale <= 0.0:
        raise ValueError("distance_scale must be finite and positive")
    count_term = min(1.0, max(0, int(neighbour_count)) / float(target_count))
    distance = max(0.0, float(mean_distance))
    proximity_term = distance_scale / (distance_scale + distance)
    return min(1.0, max(0.0, count_term * proximity_term))


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
    "MAX_CLOUD_NODES",
    "MAX_LIVE_HYPOTHESES",
    "MarketBeliefState",
    "DIRECTION_DIM",
    "DIRECTION_FEATURE_NAMES",
    "support_overlap",
    "REPRESENTATION_DIM",
    "SHAPE_COMPONENT_COUNT",
    "SHAPE_SCALE_FLOOR",
    "PathAttributes",
    "TRAJECTORY_CURVE_LENGTH",
    "TrajectoryNode",
    "belief_revision_id",
    "mode_ambiguity",
    "node_identity",
    "retrieval_confidence",
]
