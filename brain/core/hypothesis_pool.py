"""The Brain's working set, maintained by association rather than by a library.

Every clock re-extracts representative futures from the conditional cloud. Those
nodes have no identity of their own across time — the cloud is re-clustered from
scratch — so persistence is established by *matching* this clock's nodes against
the live hypotheses in the principal basis.

The five lifecycle operations fall out of that matching rather than being
separate rules:

``UPDATE``  one live hypothesis matched one node *and still rests on the same
            historical support*: it keeps its identity, its age and the path it
            has been judged against
``SPAWN``   a node matched nothing live, and inherited no live claim's support
``RETIRE``  a live hypothesis matched no node, or was out-competed on probability
``SPLIT``   one live hypothesis's supporting samples divided between two
            geometrically separated nodes
``MERGE``   two live hypotheses' supports converged onto one node that makes no
            distinguishable claim from either

Matching uses the Hungarian assignment, which is deterministic and globally
optimal. A greedy nearest-first pass would make the operation depend on
iteration order, and the whole point of recording SPLIT and MERGE is to tell a
real change from an artefact.

**Identity is decided by support, not by proximity.** A cloud's nodes are
re-clustered from scratch every minute, so "the centroid is still nearby" is a
weak claim: the same coordinates can be produced by a completely different set
of historical samples, and that is a different assertion about the market
wearing the previous one's clothes. Every node therefore carries the row
indices of the observation points that support it, and the lifecycle reads
those sets:

* a match whose support has been replaced is not an update, it is a retirement
  and a spawn that happen to coincide in space;
* a split fires when one claim's support *divides* between two nodes that are
  far enough apart to be separate claims — not when a second node merely turns
  up nearby;
* a merge, its exact dual, fires when two claims' supports converge on one node
  and the information gap between them has closed.

Distance still matters, and it is measured on the centroid **and the spread
together**. Two clouds can share a centroid and be nothing alike — one a tight
knot, the other a diffuse ring — so the spread enters the metric as one more
coordinate rather than being ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from contract.brain.forecast import (
    MAX_LIVE_HYPOTHESES,
    ConditionalCloud,
    Hypothesis,
    HypothesisStatus,
    LifecycleOperation,
    LifecycleRecord,
    TrajectoryNode,
    support_overlap,
)
from contract.market import content_hash

from .belief_updater import BeliefUpdaterConfig, evaluate, normalize_log_weights
from .trajectory import RealizedPath


class HypothesisPoolError(RuntimeError):
    """The pool refuses to advance on inputs it cannot trust."""


@dataclass(frozen=True)
class PoolConfig:
    """Every threshold that decides an association outcome."""

    max_live_hypotheses: int = MAX_LIVE_HYPOTHESES
    association_max_distance_scale: float = 0.5
    # Below this share of carried-over supporting samples, a matched pair is not
    # the same claim however close the centroids are.
    identity_minimum_overlap: float = 0.10
    # A split needs one claim's support genuinely divided: each side must take
    # at least this share of it.
    split_minimum_inheritance: float = 0.20
    # A merge needs the other claim's support to have flowed into the surviving
    # node, and the two claims to have stopped being distinguishable.
    merge_minimum_inheritance: float = 0.20
    merge_information_floor: float = 0.35
    retire_minimum_probability: float = 0.05
    retire_maximum_age_bars: int = 60
    falsification_divergence: float = 3.0
    falsification_minimum_age_bars: int = 5
    residual_floor: float = 0.05
    prior_floor: float = 0.001

    def __post_init__(self) -> None:
        if not 1 <= self.max_live_hypotheses <= MAX_LIVE_HYPOTHESES:
            raise ValueError(
                f"max_live_hypotheses must lie in [1, {MAX_LIVE_HYPOTHESES}]"
            )
        if self.association_max_distance_scale <= 0.0:
            raise ValueError("association_max_distance_scale must be positive")
        for name in (
            "identity_minimum_overlap",
            "split_minimum_inheritance",
            "merge_minimum_inheritance",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.merge_information_floor < 0.0:
            raise ValueError("merge_information_floor cannot be negative")
        if self.retire_maximum_age_bars < 1:
            raise ValueError("retire_maximum_age_bars must be positive")
        if self.falsification_divergence <= 0.0:
            raise ValueError("falsification_divergence must be positive")
        if self.falsification_minimum_age_bars < 1:
            raise ValueError("falsification_minimum_age_bars must be positive")
        if not 0.0 <= self.residual_floor < 1.0:
            raise ValueError("residual_floor must lie in [0, 1)")
        if not 0.0 < self.prior_floor < 1.0:
            raise ValueError("prior_floor must lie in (0, 1)")

    @classmethod
    def from_protocol(cls, payload: Mapping[str, Any]) -> "PoolConfig":
        section = payload.get("pool", {})
        return cls(
            max_live_hypotheses=int(section["max_live_hypotheses"]),
            association_max_distance_scale=float(
                section["association_max_distance_scale"]
            ),
            identity_minimum_overlap=float(section["identity_minimum_overlap"]),
            split_minimum_inheritance=float(section["split_minimum_inheritance"]),
            merge_minimum_inheritance=float(section["merge_minimum_inheritance"]),
            merge_information_floor=float(section["merge_information_floor"]),
            retire_minimum_probability=float(section["retire_minimum_probability"]),
            retire_maximum_age_bars=int(section["retire_maximum_age_bars"]),
            falsification_divergence=float(section["falsification_divergence"]),
            falsification_minimum_age_bars=int(
                section["falsification_minimum_age_bars"]
            ),
            residual_floor=float(section["residual_floor"]),
            prior_floor=float(section["prior_floor"]),
        )


@dataclass(frozen=True)
class PoolMember:
    """One live hypothesis: what it claims, and the path judging it."""

    hypothesis_id: str
    node: TrajectoryNode
    spawned_at: pd.Timestamp
    mass: float
    path: RealizedPath
    association_distance: float = 0.0
    lineage: tuple[str, ...] = ()
    # How much of last clock's support this claim still rests on, and how much
    # the cloud's spread around it moved. Both are published.
    support_overlap: float = 1.0
    dispersion_shift: float = 0.0

    @property
    def age(self) -> int:
        return self.path.age


@dataclass(frozen=True)
class PoolAdvance:
    """Everything one clock produced, ready for the forecast to publish."""

    asof: pd.Timestamp
    members: tuple[PoolMember, ...]
    hypotheses: tuple[Hypothesis, ...]
    residual_probability: float
    records: tuple[LifecycleRecord, ...]
    cloud: ConditionalCloud


def _hypothesis_id(node_id: str, spawned_at: pd.Timestamp, lineage: Sequence[str]) -> str:
    return content_hash(
        [node_id, pd.Timestamp(spawned_at).isoformat(), list(lineage)]
    )[:32]


def information_gap(left: TrajectoryNode, right: TrajectoryNode) -> float:
    """How distinguishable two claims are as predictions.

    The two curves are separated in units of their own pooled dispersion, so the
    answer asks the right question: not "are these numbers different" but "does
    keeping these apart say anything the cloud can actually tell apart". Two
    claims whose separation is small against the spread of futures each already
    covers are one claim written twice, and the merge is the operation that says
    so.
    """

    first = np.asarray(left.curve, dtype=float)
    second = np.asarray(right.curve, dtype=float)
    pooled = np.sqrt(
        (
            np.asarray(left.dispersion, dtype=float) ** 2
            + np.asarray(right.dispersion, dtype=float) ** 2
        )
        / 2.0
    )
    return float(np.sqrt(np.mean(((first - second) / pooled) ** 2)))


def _inheritance(parent: Sequence[int], child: Sequence[int]) -> float:
    """The share of a parent claim's support that a node has taken over."""

    supporting = set(int(value) for value in parent)
    if not supporting:
        return 0.0
    return len(supporting & set(int(value) for value in child)) / len(supporting)


def _assign(cost: np.ndarray, gate: float) -> tuple[dict[int, int], set[int], set[int]]:
    """Globally optimal one-to-one matching under a distance gate."""

    from scipy.optimize import linear_sum_assignment

    live, node_count = cost.shape
    if cost.size == 0:
        return {}, set(range(live)), set(range(node_count))
    rows, columns = linear_sum_assignment(cost)
    matched = {
        int(row): int(column)
        for row, column in zip(rows, columns)
        if cost[row, column] <= gate
    }
    return (
        matched,
        set(range(live)) - set(matched),
        set(range(node_count)) - set(matched.values()),
    )


class HypothesisPool:
    """Advances the working set by one completed bar."""

    def __init__(
        self,
        *,
        config: PoolConfig | None = None,
        updater_config: BeliefUpdaterConfig | None = None,
    ) -> None:
        self.config = config or PoolConfig()
        self.updater_config = updater_config or BeliefUpdaterConfig()
        self._members: tuple[PoolMember, ...] = ()
        self._asof: pd.Timestamp | None = None

    @property
    def members(self) -> tuple[PoolMember, ...]:
        return self._members

    @property
    def asof(self) -> pd.Timestamp | None:
        return self._asof

    def reset(self) -> None:
        self._members = ()
        self._asof = None

    def advance(
        self,
        *,
        asof: pd.Timestamp,
        close: float,
        high: float,
        low: float,
        atr: float,
        cloud: ConditionalCloud,
    ) -> PoolAdvance:
        """Run one clock: extend, score, retire, associate, normalize."""

        asof = pd.Timestamp(asof)
        if asof.tzinfo is None:
            raise HypothesisPoolError("the pool advances on timezone-aware clocks only")
        if self._asof is not None and asof <= self._asof:
            raise HypothesisPoolError(
                "the pool received a duplicate or out-of-order clock"
            )
        if not math.isfinite(atr) or atr <= 0.0:
            raise HypothesisPoolError("a live ATR is required to anchor a hypothesis")
        if cloud.asof != asof:
            raise HypothesisPoolError("the cloud belongs to a different clock")

        records: list[LifecycleRecord] = []

        # 1. Extend every live path with the completed bar.
        members = [
            replace(member, path=member.path.extend(close=close, high=high, low=low))
            for member in self._members
        ]

        # 2. Score every survivor against its realized path.
        scored = {
            member.hypothesis_id: evaluate(
                member.node.curve,
                member.node.dispersion,
                member.path,
                self.updater_config,
            )
            for member in members
        }
        # Rescoring is not a lifecycle event: it happens to every survivor on
        # every clock, and recording it would make the UPDATE count mean
        # "clocks elapsed" rather than "claims that kept their identity".

        # 3. RETIRE on expiry or falsification, before anything is matched.
        kept: list[PoolMember] = []
        expired: list[PoolMember] = []
        falsified: list[PoolMember] = []
        for member in members:
            if member.age >= self.config.retire_maximum_age_bars:
                expired.append(member)
            elif (
                # One minute of tape cannot refute a claim about an hour.
                member.age >= self.config.falsification_minimum_age_bars
                and scored[member.hypothesis_id].divergence
                > self.config.falsification_divergence
            ):
                falsified.append(member)
            else:
                kept.append(member)
        for group, reason in (
            (expired, "horizon elapsed"),
            (falsified, "path falsified the claimed curve"),
        ):
            if group:
                records.append(
                    LifecycleRecord(
                        asof=asof,
                        operation=LifecycleOperation.RETIRE,
                        hypothesis_ids=tuple(m.hypothesis_id for m in group),
                        node_ids=tuple(m.node.node_id for m in group),
                        reason=reason,
                    )
                )
        members = kept

        # 4. Associate this clock's nodes with the survivors.
        members, association_records = self._associate(members, cloud, asof, close, atr)
        records.extend(association_records)

        # 5. Normalize, then drop whatever the competition left below the floor.
        hypotheses, residual, low = self._publish(members, asof)
        if low:
            records.append(
                LifecycleRecord(
                    asof=asof,
                    operation=LifecycleOperation.RETIRE,
                    hypothesis_ids=tuple(m.hypothesis_id for m in low),
                    node_ids=tuple(m.node.node_id for m in low),
                    reason="probability fell below the retirement floor",
                )
            )
            dropped = {m.hypothesis_id for m in low}
            members = [m for m in members if m.hypothesis_id not in dropped]
            hypotheses, residual, _ = self._publish(members, asof, prune=False)

        self._members = tuple(members)
        self._asof = asof
        return PoolAdvance(
            asof=asof,
            members=self._members,
            hypotheses=hypotheses,
            residual_probability=residual,
            records=tuple(records),
            cloud=cloud,
        )

    # -- association ----------------------------------------------------------

    def _associate(
        self,
        members: Sequence[PoolMember],
        cloud: ConditionalCloud,
        asof: pd.Timestamp,
        close: float,
        atr: float,
    ) -> tuple[list[PoolMember], list[LifecycleRecord]]:
        """Match this clock's nodes to the live set and read off the operations."""

        members = list(members)
        records: list[LifecycleRecord] = []
        nodes = list(cloud.nodes)

        if not nodes:
            if members:
                records.append(
                    LifecycleRecord(
                        asof=asof,
                        operation=LifecycleOperation.RETIRE,
                        hypothesis_ids=tuple(m.hypothesis_id for m in members),
                        node_ids=tuple(m.node.node_id for m in members),
                        reason="the conditional cloud surfaced no representative future",
                    )
                )
            return [], records

        # Distance is measured on the centroid *and* the spread together. A
        # node that sits where a live claim sits but covers a far wider band of
        # futures is not that claim, and a metric that reads only the centre
        # cannot see the difference.
        cost = np.zeros((len(members), len(nodes)), dtype=float)
        for row, member in enumerate(members):
            left = np.asarray(member.node.components, dtype=float)
            for column, node in enumerate(nodes):
                centre = float(
                    np.linalg.norm(left - np.asarray(node.components, dtype=float))
                )
                spread = node.component_spread - member.node.component_spread
                cost[row, column] = math.hypot(centre, spread)
        # The gate is a fraction of the representation's own spread, not an
        # absolute distance: coordinates scale with the window's volatility, so
        # a fixed number would mean something different in every regime.
        gate = self.config.association_max_distance_scale * cloud.component_scale
        matched, unmatched_live, unmatched_nodes = _assign(cost, gate)

        survivors: list[PoolMember] = []
        # A geometric match whose support has been replaced is not the same
        # claim; it is returned to the unmatched pool so the node can spawn on
        # its own terms and the stale claim can retire on its own.
        impostors: list[int] = []
        for row, column in sorted(matched.items()):
            member, node = members[row], nodes[column]
            overlap = support_overlap(member.node.member_ids, node.member_ids)
            shift = node.component_spread - member.node.component_spread
            if (
                member.node.member_ids
                and node.member_ids
                and overlap < self.config.identity_minimum_overlap
            ):
                impostors.append(row)
                continue
            survivors.append(
                replace(
                    member,
                    node=node,
                    mass=node.mass,
                    association_distance=float(cost[row, column]),
                    support_overlap=overlap,
                    dispersion_shift=float(shift),
                )
            )
            records.append(
                LifecycleRecord(
                    asof=asof,
                    operation=LifecycleOperation.UPDATE,
                    hypothesis_ids=(member.hypothesis_id,),
                    node_ids=(node.node_id,),
                    reason="matched this clock's node on geometry and support",
                    association_distance=float(cost[row, column]),
                    support_overlap=overlap,
                    dispersion_shift=float(shift),
                )
            )
        for row in impostors:
            node = nodes[matched[row]]
            unmatched_nodes.add(matched[row])
            records.append(
                LifecycleRecord(
                    asof=asof,
                    operation=LifecycleOperation.RETIRE,
                    hypothesis_ids=(members[row].hypothesis_id,),
                    node_ids=(members[row].node.node_id,),
                    reason="the samples supporting this claim have been replaced",
                    association_distance=float(cost[row, matched[row]]),
                    support_overlap=support_overlap(
                        members[row].node.member_ids, node.member_ids
                    ),
                )
            )
        claimed = {row: column for row, column in matched.items() if row not in impostors}

        # An unmatched node is a SPLIT when it has taken over a real share of a
        # live claim's support *and* that claim's own node kept a real share
        # too — the support divided — and the two nodes are far enough apart to
        # be separate claims. Anything else is a SPAWN: a future the pool was
        # not carrying.
        #
        # Highest mass first, because a full pool can only admit a node by
        # displacing one, and the strongest candidate should get that chance.
        for column in sorted(
            unmatched_nodes, key=lambda index: (-nodes[index].mass, index)
        ):
            node = nodes[column]
            parent_row, inherited = -1, 0.0
            for row, member in enumerate(members):
                share = _inheritance(member.node.member_ids, node.member_ids)
                if share > inherited:
                    parent_row, inherited = row, share
            sibling_share, separation = 0.0, 0.0
            if parent_row in claimed:
                sibling = nodes[claimed[parent_row]]
                sibling_share = _inheritance(
                    members[parent_row].node.member_ids, sibling.member_ids
                )
                separation = float(
                    np.linalg.norm(
                        np.asarray(node.components, dtype=float)
                        - np.asarray(sibling.components, dtype=float)
                    )
                )
            is_split = (
                parent_row in claimed
                and inherited >= self.config.split_minimum_inheritance
                and sibling_share >= self.config.split_minimum_inheritance
                and separation > gate
            )
            parent_id = members[parent_row].hypothesis_id if is_split else None
            # A full pool may still admit this node by displacing its weakest
            # rival — a split cannot fire otherwise, because the Hungarian
            # assignment fills every slot before any node is left over. The
            # parent of a split is never the one displaced.
            evicted: PoolMember | None = None
            if len(survivors) >= self.config.max_live_hypotheses:
                candidates = [
                    survivor
                    for survivor in survivors
                    if survivor.hypothesis_id != parent_id
                    and survivor.mass < node.mass
                ]
                if not candidates:
                    continue
                evicted = min(candidates, key=lambda m: (m.mass, m.hypothesis_id))
            spawned = PoolMember(
                hypothesis_id=_hypothesis_id(node.node_id, asof, ()),
                node=node,
                spawned_at=asof,
                mass=node.mass,
                # A new claim is anchored at the bar that surfaced it, so its
                # path starts empty and its ATR scale is this clock's.
                path=RealizedPath(anchor_price=float(close), anchor_atr=float(atr)),
                association_distance=separation if is_split else 0.0,
                lineage=(parent_id,) if is_split else (),
                support_overlap=inherited if is_split else 0.0,
            )
            if evicted is not None:
                survivors = [
                    survivor
                    for survivor in survivors
                    if survivor.hypothesis_id != evicted.hypothesis_id
                ]
                records.append(
                    LifecycleRecord(
                        asof=asof,
                        operation=LifecycleOperation.RETIRE,
                        hypothesis_ids=(evicted.hypothesis_id,),
                        node_ids=(evicted.node.node_id,),
                        reason=(
                            f"displaced by a node carrying mass {node.mass:.3f}"
                        ),
                    )
                )
            survivors.append(spawned)
            records.append(
                LifecycleRecord(
                    asof=asof,
                    operation=(
                        LifecycleOperation.SPLIT if is_split else LifecycleOperation.SPAWN
                    ),
                    hypothesis_ids=(spawned.hypothesis_id,),
                    node_ids=(node.node_id,),
                    reason=(
                        f"a live claim's support divided: this side took "
                        f"{inherited:.2f} of it, the other {sibling_share:.2f}"
                        if is_split
                        else f"a new representative future carries mass {node.mass:.3f}"
                    ),
                    association_distance=spawned.association_distance,
                    support_overlap=spawned.support_overlap,
                )
            )

        # The exact dual: a live hypothesis whose support has flowed into a node
        # another survivor already holds, and whose claim is no longer
        # distinguishable from that survivor's, has merged.
        # Impostors are not eligible: they were retired precisely because the
        # support under them was replaced, and that is the opposite of two
        # claims converging.
        for row in sorted(unmatched_live):
            member = members[row]
            nearest = int(np.argmin(cost[row])) if cost.shape[1] else -1
            absorber = next(
                (
                    survivor
                    for survivor in survivors
                    if nearest >= 0 and survivor.node.node_id == nodes[nearest].node_id
                ),
                None,
            )
            inherited = (
                _inheritance(member.node.member_ids, nodes[nearest].member_ids)
                if nearest >= 0
                else 0.0
            )
            gap = (
                information_gap(member.node, absorber.node)
                if absorber is not None
                else float("inf")
            )
            if (
                absorber is not None
                and inherited >= self.config.merge_minimum_inheritance
                and gap <= self.config.merge_information_floor
            ):
                records.append(
                    LifecycleRecord(
                        asof=asof,
                        operation=LifecycleOperation.MERGE,
                        hypothesis_ids=(member.hypothesis_id, absorber.hypothesis_id),
                        node_ids=(nodes[nearest].node_id,),
                        reason=(
                            f"supports converged ({inherited:.2f} inherited) and the "
                            f"information gap closed to {gap:.2f}"
                        ),
                        association_distance=float(cost[row, nearest]),
                        support_overlap=inherited,
                    )
                )
                survivors = [
                    replace(
                        survivor,
                        lineage=tuple(
                            dict.fromkeys((*survivor.lineage, member.hypothesis_id))
                        ),
                    )
                    if survivor.hypothesis_id == absorber.hypothesis_id
                    else survivor
                    for survivor in survivors
                ]
            else:
                records.append(
                    LifecycleRecord(
                        asof=asof,
                        operation=LifecycleOperation.RETIRE,
                        hypothesis_ids=(member.hypothesis_id,),
                        node_ids=(member.node.node_id,),
                        reason="no node in this clock's cloud is within the gate",
                        association_distance=(
                            float(cost[row].min()) if cost.shape[1] else 0.0
                        ),
                    )
                )
        return survivors, records

    # -- publication ----------------------------------------------------------

    def _publish(
        self,
        members: Sequence[PoolMember],
        asof: pd.Timestamp,
        *,
        prune: bool = True,
    ) -> tuple[tuple[Hypothesis, ...], float, list[PoolMember]]:
        """Normalize the working set against the residual and build the output."""

        members = list(members)
        if not members:
            return (), 1.0, []

        # The residual carries the share of the conditional cloud that no live
        # claim covers. It is measured from the cloud, not asserted, and floored
        # so it can never be argued away entirely.
        claimed = sum(member.mass for member in members)
        residual_prior = max(self.config.residual_floor, 1.0 - claimed)

        evidences = []
        log_weights = []
        for member in members:
            evidence = evaluate(
                member.node.curve,
                member.node.dispersion,
                member.path,
                self.updater_config,
            )
            evidences.append(evidence)
            log_weights.append(
                math.log(max(member.mass, self.config.prior_floor))
                + evidence.evidence_log_weight
            )
        probabilities, residual = normalize_log_weights(
            tuple(log_weights), residual_log_weight=math.log(residual_prior)
        )

        if prune:
            low = [
                member
                for member, probability in zip(members, probabilities)
                if probability < self.config.retire_minimum_probability
            ]
            if low:
                return (), residual, low

        hypotheses = tuple(
            Hypothesis(
                hypothesis_id=member.hypothesis_id,
                node_id=member.node.node_id,
                spawned_at=member.spawned_at,
                asof=asof,
                age_bars=member.age,
                prior_log_weight=math.log(max(member.mass, self.config.prior_floor)),
                evidence_log_weight=evidence.evidence_log_weight,
                probability=probability,
                expected_curve=member.node.curve,
                realized_divergence=evidence.divergence,
                association_distance=member.association_distance,
                attributes=member.node.attributes,
                status=HypothesisStatus.ACTIVE,
                lineage=member.lineage,
                support_overlap=member.support_overlap,
            )
            for member, evidence, probability in zip(members, evidences, probabilities)
        )
        return hypotheses, residual, []


__all__ = [
    "HypothesisPool",
    "information_gap",
    "HypothesisPoolError",
    "PoolAdvance",
    "PoolConfig",
    "PoolMember",
]
