"""The Brain's working set of live hypotheses.

At most three modes compete at any minute, plus a residual that is never
normalized away.  The pool owns the five lifecycle operations:

``SPAWN``   a context-supported mode enters the working set
``UPDATE``  a live hypothesis is re-scored against its realized path
``SPLIT``   a hypothesis sitting between two child modes becomes both
``MERGE``   two hypotheses whose remaining futures have converged become one
``RETIRE``  a hypothesis is falsified, expired, or out-competed

Three is a working-set bound, not a claim about the market: the mode library may
hold any number of modes, and which three are live changes minute to minute.
The residual keeps the pool honest — the Brain is never forced to explain the
whole future with the modes it happens to be holding, and a pool with nothing
live publishes a residual of one.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import math
from typing import Any, Mapping, Sequence

import pandas as pd

from contract.brain.forecast import (
    MAX_LIVE_HYPOTHESES,
    TRAJECTORY_COMPONENT_HORIZON,
    TRAJECTORY_DIM,
    Hypothesis,
    HypothesisProposal,
    HypothesisStatus,
    LifecycleOperation,
    LifecycleRecord,
    ModeLibrary,
    TrajectoryMode,
)
from contract.market import content_hash

from .belief_updater import (
    BeliefUpdaterConfig,
    RealizedPath,
    evaluate,
    normalize_log_weights,
)


class HypothesisPoolError(RuntimeError):
    """The pool refuses to advance on inputs it cannot trust."""


@dataclass(frozen=True)
class PoolConfig:
    """Every threshold that decides a lifecycle transition."""

    max_live_hypotheses: int = MAX_LIVE_HYPOTHESES
    spawn_minimum_prior: float = 0.12
    spawn_displacement_margin: float = 0.05
    retire_minimum_probability: float = 0.05
    retire_maximum_age_bars: int = 60
    falsification_divergence: float = 3.0
    falsification_minimum_age_bars: int = 5
    merge_maximum_distance: float = 0.75
    split_minimum_age_bars: int = 5
    split_maximum_imbalance: float = 0.2
    residual_floor: float = 0.05
    prior_decay_per_bar: float = 0.9
    prior_floor: float = 0.001

    def __post_init__(self) -> None:
        if not 1 <= self.max_live_hypotheses <= MAX_LIVE_HYPOTHESES:
            raise ValueError(
                f"max_live_hypotheses must lie in [1, {MAX_LIVE_HYPOTHESES}]"
            )
        if self.retire_maximum_age_bars < 1:
            raise ValueError("retire_maximum_age_bars must be positive")
        if self.falsification_divergence <= 0.0:
            raise ValueError("falsification_divergence must be positive")
        if self.falsification_minimum_age_bars < 1:
            raise ValueError("falsification_minimum_age_bars must be positive")
        if not 0.0 < self.prior_decay_per_bar <= 1.0:
            raise ValueError("prior_decay_per_bar must lie in (0, 1]")
        if not 0.0 < self.prior_floor < 1.0:
            raise ValueError("prior_floor must lie in (0, 1)")
        if not 0.0 <= self.residual_floor < 1.0:
            raise ValueError("residual_floor must lie in [0, 1)")

    @classmethod
    def from_protocol(cls, payload: Mapping[str, Any]) -> "PoolConfig":
        section = payload.get("pool", {})
        return cls(
            max_live_hypotheses=int(section["max_live_hypotheses"]),
            spawn_minimum_prior=float(section["spawn_minimum_prior"]),
            spawn_displacement_margin=float(section["spawn_displacement_margin"]),
            retire_minimum_probability=float(section["retire_minimum_probability"]),
            retire_maximum_age_bars=int(section["retire_maximum_age_bars"]),
            falsification_divergence=float(section["falsification_divergence"]),
            falsification_minimum_age_bars=int(section["falsification_minimum_age_bars"]),
            merge_maximum_distance=float(section["merge_maximum_distance"]),
            split_minimum_age_bars=int(section["split_minimum_age_bars"]),
            split_maximum_imbalance=float(section["split_maximum_imbalance"]),
            residual_floor=float(section["residual_floor"]),
            prior_decay_per_bar=float(section["prior_decay_per_bar"]),
            prior_floor=float(section["prior_floor"]),
        )


@dataclass(frozen=True)
class PoolMember:
    """One live hypothesis and the path it has been judged against."""

    hypothesis_id: str
    mode_id: str
    spawned_at: pd.Timestamp
    prior: float
    path: RealizedPath
    lineage: tuple[str, ...] = ()

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


def _hypothesis_id(mode_id: str, spawned_at: pd.Timestamp, lineage: Sequence[str]) -> str:
    return content_hash(
        [mode_id, pd.Timestamp(spawned_at).isoformat(), list(lineage)]
    )[:32]


def _undecided_indices(age: int) -> tuple[int, ...]:
    """Trajectory components whose horizon has not elapsed yet."""

    return tuple(
        index
        for index in range(TRAJECTORY_DIM)
        if TRAJECTORY_COMPONENT_HORIZON[index] > age
    )


def _remaining_distance(left: TrajectoryMode, right: TrajectoryMode, age: int) -> float:
    """Dispersion-scaled distance between what two modes still claim.

    Only the undecided components count.  Two modes that disagreed about the
    first fifteen minutes but agree about the next forty-five have converged as
    far as anything still to come is concerned, and that is what merging means.
    """

    indices = _undecided_indices(age)
    if not indices:
        return 0.0
    total = 0.0
    for index in indices:
        scale = max(left.dispersion[index], right.dispersion[index], 1e-6)
        total += ((left.medoid[index] - right.medoid[index]) / scale) ** 2
    return math.sqrt(total / len(indices))


def _path_distance(mode: TrajectoryMode, path: RealizedPath) -> float:
    """Dispersion-scaled distance between a mode and the path realized so far."""

    return evaluate(mode, path).divergence


class HypothesisPool:
    """Advances the working set by one completed bar."""

    def __init__(
        self,
        *,
        library: ModeLibrary,
        config: PoolConfig | None = None,
        updater_config: BeliefUpdaterConfig | None = None,
    ) -> None:
        self.library = library
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
        proposals: Sequence[HypothesisProposal] = (),
    ) -> PoolAdvance:
        """Run one clock: extend, score, retire, merge, split, spawn, normalize."""

        asof = pd.Timestamp(asof)
        if asof.tzinfo is None:
            raise HypothesisPoolError("the pool advances on timezone-aware clocks only")
        if self._asof is not None and asof <= self._asof:
            raise HypothesisPoolError(
                "the pool received a duplicate or out-of-order clock"
            )
        if not math.isfinite(atr) or atr <= 0.0:
            raise HypothesisPoolError("a live ATR is required to anchor a hypothesis")

        records: list[LifecycleRecord] = []
        priors = {p.mode_id: p.prior for p in proposals}

        # 1. Extend every live path with the completed bar, and refresh the
        #    context prior: a mode the current context no longer retrieves keeps
        #    its claim but loses standing, rather than vanishing outright.
        members = []
        for member in self._members:
            refreshed = priors.get(
                member.mode_id,
                max(member.prior * self.config.prior_decay_per_bar, self.config.prior_floor),
            )
            members.append(
                replace(
                    member,
                    prior=refreshed,
                    path=member.path.extend(close=close, high=high, low=low),
                )
            )

        # 2. Score every survivor against its realized path.
        scored = {m.hypothesis_id: evaluate(self._mode(m), m.path, self.updater_config)
                  for m in members}
        if members:
            records.append(
                LifecycleRecord(
                    asof=asof,
                    operation=LifecycleOperation.UPDATE,
                    hypothesis_ids=tuple(m.hypothesis_id for m in members),
                    mode_ids=tuple(m.mode_id for m in members),
                    reason="rescored against the realized path",
                )
            )

        # 3. RETIRE on falsification or expiry, before anything competes.
        kept: list[PoolMember] = []
        expired: list[PoolMember] = []
        falsified: list[PoolMember] = []
        for member in members:
            if member.age >= self.config.retire_maximum_age_bars:
                expired.append(member)
            elif (
                # A claim about the next hour is not refuted by its first
                # minute: below this age the divergence rests on one or two
                # evidence terms and is mostly noise.
                member.age >= self.config.falsification_minimum_age_bars
                and scored[member.hypothesis_id].divergence
                > self.config.falsification_divergence
            ):
                falsified.append(member)
            else:
                kept.append(member)
        for group, reason in ((expired, "horizon elapsed"), (falsified, "path falsified the mode")):
            if group:
                records.append(
                    LifecycleRecord(
                        asof=asof,
                        operation=LifecycleOperation.RETIRE,
                        hypothesis_ids=tuple(m.hypothesis_id for m in group),
                        mode_ids=tuple(m.mode_id for m in group),
                        reason=reason,
                    )
                )
        members = kept

        # 4. MERGE converged siblings into their common parent.
        members, merge_records = self._merge(members, asof)
        records.extend(merge_records)

        # 5. SPLIT a hypothesis that sits between two children of its own mode.
        members, split_records = self._split(members, asof)
        records.extend(split_records)

        # 6. SPAWN from the context proposals into free (or out-competed) slots.
        members, spawn_records = self._spawn(members, asof, proposals, close, atr)
        records.extend(spawn_records)

        # 7. Normalize over the survivors plus the residual.
        hypotheses, residual, low_probability = self._publish(members, asof, scored)

        # 8. RETIRE the out-competed, then renormalize what is left.
        if low_probability:
            records.append(
                LifecycleRecord(
                    asof=asof,
                    operation=LifecycleOperation.RETIRE,
                    hypothesis_ids=tuple(m.hypothesis_id for m in low_probability),
                    mode_ids=tuple(m.mode_id for m in low_probability),
                    reason="probability fell below the retirement floor",
                )
            )
            dropped = {m.hypothesis_id for m in low_probability}
            members = tuple(m for m in members if m.hypothesis_id not in dropped)
            hypotheses, residual, _ = self._publish(members, asof, scored, prune=False)

        self._members = tuple(members)
        self._asof = asof
        return PoolAdvance(
            asof=asof,
            members=self._members,
            hypotheses=hypotheses,
            residual_probability=residual,
            records=tuple(records),
        )

    # -- lifecycle operations -------------------------------------------------

    def _mode(self, member: PoolMember) -> TrajectoryMode:
        return self.library.mode(member.mode_id)

    def _parent_of(self, mode_id: str) -> str | None:
        return self.library.mode(mode_id).parent_mode_id

    def _lowest_common_ancestor(self, left: str, right: str) -> str | None:
        """The most specific mode in the library that contains both."""

        seen: set[str] = set()
        node: str | None = left
        while node is not None:
            seen.add(node)
            node = self._parent_of(node)
        node = right
        while node is not None:
            if node in seen:
                return node
            node = self._parent_of(node)
        return None

    def _merge(
        self, members: Sequence[PoolMember], asof: pd.Timestamp
    ) -> tuple[list[PoolMember], list[LifecycleRecord]]:
        """Fold two siblings whose remaining futures agree back into the parent."""

        members = list(members)
        records: list[LifecycleRecord] = []
        for left, right in itertools.combinations(list(members), 2):
            if left not in members or right not in members:
                continue
            left_mode, right_mode = self._mode(left), self._mode(right)
            # The lowest common ancestor, not a shared immediate parent: two
            # hypotheses can converge without being adjacent leaves, and
            # requiring adjacency would make MERGE almost unreachable.
            parent_id = self._lowest_common_ancestor(left.mode_id, right.mode_id)
            if parent_id is None or parent_id in (left.mode_id, right.mode_id):
                continue
            if any(m.mode_id == parent_id for m in members):
                continue
            age = max(left.age, right.age)
            if _remaining_distance(left_mode, right_mode, age) > self.config.merge_maximum_distance:
                continue
            # The survivor keeps the older anchor, so the merged hypothesis is
            # judged against the whole path either parent was judged against.
            senior = left if left.age >= right.age else right
            lineage = tuple(
                dict.fromkeys(
                    (*left.lineage, *right.lineage, left.hypothesis_id, right.hypothesis_id)
                )
            )
            merged = PoolMember(
                hypothesis_id=_hypothesis_id(parent_id, senior.spawned_at, lineage),
                mode_id=parent_id,
                spawned_at=senior.spawned_at,
                prior=max(left.prior, right.prior),
                path=senior.path,
                lineage=lineage,
            )
            members = [m for m in members if m not in (left, right)] + [merged]
            records.append(
                LifecycleRecord(
                    asof=asof,
                    operation=LifecycleOperation.MERGE,
                    hypothesis_ids=(left.hypothesis_id, right.hypothesis_id, merged.hypothesis_id),
                    mode_ids=(left.mode_id, right.mode_id, parent_id),
                    reason="remaining futures converged onto the parent mode",
                )
            )
        return members, records

    def _split(
        self, members: Sequence[PoolMember], asof: pd.Timestamp
    ) -> tuple[list[PoolMember], list[LifecycleRecord]]:
        """Replace an undecided hypothesis with the two children it sits between."""

        members = list(members)
        records: list[LifecycleRecord] = []
        room = self.config.max_live_hypotheses - len(members)
        if room < 1:
            return members, records
        for member in sorted(members, key=lambda m: m.hypothesis_id):
            if room < 1:
                break
            if member.age < self.config.split_minimum_age_bars:
                continue
            mode = self._mode(member)
            children = [self.library.mode(cid) for cid in mode.child_mode_ids]
            children = [c for c in children if all(m.mode_id != c.mode_id for m in members)]
            if len(children) < 2:
                continue
            ranked = sorted(
                children, key=lambda c: (_path_distance(c, member.path), c.mode_id)
            )
            first, second = ranked[0], ranked[1]
            d1 = _path_distance(first, member.path)
            d2 = _path_distance(second, member.path)
            total = d1 + d2
            if total <= 0.0:
                continue
            if abs(d1 - d2) / total > self.config.split_maximum_imbalance:
                continue
            lineage = tuple(dict.fromkeys((*member.lineage, member.hypothesis_id)))
            replacements = [
                PoolMember(
                    hypothesis_id=_hypothesis_id(child.mode_id, member.spawned_at, lineage),
                    mode_id=child.mode_id,
                    spawned_at=member.spawned_at,
                    prior=member.prior * 0.5,
                    path=member.path,
                    lineage=lineage,
                )
                for child in (first, second)
            ]
            members = [m for m in members if m is not member] + replacements
            room -= 1
            records.append(
                LifecycleRecord(
                    asof=asof,
                    operation=LifecycleOperation.SPLIT,
                    hypothesis_ids=(
                        member.hypothesis_id,
                        *[r.hypothesis_id for r in replacements],
                    ),
                    mode_ids=(member.mode_id, first.mode_id, second.mode_id),
                    reason="realized path sits between two child modes",
                )
            )
        return members, records

    def _spawn(
        self,
        members: Sequence[PoolMember],
        asof: pd.Timestamp,
        proposals: Sequence[HypothesisProposal],
        close: float,
        atr: float,
    ) -> tuple[list[PoolMember], list[LifecycleRecord]]:
        """Admit context-supported modes, displacing the weakest when they beat it."""

        members = list(members)
        records: list[LifecycleRecord] = []
        live_modes = {m.mode_id for m in members}
        for proposal in proposals:
            if proposal.mode_id in live_modes:
                continue
            if proposal.prior < self.config.spawn_minimum_prior:
                continue
            if proposal.mode_id not in set(self.library.mode_ids):
                continue
            displaced: PoolMember | None = None
            if len(members) >= self.config.max_live_hypotheses:
                weakest = min(members, key=lambda m: (m.prior, m.hypothesis_id))
                if proposal.prior <= weakest.prior + self.config.spawn_displacement_margin:
                    continue
                displaced = weakest
            spawned = PoolMember(
                hypothesis_id=_hypothesis_id(proposal.mode_id, asof, ()),
                mode_id=proposal.mode_id,
                spawned_at=asof,
                prior=proposal.prior,
                # A hypothesis is anchored at the bar that spawned it, so its
                # path starts empty and its ATR scale is the one the library
                # was fitted in.
                path=RealizedPath(anchor_price=float(close), anchor_atr=float(atr)),
            )
            if displaced is not None:
                members = [m for m in members if m is not displaced]
                records.append(
                    LifecycleRecord(
                        asof=asof,
                        operation=LifecycleOperation.RETIRE,
                        hypothesis_ids=(displaced.hypothesis_id,),
                        mode_ids=(displaced.mode_id,),
                        reason="displaced by a better-supported proposal",
                    )
                )
            members.append(spawned)
            live_modes.add(proposal.mode_id)
            records.append(
                LifecycleRecord(
                    asof=asof,
                    operation=LifecycleOperation.SPAWN,
                    hypothesis_ids=(spawned.hypothesis_id,),
                    mode_ids=(spawned.mode_id,),
                    reason=f"context retrieval supports this mode at {proposal.prior:.3f}",
                )
            )
        return members, records

    def _publish(
        self,
        members: Sequence[PoolMember],
        asof: pd.Timestamp,
        scored: Mapping[str, Any],
        *,
        prune: bool = True,
    ) -> tuple[tuple[Hypothesis, ...], float, list[PoolMember]]:
        """Normalize the working set against the residual and build the output."""

        members = list(members)
        if not members:
            return (), 1.0, []

        # The residual carries the share of the retrieved neighbourhood that no
        # live hypothesis claims, floored so it can never be argued away.
        claimed = sum(m.prior for m in members)
        residual_prior = max(self.config.residual_floor, 1.0 - claimed)

        log_weights = []
        evidences = []
        for member in members:
            evidence = scored.get(member.hypothesis_id)
            if evidence is None:
                evidence = evaluate(self._mode(member), member.path, self.updater_config)
            evidences.append(evidence)
            log_weights.append(
                math.log(max(member.prior, self.config.prior_floor))
                + evidence.evidence_log_weight
            )
        probabilities, residual = normalize_log_weights(
            tuple(log_weights), residual_log_weight=math.log(residual_prior)
        )

        low: list[PoolMember] = []
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
                mode_id=member.mode_id,
                spawned_at=member.spawned_at,
                asof=asof,
                age_bars=member.age,
                prior_log_weight=math.log(max(member.prior, self.config.prior_floor)),
                evidence_log_weight=evidence.evidence_log_weight,
                probability=probability,
                expected_trajectory=self._mode(member).medoid,
                realized_divergence=evidence.divergence,
                status=HypothesisStatus.ACTIVE,
                lineage=member.lineage,
            )
            for member, evidence, probability in zip(members, evidences, probabilities)
        )
        return hypotheses, residual, []


__all__ = [
    "HypothesisPool",
    "HypothesisPoolError",
    "PoolAdvance",
    "PoolConfig",
    "PoolMember",
]
