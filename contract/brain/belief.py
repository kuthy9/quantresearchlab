"""Brain belief: the aggregate per-clock output contract.

``MarketBelief`` is what a belief producer publishes and what Decision reads.
Its path diagnostics remain shadow-only and fail closed."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from contract.market.primitives import Direction, LiquidityLevel, Playbook, PlaybookPhase, StructuralLevel, aware_timestamp
from contract.brain.vocabulary import GlobalConflictRole
from contract.brain.plan import DrawSelection, FrozenLSRContext, FrozenRangeAuctionContext, LiquidityRoute
from contract.brain.hypothesis import HypothesisBelief
from contract.brain.context import ContextThesisState, EntryEpisodeState, GlobalMarketContext, OpenMarketThesis
from brain.core.market_belief import PathBeliefUpdateRecord, PathCompetitionSetState


@dataclass(frozen=True)
class MarketBelief:
    asof: pd.Timestamp
    hypotheses: Mapping[str, HypothesisBelief]
    context_hypotheses: Mapping[str, Any] = field(default_factory=dict)
    dominant_hypothesis_id: str | None = None
    competing_hypothesis_ids: tuple[str, ...] = ()
    focus_state: Any | None = None
    cross_scale_conflicts: tuple[str, ...] = ()
    unresolved_ambiguities: tuple[str, ...] = ()
    scene_revision_id: str | None = None
    global_context: GlobalMarketContext | None = None
    thesis_candidates: Mapping[str, HypothesisBelief] = field(
        default_factory=dict
    )
    # A disappeared analytical root is retained here until a causal terminal
    # outcome is observed.  Dormant episodes are deliberately excluded from
    # the action interface; a stale executable snapshot can never re-enter.
    retained_episode_candidates: Mapping[str, HypothesisBelief] = field(
        default_factory=dict
    )
    position_management_candidates: Mapping[str, HypothesisBelief] = field(
        default_factory=dict
    )
    context_theses: Mapping[str, ContextThesisState] = field(
        default_factory=dict
    )
    entry_episodes: Mapping[str, EntryEpisodeState] = field(
        default_factory=dict
    )
    path_competition_state: PathCompetitionSetState | None = None
    path_update_records_this_clock: tuple[
        PathBeliefUpdateRecord,
        ...,
    ] = ()
    path_protocol_status: str = "development_unvalidated"
    path_authority: str = "shadow_only"

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="belief.asof"))
        object.__setattr__(
            self,
            "competing_hypothesis_ids",
            tuple(self.competing_hypothesis_ids),
        )
        object.__setattr__(
            self,
            "cross_scale_conflicts",
            tuple(self.cross_scale_conflicts),
        )
        object.__setattr__(
            self,
            "unresolved_ambiguities",
            tuple(self.unresolved_ambiguities),
        )
        path_records = tuple(self.path_update_records_this_clock)
        object.__setattr__(
            self,
            "path_update_records_this_clock",
            path_records,
        )
        path_state = self.path_competition_state
        if (
            self.path_protocol_status != "development_unvalidated"
            or self.path_authority != "shadow_only"
            or any(
                not isinstance(record, PathBeliefUpdateRecord)
                or record.asof != self.asof
                for record in path_records
            )
        ):
            raise ValueError("belief path diagnostics are not shadow-only")
        if path_state is None:
            if path_records:
                raise ValueError(
                    "belief path diagnostics require a current competition set"
                )
        else:
            if not isinstance(path_state, PathCompetitionSetState):
                raise ValueError("belief path diagnostic scope is inconsistent")
            if (
                path_state.asof != self.asof
                or path_state.protocol_status != self.path_protocol_status
                or path_state.authority != self.path_authority
                or any(
                    record.competition_set_id
                    != path_state.competition_set_id
                    for record in path_records
                )
            ):
                raise ValueError("belief path diagnostic scope is inconsistent")
        if any(
            key != hypothesis.key
            or any(
                clock > self.asof
                for clock in hypothesis.causal_observation_clocks
            )
            or hypothesis.phase_started_at > self.asof
            or any(
                evidence.observed_at > self.asof
                for evidence in (
                    *hypothesis.supporting,
                    *hypothesis.contradicting,
                )
            )
            or (
                hypothesis.invalidation is not None
                and hypothesis.invalidation.observed_at > self.asof
            )
            or any(
                target.confirmed_at > self.asof
                for target in hypothesis.deliverable_targets
            )
            or (
                hypothesis.thesis_draw is not None
                and hypothesis.thesis_draw.confirmed_at > self.asof
            )
            or (
                hypothesis.sequence is not None
                and (
                    (
                        hypothesis.sequence.started_at is not None
                        and hypothesis.sequence.started_at > self.asof
                    )
                    or any(
                        step.observed_at is not None
                        and step.observed_at > self.asof
                        for step in hypothesis.sequence.steps
                    )
                )
            )
            or (
                hypothesis.terminal_at is not None
                and hypothesis.terminal_at > self.asof
            )
            or (
                hypothesis.selected_trigger is not None
                and hypothesis.selected_trigger.observed_at > self.asof
            )
            or (
                hypothesis.draw_selection is not None
                and hypothesis.draw_selection.selected_at > self.asof
            )
            or (
                hypothesis.liquidity_route is not None
                and hypothesis.liquidity_route.selected_at > self.asof
            )
            for key, hypothesis in self.hypotheses.items()
        ):
            raise ValueError(
                "belief mapping identity or causal clock is invalid"
            )
        context_values = tuple(self.context_hypotheses.values())
        if any(
            key != getattr(value, "hypothesis_id", None)
            for key, value in self.context_hypotheses.items()
        ):
            raise ValueError("context hypothesis mapping identity is invalid")
        if any(
            hypothesis.record_kind != "summary"
            or (
                hypothesis.summary_source_candidate_id is not None
                and hypothesis.summary_source_candidate_id
                not in self.thesis_candidates
                and hypothesis.summary_source_candidate_id
                not in self.position_management_candidates
            )
            for hypothesis in self.hypotheses.values()
        ):
            raise ValueError("six-slot beliefs must be read-only summaries")
        candidates = tuple(self.thesis_candidates.values())
        if any(
            key != candidate.candidate_id
            or candidate.required_root_id is None
            or candidate.record_kind != "root_candidate"
            or any(
                clock > self.asof
                for clock in candidate.causal_observation_clocks
            )
            or candidate.phase_started_at > self.asof
            or (
                candidate.market_thesis_root_id
                != candidate.required_root_id
            )
            for key, candidate in self.thesis_candidates.items()
        ):
            raise ValueError("root-specific thesis candidate mapping is invalid")
        retained_candidates = tuple(
            self.retained_episode_candidates.values()
        )
        if (
            not set(self.thesis_candidates).isdisjoint(
                self.retained_episode_candidates
            )
            or any(
            key != candidate.candidate_id
            or candidate.required_root_id is None
            or candidate.record_kind != "retained_episode"
            or any(
                clock > self.asof
                for clock in candidate.causal_observation_clocks
            )
            or candidate.phase_started_at > self.asof
            or candidate.market_thesis_root_id
            != candidate.required_root_id
            or candidate.phase is PlaybookPhase.EXECUTABLE
            for key, candidate in self.retained_episode_candidates.items()
            )
        ):
            raise ValueError("retained entry episode mapping is invalid")
        candidate_slots = {
            (
                candidate.required_root_id,
                candidate.playbook,
                candidate.direction,
                candidate.episode_id,
            )
            for candidate in candidates
        }
        if len(candidate_slots) != len(candidates):
            raise ValueError(
                "root-specific EntryEpisode candidates contain duplicates"
            )
        management_candidates = tuple(
            self.position_management_candidates.values()
        )
        if (
            len(management_candidates) > 1
            or any(
                key != candidate.candidate_id
                or candidate.required_root_id is None
                or candidate.record_kind != "position_management"
                or any(
                    clock > self.asof
                    for clock in candidate.causal_observation_clocks
                )
                or candidate.phase_started_at > self.asof
                or candidate.market_thesis_root_id
                != candidate.required_root_id
                or not candidate.market_thesis_action_bound
                or candidate.market_thesis_match_status
                != "exact_root_bound"
                or candidate.phase
                not in {
                    PlaybookPhase.ENTERED,
                    PlaybookPhase.DELIVERING,
                    PlaybookPhase.WEAKENING,
                    PlaybookPhase.COMPLETED,
                    PlaybookPhase.INVALIDATED,
                }
                for key, candidate
                in self.position_management_candidates.items()
            )
            or not set(self.position_management_candidates).isdisjoint(
                self.thesis_candidates
            )
            or not set(self.position_management_candidates).isdisjoint(
                self.retained_episode_candidates
            )
        ):
            raise ValueError(
                "position-management thesis candidate mapping is invalid"
            )
        candidate_ids = {
            *self.thesis_candidates,
            *self.retained_episode_candidates,
            *self.position_management_candidates,
        }
        realtime_candidate_ids = {
            *self.thesis_candidates,
            *self.position_management_candidates,
        }
        context_ids = set(self.context_hypotheses)
        if not context_ids.issubset(realtime_candidate_ids):
            raise ValueError(
                "candidate context views must reference realtime candidates"
            )
        if (
            self.dominant_hypothesis_id is not None
            and self.dominant_hypothesis_id not in realtime_candidate_ids
        ):
            raise ValueError("dominant root candidate is not realtime")
        if (
            len(self.competing_hypothesis_ids)
            != len(set(self.competing_hypothesis_ids))
            or not set(self.competing_hypothesis_ids).issubset(
                realtime_candidate_ids
            )
            or self.dominant_hypothesis_id in self.competing_hypothesis_ids
        ):
            raise ValueError(
                "competing realtime candidate identities are invalid"
            )
        if self.focus_state is not None and self.focus_state.asof != self.asof:
            raise ValueError("belief focus and belief clocks disagree")
        if (
            self.focus_state is not None
            and self.focus_state.hypothesis_id is not None
            and self.focus_state.hypothesis_id not in realtime_candidate_ids
        ):
            raise ValueError("focus must reference a realtime root candidate")
        if self.scene_revision_id is not None and not self.scene_revision_id:
            raise ValueError("belief scene revision cannot be empty")
        if self.global_context is not None:
            if self.global_context.updated_at != self.asof:
                raise ValueError("belief global context and belief clocks disagree")
            if (
                self.scene_revision_id is not None
                and self.global_context.scene_revision_id
                != self.scene_revision_id
            ):
                raise ValueError(
                    "belief global context and scene revisions disagree"
                )
            authoritative_conflict_ids = {
                conflict.conflict_id
                for conflict in self.global_context.material_conflicts
                if GlobalConflictRole(conflict.role)
                is not GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY
            }
            if not set(self.cross_scale_conflicts).issubset(
                authoritative_conflict_ids
            ):
                raise ValueError(
                    "belief conflicts must be filtered GlobalMarketContext IDs"
                )
            # Discovery-root visibility is not an EntryEpisode lifetime
            # signal.  A root-specific action candidate may therefore retain
            # its frozen exact binding while the open-thesis projection is
            # absent, provided PlaybookBrain can still resolve the identical
            # live setup/location/path.  PlaybookBrain's exact frozen-source
            # matching and mapping disjointness—not current root visibility—
            # form the action-authority boundary.
        elif candidates or retained_candidates or management_candidates:
            raise ValueError(
                "market thesis candidates require GlobalMarketContext"
            )
        context_theses = dict(self.context_theses)
        entry_episodes = dict(self.entry_episodes)
        if any(
            key != thesis.context_thesis_id
            or thesis.updated_at > self.asof
            for key, thesis in context_theses.items()
        ):
            raise ValueError("context thesis mapping identity or clock is invalid")
        if any(
            key != episode.candidate_id
            or episode.parent_context_thesis_id not in context_theses
            or any(
                clock > self.asof
                for clock in episode.causal_observation_clocks
            )
            or episode.updated_at > self.asof
            or key not in candidate_ids
            for key, episode in entry_episodes.items()
        ):
            raise ValueError("entry episode mapping identity or clock is invalid")
        if any(
            candidate.context_thesis_id is not None
            and (
                candidate.context_thesis_id not in context_theses
                or context_theses[candidate.context_thesis_id].direction
                is not candidate.direction
            )
            for candidate in (
                *candidates,
                *retained_candidates,
                *management_candidates,
            )
        ):
            raise ValueError("candidate references an unknown context thesis")
        if any(
            candidate.episode_id is not None
            and (
                candidate.candidate_id not in entry_episodes
                or entry_episodes[candidate.candidate_id].episode_id
                != candidate.episode_id
                or entry_episodes[candidate.candidate_id]
                .parent_context_thesis_id
                != candidate.parent_context_thesis_id
                or entry_episodes[candidate.candidate_id].playbook
                is not candidate.playbook
                or entry_episodes[candidate.candidate_id].direction
                is not candidate.direction
                or entry_episodes[candidate.candidate_id].phase
                is not candidate.phase
                or entry_episodes[candidate.candidate_id].entry_location_id
                != candidate.entry_location_id
                or entry_episodes[candidate.candidate_id].entry_path_id
                != candidate.entry_path_id
                or entry_episodes[candidate.candidate_id].selected_trigger
                != candidate.selected_trigger
                or entry_episodes[candidate.candidate_id].plan
                != candidate.plan
                or entry_episodes[candidate.candidate_id].invalidation
                != candidate.invalidation
                or entry_episodes[candidate.candidate_id].deadline
                != candidate.episode_deadline
            )
            for candidate in (
                *candidates,
                *retained_candidates,
                *management_candidates,
            )
        ):
            raise ValueError("candidate and entry episode lifecycle disagree")
        if any(
            not {
                episode.episode_id
                for episode in entry_episodes.values()
                if episode.parent_context_thesis_id == identity
            }.issubset(thesis.child_episode_ids)
            for identity, thesis in context_theses.items()
        ):
            raise ValueError("context thesis children disagree with entry episodes")
        # Exact entry-path ownership is an action/position invariant.  A
        # dormant analytical projection may overlap the position owner (or
        # another unresolved dormant root) while the graph is unable to prove
        # which root owned the historical path.  Those snapshots have no
        # action authority and must not crash the whole completed-bar update.
        realtime_episode_ids = {
            *self.thesis_candidates,
            *self.position_management_candidates,
        }
        active_episodes = tuple(
            episode
            for candidate_id, episode in entry_episodes.items()
            if candidate_id in realtime_episode_ids
            and episode.phase
            not in {PlaybookPhase.COMPLETED, PlaybookPhase.INVALIDATED}
        )
        for field_name in ("entry_location_id", "entry_path_id"):
            identities = tuple(
                getattr(episode, field_name)
                for episode in active_episodes
                if getattr(episode, field_name) is not None
            )
            if len(identities) != len(set(identities)):
                raise ValueError(
                    f"active entry episodes cannot share {field_name}"
                )
        trigger_ids = tuple(
            episode.selected_trigger.trigger_id
            for episode in active_episodes
            if episode.selected_trigger is not None
        )
        if len(trigger_ids) != len(set(trigger_ids)):
            raise ValueError("active entry episodes cannot share a trigger")

    def candidates(self) -> tuple[HypothesisBelief, ...]:
        """Compatibility view of the six playbook-direction summaries."""

        return tuple(self.hypotheses.values())

    def owns_actionable_entry_episode(
        self,
        candidate_id: str,
        hypothesis: HypothesisBelief,
    ) -> bool:
        """Return whether one action root owns its exact live projections."""

        candidate = self.thesis_candidates.get(candidate_id)
        context_id = hypothesis.context_thesis_id
        episode_id = hypothesis.episode_id
        plan = hypothesis.plan
        if (
            candidate is not hypothesis
            or hypothesis.candidate_id != candidate_id
            or not isinstance(context_id, str)
            or not context_id
            or not isinstance(episode_id, str)
            or not episode_id
            or hypothesis.parent_context_thesis_id != context_id
            or plan is None
            or plan.setup_id != episode_id
            or hypothesis.setup_context_id != episode_id
        ):
            return False
        context = self.context_theses.get(context_id)
        episode = self.entry_episodes.get(candidate_id)
        return bool(
            context is not None
            and episode is not None
            and context.direction is hypothesis.direction
            and context.lifecycle
            not in {"completed", "invalidated", "censored"}
            and episode_id in context.child_episode_ids
            and episode.candidate_id == candidate_id
            and episode.episode_id == episode_id
            and episode.parent_context_thesis_id == context_id
            and episode.playbook is hypothesis.playbook
            and episode.direction is hypothesis.direction
            and episode.phase is hypothesis.phase
            and episode.entry_location_id == hypothesis.entry_location_id
            and episode.entry_path_id == hypothesis.entry_path_id
            and episode.selected_trigger == hypothesis.selected_trigger
            and episode.plan == plan
            and episode.invalidation == hypothesis.invalidation
            and episode.deadline == hypothesis.episode_deadline
        )

    def action_candidate_items(
        self,
    ) -> tuple[tuple[str, HypothesisBelief], ...]:
        """Stable action identities paired with their root-specific beliefs."""

        # The six playbook-direction slots are read-only summaries.  A
        # missing graph/context therefore yields no action identity instead
        # of silently promoting a summary into an executable hypothesis.
        # Graph-free construction needed by unit tests belongs in a test
        # helper, not in this production contract.
        return tuple(self.thesis_candidates.items())

    def lifecycle_candidate_items(
        self,
    ) -> tuple[tuple[str, HypothesisBelief], ...]:
        """All candidates whose causal lifecycle still needs observation.

        Unlike :meth:`action_candidate_items`, this includes dormant roots.
        Calibration/diagnostic recorders may observe them, but Decision must
        never use this interface to authorize an entry.
        """

        return (
            *tuple(self.thesis_candidates.items()),
            *tuple(self.retained_episode_candidates.items()),
            *tuple(self.position_management_candidates.items()),
        )

    def position_candidate_items(
        self,
    ) -> tuple[tuple[str, HypothesisBelief], ...]:
        """Candidates resolvable for an already-open frozen position.

        A closed analytical root may remain here only to manage the exact
        position that it created.  It is deliberately excluded from
        :meth:`action_candidate_items`, so it can never authorize a new
        ``ENTER``.
        """

        return (
            *self.action_candidate_items(),
            *tuple(self.position_management_candidates.items()),
        )

    @property
    def market_theses(self) -> tuple[OpenMarketThesis, ...]:
        """Current playbook-neutral theses, never direct action candidates."""

        return (
            ()
            if self.global_context is None
            else self.global_context.open_market_theses
        )

    def resolve_hypothesis(self, identity: str | None) -> HypothesisBelief | None:
        if identity is None:
            return None
        direct = self.thesis_candidates.get(identity)
        if direct is not None:
            return direct
        direct = self.retained_episode_candidates.get(identity)
        if direct is not None:
            return direct
        direct = self.position_management_candidates.get(identity)
        if direct is not None:
            return direct
        direct = self.hypotheses.get(identity)
        if direct is not None:
            return direct
        context = self.context_hypotheses.get(identity)
        if context is None:
            return None
        return self.thesis_candidates.get(identity) or (
            self.position_management_candidates.get(identity)
        )

    def for_slot(
        self,
        playbook: Playbook,
        direction: Direction,
    ) -> tuple[HypothesisBelief, ...]:
        value = self.hypotheses.get(f"{playbook.value}:{direction.value}")
        return () if value is None else (value,)

    def ranked(self) -> list[HypothesisBelief]:
        def rank_key(
            belief: HypothesisBelief,
        ) -> tuple[float, float, float, float, float]:
            phase_rank = {
                PlaybookPhase.INVALIDATED: 0.0,
                PlaybookPhase.COMPLETED: 0.0,
                PlaybookPhase.INACTIVE: 1.0,
                PlaybookPhase.FORMING: 2.0,
                PlaybookPhase.ARMED: 3.0,
                PlaybookPhase.WAITING_LOCATION: 4.0,
                PlaybookPhase.WAITING_TRIGGER: 5.0,
                PlaybookPhase.EXECUTABLE: 6.0,
                PlaybookPhase.WEAKENING: 7.0,
                PlaybookPhase.DELIVERING: 8.0,
                PlaybookPhase.ENTERED: 9.0,
            }[belief.phase]
            return (
                float(belief.eligible),
                phase_rank,
                float(belief.entry_readiness or 0.0),
                float(belief.thesis_strength),
                -belief.uncertainty,
            )

        source = (
            {
                **self.thesis_candidates,
                **self.position_management_candidates,
            }.values()
            if self.global_context is not None
            else self.hypotheses.values()
        )
        return sorted(
            source,
            key=rank_key,
            reverse=True,
        )


@dataclass(frozen=True)
class FrozenThesis:
    thesis_hash: str
    created_at: pd.Timestamp
    playbook: Playbook
    direction: Direction
    entry: float
    original_invalidation: StructuralLevel
    original_targets: tuple[LiquidityLevel, ...]
    deadline: pd.Timestamp
    setup_id: str | None = None
    entry_location_id: str | None = None
    entry_path_id: str | None = None
    draw_selection: DrawSelection | None = None
    range_auction: FrozenRangeAuctionContext | None = None
    lsr_context: FrozenLSRContext | None = None
    liquidity_route: LiquidityRoute | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", aware_timestamp(self.created_at, name="thesis.created_at"))
        object.__setattr__(self, "deadline", aware_timestamp(self.deadline, name="thesis.deadline"))
        typed_identity = (
            self.setup_id,
            self.entry_location_id,
            self.entry_path_id,
        )
        if any(value is not None for value in typed_identity) and any(
            not isinstance(value, str) or not value
            for value in typed_identity
        ):
            raise ValueError(
                "frozen thesis typed identities must be complete "
                "non-empty text"
            )
        if self.deadline <= self.created_at or not self.original_targets:
            raise ValueError("frozen thesis requires a future deadline and targets")
        if (
            self.playbook is Playbook.FAILED_AUCTION_VALUE_RETURN
            and self.setup_id is not None
        ):
            if (
                self.range_auction is None
                or self.draw_selection is None
            ):
                raise ValueError(
                    "frozen FAVR thesis lacks its range-auction context"
                )
        elif self.range_auction is not None:
            raise ValueError("non-FAVR thesis cannot own a range auction")
        if (
            self.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
            and self.setup_id is not None
        ):
            if self.lsr_context is None:
                raise ValueError(
                    "frozen LSR thesis lacks its parent Context provenance"
                )
        elif self.lsr_context is not None:
            raise ValueError("non-LSR thesis cannot own LSR Context provenance")


__all__ = [
    "FrozenThesis",
    "MarketBelief",
]
