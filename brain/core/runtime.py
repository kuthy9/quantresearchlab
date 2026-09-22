"""The SLEEP ↔ ACTIVE machine, one step per completed bar.

```text
SLEEP  ──wake event──►  ACTIVE  (new episode; LLM reasons from nothing)
ACTIVE ──new evidence──► LLM update → reduce ──five exit conditions hold──► archive → SLEEP
ACTIVE ──no evidence──►  TICK (no LLM; bookkeeping only)
ACTIVE ──LLM failure──►  incident, state carried forward, still ACTIVE
ACTIVE ──release window──► EVENT_SLEEP: archive without a call → SLEEP (2026-09-21)
SLEEP  ──window ends──►  WAKE on the first bar after it, Eye event or not
```

The runtime owns the episode's ``ObjectRegistry``, the 1m tape accumulated
between LLM calls, the 5m+ bookkeeping evidence of the TICK bars since the
last call (delivered with the next one), the idle-update counter behind the
idle archive rule, the price relations of the watched objects (a change is
evidence), the ``known_at`` of the last LLM call (an interaction path that
stepped after it is *open* for the sleep gate), the per-day episode counter,
and the journal.  It asserts that
``known_at`` strictly increases: a bar can never be replayed out of order."""
from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from brain.core.eye_view import EyeContext, build_eye_context, evidence_id
from contract.brain.state import EvidenceItem
from brain.core.journal import BrainJournal
from brain.core.main_brain import BrainStep, MainBrain
from brain.core.object_registry import ObjectRegistry
from brain.core.opportunity_geometry import GeometryError, resolve_geometry
from brain.core.position_ledger import PositionLedger, engaged
from brain.core.reducer import ReduceContext, ReduceResult, apply
from brain.core.sleep_controller import ControllerConfig, Decision, decide
from contract.brain.llm import LLMInput
from contract.brain.state import BrainState, Opportunity, OpportunityState, isoformat_utc  # noqa: E402
from contract.eye import MarketEvent, MarketObservation
from shares.core.timing import NO_TIMINGS, Timings, timed


class RuntimeStatus(str, Enum):
    SLEEP = "SLEEP"
    ACTIVE = "ACTIVE"


@dataclass(frozen=True)
class StepResult:
    known_at: pd.Timestamp
    decision: Decision
    status_after: RuntimeStatus
    episode_id: str | None
    revision: int | None
    llm_called: bool
    incident: str | None
    slept: bool
    rejections: tuple[str, ...]
    # The LLM's own reply latency when this step called it and got a reply.
    llm_latency_ms: int | None = None
    # The release window that put the episode to sleep on this bar
    # (``event:<kind>:<release>``), for the executor to withdraw every expression.
    event: str | None = None


StepHook = Callable[[StepResult, BrainState | None, LLMInput | None], None]


class _Tape:
    """1m reaction events since the last LLM call — context, never evidence."""

    def __init__(self, limit: int) -> None:
        self.bars = 0
        self.counts: Counter[str] = Counter()
        self.recent: deque[dict[str, Any]] = deque(maxlen=limit)

    def add(self, events: tuple[MarketEvent, ...], config: ControllerConfig, registry: ObjectRegistry) -> None:
        self.bars += 1
        for event in events:
            if event.timeframe.value != config.tape_timeframe:
                continue
            if not config.evidence.is_transition(event.kind.value):
                continue
            self.counts[event.kind.value] += 1
            if config.is_tape_event(event):
                self.recent.append(
                    {
                        "evidence_id": evidence_id(event),
                        "kind": event.kind.value,
                        "object_id": registry.alias_of(event.entity_id),
                        "known_at": isoformat_utc(event.known_at),
                    }
                )

    def payload(self) -> dict[str, Any]:
        return {
            "bars": self.bars,
            "counts": dict(sorted(self.counts.items())),
            "recent": list(self.recent),
        }


class BrainRuntime:
    def __init__(
        self,
        *,
        controller: ControllerConfig,
        brain: MainBrain,
        journal: BrainJournal | None,
        ledger: PositionLedger,
        tick: float,
        timezone: str = "America/New_York",
        on_step: StepHook | None = None,
        timings: Timings = NO_TIMINGS,
    ) -> None:
        self._controller = controller
        self._brain = brain
        self._journal = journal
        self._timings = timings
        self._ledger = ledger
        self._tick = float(tick)
        self._timezone = timezone
        self._on_step = on_step
        self._status = RuntimeStatus.SLEEP
        self._state: BrainState | None = None
        self._last_archived: BrainState | None = None
        self._registry: ObjectRegistry | None = None
        self._relations: dict[str, str | None] = {}
        # The last bar a watched alias's relation flip triggered a call (debounce).
        self._relation_triggered: dict[str, pd.Timestamp] = {}
        self._tape = _Tape(controller.tape_recent_limit)
        self._episode_counter: dict[str, int] = {}
        self._last_known_at: pd.Timestamp | None = None
        self._last_llm_known_at: pd.Timestamp | None = None
        self._deferred: list[EvidenceItem] = []
        self._idle_updates = 0
        self._last_context: EyeContext | None = None

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def last_context(self) -> EyeContext | None:
        """The context built on the last step; None while asleep."""
        return self._last_context

    @property
    def state(self) -> BrainState | None:
        return self._state

    @property
    def last_archived(self) -> BrainState | None:
        return self._last_archived

    # ------------------------------------------------------------------ helpers

    def _next_episode_id(self, known_at: pd.Timestamp) -> str:
        day = known_at.tz_convert(self._timezone).strftime("%Y%m%d")
        number = self._episode_counter.get(day, 0) + 1
        self._episode_counter[day] = number
        return f"EP_{day}_{number:03d}"

    def _watched(self, state: BrainState) -> tuple[str, ...]:
        """The watched and opportunity objects whose relation flips count:
        those on the controller's ``relation_change_timeframes``."""
        aliases = [item.object_id for item in state.watch_next]
        aliases.extend(state.opportunity.object_ids)
        scales = self._controller.relation_change_timeframes
        registry = state.object_registry
        return tuple(
            alias for alias in dict.fromkeys(aliases)
            if alias in registry and registry[alias].timeframe in scales
        )

    def _remember_relations(self, state: BrainState, context: EyeContext) -> None:
        self._relations = {alias: context.relation_of(alias) for alias in self._watched(state)}

    def _relation_changes(self, context: EyeContext) -> tuple[str, ...]:
        """The watched aliases whose side of price flipped since the last
        call, less those that triggered within the last
        ``relation_change_debounce_bars`` minutes (2026-09-18)."""
        window = pd.Timedelta(minutes=self._controller.relation_change_debounce_bars)
        now_at = pd.Timestamp(context.known_at)
        changed = []
        for alias, relation in self._relations.items():
            now = context.relation_of(alias)
            if now is None or now == relation:
                continue
            last = self._relation_triggered.get(alias)
            if last is not None and now_at - last < window:
                continue
            changed.append(alias)
        return tuple(changed)

    def _relation_triggered_at(self, aliases: Sequence[str], known_at: pd.Timestamp) -> None:
        for alias in aliases:
            self._relation_triggered[alias] = pd.Timestamp(known_at)

    @staticmethod
    def _latency(step: BrainStep) -> int | None:
        reply = step.outcome.reply
        return None if reply is None else int(reply.latency_ms)

    def _journal_llm(self, episode_id: str, context: EyeContext, step: BrainStep) -> None:
        if self._journal is None:
            return
        outcome = step.outcome
        self._journal.write(
            "llm_call",
            episode_id=episode_id,
            known_at=context.known_at,
            payload={
                "input_sha": step.llm_input.input_sha,
                "input": step.llm_input.to_dict(),
                "reply": None if outcome.reply is None else outcome.reply.to_dict(),
                "attempts": outcome.attempts,
                "repaired": outcome.repaired,
                "repair_reason": outcome.repair_reason,
                "rejected_reply": None if outcome.rejected_reply is None else outcome.rejected_reply.to_dict(),
            },
        )
        if step.result.incident is not None:
            self._journal.write(
                "incident",
                episode_id=episode_id,
                known_at=context.known_at,
                payload={
                    "kind": step.result.incident,
                    "message": outcome.incident_message,
                    "attempts": outcome.attempts,
                    "input_sha": step.llm_input.input_sha,
                },
            )

    def _journal_state(self, episode_id: str, context: EyeContext, result: ReduceResult, previous: BrainState | None) -> None:
        if self._journal is None:
            return
        state = result.state
        self._journal.write(
            "state",
            episode_id=episode_id,
            known_at=context.known_at,
            payload={"revision": state.revision, "state": state.to_dict(), "rejections": list(result.rejections)},
        )
        before = previous.opportunity if previous is not None else Opportunity()
        if state.opportunity != before and state.opportunity.state is not OpportunityState.NONE:
            try:
                geometry = resolve_geometry(
                    state.opportunity, context.geometries(), close=context.close, tick=self._tick, atr_1m=context.atr_1m
                ).to_dict()
            except GeometryError as error:  # the reducer already vetted it; record the reason if not
                geometry = {"error": str(error)}
            self._journal.write(
                "opportunity",
                episode_id=episode_id,
                known_at=context.known_at,
                payload={"opportunity": state.opportunity.to_dict(), "geometry": geometry},
            )

    def _archive(self, episode_id: str, known_at: pd.Timestamp, reason: str) -> None:
        state = self._state
        assert state is not None
        if self._journal is not None:
            self._journal.close_episode(
                episode_id, known_at,
                payload={
                    "revisions": state.revision + 1,
                    "reason": reason,
                    "summary": state.market_understanding,
                    "confidence": state.reasoning_confidence.value,
                },
            )
        self._last_archived = state
        self._state = None
        self._registry = None
        self._relations = {}
        self._relation_triggered = {}
        self._last_llm_known_at = None
        self._deferred = []
        self._idle_updates = 0
        self._status = RuntimeStatus.SLEEP

    def _count_idle(self, step: BrainStep) -> None:
        """An accepted update with no opportunity and an unchanged understanding
        is idle; anything else resets the run.  An incident leaves it alone."""
        update = step.outcome.update
        if update is None:
            return
        if update.understanding_holds and step.result.state.opportunity.state is OpportunityState.NONE:
            self._idle_updates += 1
        else:
            self._idle_updates = 0

    def _finish(self, result: StepResult, state: BrainState | None, llm_input: LLMInput | None) -> StepResult:
        if self._on_step is not None:
            self._on_step(result, state, llm_input)
        return result

    # --------------------------------------------------------------------- step

    def step(self, observation: MarketObservation) -> StepResult:
        known_at = pd.Timestamp(observation.asof).tz_convert("UTC")
        if self._last_known_at is not None and known_at <= self._last_known_at:
            raise ValueError(
                f"known_at {isoformat_utc(known_at)} does not advance past {isoformat_utc(self._last_known_at)}"
            )
        previous_known_at = self._last_known_at
        self._last_known_at = known_at
        events = observation.events_this_update

        if self._status is RuntimeStatus.SLEEP:
            with timed(self._timings, "controller"):
                decision = decide(events, active=False, config=self._controller, known_at=known_at, previous_known_at=previous_known_at)
            if decision.decision is Decision.STAY_ASLEEP:
                self._last_context = None
                return self._finish(
                    StepResult(known_at, Decision.STAY_ASLEEP, RuntimeStatus.SLEEP, None, None, False, None, False, ()),
                    None, None,
                )
            registry = ObjectRegistry()
            self._registry = registry
            context = build_eye_context(observation, registry, rule=self._controller.evidence, interaction_since=None)
            self._last_context = context
            episode_id = self._next_episode_id(known_at)
            if self._journal is not None:
                self._journal.open_episode(episode_id, known_at)
                self._journal.write(
                    "wake",
                    episode_id=episode_id,
                    known_at=known_at,
                    payload={"reasons": list(decision.reasons), "aliases": sorted(context.visible_aliases())},
                )
            self._tape = _Tape(self._controller.tape_recent_limit)
            self._deferred = []
            self._idle_updates = 0
            step = self._brain.step(
                episode_id=episode_id, context=context, trigger_kind="WAKE", reasons=decision.reasons,
                tape=self._tape.payload(), prior=None, registry=registry, tick=self._tick,
            )
            with timed(self._timings, "journal"):
                self._journal_llm(episode_id, context, step)
                self._journal_state(episode_id, context, step.result, None)
            self._state = step.result.state
            self._last_llm_known_at = known_at
            self._status = RuntimeStatus.ACTIVE
            self._tape = _Tape(self._controller.tape_recent_limit)
            self._remember_relations(self._state, context)
            if step.result.slept:
                self._archive(episode_id, known_at, step.result.sleep_reason or "continue_active=false")
            return self._finish(
                StepResult(
                    known_at, Decision.WAKE, self._status, episode_id, step.result.state.revision, True,
                    step.result.incident, step.result.slept, step.result.rejections, self._latency(step),
                ),
                step.result.state, step.llm_input,
            )

        # ACTIVE
        assert self._state is not None and self._registry is not None
        prev = self._state
        registry = self._registry
        context = build_eye_context(
            observation, registry, rule=self._controller.evidence, interaction_since=self._last_llm_known_at
        )
        self._last_context = context
        with timed(self._timings, "controller"):
            decision = decide(
                events, active=True, config=self._controller, relation_changes=self._relation_changes(context),
                known_at=known_at, previous_known_at=previous_known_at,
            )
        if decision.decision is Decision.UPDATE:
            self._relation_triggered_at([r for r in decision.reasons if not r.startswith("ev_")], known_at)
        episode_id = prev.episode_id
        if decision.decision is Decision.EVENT_SLEEP:
            # A scheduled release: the episode ends here, without a call, and
            # the executor withdraws every expression on this same bar.
            reason = decision.reasons[0]
            self._archive(episode_id, known_at, reason)
            return self._finish(
                StepResult(known_at, Decision.EVENT_SLEEP, RuntimeStatus.SLEEP, episode_id, prev.revision, False, None, True, (), event=reason),
                None, None,
            )
        if decision.decision is Decision.TICK:
            self._tape.add(events, self._controller, registry)
            self._deferred.extend(context.events)
            ctx = ReduceContext(
                known_at=known_at,
                has_open_position=engaged(self._ledger),
                open_interaction=context.open_interaction,
                visible_aliases=context.visible_aliases(),
                registry=registry.snapshot(),
                coherence=lambda opportunity: None,
                max_pending=self._brain.config.max_pending_evidence,
            )
            result = apply(prev, episode_id=episode_id, evidence=(), update=None, ctx=ctx)
            self._state = result.state
            if self._journal is not None:
                with timed(self._timings, "journal"):
                    self._journal.write(
                        "tick", episode_id=episode_id, known_at=known_at,
                        payload={"revision": result.state.revision, "tape_bars": self._tape.bars},
                    )
            return self._finish(
                StepResult(known_at, Decision.TICK, RuntimeStatus.ACTIVE, episode_id, result.state.revision, False, None, False, ()),
                self._state, None,
            )

        # UPDATE
        self._tape.add(events, self._controller, registry)
        step = self._brain.step(
            episode_id=episode_id, context=context, trigger_kind="UPDATE", reasons=decision.reasons,
            tape=self._tape.payload(), prior=prev, registry=registry, tick=self._tick,
            deferred=tuple(self._deferred), idle_updates=self._idle_updates,
            idle_archive_after=self._controller.idle_archive_after_updates,
        )
        with timed(self._timings, "journal"):
            self._journal_llm(episode_id, context, step)
            self._journal_state(episode_id, context, step.result, prev)
        self._state = step.result.state
        self._last_llm_known_at = known_at
        self._tape = _Tape(self._controller.tape_recent_limit)
        self._deferred = []
        self._count_idle(step)
        self._remember_relations(self._state, context)
        slept = step.result.slept
        if slept:
            self._archive(episode_id, known_at, step.result.sleep_reason or "continue_active=false")
        return self._finish(
            StepResult(
                known_at, Decision.UPDATE, self._status, episode_id, step.result.state.revision, True,
                step.result.incident, slept, step.result.rejections, self._latency(step),
            ),
            step.result.state, step.llm_input,
        )


__all__ = ["BrainRuntime", "RuntimeStatus", "StepHook", "StepResult"]
