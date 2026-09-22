"""Eye → Brain → Risk → Execution, one bar at a time.

``TradingStack.step`` runs the Brain on the observation, turns an ACTIONABLE
opportunity into a ``TradePlan`` on the same bar's context, and hands the
order machine the plan and the bar — together with whether the LLM was
called on this bar (so a plan the LLM re-proposed after a veto is counted
as a proposal), the scales whose bar completed on it (the close-beyond
exit watches them), the Brain's bias direction (a position on the other
side is flattened, 2026-09-19) and the release window that put the Brain to
sleep on this bar (every expression is withdrawn, 2026-09-21).  ``halted`` mirrors the machine's drawdown halt so the
runner can stop.  Without a machine it is the Brain alone.  The machine's
ledger is the Brain's ``PositionLedger``, so the Brain cannot sleep while
an order works or a position is open, and its ``execution_view`` is what
the next LLM call reads in ``prior_state.execution``."""
from __future__ import annotations

from brain.core.runtime import BrainRuntime, StepResult
from contract.brain.state import BiasDirection
from contract.eye import EventKind, MarketObservation
from contract.market.primitives import Bar
from execution.core.order_fsm import OrderMachine
from execution.core.plan import plan_from_state
from shares.core.timing import NO_TIMINGS, Timings, timed


class TradingStack:
    def __init__(self, runtime: BrainRuntime, machine: OrderMachine | None, *, tick: float, timings: Timings = NO_TIMINGS) -> None:
        self.runtime = runtime
        self.machine = machine
        self._tick = float(tick)
        self._timings = timings
        self.last_trade_kinds: tuple[str, ...] = ()

    @property
    def halted(self) -> bool:
        return self.machine is not None and self.machine.halted

    @staticmethod
    def closed_timeframes(observation: MarketObservation) -> frozenset[str]:
        """The scales whose bar completed on this 1m bar."""
        return frozenset(
            event.timeframe.value for event in observation.events_this_update if event.kind is EventKind.BAR_COMPLETED
        )

    def step(self, observation: MarketObservation, bar: Bar | None) -> StepResult:
        result = self.runtime.step(observation)
        self.last_trade_kinds = ()
        if self.machine is None:
            return result
        context = self.runtime.last_context
        state = self.runtime.state
        with timed(self._timings, "plan"):
            plan = None if state is None or context is None else plan_from_state(state, context, tick=self._tick)
        visible_aliases = frozenset() if context is None else context.visible_aliases()
        with timed(self._timings, "machine"):
            self.last_trade_kinds = self.machine.on_bar(
                observation.asof, bar, plan, episode_id=result.episode_id, visible=lambda alias: alias in visible_aliases,
                llm_called=result.llm_called, closed_timeframes=self.closed_timeframes(observation),
                bias_direction=None if state is None or state.bias.direction is BiasDirection.NEUTRAL else state.bias.direction.value,
                event_sleep=result.event,
            )
        return result


__all__ = ["TradingStack"]
