from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from smc_trader.foundation_registry import FOUNDATION_VERSION
from smc_trader.market_state import DeliveryPhase, RelationRole, RelationState
from smc_trader.model import Direction, EventKind, MarketEvent, Timeframe
from smc_trader.semantic_foundation import (
    FoundationRecord,
    FoundationRecordStatus,
)
from smc_trader.semantic_lifecycle import (
    BoundaryAttackFact,
    DeliveryPhaseGeneration,
    GenerationLifecycle,
    LiquidityInteractionLifecycle,
    LiquidityInteractionTerminal,
    LiquidityLevelLifecycle,
    NormalizedLifecycleTransition,
    NormalizedTransitionKind,
    RelationGeneration,
    SemanticLifecycleReducer,
    StructureGenerationLifecycle,
    StructureTransitionLifecycle,
)


TZ = "America/New_York"


def _clock(minutes: int) -> pd.Timestamp:
    return pd.Timestamp("2024-06-03 09:30", tz=TZ) + pd.Timedelta(
        minutes, unit="m"
    )


def _fact(
    kind: NormalizedTransitionKind,
    minutes: int,
    *,
    timeframe: Timeframe | None = Timeframe.H1,
    payload: dict[str, object] | None = None,
    source_event_ids: tuple[str, ...] | None = None,
    fact_id: str | None = None,
) -> NormalizedLifecycleTransition:
    identity = fact_id or f"fact:{minutes}:{kind.value}:{timeframe}"
    return NormalizedLifecycleTransition(
        fact_id=identity,
        kind=kind,
        known_at=_clock(minutes),
        timeframe=timeframe,
        source_event_ids=source_event_ids or (f"source:{identity}",),
        payload=payload or {},
    )


def _create_level(
    state,
    *,
    minutes: int = 0,
    source_identity: str = "v1.2-level-a",
    side: str = "above",
    price_ticks: int = 400,
    source_timeframe: Timeframe = Timeframe.H1,
    interaction_timeframe: Timeframe = Timeframe.H1,
):
    created = _fact(
        NormalizedTransitionKind.LIQUIDITY_LEVEL_CREATED,
        minutes,
        timeframe=source_timeframe,
        payload={
            "source_kind": "confirmed_swing",
            "source_identity": source_identity,
            "side": side,
            "price_ticks": price_ticks,
            "tick_size": 0.25,
            "interaction_timeframe": interaction_timeframe.value,
        },
        source_event_ids=(f"level-source:{source_identity}",),
    )
    state = SemanticLifecycleReducer.reduce(state, created)
    return state, state.levels[-1]


def _real_bar(state, minutes: int, timeframe: Timeframe = Timeframe.H1):
    bar_id = f"bar:{timeframe.value}:{minutes}"
    fact = _fact(
        NormalizedTransitionKind.REAL_BAR_COMPLETED,
        minutes,
        timeframe=timeframe,
        payload={"bar_event_id": bar_id, "real_completed": True},
        source_event_ids=(bar_id,),
    )
    return SemanticLifecycleReducer.reduce(state, fact), bar_id


def _touch_penetrate_terminal(
    state,
    level_id: str,
    generation_id: str,
    *,
    minutes: int,
    terminal_kind: NormalizedTransitionKind,
    interaction_timeframe: Timeframe = Timeframe.H1,
):
    level = state.level(level_id)
    state, bar_id = _real_bar(state, minutes, interaction_timeframe)
    high_ticks = level.upper_bound_ticks + 2
    low_ticks = level.lower_bound_ticks - 2
    is_acceptance = (
        terminal_kind
        is NormalizedTransitionKind.LIQUIDITY_ACCEPTANCE_TERMINAL
    )
    first_close_ticks = (
        level.upper_bound_ticks + 1
        if level.side == "above" and is_acceptance
        else level.lower_bound_ticks - 1
        if level.side == "below" and is_acceptance
        else level.price_ticks
    )
    state = SemanticLifecycleReducer.reduce(
        state,
        _fact(
            NormalizedTransitionKind.LIQUIDITY_TOUCHED,
            minutes,
            timeframe=interaction_timeframe,
            payload={"level_id": level_id, "bar_event_id": bar_id},
            source_event_ids=(bar_id, f"touch-event:{minutes}"),
            fact_id=f"touch:{level_id}:{minutes}",
        ),
    )
    state = SemanticLifecycleReducer.reduce(
        state,
        _fact(
            NormalizedTransitionKind.LIQUIDITY_PENETRATED,
            minutes,
            timeframe=interaction_timeframe,
            payload={
                "level_id": level_id,
                "bar_event_id": bar_id,
                "penetration_ticks": 2,
                "high_ticks": high_ticks,
                "low_ticks": low_ticks,
                "close_ticks": first_close_ticks,
            },
            source_event_ids=(bar_id, f"penetration-event:{minutes}"),
            fact_id=f"penetration:{level_id}:{minutes}",
        ),
    )
    if is_acceptance:
        terminal_minutes = minutes + 1
        state, terminal_bar_id = _real_bar(
            state, terminal_minutes, interaction_timeframe
        )
        terminal_high = level.upper_bound_ticks + 3
        terminal_low = level.lower_bound_ticks - 2
        terminal_close = (
            level.upper_bound_ticks + 1
            if level.side == "above"
            else level.lower_bound_ticks - 1
        )
        ids = (bar_id, terminal_bar_id, terminal_bar_id)
        roles = ("penetration", "hold", "confirmation")
        clocks = (
            _clock(minutes),
            _clock(terminal_minutes),
            _clock(terminal_minutes),
        )
        highs = (high_ticks, terminal_high, terminal_high)
        lows = (low_ticks, terminal_low, terminal_low)
        closes = (first_close_ticks, terminal_close, terminal_close)
        maximum = 3
        first_inside = None
        first_outside = _clock(minutes)
    else:
        terminal_minutes = minutes
        terminal_bar_id = bar_id
        ids = (bar_id,) * 4
        roles = ("penetration", "reentry", "hold", "confirmation")
        clocks = (_clock(minutes),) * 4
        highs = (high_ticks,) * 4
        lows = (low_ticks,) * 4
        closes = (first_close_ticks,) * 4
        maximum = 2
        first_inside = _clock(minutes)
        first_outside = None
    state = SemanticLifecycleReducer.reduce(
        state,
        _fact(
            terminal_kind,
            terminal_minutes,
            timeframe=interaction_timeframe,
            payload={
                "level_id": level_id,
                "interaction_generation_id": generation_id,
                "terminal_event_id": f"terminal-event:{terminal_minutes}",
                "constituent_bar_ids": ids,
                "constituent_bar_roles": roles,
                "constituent_bar_known_at": clocks,
                "constituent_bar_high_ticks": highs,
                "constituent_bar_low_ticks": lows,
                "constituent_bar_close_ticks": closes,
                "max_penetration_ticks": maximum,
                "first_inside_close_at": first_inside,
                "first_outside_close_at": first_outside,
            },
            source_event_ids=tuple(
                dict.fromkeys(
                    (
                        bar_id,
                        terminal_bar_id,
                        f"terminal-event:{terminal_minutes}",
                    )
                )
            ),
            fact_id=(
                f"terminal:{terminal_kind.value}:{level_id}:{terminal_minutes}"
            ),
        ),
    )
    return state, terminal_bar_id


def _start_structure(
    state,
    *,
    minutes: int,
    timeframe: Timeframe,
    scope: str,
    direction: Direction,
    label: str,
):
    origin_event = f"origin-event:{label}"
    started = _fact(
        NormalizedTransitionKind.STRUCTURE_GENERATION_STARTED,
        minutes,
        timeframe=timeframe,
        payload={
            "scope": scope,
            "direction": direction.value,
            "origin_event_id": origin_event,
            "origin_swing_id": f"origin-swing:{label}",
        },
        source_event_ids=(origin_event,),
        fact_id=f"structure-start:{label}",
    )
    state = SemanticLifecycleReducer.reduce(state, started)
    generation = state.structure_generations[-1]
    confirmation_event = f"structure-confirmation:{label}"
    confirmed = _fact(
        NormalizedTransitionKind.STRUCTURE_GENERATION_CONFIRMED,
        minutes + 1,
        timeframe=timeframe,
        payload={
            "structure_generation_id": generation.structure_generation_id,
            "confirmation_event_id": confirmation_event,
        },
        source_event_ids=(confirmation_event,),
        fact_id=f"structure-confirm:{label}",
    )
    state = SemanticLifecycleReducer.reduce(state, confirmed)
    return state, state.structure(generation.structure_generation_id)


def test_market_event_adapter_preserves_v1_level_as_source_not_foundation_id() -> None:
    event = MarketEvent(
        event_id="v1-level-event",
        kind=EventKind.LIQUIDITY_LEVEL_CREATED,
        observed_at=_clock(0),
        timeframe=Timeframe.H1,
        side="above",
        price=100.0,
        strength=0.5,
        evidence={
            "level_id": "v1.2-swing-level",
            "source_kind": "confirmed_swing",
        },
    )
    normalized = NormalizedLifecycleTransition.from_market_event(
        event,
        kind=NormalizedTransitionKind.LIQUIDITY_LEVEL_CREATED,
        payload={
            "price_ticks": 400,
            "tick_size": 0.25,
            "interaction_timeframe": Timeframe.H1.value,
        },
    )
    state = SemanticLifecycleReducer.reduce(
        SemanticLifecycleReducer.initial_state(), normalized
    )
    level = state.levels[0]
    assert level.semantic_version == FOUNDATION_VERSION
    assert level.source_identity == "v1.2-swing-level"
    assert level.side == "above"
    assert level.level_id != "v1.2-swing-level"
    assert state.interactions[0].known_at == _clock(0)


def test_sweep_is_unique_terminal_and_rearm_requires_later_real_bar_and_departure() -> None:
    state, level = _create_level(
        SemanticLifecycleReducer.initial_state(),
        interaction_timeframe=Timeframe.M1,
    )
    first_generation_id = level.active_generation_id
    state, _ = _touch_penetrate_terminal(
        state,
        level.level_id,
        first_generation_id,
        minutes=1,
        terminal_kind=NormalizedTransitionKind.LIQUIDITY_SWEEP_TERMINAL,
        interaction_timeframe=Timeframe.M1,
    )
    terminal = state.interaction(first_generation_id)
    assert terminal.terminal_state is LiquidityInteractionTerminal.SWEEP
    assert state.level(level.level_id).lifecycle is LiquidityLevelLifecycle.DISARMED

    conflicting = _fact(
        NormalizedTransitionKind.LIQUIDITY_ACCEPTANCE_TERMINAL,
        1,
        timeframe=Timeframe.M1,
        payload={
            "level_id": level.level_id,
            "interaction_generation_id": first_generation_id,
            "constituent_bar_ids": ("bar:1m:1",),
        },
        source_event_ids=("bar:1m:1", "conflicting-acceptance"),
        fact_id="conflicting-terminal",
    )
    frozen_state = state
    with pytest.raises(ValueError):
        SemanticLifecycleReducer.reduce(state, conflicting)
    assert state == frozen_state
    assert state.interaction(first_generation_id) == terminal

    same_clock = _fact(
        NormalizedTransitionKind.LIQUIDITY_LEVEL_REARMABLE,
        1,
        timeframe=Timeframe.M1,
        payload={
            "level_id": level.level_id,
            "prior_generation_id": first_generation_id,
            "departure_price_ticks": 399,
            "departure_bar_event_id": "bar:1m:1",
        },
        source_event_ids=("bar:1m:1", "rearm:same-clock"),
    )
    with pytest.raises(ValueError, match="strictly later"):
        SemanticLifecycleReducer.reduce(state, same_clock)

    state, later_bar = _real_bar(state, 2, Timeframe.M1)
    wrong_side_departure = _fact(
        NormalizedTransitionKind.LIQUIDITY_LEVEL_REARMABLE,
        2,
        timeframe=Timeframe.M1,
        payload={
            "level_id": level.level_id,
            "prior_generation_id": first_generation_id,
            "departure_price_ticks": 401,
            "departure_bar_event_id": later_bar,
        },
        source_event_ids=(later_bar, "rearm:wrong-side"),
    )
    with pytest.raises(ValueError, match="one tick departure"):
        SemanticLifecycleReducer.reduce(state, wrong_side_departure)

    rearm = _fact(
        NormalizedTransitionKind.LIQUIDITY_LEVEL_REARMABLE,
        2,
        timeframe=Timeframe.M1,
        payload={
            "level_id": level.level_id,
            "prior_generation_id": first_generation_id,
            "departure_price_ticks": 399,
            "departure_ticks": 1,
            "departure_bar_event_id": later_bar,
        },
        source_event_ids=(later_bar, "rearm:qualified"),
        fact_id="rearm:qualified",
    )
    state = SemanticLifecycleReducer.reduce(state, rearm)
    eligible = state.level(level.level_id)
    assert eligible.lifecycle is LiquidityLevelLifecycle.REARMABLE
    assert eligible.active_generation_id is None
    activation = _fact(
        NormalizedTransitionKind.LIQUIDITY_LEVEL_REARMED,
        2,
        timeframe=Timeframe.M1,
        payload={
            "level_id": level.level_id,
            "prior_generation_id": first_generation_id,
            "rearmable_fact_id": rearm.fact_id,
            "departure_bar_event_id": later_bar,
        },
        source_event_ids=(later_bar, "rearm:activation"),
        fact_id="rearm:activation",
    )
    state = SemanticLifecycleReducer.reduce(state, activation)
    rearmed_level = state.level(level.level_id)
    second = state.interaction(rearmed_level.active_generation_id)
    assert rearmed_level.lifecycle is LiquidityLevelLifecycle.REARMED
    assert rearmed_level.rearmable_at == _clock(2)
    assert second.previous_generation_id == first_generation_id
    assert second.generation_number == 2
    assert terminal.terminal_state is LiquidityInteractionTerminal.SWEEP


def test_acceptance_permanently_retires_same_source_and_reset_censors_active() -> None:
    state, level = _create_level(
        SemanticLifecycleReducer.initial_state(),
        interaction_timeframe=Timeframe.M1,
    )
    state, _ = _touch_penetrate_terminal(
        state,
        level.level_id,
        level.active_generation_id,
        minutes=1,
        terminal_kind=NormalizedTransitionKind.LIQUIDITY_ACCEPTANCE_TERMINAL,
        interaction_timeframe=Timeframe.M1,
    )
    retired = state.level(level.level_id)
    assert retired.lifecycle is LiquidityLevelLifecycle.RETIRED
    assert retired.retirement_reason == "acceptance"
    recreate = _fact(
        NormalizedTransitionKind.LIQUIDITY_LEVEL_CREATED,
        2,
        payload={
            "source_kind": retired.source_kind,
            "source_identity": retired.source_identity,
            "side": retired.side,
            "price_ticks": retired.price_ticks,
            "tick_size": retired.tick_size,
            "interaction_timeframe": Timeframe.H1.value,
        },
        source_event_ids=("same-source-recreate",),
    )
    with pytest.raises(ValueError, match="same source"):
        SemanticLifecycleReducer.reduce(state, recreate)

    live_state, live_level = _create_level(
        SemanticLifecycleReducer.initial_state(), source_identity="reset-level"
    )
    reset = _fact(
        NormalizedTransitionKind.RESET,
        1,
        timeframe=None,
        payload={"reason": "semantic_reset"},
        source_event_ids=("epoch-reset",),
    )
    live_state = SemanticLifecycleReducer.reduce(live_state, reset)
    assert (
        live_state.interaction(live_level.active_generation_id).terminal_state
        is LiquidityInteractionTerminal.CENSORED
    )
    assert live_state.level(live_level.level_id).lifecycle is LiquidityLevelLifecycle.ARCHIVED


def test_h1_source_level_uses_m1_interaction_and_rearm_real_bar_clock() -> None:
    state, level = _create_level(
        SemanticLifecycleReducer.initial_state(),
        source_identity="h1-source-m1-crossing",
        source_timeframe=Timeframe.H1,
        interaction_timeframe=Timeframe.M1,
    )
    generation_id = level.active_generation_id
    generation = state.interaction(generation_id)
    assert level.source_timeframe is Timeframe.H1
    assert generation.source_timeframe is Timeframe.H1
    assert generation.interaction_timeframe is Timeframe.M1

    wrong_tf_touch = _fact(
        NormalizedTransitionKind.LIQUIDITY_TOUCHED,
        1,
        timeframe=Timeframe.H1,
        payload={"level_id": level.level_id, "bar_event_id": "h1-wrong-bar"},
        source_event_ids=("h1-wrong-bar",),
    )
    with pytest.raises(ValueError, match="foreign level"):
        SemanticLifecycleReducer.reduce(state, wrong_tf_touch)

    state, _ = _touch_penetrate_terminal(
        state,
        level.level_id,
        generation_id,
        minutes=1,
        terminal_kind=NormalizedTransitionKind.LIQUIDITY_SWEEP_TERMINAL,
        interaction_timeframe=Timeframe.M1,
    )
    state, m1_departure_bar = _real_bar(state, 2, Timeframe.M1)
    rearmable = _fact(
        NormalizedTransitionKind.LIQUIDITY_LEVEL_REARMABLE,
        2,
        timeframe=Timeframe.M1,
        payload={
            "level_id": level.level_id,
            "prior_generation_id": generation_id,
            "departure_price_ticks": 399,
            "departure_bar_event_id": m1_departure_bar,
        },
        source_event_ids=(m1_departure_bar, "m1-rearmable"),
        fact_id="m1-rearmable",
    )
    state = SemanticLifecycleReducer.reduce(state, rearmable)
    assert state.level(level.level_id).lifecycle is LiquidityLevelLifecycle.REARMABLE
    rearm = _fact(
        NormalizedTransitionKind.LIQUIDITY_LEVEL_REARMED,
        2,
        timeframe=Timeframe.M1,
        payload={
            "level_id": level.level_id,
            "prior_generation_id": generation_id,
            "rearmable_fact_id": rearmable.fact_id,
            "departure_bar_event_id": m1_departure_bar,
        },
        source_event_ids=(m1_departure_bar, "m1-rearm"),
    )
    state = SemanticLifecycleReducer.reduce(state, rearm)
    rearmed = state.interaction(state.level(level.level_id).active_generation_id)
    assert rearmed.interaction_timeframe is Timeframe.M1
    assert rearmed.previous_generation_id == generation_id


def test_multibar_interaction_ancestry_has_ordered_roles_and_strict_response_window() -> None:
    state, level = _create_level(
        SemanticLifecycleReducer.initial_state(),
        interaction_timeframe=Timeframe.M1,
    )
    generation_id = level.active_generation_id
    state, penetration_bar = _real_bar(state, 1, Timeframe.M1)
    state = SemanticLifecycleReducer.reduce(
        state,
        _fact(
            NormalizedTransitionKind.LIQUIDITY_TOUCHED,
            1,
            timeframe=Timeframe.M1,
            payload={"level_id": level.level_id, "bar_event_id": penetration_bar},
            source_event_ids=(penetration_bar, "multibar-touch"),
        ),
    )
    state = SemanticLifecycleReducer.reduce(
        state,
        _fact(
            NormalizedTransitionKind.LIQUIDITY_PENETRATED,
            1,
            timeframe=Timeframe.M1,
            payload={
                "level_id": level.level_id,
                "bar_event_id": penetration_bar,
                "penetration_ticks": 2,
                "high_ticks": 402,
                "low_ticks": 398,
                "close_ticks": 401,
            },
            source_event_ids=(penetration_bar, "multibar-penetration"),
        ),
    )
    state, reentry_bar = _real_bar(state, 2, Timeframe.M1)
    state, hold_bar = _real_bar(state, 3, Timeframe.M1)
    state, second_hold_bar = _real_bar(state, 4, Timeframe.M1)
    state, confirmation_bar = _real_bar(state, 5, Timeframe.M1)
    terminal = _fact(
        NormalizedTransitionKind.LIQUIDITY_SWEEP_TERMINAL,
        5,
        timeframe=Timeframe.M1,
        payload={
            "level_id": level.level_id,
            "interaction_generation_id": generation_id,
            "terminal_event_id": "multibar-terminal",
            "constituent_bar_ids": (
                penetration_bar,
                reentry_bar,
                hold_bar,
                second_hold_bar,
                confirmation_bar,
            ),
            "constituent_bar_roles": (
                "penetration",
                "hold",
                "hold",
                "reentry",
                "confirmation",
            ),
            "constituent_bar_known_at": (
                _clock(1),
                _clock(2),
                _clock(3),
                _clock(4),
                _clock(5),
            ),
            "constituent_bar_high_ticks": (402, 403, 408, 401, 401),
            "constituent_bar_low_ticks": (398, 399, 399, 398, 398),
            "constituent_bar_close_ticks": (401, 402, 401, 400, 399),
            "max_penetration_ticks": 8,
            "first_inside_close_at": _clock(4),
            "first_outside_close_at": _clock(1),
        },
        source_event_ids=(
            penetration_bar,
            reentry_bar,
            hold_bar,
            second_hold_bar,
            confirmation_bar,
            "multibar-terminal",
        ),
    )
    frozen = state
    wrong_maximum = replace(
        terminal,
        fact_id="multibar-terminal-wrong-maximum",
        payload={**dict(terminal.payload), "max_penetration_ticks": 2},
    )
    with pytest.raises(ValueError, match="max penetration"):
        SemanticLifecycleReducer.reduce(state, wrong_maximum)
    assert state == frozen
    wrong_roles = replace(
        terminal,
        fact_id="multibar-terminal-wrong-roles",
        payload={
            **dict(terminal.payload),
            "constituent_bar_roles": (
                "penetration",
                "reentry",
                "hold",
                "hold",
                "confirmation",
            ),
        },
    )
    with pytest.raises(ValueError, match="roles disagree"):
        SemanticLifecycleReducer.reduce(state, wrong_roles)
    assert state == frozen
    state = SemanticLifecycleReducer.reduce(state, terminal)
    resolved = state.interaction(generation_id)
    assert resolved.constituent_bar_roles == (
        "penetration",
        "hold",
        "hold",
        "reentry",
        "confirmation",
    )
    assert resolved.max_penetration_ticks == 8
    assert resolved.constituent_bar_ids[-1] == confirmation_bar
    with pytest.raises(ValueError, match="strictly after"):
        resolved.require_response_clock(_clock(5))
    assert resolved.require_response_clock(_clock(6)) == _clock(6)


def test_reference_retirement_expires_live_interaction_and_supersession_is_atomic() -> None:
    state, old = _create_level(
        SemanticLifecycleReducer.initial_state(),
        source_identity="previous-day-high:2024-05-31",
    )
    retired_fact = _fact(
        NormalizedTransitionKind.LIQUIDITY_LEVEL_RETIRED,
        1,
        payload={"level_id": old.level_id, "reason": "reference_rollover"},
        source_event_ids=("reference-rollover",),
    )
    state = SemanticLifecycleReducer.reduce(state, retired_fact)
    assert state.level(old.level_id).lifecycle is LiquidityLevelLifecycle.RETIRED
    assert (
        state.interaction(old.active_generation_id).terminal_state
        is LiquidityInteractionTerminal.EXPIRED
    )

    state, superseded = _create_level(
        SemanticLifecycleReducer.initial_state(), source_identity="swing-old"
    )
    create_replacement = _fact(
        NormalizedTransitionKind.LIQUIDITY_LEVEL_CREATED,
        1,
        payload={
            "source_kind": "confirmed_swing",
            "source_identity": "swing-new",
            "side": "above",
            "price_ticks": 404,
            "tick_size": 0.25,
            "interaction_timeframe": Timeframe.H1.value,
            "supersedes_level_id": superseded.level_id,
        },
        source_event_ids=("new-swing-level",),
    )
    state = SemanticLifecycleReducer.reduce(state, create_replacement)
    old_after = state.level(superseded.level_id)
    assert old_after.retirement_reason == "supersession"
    assert old_after.superseded_by_level_id == state.levels[-1].level_id
    assert state.levels[-1].side == "above"


def test_external_structure_owns_and_retires_all_same_timeframe_swing_levels() -> None:
    state, level = _create_level(
        SemanticLifecycleReducer.initial_state(),
        minutes=0,
        source_identity="owned-swing-before-generation",
    )
    assert level.owner_structure_generation_id is None
    state, generation = _start_structure(
        state,
        minutes=1,
        timeframe=Timeframe.H1,
        scope="external",
        direction=Direction.LONG,
        label="liquidity-owner",
    )
    owned = state.level(level.level_id)
    assert owned.owner_structure_generation_id == generation.generation_id

    state, later = _create_level(
        state,
        minutes=3,
        source_identity="owned-swing-created-inside-generation",
    )
    assert later.owner_structure_generation_id == generation.generation_id
    terminal_fact = _fact(
        NormalizedTransitionKind.STRUCTURE_GENERATION_TERMINATED,
        4,
        timeframe=Timeframe.H1,
        payload={
            "structure_generation_id": generation.generation_id,
            "reason": "scope_rollover",
        },
        source_event_ids=("structure-owner-rollover",),
    )
    state = SemanticLifecycleReducer.reduce(state, terminal_fact)

    for prior in (owned, later):
        retired = state.level(prior.level_id)
        assert retired.lifecycle is LiquidityLevelLifecycle.RETIRED
        assert retired.retirement_reason == "structure_generation_terminated"
        assert retired.active_generation_id is None
        interaction = state.interaction(prior.active_generation_id)
        assert interaction.terminal_state is LiquidityInteractionTerminal.EXPIRED
        assert interaction.terminal_reason == "structure_generation_terminated"


def test_mss_starts_transition_but_only_resumption_can_fail_it() -> None:
    state, incumbent = _start_structure(
        SemanticLifecycleReducer.initial_state(),
        minutes=0,
        timeframe=Timeframe.H1,
        scope="external",
        direction=Direction.LONG,
        label="incumbent-failure",
    )
    mss_id = "mss:failure"
    state = SemanticLifecycleReducer.reduce(
        state,
        _fact(
            NormalizedTransitionKind.MSS_TRANSITION_STARTED,
            2,
            payload={
                "incumbent_structure_generation_id": incumbent.structure_generation_id,
                "challenger_direction": Direction.SHORT.value,
                "mss_event_id": mss_id,
            },
            source_event_ids=(mss_id,),
        ),
    )
    candidate = state.structure_transitions[-1]
    assert candidate.lifecycle is StructureTransitionLifecycle.STARTED
    assert state.structure(incumbent.structure_generation_id).direction is Direction.LONG
    assert state.structure(incumbent.structure_generation_id).lifecycle is StructureGenerationLifecycle.CONFIRMED

    impossible_confirmation = _fact(
        NormalizedTransitionKind.STRUCTURE_TRANSITION_CONFIRMED,
        3,
        payload={
            "structure_transition_id": candidate.structure_transition_id,
            "protected_acceptance_event_id": "missing-acceptance",
            "opposite_structure_generation_id": incumbent.structure_generation_id,
            "opposite_confirmation_event_id": incumbent.confirmation_event_id,
        },
        source_event_ids=("missing-acceptance", incumbent.confirmation_event_id),
    )
    before = state
    with pytest.raises(ValueError):
        SemanticLifecycleReducer.reduce(state, impossible_confirmation)
    assert state == before

    resumption_event = "original-direction-resumed"
    state = SemanticLifecycleReducer.reduce(
        state,
        _fact(
            NormalizedTransitionKind.STRUCTURE_DIRECTION_RESUMED,
            4,
            payload={
                "structure_transition_id": candidate.structure_transition_id,
                "resumed_structure_generation_id": incumbent.structure_generation_id,
                "resumption_event_id": resumption_event,
            },
            source_event_ids=(resumption_event,),
        ),
    )
    failed = state.transition(candidate.structure_transition_id)
    assert failed.lifecycle is StructureTransitionLifecycle.FAILED
    assert failed.terminal_reason == "original_direction_resumed"


def test_structure_dtos_and_records_reject_partial_or_invented_provenance() -> None:
    state, incumbent = _start_structure(
        SemanticLifecycleReducer.initial_state(),
        minutes=0,
        timeframe=Timeframe.H1,
        scope="external",
        direction=Direction.LONG,
        label="provenance-hardening",
    )
    with pytest.raises(ValueError, match="provenance must be complete"):
        replace(incumbent, protected_swing_id="protected-low")
    with pytest.raises(ValueError, match="exact ancestry"):
        replace(
            incumbent,
            protected_swing_id="protected-low",
            protected_swing_assignment_event_id="invented-assignment",
        )
    assigned = replace(
        incumbent,
        protected_swing_id="protected-low",
        protected_swing_assignment_event_id="protected-assignment",
        source_event_ids=(
            *incumbent.source_event_ids,
            "protected-assignment",
        ),
    )
    FoundationRecord.from_dto(assigned)

    incumbent_record = FoundationRecord.from_dto(incumbent)
    partial_generation_payload = dict(incumbent_record.payload)
    partial_generation_payload["protected_swing_id"] = "protected-low"
    with pytest.raises(ValueError, match="provenance must be complete"):
        FoundationRecord(
            object_type=incumbent_record.object_type,
            object_id=incumbent_record.object_id,
            status=incumbent_record.status,
            known_at=incumbent_record.known_at,
            payload=partial_generation_payload,
            source_event_ids=incumbent_record.source_event_ids,
        )

    mss_event_id = "provenance-mss"
    state = SemanticLifecycleReducer.reduce(
        state,
        _fact(
            NormalizedTransitionKind.MSS_TRANSITION_STARTED,
            2,
            payload={
                "incumbent_structure_generation_id": (
                    incumbent.structure_generation_id
                ),
                "challenger_direction": Direction.SHORT.value,
                "mss_event_id": mss_event_id,
            },
            source_event_ids=(mss_event_id,),
        ),
    )
    started_record = FoundationRecord.from_dto(
        state.structure_transitions[-1]
    )
    invented_sources = (
        *started_record.source_event_ids,
        "invented-resumption",
    )
    invented_started_payload = dict(started_record.payload)
    invented_started_payload.update(
        {
            "resumption_event_id": "invented-resumption",
            "source_event_ids": invented_sources,
        }
    )
    with pytest.raises(ValueError, match="terminal evidence"):
        FoundationRecord(
            object_type=started_record.object_type,
            object_id=started_record.object_id,
            status=FoundationRecordStatus.ACTIVE,
            known_at=started_record.known_at,
            payload=invented_started_payload,
            source_event_ids=invented_sources,
        )

    confirmed_sources = (
        *started_record.source_event_ids,
        "protected-acceptance",
    )
    partial_confirmed_payload = dict(started_record.payload)
    partial_confirmed_payload.update(
        {
            "lifecycle": StructureTransitionLifecycle.CONFIRMED.value,
            "updated_at": _clock(3).isoformat(),
            "terminal_at": _clock(3).isoformat(),
            "terminal_reason": "invented-confirmation",
            "protected_acceptance_event_id": "protected-acceptance",
            "opposite_structure_generation_id": "opposite-generation",
            "opposite_confirmation_event_id": None,
            "source_event_ids": confirmed_sources,
        }
    )
    with pytest.raises(ValueError, match="opposite confirmation is incomplete"):
        FoundationRecord(
            object_type=started_record.object_type,
            object_id=started_record.object_id,
            status=FoundationRecordStatus.TERMINAL,
            known_at=_clock(3),
            payload=partial_confirmed_payload,
            source_event_ids=confirmed_sources,
        )


def test_transition_confirmation_requires_exact_acceptance_and_opposite_generation() -> None:
    state, incumbent = _start_structure(
        SemanticLifecycleReducer.initial_state(),
        minutes=0,
        timeframe=Timeframe.H1,
        scope="external",
        direction=Direction.LONG,
        label="incumbent-success",
    )
    mss_id = "mss:success"
    state = SemanticLifecycleReducer.reduce(
        state,
        _fact(
            NormalizedTransitionKind.MSS_TRANSITION_STARTED,
            2,
            payload={
                "incumbent_structure_generation_id": incumbent.structure_generation_id,
                "challenger_direction": Direction.SHORT.value,
                "mss_event_id": mss_id,
            },
            source_event_ids=(mss_id,),
        ),
    )
    candidate = state.structure_transitions[-1]
    acceptance = "acceptance:protected-low"
    state = SemanticLifecycleReducer.reduce(
        state,
        _fact(
            NormalizedTransitionKind.STRUCTURE_GENERATION_TERMINATED,
            3,
            payload={
                "structure_generation_id": incumbent.structure_generation_id,
                "reason": "protected_break_accepted",
                "protected_acceptance_event_id": acceptance,
            },
            source_event_ids=(acceptance,),
        ),
    )
    acceptance_evidence = _fact(
        NormalizedTransitionKind.STRUCTURE_TRANSITION_EVIDENCE,
        3,
        payload={
            "structure_transition_id": candidate.structure_transition_id,
            "protected_acceptance_event_id": acceptance,
        },
        source_event_ids=(acceptance,),
        fact_id="transition-acceptance-evidence",
    )
    state = SemanticLifecycleReducer.reduce(state, acceptance_evidence)
    staged = state.transition(candidate.structure_transition_id)
    assert staged.lifecycle is StructureTransitionLifecycle.STARTED
    assert staged.protected_acceptance_event_id == acceptance
    checkpoint = SemanticLifecycleReducer.checkpoint(state)
    state = SemanticLifecycleReducer.restore(checkpoint)
    state, opposite = _start_structure(
        state,
        minutes=3,
        timeframe=Timeframe.H1,
        scope="external",
        direction=Direction.SHORT,
        label="opposite-success",
    )
    state = SemanticLifecycleReducer.reduce(
        state,
        _fact(
            NormalizedTransitionKind.STRUCTURE_TRANSITION_CONFIRMED,
            4,
            payload={
                "structure_transition_id": candidate.structure_transition_id,
                "protected_acceptance_event_id": acceptance,
                "opposite_structure_generation_id": opposite.structure_generation_id,
                "opposite_confirmation_event_id": opposite.confirmation_event_id,
            },
            source_event_ids=(acceptance, opposite.confirmation_event_id),
        ),
    )
    confirmed = state.transition(candidate.structure_transition_id)
    assert confirmed.lifecycle is StructureTransitionLifecycle.CONFIRMED
    assert confirmed.opposite_structure_generation_id == opposite.structure_generation_id
    immutable = confirmed
    with pytest.raises(ValueError):
        SemanticLifecycleReducer.reduce(
            state,
            _fact(
                NormalizedTransitionKind.STRUCTURE_DIRECTION_RESUMED,
                5,
                payload={
                    "structure_transition_id": candidate.structure_transition_id,
                    "resumed_structure_generation_id": opposite.structure_generation_id,
                    "resumption_event_id": "late-conflict",
                },
                source_event_ids=("late-conflict",),
            ),
        )
    assert state.transition(candidate.structure_transition_id) == immutable


def _relation(
    minutes: int,
    role: RelationRole,
    *,
    parent_direction: Direction = Direction.LONG,
    child_direction: Direction = Direction.SHORT,
) -> RelationState:
    return RelationState(
        relation_id="1H__5m",
        parent_tf=Timeframe.H1,
        child_tf=Timeframe.M5,
        role=role,
        parent_direction=parent_direction,
        child_direction=child_direction,
        parent_protected_swing_intact=True,
        parent_state_invalidated=False,
        reversal_warning=False,
        child_location_in_parent_range=0.45,
        parent_invalidation_distance_atr=1.2,
        known_at=_clock(minutes),
        parent_source_cutoff=_clock(minutes),
        child_source_cutoff=_clock(minutes),
    )


def test_relation_generation_updates_once_per_signature_then_reclassifies() -> None:
    state, parent = _start_structure(
        SemanticLifecycleReducer.initial_state(),
        minutes=0,
        timeframe=Timeframe.H1,
        scope="external",
        direction=Direction.LONG,
        label="relation-parent",
    )
    state, child = _start_structure(
        state,
        minutes=1,
        timeframe=Timeframe.M5,
        scope="internal",
        direction=Direction.SHORT,
        label="relation-child",
    )
    state = SemanticLifecycleReducer.observe_relation(
        state,
        _relation(2, RelationRole.PARENT_RETRACEMENT),
        parent_structure_generation_id=parent.structure_generation_id,
        child_structure_generation_id=child.structure_generation_id,
        source_event_ids=("relation-observation:2",),
    )
    first_id = state.relation_generations[-1].relation_generation_id
    state = SemanticLifecycleReducer.observe_relation(
        state,
        _relation(3, RelationRole.PARENT_RETRACEMENT),
        parent_structure_generation_id=parent.structure_generation_id,
        child_structure_generation_id=child.structure_generation_id,
        source_event_ids=("relation-observation:3",),
    )
    assert len(state.relation_generations) == 1
    assert state.relation_generations[0].relation_generation_id == first_id
    assert state.relation_generations[0].observation_count == 2

    state = SemanticLifecycleReducer.observe_relation(
        state,
        _relation(4, RelationRole.REVERSAL_ATTEMPT),
        parent_structure_generation_id=parent.structure_generation_id,
        child_structure_generation_id=child.structure_generation_id,
        source_event_ids=("relation-observation:4",),
    )
    assert len(state.relation_generations) == 2
    assert state.relation_generations[0].termination_reason == "relation_reclassified"
    assert state.relation_generations[1].lifecycle is GenerationLifecycle.ACTIVE
    active_id = state.relation_generations[1].relation_generation_id
    state = SemanticLifecycleReducer.reduce(
        state,
        _fact(
            NormalizedTransitionKind.RELATION_TERMINATED,
            5,
            timeframe=Timeframe.M5,
            payload={
                "relation_generation_id": active_id,
                "reason": "semantic_reset",
            },
            source_event_ids=("relation-owner-disappeared",),
        ),
    )
    assert state.relation_generations[1].termination_reason == "semantic_reset"


@pytest.mark.parametrize(
    "reason",
    ("contract_reset", "data_reset", "semantic_reset"),
)
def test_reset_preserves_exact_relation_termination_reason(reason: str) -> None:
    state, parent = _start_structure(
        SemanticLifecycleReducer.initial_state(),
        minutes=0,
        timeframe=Timeframe.H1,
        scope="external",
        direction=Direction.LONG,
        label=f"reset-parent:{reason}",
    )
    state, child = _start_structure(
        state,
        minutes=1,
        timeframe=Timeframe.M5,
        scope="external",
        direction=Direction.SHORT,
        label=f"reset-child:{reason}",
    )
    state = SemanticLifecycleReducer.observe_relation(
        state,
        _relation(2, RelationRole.PARENT_RETRACEMENT),
        parent_structure_generation_id=parent.structure_generation_id,
        child_structure_generation_id=child.structure_generation_id,
        source_event_ids=(f"relation-before-reset:{reason}",),
    )

    state = SemanticLifecycleReducer.reduce(
        state,
        _fact(
            NormalizedTransitionKind.RESET,
            3,
            timeframe=None,
            payload={"reason": reason},
            source_event_ids=(f"epoch-reset:{reason}",),
        ),
    )

    assert state.relation_generations[-1].lifecycle is GenerationLifecycle.TERMINATED
    assert state.relation_generations[-1].termination_reason == reason


def test_delivery_generation_updates_age_extrema_and_coalesces_same_clock() -> None:
    state, parent = _start_structure(
        SemanticLifecycleReducer.initial_state(),
        minutes=0,
        timeframe=Timeframe.H1,
        scope="external",
        direction=Direction.LONG,
        label="delivery-parent",
    )
    origin = "delivery-origin:2"
    state = SemanticLifecycleReducer.observe_delivery_phase(
        state,
        DeliveryPhase.RETRACEMENT,
        timeframe=Timeframe.H1,
        known_at=_clock(2),
        parent_structure_generation_id=parent.structure_generation_id,
        origin_event_id=origin,
        source_event_ids=(origin,),
        current_price_ticks=400,
    )
    first_id = state.delivery_generations[-1].delivery_generation_id
    state, _ = _real_bar(state, 3)
    update_origin = "delivery-origin:3"
    state = SemanticLifecycleReducer.observe_delivery_phase(
        state,
        DeliveryPhase.RETRACEMENT,
        timeframe=Timeframe.H1,
        known_at=_clock(3),
        parent_structure_generation_id=parent.structure_generation_id,
        origin_event_id=update_origin,
        source_event_ids=(update_origin,),
        current_price_ticks=395,
        retracement_ticks=5.0,
    )
    assert len(state.delivery_generations) == 1
    updated = state.delivery_generations[0]
    assert updated.delivery_generation_id == first_id
    assert updated.observation_count == 2
    assert updated.origin_price_ticks == 400
    assert updated.current_price_ticks == 395
    assert updated.max_retracement_ticks == 5.0
    dishonest = NormalizedLifecycleTransition.from_delivery_phase(
        DeliveryPhase.RETRACEMENT,
        timeframe=Timeframe.H1,
        known_at=_clock(3),
        parent_structure_generation_id=parent.structure_generation_id,
        origin_event_id="dishonest-extrema",
        source_event_ids=("dishonest-extrema",),
        current_price_ticks=395,
        retracement_ticks=99.0,
    )
    with pytest.raises(ValueError, match="disagree"):
        SemanticLifecycleReducer.reduce(state, dishonest)

    expansion_origin = "delivery-origin:4"
    state = SemanticLifecycleReducer.observe_delivery_phase(
        state,
        DeliveryPhase.EXPANSION,
        timeframe=Timeframe.H1,
        known_at=_clock(4),
        parent_structure_generation_id=parent.structure_generation_id,
        origin_event_id=expansion_origin,
        source_event_ids=(expansion_origin,),
        current_price_ticks=403,
    )
    assert len(state.delivery_generations) == 2
    ended_retracement = state.delivery_generations[0]
    assert ended_retracement.next_phase == "expansion"
    assert ended_retracement.current_price_ticks == 403
    assert ended_retracement.observation_count == 3
    assert ended_retracement.max_extension_ticks == 3.0
    assert state.delivery_generations[1].lifecycle is GenerationLifecycle.ACTIVE

    same_clock_a = NormalizedLifecycleTransition.from_delivery_phase(
        DeliveryPhase.EXPANSION,
        timeframe=Timeframe.H1,
        known_at=_clock(5),
        parent_structure_generation_id=parent.structure_generation_id,
        origin_event_id="same-clock-a",
        source_event_ids=("same-clock-a",),
        current_price_ticks=404,
    )
    same_clock_b = NormalizedLifecycleTransition.from_delivery_phase(
        DeliveryPhase.RETRACEMENT,
        timeframe=Timeframe.H1,
        known_at=_clock(5),
        parent_structure_generation_id=parent.structure_generation_id,
        origin_event_id="same-clock-b",
        source_event_ids=("same-clock-b",),
        current_price_ticks=404,
    )
    frozen = state
    with pytest.raises(ValueError, match="coalesced"):
        SemanticLifecycleReducer.replay(
            (same_clock_a, same_clock_b), initial_state=state
        )
    assert state == frozen
    active_delivery_id = state.delivery_generations[-1].delivery_generation_id
    state = SemanticLifecycleReducer.reduce(
        state,
        _fact(
            NormalizedTransitionKind.DELIVERY_PHASE_TERMINATED,
            6,
            payload={
                "delivery_generation_id": active_delivery_id,
                "reason": "data_reset",
            },
            source_event_ids=("delivery-owner-disappeared",),
        ),
    )
    assert state.delivery_generations[-1].termination_reason == "data_reset"


def test_boundary_attack_is_strict_immutable_and_directional() -> None:
    state = SemanticLifecycleReducer.initial_state()
    attack = _fact(
        NormalizedTransitionKind.BOUNDARY_ATTACK_OBSERVED,
        0,
        payload={
            "bos_generation_id": "bos-generation:1",
            "direction": Direction.LONG.value,
            "target_swing_event_id": "target-swing-event",
            "bar_event_id": "attack-bar:1",
            "boundary_ticks": 400,
            "high_ticks": 402,
            "low_ticks": 397,
            "close_ticks": 400,
        },
        source_event_ids=("target-swing-event", "attack-bar:1"),
    )
    state = SemanticLifecycleReducer.reduce(state, attack)
    fact = state.boundary_attacks[0]
    assert fact.penetration_ticks == 2
    assert fact.attempt_ordinal == 1
    with pytest.raises(FrozenInstanceError):
        fact.close_ticks = 399
    checkpoint = SemanticLifecycleReducer.checkpoint(state)
    restored = SemanticLifecycleReducer.restore(checkpoint)
    assert restored == state
    assert SemanticLifecycleReducer.reduce(restored, attack) == state
    duplicate_semantic_bar = replace(attack, fact_id="duplicate-attack-input")
    with pytest.raises(ValueError, match="duplicate an attack BAR"):
        SemanticLifecycleReducer.reduce(state, duplicate_semantic_bar)

    equal_wick = _fact(
        NormalizedTransitionKind.BOUNDARY_ATTACK_OBSERVED,
        1,
        payload={
            "bos_generation_id": "bos-generation:1",
            "direction": Direction.LONG.value,
            "target_swing_event_id": "target-swing-event",
            "bar_event_id": "attack-bar:2",
            "boundary_ticks": 400,
            "high_ticks": 400,
            "low_ticks": 397,
            "close_ticks": 399,
        },
        source_event_ids=("target-swing-event", "attack-bar:2"),
    )
    with pytest.raises(ValueError, match="strict high"):
        SemanticLifecycleReducer.reduce(state, equal_wick)
    close_break = replace(
        equal_wick,
        fact_id="close-break-is-not-attack",
        payload={**dict(equal_wick.payload), "high_ticks": 402, "close_ticks": 401},
    )
    with pytest.raises(ValueError, match="without close break"):
        SemanticLifecycleReducer.reduce(state, close_break)


def test_checkpoint_resume_and_duplicate_replay_are_deterministic() -> None:
    create = _fact(
        NormalizedTransitionKind.LIQUIDITY_LEVEL_CREATED,
        0,
        payload={
            "source_kind": "confirmed_swing",
            "source_identity": "checkpoint-level",
            "side": "below",
            "price_ticks": 396,
            "tick_size": 0.25,
            "interaction_timeframe": Timeframe.H1.value,
        },
        source_event_ids=("checkpoint-level-source",),
    )
    bar = _fact(
        NormalizedTransitionKind.REAL_BAR_COMPLETED,
        1,
        payload={"bar_event_id": "checkpoint-bar", "real_completed": True},
        source_event_ids=("checkpoint-bar",),
    )
    prefix = SemanticLifecycleReducer.replay((create, bar))
    level = prefix.levels[0]
    touch = _fact(
        NormalizedTransitionKind.LIQUIDITY_TOUCHED,
        1,
        payload={"level_id": level.level_id, "bar_event_id": "checkpoint-bar"},
        source_event_ids=("checkpoint-bar", "checkpoint-touch"),
    )
    penetration = _fact(
        NormalizedTransitionKind.LIQUIDITY_PENETRATED,
        1,
        payload={
            "level_id": level.level_id,
            "bar_event_id": "checkpoint-bar",
            "penetration_ticks": 1,
            "high_ticks": 398,
            "low_ticks": 395,
            "close_ticks": 396,
        },
        source_event_ids=("checkpoint-bar", "checkpoint-penetration"),
    )
    full = SemanticLifecycleReducer.replay((create, bar, touch, penetration))
    checkpoint = SemanticLifecycleReducer.checkpoint(prefix)
    restored = SemanticLifecycleReducer.restore(checkpoint)
    resumed = SemanticLifecycleReducer.replay(
        (touch, penetration), initial_state=restored
    )
    assert resumed == full
    assert SemanticLifecycleReducer.reduce(resumed, penetration) == resumed
    with pytest.raises(ValueError, match="checkpoint"):
        replace(checkpoint, state_digest="0" * 64)


def test_common_generation_aliases_cover_active_and_terminal_dtos() -> None:
    liquidity_state, level = _create_level(
        SemanticLifecycleReducer.initial_state(), source_identity="common-alias"
    )
    interaction_active = liquidity_state.interaction(level.active_generation_id)
    interaction_terminal = replace(
        interaction_active,
        lifecycle=LiquidityInteractionLifecycle.TERMINAL,
        updated_at=_clock(10),
        terminal_event_id="common-reset",
        terminal_state=LiquidityInteractionTerminal.CENSORED,
        terminal_at=_clock(10),
        terminal_reason="semantic_reset",
        terminal_real_bar_ordinal=0,
    )

    structure_state, parent = _start_structure(
        SemanticLifecycleReducer.initial_state(),
        minutes=0,
        timeframe=Timeframe.H1,
        scope="external",
        direction=Direction.LONG,
        label="common-parent",
    )
    structure_active = parent
    structure_terminal = replace(
        structure_active,
        lifecycle=StructureGenerationLifecycle.TERMINATED,
        updated_at=_clock(10),
        terminated_at=_clock(10),
        termination_reason="scope_rollover",
    )
    structure_state, child = _start_structure(
        structure_state,
        minutes=1,
        timeframe=Timeframe.M5,
        scope="internal",
        direction=Direction.SHORT,
        label="common-child",
    )
    structure_state = SemanticLifecycleReducer.observe_relation(
        structure_state,
        _relation(3, RelationRole.PARENT_RETRACEMENT),
        parent_structure_generation_id=parent.structure_generation_id,
        child_structure_generation_id=child.structure_generation_id,
        source_event_ids=("common-relation",),
    )
    relation_active = structure_state.relation_generations[-1]
    relation_terminal = replace(
        relation_active,
        lifecycle=GenerationLifecycle.TERMINATED,
        last_updated_at=_clock(10),
        terminated_at=_clock(10),
        termination_reason="semantic_reset",
    )
    structure_state = SemanticLifecycleReducer.observe_delivery_phase(
        structure_state,
        DeliveryPhase.RETRACEMENT,
        timeframe=Timeframe.H1,
        known_at=_clock(4),
        parent_structure_generation_id=parent.structure_generation_id,
        origin_event_id="common-delivery",
        source_event_ids=("common-delivery",),
        current_price_ticks=400,
    )
    delivery_active = structure_state.delivery_generations[-1]
    delivery_terminal = replace(
        delivery_active,
        lifecycle=GenerationLifecycle.TERMINATED,
        last_updated_at=_clock(10),
        duration_seconds=360.0,
        terminated_at=_clock(10),
        termination_reason="semantic_reset",
    )

    for generation in (
        interaction_active,
        interaction_terminal,
        structure_active,
        structure_terminal,
        relation_active,
        relation_terminal,
        delivery_active,
        delivery_terminal,
    ):
        assert generation.generation_id
        assert generation.started_at <= generation.known_at <= generation.updated_at
        if generation.terminated_at is None:
            assert generation.termination_reason is None
        else:
            assert generation.updated_at <= generation.terminated_at
            assert generation.termination_reason


def test_generation_dtos_reject_timeframe_phase_and_boundary_counterexamples() -> None:
    with pytest.raises(ValueError, match="relation generation"):
        RelationGeneration(
            relation_generation_id="bad-relation",
            source_relation_id="1m:5m",
            parent_tf=Timeframe.M1,
            child_tf=Timeframe.M5,
            parent_structure_generation_id="parent",
            child_structure_generation_id="child",
            role="parent_retracement",
            lifecycle=GenerationLifecycle.ACTIVE,
            entered_at=_clock(0),
            known_at=_clock(0),
            last_updated_at=_clock(0),
            observation_count=1,
            latest_relation_digest="digest",
            source_event_ids=("relation-source",),
        )

    with pytest.raises(ValueError, match="active delivery"):
        DeliveryPhaseGeneration(
            delivery_generation_id="bad-delivery",
            timeframe=Timeframe.M5,
            phase="retracement",
            parent_structure_generation_id="parent",
            lifecycle=GenerationLifecycle.ACTIVE,
            entered_at=_clock(0),
            known_at=_clock(0),
            last_updated_at=_clock(0),
            entered_real_bar_ordinal=1,
            duration_bars=0,
            duration_seconds=0.0,
            origin_event_id="delivery-origin",
            observation_count=1,
            origin_price_ticks=400,
            current_price_ticks=400,
            max_extension_ticks=0.0,
            max_retracement_ticks=0.0,
            next_phase="expansion",
            source_event_ids=("delivery-origin",),
        )

    valid_attack = BoundaryAttackFact(
        boundary_attack_id="attack",
        bos_generation_id="bos",
        timeframe=Timeframe.M5,
        direction=Direction.LONG,
        target_swing_event_id="swing",
        bar_event_id="bar",
        boundary_ticks=400,
        extreme_ticks=402,
        close_ticks=400,
        penetration_ticks=2,
        attempt_ordinal=1,
        known_at=_clock(1),
        source_event_ids=("swing", "bar"),
    )
    with pytest.raises(ValueError, match="boundary-attack"):
        replace(valid_attack, close_ticks=401)
    with pytest.raises(ValueError, match="boundary-attack"):
        replace(valid_attack, penetration_ticks=1)
