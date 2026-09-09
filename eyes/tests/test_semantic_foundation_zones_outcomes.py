from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import pickle
from typing import Any

import pandas as pd
import pytest

from eyes.core.foundation_registry import FOUNDATION_VERSION
from contract.market import (
    Direction,
    Timeframe,
)
from contract.eye import (
    BOSLifecycle,
    BOSScope,
    EventKind,
    EventOrigin,
    MarketEvent,
)
from eyes.core.observation import CausalObserver, ObserverConfig
from eyes.core.semantic_zones import (
    BaseOriginCore,
    CompatibleStructureKind,
    CompletedZoneBar,
    FVGAvailability,
    FVGStructuralLifecycle,
    FVGTerminationCause,
    ZoneEntrySide,
    ZoneFirstReinteractionTracker,
    ZoneFirstRetestSpec,
    ZoneObjectKind,
    bind_fvg_structural_context,
    qualify_order_block,
    reduce_fvg_termination,
)
from eyes.core.structural_outcome import (
    OutcomeBar,
    OutcomeTerminal,
    StructuralOutcomeEngine,
    StructuralOutcomeSpec,
    project_conservative_execution,
)
from eyes.tests.test_v3_displacement_replay import (
    CORE_TEST_SCALE_SPECS,
    GROUP12_PROTOCOL_PATH,
    GROUP3_PROTOCOL_PATH,
    PROTOCOL_PATH,
    _m5 as _observer_m5,
    _update as _observer_update,
)
from eyes.tests.test_zone_primitives import (
    _form_fvg,
    _form_order_block,
)


_START = pd.Timestamp("2024-06-03T14:30:00Z")


def _at(*, minutes: int = 0, seconds: int = 0) -> pd.Timestamp:
    return _START + pd.Timedelta(minutes * 60 + seconds, unit="s")


def _base_origin_core() -> BaseOriginCore:
    return BaseOriginCore(
        symbol="ES",
        instrument_id=123,
        timeframe=Timeframe.M5,
        direction=Direction.LONG,
        source_displacement_id="disp-1",
        source_displacement_event_id="event-disp-1",
        anchor_bar_event_ids=("bar-anchor-1", "bar-anchor-2"),
        anchor_candle_ids=("candle-anchor-1", "candle-anchor-2"),
        anchor_completed_at=(_at(minutes=-10), _at(minutes=-5)),
        lower_bound=99.0,
        upper_bound=101.0,
        body_lower_bound=99.25,
        body_upper_bound=100.75,
        tick_size=0.25,
        formed_at=_at(minutes=-5),
        known_at=_START,
    )


def test_base_origin_core_rejects_a_gapped_anchor_cluster() -> None:
    values = _base_origin_core().__dict__.copy()
    values.pop("core_id")
    values["anchor_completed_at"] = (
        _at(minutes=-15),
        _at(minutes=-5),
    )

    with pytest.raises(ValueError, match="not contiguous"):
        BaseOriginCore(**values)


def _retest_spec() -> ZoneFirstRetestSpec:
    return ZoneFirstRetestSpec(
        object_kind=ZoneObjectKind.FVG,
        object_id="fvg-1",
        creation_event_id="event-fvg-1",
        symbol="ES",
        instrument_id=123,
        timeframe=Timeframe.M5,
        direction=Direction.LONG,
        lower_bound=100.0,
        upper_bound=102.0,
        tick_size=0.25,
        object_created_at=_at(minutes=-5),
        object_known_at=_START,
        departure_confirmed_at=_START,
        departure_source_event_id="event-departure-1",
        creation_declared_departed=True,
    )


def _zone_bar(
    minutes: int,
    *,
    bar_event_id: str | None = None,
    open_: float = 103.0,
    high: float = 104.0,
    low: float = 102.25,
    close: float = 103.5,
    instrument_id: int = 123,
    timeframe: Timeframe = Timeframe.M5,
) -> CompletedZoneBar:
    return CompletedZoneBar(
        bar_event_id=bar_event_id or f"bar-{minutes}",
        symbol="ES",
        instrument_id=instrument_id,
        timeframe=timeframe,
        known_at=_at(minutes=minutes),
        open=open_,
        high=high,
        low=low,
        close=close,
        session="RTH",
        context_event_ids=("context-parent-1",),
    )


def _fvg_lifecycle() -> FVGStructuralLifecycle:
    return FVGStructuralLifecycle(
        fvg_id="fvg-1",
        source_creation_event_id="event-fvg-1",
        symbol="ES",
        instrument_id=123,
        timeframe=Timeframe.M5,
        created_at=_at(minutes=-5),
        known_at=_START,
        parent_structure_generation_id="structure-generation-1",
        structural_range_id="structural-range-1",
        context_source_event_ids=(
            "event-structure-generation-1",
            "event-structural-range-1",
        ),
    )


def _outcome_spec(
    *,
    horizon_bars: int = 3,
    horizon_seconds: int = 900,
    window_end_minutes: int = 60,
) -> StructuralOutcomeSpec:
    return StructuralOutcomeSpec(
        source_event_id="sweep-1",
        symbol="ES",
        instrument_id=123,
        timeframe=Timeframe.M5,
        direction=Direction.LONG,
        observation_start_known_at=_START,
        observation_window_end_exclusive=_at(minutes=window_end_minutes),
        reference_price=100.0,
        target_price=102.0,
        invalidation_price=98.0,
        target_definition="plus_one_atr",
        invalidation_definition="minus_one_atr",
        atr_at_start=2.0,
        tick_size=0.25,
        horizon_bars=horizon_bars,
        horizon_seconds=horizon_seconds,
    )


def _outcome_bar(
    minutes: int,
    *,
    bar_event_id: str | None = None,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
    instrument_id: int = 123,
    timeframe: Timeframe = Timeframe.M5,
) -> OutcomeBar:
    return OutcomeBar(
        bar_event_id=bar_event_id or f"outcome-bar-{minutes}",
        symbol="ES",
        instrument_id=instrument_id,
        timeframe=timeframe,
        known_at=_at(minutes=minutes),
        open=open_,
        high=high,
        low=low,
        close=close,
    )




def test_base_origin_core_freezes_existing_full_cluster_geometry() -> None:
    core = _base_origin_core()

    assert core.semantic_version == FOUNDATION_VERSION
    assert core.lower_bound == 99.0
    assert core.upper_bound == 101.0
    assert core.width_ticks == 8
    assert core.source_event_ids == (
        "event-disp-1",
        "bar-anchor-1",
        "bar-anchor-2",
    )
    assert core.core_id == _base_origin_core().core_id
    with pytest.raises(FrozenInstanceError):
        core.lower_bound = 98.0  # type: ignore[misc]


def test_base_origin_core_rejects_unordered_or_off_grid_ancestry() -> None:
    values = _base_origin_core().__dict__.copy()
    values.pop("core_id")
    values["anchor_completed_at"] = (_at(minutes=-5), _at(minutes=-10))
    with pytest.raises(ValueError, match="ordered one-to-one"):
        BaseOriginCore(**values)

    values["anchor_completed_at"] = (_at(minutes=-10), _at(minutes=-5))
    values["lower_bound"] = 99.1
    with pytest.raises(ValueError, match="off-grid"):
        BaseOriginCore(**values)


@pytest.mark.parametrize(
    "kind",
    (
        CompatibleStructureKind.QUALIFIED_BOS,
        CompatibleStructureKind.MSS_CORE_CONFIRMED,
    ),
)
def test_qualified_order_block_binds_exact_core_and_structure_fact(
    kind: CompatibleStructureKind,
) -> None:
    core = _base_origin_core()
    qualified = qualify_order_block(
        core,
        source_displacement_id="disp-1",
        source_displacement_event_id="event-disp-1",
        compatible_structure_event_id=f"event-{kind.value}",
        compatible_structure_kind=kind,
        qualified_at=_at(minutes=5),
        known_at=_at(minutes=5),
    )

    assert qualified.base_origin_core_id == core.core_id
    assert qualified.source_displacement_id == core.source_displacement_id
    assert qualified.compatible_structure_kind is kind
    assert (
        qualified.lower_bound,
        qualified.upper_bound,
        qualified.body_lower_bound,
        qualified.body_upper_bound,
    ) == (
        core.lower_bound,
        core.upper_bound,
        core.body_lower_bound,
        core.body_upper_bound,
    )


@pytest.mark.parametrize(
    ("displacement_id", "displacement_event_id"),
    (("sibling-disp", "event-disp-1"), ("disp-1", "sibling-event")),
)
def test_qualified_order_block_rejects_sibling_displacement_borrowing(
    displacement_id: str,
    displacement_event_id: str,
) -> None:
    with pytest.raises(ValueError, match="exact core"):
        qualify_order_block(
            _base_origin_core(),
            source_displacement_id=displacement_id,
            source_displacement_event_id=displacement_event_id,
            compatible_structure_event_id="event-qualified-bos",
            compatible_structure_kind=CompatibleStructureKind.QUALIFIED_BOS,
            qualified_at=_at(minutes=5),
            known_at=_at(minutes=5),
        )


def test_qualified_order_block_cannot_predate_its_core() -> None:
    with pytest.raises(ValueError, match="predates"):
        qualify_order_block(
            _base_origin_core(),
            source_displacement_id="disp-1",
            source_displacement_event_id="event-disp-1",
            compatible_structure_event_id="event-qualified-bos",
            compatible_structure_kind=CompatibleStructureKind.QUALIFIED_BOS,
            qualified_at=_at(minutes=-1),
            known_at=_at(minutes=5),
        )

    with pytest.raises(ValueError):
        qualify_order_block(
            _base_origin_core(),
            source_displacement_id="disp-1",
            source_displacement_event_id="event-disp-1",
            compatible_structure_event_id="event-raw-break",
            compatible_structure_kind="raw_boundary_break",  # type: ignore[arg-type]
            qualified_at=_at(minutes=5),
            known_at=_at(minutes=5),
        )


def test_first_retest_registration_supports_later_observed_departure() -> None:
    values = _retest_spec().__dict__.copy()
    values["creation_declared_departed"] = False
    values["departure_confirmed_at"] = None
    values["departure_source_event_id"] = None
    waiting = ZoneFirstRetestSpec(**values)
    assert waiting.creation_declared_departed is False

    values["creation_declared_departed"] = True
    values["departure_confirmed_at"] = _at(minutes=1)
    with pytest.raises(ValueError, match="departure"):
        ZoneFirstRetestSpec(**values)

    values["departure_confirmed_at"] = _at(minutes=-10)
    with pytest.raises(ValueError, match="departure"):
        ZoneFirstRetestSpec(**values)


def test_first_retest_waits_for_future_departure_then_first_reentry() -> None:
    values = _retest_spec().__dict__.copy()
    values.update(
        creation_declared_departed=False,
        departure_confirmed_at=None,
        departure_source_event_id=None,
    )
    tracker = ZoneFirstReinteractionTracker(ZoneFirstRetestSpec(**values))

    still_inside = tracker.on_completed_bar(
        _zone_bar(
            5,
            open_=101.0,
            high=101.75,
            low=100.25,
            close=101.5,
        )
    )
    assert still_inside.departure_confirmed_at is None
    assert still_inside.first_retest is None

    departed = still_inside.on_completed_bar(
        _zone_bar(
            10,
            open_=101.5,
            high=103.0,
            low=101.25,
            close=102.5,
        )
    )
    assert departed.departure_confirmed_at == _at(minutes=10)
    assert departed.departure_source_event_id == "bar-10"
    assert departed.first_retest is None

    reentered = departed.on_completed_bar(
        _zone_bar(
            15,
            open_=103.0,
            high=103.25,
            low=102.0,
            close=102.5,
        )
    )
    assert reentered.first_retest is not None
    assert reentered.first_retest.departure_source_event_id == "bar-10"
    assert reentered.first_retest.source_bar_event_id == "bar-15"
    assert reentered.first_retest.age_bars == 3


def test_first_retest_is_strictly_later_and_boundary_equality_counts() -> None:
    tracker = ZoneFirstReinteractionTracker(_retest_spec())
    with pytest.raises(ValueError, match="strictly later"):
        tracker.on_completed_bar(
            _zone_bar(
                0,
                open_=101.0,
                high=102.0,
                low=100.0,
                close=101.0,
            )
        )

    untouched = tracker.on_completed_bar(_zone_bar(5))
    terminal = untouched.on_completed_bar(
        _zone_bar(
            10,
            open_=103.0,
            high=104.0,
            low=102.0,
            close=102.5,
        )
    )
    event = terminal.first_retest

    assert event is not None
    assert event.entry_side is ZoneEntrySide.FROM_ABOVE
    assert event.fill_fraction == 0.0
    assert event.age_bars == 2
    assert event.age_seconds == 600
    assert event.source_bar_event_id == "bar-10"
    assert event.source_event_ids == (
        "event-fvg-1",
        "event-departure-1",
        "bar-10",
    )
    assert event.session == "RTH"
    assert event.context_event_ids == ("context-parent-1",)


def test_first_retest_gap_open_inside_and_future_path_invariance() -> None:
    initial = ZoneFirstReinteractionTracker(_retest_spec())
    retest_bar = _zone_bar(
        5,
        open_=101.0,
        high=101.5,
        low=100.5,
        close=101.25,
    )
    prefix_result = initial.on_completed_bar(retest_bar)
    event = prefix_result.first_retest
    assert event is not None
    assert event.entry_side is ZoneEntrySide.GAP_OPENED_INSIDE
    assert event.fill_fraction == 0.75

    future_extreme = _zone_bar(
        10,
        open_=103.0,
        high=110.0,
        low=90.0,
        close=100.0,
    )
    full_path_result = prefix_result.on_completed_bar(future_extreme)
    assert full_path_result is prefix_result
    assert full_path_result.first_retest is event
    assert set(event.__dict__).isdisjoint(
        {
            "eventual_midpoint_touch",
            "eventual_full_fill",
            "eventual_invalidation",
            "eventual_continuation",
        }
    )


def test_first_retest_rejects_non_native_timeframe() -> None:
    with pytest.raises(ValueError, match="scope differs"):
        ZoneFirstReinteractionTracker(_retest_spec()).on_completed_bar(
            _zone_bar(5, timeframe=Timeframe.M1)
        )


def test_first_retest_rejects_a_missing_native_bar_before_later_contact() -> None:
    tracker = ZoneFirstReinteractionTracker(_retest_spec()).on_completed_bar(
        _zone_bar(5)
    )

    with pytest.raises(ValueError, match="not contiguous"):
        tracker.on_completed_bar(
            _zone_bar(
                15,
                open_=103.0,
                high=104.0,
                low=102.0,
                close=102.5,
            )
        )


def test_first_retest_advances_across_registered_memorial_closure() -> None:
    memorial_close = pd.Timestamp("2024-05-27 13:00", tz="America/New_York")
    prior_completion = memorial_close - pd.Timedelta(5, unit="min")
    spec = replace(
        _retest_spec(),
        object_created_at=prior_completion - pd.Timedelta(5, unit="min"),
        object_known_at=prior_completion,
        departure_confirmed_at=prior_completion,
    )
    at_close = replace(
        _zone_bar(5),
        bar_event_id="memorial-close-m5",
        known_at=memorial_close,
    )
    first_reopen = replace(
        _zone_bar(10),
        bar_event_id="memorial-reopen-m5",
        known_at=pd.Timestamp(
            "2024-05-27 18:05",
            tz="America/New_York",
        ),
    )

    tracker = ZoneFirstReinteractionTracker(spec).on_completed_bar(at_close)
    advanced = tracker.on_completed_bar(first_reopen)

    assert advanced.observed_native_bars == 2
    assert advanced.last_observed_at == first_reopen.known_at
    assert advanced.first_retest is None


def test_first_retest_rejects_skipping_first_memorial_reopen_bucket() -> None:
    memorial_close = pd.Timestamp("2024-05-27 13:00", tz="America/New_York")
    prior_completion = memorial_close - pd.Timedelta(5, unit="min")
    spec = replace(
        _retest_spec(),
        object_created_at=prior_completion - pd.Timedelta(5, unit="min"),
        object_known_at=prior_completion,
        departure_confirmed_at=prior_completion,
    )
    tracker = ZoneFirstReinteractionTracker(spec).on_completed_bar(
        replace(
            _zone_bar(5),
            bar_event_id="memorial-close-m5",
            known_at=memorial_close,
        )
    )

    with pytest.raises(ValueError, match="not contiguous"):
        tracker.on_completed_bar(
            replace(
                _zone_bar(15),
                bar_event_id="memorial-skipped-reopen-m5",
                known_at=pd.Timestamp(
                    "2024-05-27 18:10",
                    tz="America/New_York",
                ),
            )
        )


@pytest.mark.parametrize(
    ("cause", "availability"),
    (
        (
            FVGTerminationCause.CLOSE_THROUGH_FAR_EDGE,
            FVGAvailability.INVALIDATED,
        ),
        (FVGTerminationCause.CONTRACT_ROLLOVER, FVGAvailability.EXPIRED),
        (FVGTerminationCause.SEMANTIC_RESET, FVGAvailability.EXPIRED),
        (FVGTerminationCause.DATA_GAP, FVGAvailability.CENSORED),
    ),
)
def test_fvg_terminal_causes_have_frozen_dispositions(
    cause: FVGTerminationCause,
    availability: FVGAvailability,
) -> None:
    state = _fvg_lifecycle()
    terminal = reduce_fvg_termination(
        state,
        cause=cause,
        known_at=_at(minutes=5),
        cause_event_ids=(f"event-{cause.value}",),
    )

    assert terminal.availability is availability
    assert terminal.terminal_reason is cause
    assert reduce_fvg_termination(
        terminal,
        cause=cause,
        known_at=_at(minutes=5),
        cause_event_ids=(f"event-{cause.value}",),
    ) is terminal


@pytest.mark.parametrize(
    ("cause", "related_entity_id"),
    (
        (
            FVGTerminationCause.PARENT_STRUCTURE_TERMINATED,
            "structure-generation-1",
        ),
        (
            FVGTerminationCause.STRUCTURAL_RANGE_REPLACED,
            "structural-range-1",
        ),
    ),
)
def test_fvg_structural_expiry_binds_exact_parent_or_range(
    cause: FVGTerminationCause,
    related_entity_id: str,
) -> None:
    terminal = reduce_fvg_termination(
        _fvg_lifecycle(),
        cause=cause,
        known_at=_at(minutes=5),
        cause_event_ids=(f"event-{cause.value}",),
        related_entity_id=related_entity_id,
    )
    assert terminal.availability is FVGAvailability.EXPIRED
    assert terminal.terminal_source_event_ids[:3] == (
        "event-fvg-1",
        "event-structure-generation-1",
        "event-structural-range-1",
    )

    with pytest.raises(ValueError, match="exact"):
        reduce_fvg_termination(
            _fvg_lifecycle(),
            cause=cause,
            known_at=_at(minutes=5),
            cause_event_ids=(f"event-{cause.value}",),
            related_entity_id="sibling-entity",
        )


def test_fvg_creation_context_cannot_be_bound_after_age_or_rewritten() -> None:
    unbound = replace(
        _fvg_lifecycle(),
        parent_structure_generation_id=None,
        structural_range_id=None,
        context_source_event_ids=(),
    )
    bound = bind_fvg_structural_context(
        unbound,
        parent_structure_generation_id="structure-generation-1",
        structural_range_id="structural-range-1",
        source_event_ids=(
            "event-structure-generation-1",
            "event-structural-range-1",
        ),
    )
    assert bound.parent_structure_generation_id == "structure-generation-1"
    assert bind_fvg_structural_context(
        bound,
        parent_structure_generation_id="structure-generation-1",
        structural_range_id="structural-range-1",
        source_event_ids=bound.context_source_event_ids,
    ) is bound

    with pytest.raises(ValueError, match="immutable"):
        bind_fvg_structural_context(
            bound,
            parent_structure_generation_id="structure-generation-2",
            structural_range_id="structural-range-2",
            source_event_ids=("event-structure-generation-2",),
        )
    with pytest.raises(ValueError, match="creation time"):
        bind_fvg_structural_context(
            replace(
                unbound,
                age_bars=1,
                age_seconds=300,
                last_updated_at=_at(minutes=5),
            ),
            parent_structure_generation_id="structure-generation-1",
            structural_range_id=None,
            source_event_ids=("event-structure-generation-1",),
        )


def test_fvg_has_no_ttl_and_terminal_generation_is_immutable() -> None:
    with pytest.raises(ValueError):
        FVGTerminationCause("age_ttl")

    expired = reduce_fvg_termination(
        _fvg_lifecycle(),
        cause=FVGTerminationCause.SEMANTIC_RESET,
        known_at=_at(minutes=5),
        cause_event_ids=("event-semantic-reset",),
    )
    with pytest.raises(ValueError, match="immutable"):
        reduce_fvg_termination(
            expired,
            cause=FVGTerminationCause.DATA_GAP,
            known_at=_at(minutes=10),
            cause_event_ids=("event-data-gap",),
        )


def test_outcome_scans_strictly_after_start_and_freezes_full_spec() -> None:
    spec = _outcome_spec()
    outcome = StructuralOutcomeEngine.evaluate(
        spec,
        (
            _outcome_bar(
                0,
                bar_event_id="source-bar",
                high=103.0,
                low=97.0,
            ),
            _outcome_bar(5, high=102.0, low=99.0, close=101.0),
            _outcome_bar(10, high=101.0, low=97.75, close=99.0),
            _outcome_bar(15, high=100.5, low=99.5, close=100.0),
        ),
    )

    assert outcome.terminal is OutcomeTerminal.TARGET_FIRST
    assert outcome.target_hit_at == _at(minutes=5)
    assert outcome.invalidation_hit_at == _at(minutes=10)
    assert outcome.source_bar_event_ids == (
        "outcome-bar-5",
        "outcome-bar-10",
        "outcome-bar-15",
    )
    assert outcome.mfe_atr == 1.0
    assert outcome.mae_atr == 1.125
    assert outcome.atr_at_start == spec.atr_at_start
    assert outcome.observation_window_end_exclusive == (
        spec.observation_window_end_exclusive
    )
    assert (outcome.symbol, outcome.instrument_id) == (
        spec.symbol,
        spec.instrument_id,
    )


def test_outcome_same_bar_is_factually_ambiguous_and_projection_is_separate() -> None:
    outcome = StructuralOutcomeEngine.evaluate(
        _outcome_spec(horizon_bars=1),
        (_outcome_bar(5, high=102.0, low=98.0),),
    )
    factual_id = outcome.outcome_id

    assert outcome.terminal is OutcomeTerminal.AMBIGUOUS_SAME_BAR
    projection = project_conservative_execution(outcome)
    assert projection.factual_outcome_id == factual_id
    assert projection.factual_terminal is OutcomeTerminal.AMBIGUOUS_SAME_BAR
    assert projection.execution_terminal is OutcomeTerminal.INVALIDATION_FIRST
    assert projection.conservative_assumption_applied
    assert outcome.terminal is OutcomeTerminal.AMBIGUOUS_SAME_BAR
    assert outcome.outcome_id == factual_id


def test_outcome_contract_change_and_incomplete_paths_are_censored() -> None:
    contract_censored = StructuralOutcomeEngine.evaluate(
        _outcome_spec(horizon_bars=2),
        (_outcome_bar(5, instrument_id=999),),
    )
    assert contract_censored.terminal is OutcomeTerminal.CENSORED
    assert contract_censored.observed_bars == 0
    assert contract_censored.path_censor_reason == "contract_change"
    assert contract_censored.resolved_at == _at(minutes=5)

    incomplete = StructuralOutcomeEngine.evaluate(
        _outcome_spec(horizon_bars=3),
        (_outcome_bar(5),),
    )
    assert incomplete.terminal is OutcomeTerminal.CENSORED
    assert incomplete.observed_bars == 1
    assert incomplete.path_censor_reason == "incomplete_completed_bar_census"
    assert incomplete.mfe_atr is None
    assert incomplete.mae_atr is None


def test_outcome_window_boundary_is_explicit_censoring() -> None:
    outcome = StructuralOutcomeEngine.evaluate(
        _outcome_spec(
            horizon_bars=3,
            horizon_seconds=1_800,
            window_end_minutes=10,
        ),
        (_outcome_bar(5), _outcome_bar(10)),
    )
    assert outcome.terminal is OutcomeTerminal.CENSORED
    assert outcome.path_censor_reason == "observation_window_end"
    assert outcome.resolved_at == _at(minutes=10)
    assert outcome.source_bar_event_ids == ("outcome-bar-5",)


def test_outcome_seconds_horizon_and_native_bar_clocks_are_frozen() -> None:
    elapsed = StructuralOutcomeEngine.evaluate(
        _outcome_spec(horizon_bars=4, horizon_seconds=600),
        (_outcome_bar(5), _outcome_bar(10)),
    )
    assert elapsed.terminal is OutcomeTerminal.CENSORED
    assert elapsed.path_censor_reason == "horizon_seconds_elapsed"
    assert elapsed.resolved_at == _at(minutes=10)

    with pytest.raises(ValueError, match="duplicated or out of order"):
        StructuralOutcomeEngine.evaluate(
            _outcome_spec(horizon_bars=2),
            (
                _outcome_bar(5, bar_event_id="bar-a"),
                _outcome_bar(5, bar_event_id="bar-b"),
            ),
        )


def test_outcome_censors_at_first_missing_native_completed_bar() -> None:
    outcome = StructuralOutcomeEngine.evaluate(
        _outcome_spec(horizon_bars=2, horizon_seconds=3_600),
        (
            _outcome_bar(5),
            _outcome_bar(50, high=110.0, low=90.0),
        ),
    )

    assert outcome.terminal is OutcomeTerminal.CENSORED
    assert outcome.observed_bars == 1
    assert outcome.source_bar_event_ids == ("outcome-bar-5",)
    assert outcome.path_censor_reason == "missing_native_completed_bar"
    assert outcome.resolved_at == _at(minutes=10)
    assert outcome.target_hit_at is None
    assert outcome.invalidation_hit_at is None


def test_outcome_is_deterministic_and_invariant_to_bars_after_horizon() -> None:
    spec = _outcome_spec(horizon_bars=2)
    prefix = (_outcome_bar(5), _outcome_bar(10))
    first = StructuralOutcomeEngine.evaluate(spec, prefix)
    repeated = StructuralOutcomeEngine.evaluate(spec, prefix)
    with_future = StructuralOutcomeEngine.evaluate(
        spec,
        (
            *prefix,
            _outcome_bar(15, high=110.0, low=90.0),
        ),
    )

    assert first.terminal is OutcomeTerminal.CENSORED
    assert first.full_horizon_observed
    assert first.path_censor_reason is None
    assert first.outcome_id == repeated.outcome_id == with_future.outcome_id


def test_outcome_rejects_off_grid_input_and_wrong_timeframe() -> None:
    spec = _outcome_spec(horizon_bars=1)
    with pytest.raises(ValueError, match="off-grid"):
        StructuralOutcomeEngine.evaluate(
            spec,
            (_outcome_bar(5, high=101.1),),
        )
    with pytest.raises(ValueError, match="timeframe differs"):
        StructuralOutcomeEngine.evaluate(
            spec,
            (_outcome_bar(5, timeframe=Timeframe.M1),),
        )
