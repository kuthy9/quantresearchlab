from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from smc_trader.signal_research import (
    BranchingResearchEpisode,
    ControlDirectionPolicy,
    MatchSpec,
    PseudoLevelSpec,
    ResearchContractError,
    ResearchLinkMode,
    TimeShiftSpec,
    TypedLinkSpec,
    build_forward_time_shift_controls,
    canonical_treatment_episodes,
    construct_pseudo_levels,
    deterministic_maximum_cardinality_match,
    exact_mcnemar,
    find_prior_typed_link,
    find_prior_typed_links,
    holm_adjust_fixed_family,
    project_branching_research_episode,
)


def _clock(minute: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-02T09:30:00-05:00") + pd.Timedelta(
        int(minute), unit="min"
    )


def _link_event(
    event_id: str,
    minute: int,
    *,
    kind: str,
    timeframe: str,
    lineage: tuple[str, ...] = (),
    constituent_bars: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "known_at": _clock(minute),
        "kind": kind,
        "timeframe": timeframe,
        "direction": "long",
        "symbol": "NQ",
        "instrument_id": 123,
        "lineage_tokens": tuple(
            sorted({*lineage, *(f"event:{value}" for value in constituent_bars)})
        ),
        "constituent_bar_event_ids": constituent_bars,
    }


def test_typed_strict_source_link_does_not_fall_back_to_time() -> None:
    prior = _link_event(
        "sweep:1",
        0,
        kind="sweep_confirmed",
        timeframe="1m",
        lineage=("event:m1-bar",),
    )
    current = _link_event(
        "disp:1",
        2,
        kind="displacement_observed",
        timeframe="5m",
        lineage=("event:m5-bar",),
    )
    spec = TypedLinkSpec(
        previous_kind="sweep_confirmed",
        previous_timeframe="1m",
        current_kind="displacement_observed",
        current_timeframe="5m",
        maximum_completed_bars=2,
    )

    assert (
        find_prior_typed_link(
            [prior],
            current,
            completed_index={_clock(0): 0, _clock(2): 1},
            spec=spec,
        )
        is None
    )


def test_typed_strict_source_link_requires_canonical_lookup() -> None:
    prior = _link_event(
        "sweep:1",
        0,
        kind="sweep_confirmed",
        timeframe="1m",
        lineage=("event:shared",),
    )
    current = _link_event(
        "disp:1",
        2,
        kind="displacement_observed",
        timeframe="5m",
        lineage=("event:sweep:1",),
    )
    spec = TypedLinkSpec(
        previous_kind="sweep_confirmed",
        previous_timeframe="1m",
        current_kind="displacement_observed",
        current_timeframe="5m",
        maximum_completed_bars=2,
    )
    completed = {_clock(0): 0, _clock(2): 1}

    with pytest.raises(ResearchContractError, match="canonical event lookup"):
        find_prior_typed_link([prior], current, completed_index=completed, spec=spec)

    canonical = {
        "sweep:1": SimpleNamespace(
            event_id="sweep:1",
            kind="sweep_confirmed",
            timeframe="1m",
            direction="long",
            symbol="NQ",
            instrument_id=123,
            origin="semantic_atomic",
            known_at=_clock(0),
            source_event_ids=(),
            context_event_ids=(),
        ),
        "disp:1": SimpleNamespace(
            event_id="disp:1",
            kind="displacement_observed",
            timeframe="5m",
            direction="long",
            symbol="NQ",
            instrument_id=123,
            origin="semantic_atomic",
            known_at=_clock(2),
            source_event_ids=("sweep:1",),
            context_event_ids=(),
        ),
    }
    link = find_prior_typed_link(
        [prior],
        current,
        completed_index=completed,
        spec=spec,
        event_lookup=canonical.get,
    )
    assert link is not None
    assert link.source_ancestry_proven is True
    assert link.composition_proven is False
    assert link.shared_event_ids == ("sweep:1",)


def test_typed_strict_source_link_rejects_siblings_with_shared_ancestor() -> None:
    prior = _link_event(
        "sweep:1",
        0,
        kind="sweep_confirmed",
        timeframe="1m",
        lineage=("event:shared",),
    )
    current = _link_event(
        "disp:1",
        2,
        kind="displacement_observed",
        timeframe="5m",
        lineage=("event:shared",),
    )
    common = {
        "direction": "long",
        "symbol": "NQ",
        "instrument_id": 123,
        "origin": "semantic_atomic",
        "source_event_ids": ("shared",),
        "context_event_ids": (),
    }
    canonical = {
        "shared": SimpleNamespace(
            event_id="shared",
            kind="swing_confirmed",
            timeframe="1m",
            direction="long",
            symbol="NQ",
            instrument_id=123,
            origin="semantic_atomic",
            known_at=_clock(0),
            source_event_ids=(),
            context_event_ids=(),
        ),
        "sweep:1": SimpleNamespace(
            **common,
            event_id="sweep:1",
            kind="sweep_confirmed",
            timeframe="1m",
            known_at=_clock(0),
        ),
        "disp:1": SimpleNamespace(
            **common,
            event_id="disp:1",
            kind="displacement_observed",
            timeframe="5m",
            known_at=_clock(2),
        ),
    }
    spec = TypedLinkSpec(
        previous_kind="sweep_confirmed",
        previous_timeframe="1m",
        current_kind="displacement_observed",
        current_timeframe="5m",
        maximum_completed_bars=2,
    )

    assert (
        find_prior_typed_link(
            [prior],
            current,
            completed_index={_clock(0): 0, _clock(2): 1},
            spec=spec,
            event_lookup=canonical.get,
        )
        is None
    )


def test_typed_strict_source_link_rejects_context_only_predecessor() -> None:
    prior = _link_event("sweep:1", 0, kind="sweep_confirmed", timeframe="1m")
    current = _link_event(
        "disp:1",
        2,
        kind="displacement_observed",
        timeframe="5m",
        lineage=("event:sweep:1",),
    )
    canonical = {
        "sweep:1": SimpleNamespace(
            **prior,
            origin="semantic_atomic",
            source_event_ids=(),
            context_event_ids=(),
        ),
        "disp:1": SimpleNamespace(
            **current,
            origin="semantic_atomic",
            source_event_ids=(),
            context_event_ids=("sweep:1",),
        ),
    }
    spec = TypedLinkSpec(
        previous_kind="sweep_confirmed",
        previous_timeframe="1m",
        current_kind="displacement_observed",
        current_timeframe="5m",
        maximum_completed_bars=2,
    )

    assert (
        find_prior_typed_link(
            [prior],
            current,
            completed_index={_clock(0): 0, _clock(2): 1},
            spec=spec,
            event_lookup=canonical.get,
        )
        is None
    )


def test_typed_strict_source_link_accepts_transitive_source_predecessor() -> None:
    prior = _link_event("sweep:1", 0, kind="sweep_confirmed", timeframe="1m")
    current = _link_event(
        "disp:1",
        2,
        kind="displacement_observed",
        timeframe="5m",
        lineage=("event:bridge:1", "event:sweep:1"),
    )
    canonical = {
        "sweep:1": SimpleNamespace(
            **prior,
            origin="semantic_atomic",
            source_event_ids=(),
            context_event_ids=(),
        ),
        "bridge:1": SimpleNamespace(
            event_id="bridge:1",
            kind="structural_leg_created",
            timeframe="5m",
            direction="long",
            symbol="NQ",
            instrument_id=123,
            origin="semantic_atomic",
            known_at=_clock(1),
            source_event_ids=("sweep:1",),
            context_event_ids=(),
        ),
        "disp:1": SimpleNamespace(
            **current,
            origin="semantic_atomic",
            source_event_ids=("bridge:1",),
            context_event_ids=(),
        ),
    }
    spec = TypedLinkSpec(
        previous_kind="sweep_confirmed",
        previous_timeframe="1m",
        current_kind="displacement_observed",
        current_timeframe="5m",
        maximum_completed_bars=2,
    )

    link = find_prior_typed_link(
        [prior],
        current,
        completed_index={_clock(0): 0, _clock(2): 1},
        spec=spec,
        event_lookup=canonical.get,
    )

    assert link is not None
    assert link.source_ancestry_proven is True
    assert link.composition_proven is False


def test_registered_temporal_episode_is_explicitly_not_ancestry() -> None:
    prior = _link_event(
        "sweep:1",
        0,
        kind="sweep_confirmed",
        timeframe="1m",
    )
    current = _link_event(
        "disp:1",
        2,
        kind="displacement_observed",
        timeframe="5m",
    )
    spec = TypedLinkSpec(
        previous_kind="sweep_confirmed",
        previous_timeframe="1m",
        current_kind="displacement_observed",
        current_timeframe="5m",
        maximum_completed_bars=2,
        mode=ResearchLinkMode.REGISTERED_TEMPORAL_EPISODE,
        registered_episode_definition="same-direction-prior-window-v1",
    )

    link = find_prior_typed_link(
        [prior],
        current,
        completed_index={_clock(0): 0, _clock(2): 1},
        spec=spec,
    )

    assert link is not None
    assert link.mode is ResearchLinkMode.REGISTERED_TEMPORAL_EPISODE
    assert link.source_ancestry_proven is False
    assert link.composition_proven is False
    assert link.shared_event_ids == ()
    assert link.shared_tokens == ()
    assert link.registered_episode_definition == "same-direction-prior-window-v1"


def test_typed_link_rejects_a_timezone_naive_prior_before_mixed_clock_sort() -> None:
    aware_prior = _link_event(
        "sweep:aware",
        0,
        kind="sweep_confirmed",
        timeframe="1m",
    )
    prior = _link_event(
        "sweep:naive",
        0,
        kind="sweep_confirmed",
        timeframe="1m",
    )
    prior["known_at"] = pd.Timestamp("2024-06-24 09:30:00")
    current = _link_event(
        "disp:aware",
        1,
        kind="displacement_observed",
        timeframe="5m",
    )
    spec = TypedLinkSpec(
        previous_kind="sweep_confirmed",
        previous_timeframe="1m",
        current_kind="displacement_observed",
        current_timeframe="5m",
        maximum_completed_bars=2,
        mode=ResearchLinkMode.REGISTERED_TEMPORAL_EPISODE,
        registered_episode_definition="same-direction-prior-window-v1",
    )

    with pytest.raises(
        ResearchContractError,
        match="prior known_at must be timezone aware",
    ):
        find_prior_typed_link(
            [aware_prior, prior],
            current,
            completed_index={prior["known_at"]: 0, _clock(1): 1},
            spec=spec,
        )


@pytest.mark.parametrize(
    ("prior_minute", "current_minute", "expected_distance"),
    (
        pytest.param(0, 1, 1, id="one-completed-bar"),
        pytest.param(0, 2, 2, id="registered-window-boundary"),
        pytest.param(0, 3, None, id="one-past-registered-window"),
        pytest.param(0, 0, None, id="same-clock"),
        pytest.param(1, 0, None, id="reverse-clock"),
    ),
)
def test_registered_temporal_episode_completed_bar_boundaries(
    prior_minute: int,
    current_minute: int,
    expected_distance: int | None,
) -> None:
    prior = _link_event(
        "sweep:boundary",
        prior_minute,
        kind="sweep_confirmed",
        timeframe="1m",
    )
    current = _link_event(
        "disp:boundary",
        current_minute,
        kind="displacement_observed",
        timeframe="5m",
    )
    # Reuse the existing synthetic W=2 primitive contract; this is not a new
    # production temporal-window registration.
    spec = TypedLinkSpec(
        previous_kind="sweep_confirmed",
        previous_timeframe="1m",
        current_kind="displacement_observed",
        current_timeframe="5m",
        maximum_completed_bars=2,
        mode=ResearchLinkMode.REGISTERED_TEMPORAL_EPISODE,
        registered_episode_definition="same-direction-prior-window-v1",
    )
    completed_index = {_clock(minute): minute for minute in range(4)}

    link = find_prior_typed_link(
        [prior],
        current,
        completed_index=completed_index,
        spec=spec,
    )

    if expected_distance is None:
        assert link is None
    else:
        assert link is not None
        assert link.completed_bars == expected_distance
        assert link.source_ancestry_proven is False
        assert link.composition_proven is False


def test_cross_timeframe_composition_requires_verified_constituent_bar() -> None:
    prior = _link_event(
        "sweep:1",
        0,
        kind="sweep_confirmed",
        timeframe="1m",
        constituent_bars=("m1-bar:1",),
    )
    current = _link_event(
        "disp:1",
        2,
        kind="displacement_observed",
        timeframe="5m",
        constituent_bars=("m1-bar:1", "m1-bar:2"),
    )
    canonical = {
        "m1-bar:1": SimpleNamespace(
            kind="bar_completed",
            timeframe="1m",
            origin="normalized_data",
            known_at=_clock(0),
        ),
        "m1-bar:2": SimpleNamespace(
            kind="bar_completed",
            timeframe="1m",
            origin="normalized_data",
            known_at=_clock(0),
        ),
        "sweep:1": SimpleNamespace(
            **prior,
            origin="semantic_atomic",
            source_event_ids=("m1-bar:1",),
            context_event_ids=(),
        ),
        "disp:1": SimpleNamespace(
            **current,
            origin="semantic_atomic",
            source_event_ids=("m1-bar:1", "m1-bar:2"),
            context_event_ids=(),
        ),
    }
    spec = TypedLinkSpec(
        previous_kind="sweep_confirmed",
        previous_timeframe="1m",
        current_kind="displacement_observed",
        current_timeframe="5m",
        maximum_completed_bars=2,
        mode=ResearchLinkMode.CROSS_TIMEFRAME_CONSTITUENT_BAR,
        constituent_bar_timeframe="1m",
    )

    link = find_prior_typed_link(
        [prior],
        current,
        completed_index={_clock(0): 0, _clock(2): 1},
        spec=spec,
        event_lookup=canonical.get,
    )

    assert link is not None
    assert link.source_ancestry_proven is False
    assert link.composition_proven is True
    assert link.shared_event_ids == ("m1-bar:1",)

    unbound = {**current, "lineage_tokens": ()}
    with pytest.raises(ResearchContractError, match="canonical lineage"):
        find_prior_typed_link(
            [prior],
            unbound,
            completed_index={_clock(0): 0, _clock(2): 1},
            spec=spec,
            event_lookup=canonical.get,
        )

    canonical["m1-bar:1"] = SimpleNamespace(
        kind="bar_completed",
        timeframe="5m",
        origin="normalized_data",
        known_at=_clock(0),
    )
    with pytest.raises(ResearchContractError, match="registered normalized 1m BAR"):
        find_prior_typed_link(
            [prior],
            current,
            completed_index={_clock(0): 0, _clock(2): 1},
            spec=spec,
            event_lookup=canonical.get,
        )


def test_constituent_bar_composition_rejects_context_only_bar() -> None:
    prior = _link_event(
        "sweep:1",
        0,
        kind="sweep_confirmed",
        timeframe="1m",
        constituent_bars=("m1-bar:1",),
    )
    current = _link_event(
        "disp:1",
        2,
        kind="displacement_observed",
        timeframe="5m",
        constituent_bars=("m1-bar:1",),
    )
    canonical = {
        "m1-bar:1": SimpleNamespace(
            kind="bar_completed",
            timeframe="1m",
            origin="normalized_data",
            known_at=_clock(0),
        ),
        "sweep:1": SimpleNamespace(
            **prior,
            origin="semantic_atomic",
            source_event_ids=("m1-bar:1",),
            context_event_ids=(),
        ),
        "disp:1": SimpleNamespace(
            **current,
            origin="semantic_atomic",
            source_event_ids=(),
            context_event_ids=("m1-bar:1",),
        ),
    }
    spec = TypedLinkSpec(
        previous_kind="sweep_confirmed",
        previous_timeframe="1m",
        current_kind="displacement_observed",
        current_timeframe="5m",
        maximum_completed_bars=2,
        mode=ResearchLinkMode.CROSS_TIMEFRAME_CONSTITUENT_BAR,
        constituent_bar_timeframe="1m",
    )

    with pytest.raises(ResearchContractError, match="source-only lineage"):
        find_prior_typed_link(
            [prior],
            current,
            completed_index={_clock(0): 0, _clock(2): 1},
            spec=spec,
            event_lookup=canonical.get,
        )


def test_canonical_treatment_episode_is_order_independent_and_keeps_members() -> None:
    common = {
        "kind": "level_touched",
        "known_at": _clock(0),
        "symbol": "NQ",
        "instrument_id": 123,
        "direction": "short",
        "timeframe": "1m",
    }
    first = {**common, "event_id": "touch:b"}
    second = {**common, "event_id": "touch:a"}

    left = canonical_treatment_episodes([first, second])
    right = canonical_treatment_episodes([second, first])

    assert left == right
    assert len(left) == 1
    assert left[0]["constituent_event_ids"] == ("touch:a", "touch:b")
    assert left[0]["constituent_event_count"] == 2
    assert left[0]["known_at"] == _clock(0)


def _synthetic_temporal_spec(previous_kind: str, current_kind: str) -> TypedLinkSpec:
    # Reuse the existing synthetic W=2 primitive contract.  These tests do not
    # register a production window or authorize a probability model.
    return TypedLinkSpec(
        previous_kind=previous_kind,
        previous_timeframe="5m",
        current_kind=current_kind,
        current_timeframe="5m",
        maximum_completed_bars=2,
        mode=ResearchLinkMode.REGISTERED_TEMPORAL_EPISODE,
        registered_episode_definition="same-direction-prior-window-v1",
    )


def test_plural_typed_links_retains_every_registered_predecessor() -> None:
    early = _link_event(
        "acceptance:early", 0, kind="acceptance", timeframe="5m"
    )
    late = _link_event("acceptance:late", 1, kind="acceptance", timeframe="5m")
    response = _link_event(
        "response:1", 2, kind="same_direction_response", timeframe="5m"
    )

    links = find_prior_typed_links(
        [late, early],
        response,
        completed_index={_clock(value): value for value in range(3)},
        spec=_synthetic_temporal_spec("acceptance", "same_direction_response"),
    )

    assert tuple(str(link.prior["event_id"]) for link in links) == (
        "acceptance:late",
        "acceptance:early",
    )
    assert tuple(link.completed_bars for link in links) == (1, 2)
    assert all(link.source_ancestry_proven is False for link in links)
    assert all(link.composition_proven is False for link in links)


def test_branching_episode_projection_keeps_parallel_paths_and_typed_ledger() -> None:
    records = [
        _link_event("interaction:1", 0, kind="interaction", timeframe="5m"),
        _link_event("acceptance:1", 1, kind="acceptance", timeframe="5m"),
        _link_event("sweep:1", 1, kind="sweep", timeframe="5m"),
        _link_event(
            "response:continuation",
            2,
            kind="same_direction_response",
            timeframe="5m",
        ),
        _link_event(
            "response:opposite", 3, kind="opposite_flow", timeframe="5m"
        ),
        _link_event("unrelated:1", 3, kind="balance", timeframe="5m"),
    ]
    specs = {
        "interaction_to_acceptance": _synthetic_temporal_spec(
            "interaction", "acceptance"
        ),
        "interaction_to_sweep": _synthetic_temporal_spec("interaction", "sweep"),
        "acceptance_to_same_direction_response": _synthetic_temporal_spec(
            "acceptance", "same_direction_response"
        ),
        "sweep_to_opposite_flow": _synthetic_temporal_spec(
            "sweep", "opposite_flow"
        ),
    }
    completed = {_clock(value): value for value in range(4)}

    left = project_branching_research_episode(
        records,
        root_event_id="interaction:1",
        completed_index=completed,
        link_specs=specs,
    )
    right = project_branching_research_episode(
        list(reversed(records)),
        root_event_id="interaction:1",
        completed_index=completed,
        link_specs=dict(reversed(tuple(specs.items()))),
    )

    assert isinstance(left, BranchingResearchEpisode)
    assert left == right
    assert left.event_ids == (
        "interaction:1",
        "acceptance:1",
        "sweep:1",
        "response:continuation",
        "response:opposite",
    )
    assert left.terminal_event_ids == (
        "response:continuation",
        "response:opposite",
    )
    assert {
        (edge.relation_id, edge.prior_event_id, edge.current_event_id)
        for edge in left.edge_ledger
    } == {
        ("interaction_to_acceptance", "interaction:1", "acceptance:1"),
        ("interaction_to_sweep", "interaction:1", "sweep:1"),
        (
            "acceptance_to_same_direction_response",
            "acceptance:1",
            "response:continuation",
        ),
        ("sweep_to_opposite_flow", "sweep:1", "response:opposite"),
    }
    assert all(
        edge.mode is ResearchLinkMode.REGISTERED_TEMPORAL_EPISODE
        and edge.source_ancestry_proven is False
        and edge.composition_proven is False
        and edge.registered_episode_definition
        == "same-direction-prior-window-v1"
        for edge in left.edge_ledger
    )


def test_branching_projection_does_not_require_a_complete_linear_chain() -> None:
    interaction = _link_event(
        "interaction:1", 0, kind="interaction", timeframe="5m"
    )
    sweep = _link_event("sweep:1", 1, kind="sweep", timeframe="5m")
    response = _link_event(
        "response:opposite", 2, kind="opposite_flow", timeframe="5m"
    )
    specs = {
        "interaction_to_acceptance": _synthetic_temporal_spec(
            "interaction", "acceptance"
        ),
        "interaction_to_sweep": _synthetic_temporal_spec("interaction", "sweep"),
        "acceptance_to_same_direction_response": _synthetic_temporal_spec(
            "acceptance", "same_direction_response"
        ),
        "sweep_to_opposite_flow": _synthetic_temporal_spec(
            "sweep", "opposite_flow"
        ),
    }

    episode = project_branching_research_episode(
        [interaction, sweep, response],
        root_event_id="interaction:1",
        completed_index={_clock(value): value for value in range(3)},
        link_specs=specs,
    )

    assert episode.event_ids == (
        "interaction:1",
        "sweep:1",
        "response:opposite",
    )
    assert tuple(edge.relation_id for edge in episode.edge_ledger) == (
        "interaction_to_sweep",
        "sweep_to_opposite_flow",
    )
    assert episode.terminal_event_ids == ("response:opposite",)


def test_branching_projection_cannot_infer_an_unregistered_relation_window() -> None:
    interaction = _link_event(
        "interaction:1", 0, kind="interaction", timeframe="5m"
    )

    with pytest.raises(ResearchContractError, match="explicit typed link specs"):
        project_branching_research_episode(
            [interaction],
            root_event_id="interaction:1",
            completed_index={_clock(0): 0},
            link_specs={},
        )
    with pytest.raises(ResearchContractError, match="lacks TypedLinkSpec"):
        project_branching_research_episode(
            [interaction],
            root_event_id="interaction:1",
            completed_index={_clock(0): 0},
            link_specs={"interaction_to_response": {}},  # type: ignore[dict-item]
        )


def _match_record(
    identity: str,
    minute: int,
    *,
    candidate: bool,
) -> dict[str, object]:
    value = {
        "known_at": _clock(minute),
        "symbol": "NQ",
        "instrument_id": 123,
        "session_phase": "new_york_am",
        "direction": "long",
    }
    value["candidate_id" if candidate else "event_id"] = identity
    return value


def test_maximum_cardinality_matching_is_deterministic_and_not_greedy() -> None:
    # t:narrow can only use c:shared; t:flex can use c:shared or c:alternate.
    # A one-pass greedy matcher processing t:flex first would return one pair.
    narrow = _match_record("t:narrow", 0, candidate=False)
    flexible = _match_record("t:flex", 2, candidate=False)
    shared = _match_record("c:shared", 3, candidate=True)
    alternate = _match_record("c:alternate", 5, candidate=True)
    completed = {_clock(value): value for value in range(6)}
    spec = MatchSpec(
        exact_fields=("symbol", "instrument_id", "session_phase"),
        maximum_completed_bar_offset=3,
        outcome_horizon_completed_bars=0,
    )

    left = deterministic_maximum_cardinality_match(
        [flexible, narrow],
        [alternate, shared],
        completed_index=completed,
        spec=spec,
    )
    right = deterministic_maximum_cardinality_match(
        [narrow, flexible],
        [shared, alternate],
        completed_index=completed,
        spec=spec,
    )

    assert left.pairs == right.pairs
    assert left.matched == 2
    assert {(pair.treatment_id, pair.candidate_id) for pair in left.pairs} == {
        ("t:narrow", "c:shared"),
        ("t:flex", "c:alternate"),
    }
    assert all(
        pair.direction_known_at <= pd.Timestamp(pair.candidate["known_at"])
        for pair in left.pairs
    )


def test_matching_enforces_forward_horizon_embargo_and_caliper() -> None:
    treatment = _match_record("t:1", 0, candidate=False)
    too_early = _match_record("c:early", 3, candidate=True)
    eligible = _match_record("c:ok", 4, candidate=True)
    completed = {_clock(value): value for value in range(9)}
    spec = MatchSpec(
        exact_fields=("symbol", "instrument_id", "session_phase"),
        maximum_completed_bar_offset=5,
        outcome_horizon_completed_bars=2,
        embargo_completed_bars=1,
        direction_policy=ControlDirectionPolicy.INHERIT_TREATMENT_AFTER_KNOWN_AT,
    )

    result = deterministic_maximum_cardinality_match(
        [treatment],
        [too_early, eligible],
        completed_index=completed,
        spec=spec,
    )

    assert result.matched == 1
    assert result.pairs[0].candidate_id == "c:ok"
    assert result.pairs[0].completed_bar_offset == 4


def test_matching_replacement_limit_is_explicit_capacity() -> None:
    first = _match_record("t:1", 0, candidate=False)
    second = _match_record("t:2", 1, candidate=False)
    candidate = _match_record("c:1", 3, candidate=True)
    completed = {_clock(value): value for value in range(5)}

    without_replacement = deterministic_maximum_cardinality_match(
        [first, second],
        [candidate],
        completed_index=completed,
        spec=MatchSpec(
            exact_fields=("symbol", "instrument_id", "session_phase"),
            maximum_completed_bar_offset=3,
            outcome_horizon_completed_bars=0,
            replacement_limit=1,
        ),
    )
    with_capacity_two = deterministic_maximum_cardinality_match(
        [first, second],
        [candidate],
        completed_index=completed,
        spec=MatchSpec(
            exact_fields=("symbol", "instrument_id", "session_phase"),
            maximum_completed_bar_offset=3,
            outcome_horizon_completed_bars=0,
            replacement_limit=2,
        ),
    )

    assert without_replacement.matched == 1
    assert set(without_replacement.unmatched.values()) == {
        "candidate_capacity_exhausted"
    }
    assert with_capacity_two.matched == 2


def test_candidate_local_matching_requires_causally_known_direction() -> None:
    treatment = _match_record("t:1", 0, candidate=False)
    candidate = _match_record("c:1", 2, candidate=True)
    completed = {_clock(value): value for value in range(4)}
    spec = MatchSpec(
        exact_fields=("symbol", "instrument_id", "session_phase"),
        maximum_completed_bar_offset=3,
        outcome_horizon_completed_bars=0,
        direction_policy=ControlDirectionPolicy.CANDIDATE_LOCAL,
    )

    with pytest.raises(ResearchContractError, match="direction_known_at"):
        deterministic_maximum_cardinality_match(
            [treatment],
            [candidate],
            completed_index=completed,
            spec=spec,
        )

    future_known = {**candidate, "direction_known_at": _clock(3)}
    with pytest.raises(ResearchContractError, match="future-known"):
        deterministic_maximum_cardinality_match(
            [treatment],
            [future_known],
            completed_index=completed,
            spec=spec,
        )

    causal = {**candidate, "direction_known_at": _clock(2)}
    result = deterministic_maximum_cardinality_match(
        [treatment],
        [causal],
        completed_index=completed,
        spec=spec,
    )
    assert result.matched == 1
    assert result.pairs[0].direction_known_at == _clock(2)


def test_pseudo_levels_use_only_exact_snapshot_and_known_levels() -> None:
    candidate = {
        "event_id": "level:1",
        "known_at": _clock(0),
        "symbol": "NQ",
        "instrument_id": 123,
    }
    snapshot = {
        "snapshot_id": "snapshot:1",
        "asof": _clock(0),
        "symbol": "NQ",
        "instrument_id": 123,
        "range_low": 90.0,
        "range_high": 110.0,
        "current_price": 100.0,
    }
    spec = PseudoLevelSpec(
        protocol_id="pseudo-v1",
        relative_locations=(0.25, 0.5, 0.75),
        tick_size=0.25,
        minimum_separation_ticks=1,
    )

    result = construct_pseudo_levels(
        candidate,
        snapshot,
        [
            {
                "known_at": _clock(0),
                "symbol": "NQ",
                "instrument_id": 123,
                "price": 95.0,
            }
        ],
        spec=spec,
    )

    assert [(level.price, level.direction) for level in result.levels] == [
        (105.0, "short")
    ]
    assert {item["reason"] for item in result.exclusions} == {
        "too_close_to_known_real_level",
        "equal_to_current_price",
    }
    assert result == construct_pseudo_levels(
        candidate,
        snapshot,
        [
            {
                "known_at": _clock(0),
                "symbol": "NQ",
                "instrument_id": 123,
                "price": 95.0,
            }
        ],
        spec=spec,
    )

    with pytest.raises(ResearchContractError, match="future-known"):
        construct_pseudo_levels(
            candidate,
            snapshot,
            [
                {
                    "known_at": _clock(1),
                    "symbol": "NQ",
                    "instrument_id": 123,
                    "price": 95.0,
                }
            ],
            spec=spec,
        )

    with pytest.raises(ResearchContractError, match="contract identity"):
        construct_pseudo_levels(
            candidate,
            {**snapshot, "symbol": "ES"},
            [],
            spec=spec,
        )


def test_forward_time_shift_is_causal_and_outcome_separated() -> None:
    rows = [
        {
            "asof": _clock(index),
            "symbol": "NQ",
            "instrument_id": 123,
            "session_phase": "new_york_am",
        }
        for index in range(10)
    ]
    completed = {row["asof"]: index for index, row in enumerate(rows)}
    event = {
        "event_id": "sweep:1",
        "known_at": _clock(0),
        "direction": "short",
        "symbol": "NQ",
        "instrument_id": 123,
        "session_phase": "new_york_am",
    }
    spec = TimeShiftSpec(
        protocol_id="shift-v1",
        forward_offsets_completed_bars=(4,),
        outcome_horizon_completed_bars=2,
        embargo_completed_bars=1,
        exclude_other_treatment_windows=False,
    )

    result = build_forward_time_shift_controls(
        [event],
        rows,
        completed_index=completed,
        spec=spec,
    )

    assert len(result.controls) == 1
    control = result.controls[0]
    assert control["known_at"] == _clock(4)
    assert control["direction"] == "short"
    assert control["direction_known_at"] == _clock(0)
    assert control["construction_known_at"] < control["known_at"]

    with pytest.raises(ResearchContractError, match="outcome horizon"):
        TimeShiftSpec(
            protocol_id="bad-shift",
            forward_offsets_completed_bars=(3,),
            outcome_horizon_completed_bars=2,
            embargo_completed_bars=1,
        )

    other_treatment = {
        **event,
        "event_id": "sweep:2",
        "known_at": _clock(4),
    }
    overlap_result = build_forward_time_shift_controls(
        [event, other_treatment],
        rows,
        completed_index=completed,
        spec=TimeShiftSpec(
            protocol_id="shift-overlap-v1",
            forward_offsets_completed_bars=(4,),
            outcome_horizon_completed_bars=2,
            embargo_completed_bars=1,
        ),
    )
    assert any(
        item["source_event_id"] == "sweep:1"
        and item["reason"] == "shifted_outcome_overlaps_other_treatment"
        for item in overlap_result.exclusions
    )


def test_exact_mcnemar_and_fixed_family_holm_have_known_answers() -> None:
    mcnemar = exact_mcnemar(
        [
            (True, False),
            (True, False),
            (True, False),
            (False, True),
            (True, True),
            (None, True),
        ]
    )

    assert mcnemar.paired_n == 5
    assert mcnemar.excluded_n == 1
    assert mcnemar.treatment_only_successes == 3
    assert mcnemar.control_only_successes == 1
    assert mcnemar.p_value == pytest.approx(0.625)

    holm = holm_adjust_fixed_family(
        {"primary": 0.01, "secondary": 0.04, "underpowered": None},
        family_order=("primary", "secondary", "underpowered"),
        alpha=0.05,
    )

    assert holm.adjusted_p_values == {
        "primary": pytest.approx(0.03),
        "secondary": pytest.approx(0.08),
        "underpowered": 1.0,
    }
    assert holm.rejected == {
        "primary": True,
        "secondary": False,
        "underpowered": False,
    }
    assert holm.missing_as_one == ("underpowered",)

    with pytest.raises(ResearchContractError, match="exact registered family"):
        holm_adjust_fixed_family(
            {"primary": 0.01},
            family_order=("primary", "missing"),
        )
