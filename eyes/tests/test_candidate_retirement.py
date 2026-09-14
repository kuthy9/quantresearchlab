"""A candidate that is too old or too far away leaves the candidate set.

The liquidity candidate collection was append-only by design: a sweep disarmed
a level and kept it so a later re-approach could re-arm the same identity, and
nothing ever removed one.  On the real tape the 1m set grew by 0.17 levels per
bar (102 at bar 500, 508 at bar 3,000), every bar's cost grew with it, and the
Brain's ``unswept_bsl`` / ``unswept_ssl`` counts were taken over every level
the Eye had created since it started.  A candidate now retires when it has
outlived ``candidate_retirement_max_native_age_bars`` of its own scale or sits
beyond ``candidate_retirement_max_distance_atr``; the retirement is an atomic
``LIQUIDITY_RETIRED`` fact the reducer applies, so a cold replay agrees with
the hot view, and a retired item that was never reached also ends as
``LEVEL_INVALIDATED``.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from contract.eye import EventKind, EventOrigin
from contract.market import Timeframe
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import (
    _LIFECYCLE_OWNED_CANDIDATE_KINDS as LIFECYCLE_OWNED,
    CausalObserver,
    ObserverConfig,
)

from eyes.tests.test_facts_on_every_scale import _noisy
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars


MAX_NATIVE_AGE_BARS = 120
MAX_DISTANCE_ATR = 6.0


def _tight_protocol(tmp_path: Path) -> Path:
    payload = json.loads(
        Path("configs/primitives_structure_liquidity.json").read_text(encoding="utf-8")
    )
    payload["liquidity_parameters"]["candidate_retirement_max_native_age_bars"] = (
        MAX_NATIVE_AGE_BARS
    )
    payload["liquidity_parameters"]["candidate_retirement_max_distance_atr"] = (
        MAX_DISTANCE_ATR
    )
    path = tmp_path / "primitives_structure_liquidity.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    protocol = _tight_protocol(tmp_path_factory.mktemp("protocol"))
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=str(protocol),
            liquidity_protocol=str(protocol),
            displacement_protocol="configs/primitives_displacement.json",
            zone_protocol="configs/primitives_zones.json",
            # Group 4 pins the Group 1-2 protocol bytes, and this test's
            # protocol is a tightened copy; the range auction is not under test.
            range_auction_protocol=None,
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    observations = [
        observer.observe(reader.on_bar(bar)) for bar in _noisy(session_bars(1)[:900])
    ]
    return observer.audit_store.events(), observations


def _retirements(events):
    found = [
        event
        for event in events
        if event.kind is EventKind.LIQUIDITY_RETIRED
        and event.origin is EventOrigin.SEMANTIC_ATOMIC
    ]
    assert found, "no candidate retired"
    return found


def test_candidates_retire_for_age_and_for_distance(replay) -> None:
    events, _ = replay
    reasons = Counter(event.details["reason"] for event in _retirements(events))
    assert reasons["candidate_aged_out"]
    assert reasons["candidate_out_of_reach"]


def test_a_retired_candidate_leaves_the_candidate_set(replay) -> None:
    events, observations = replay
    retired_at = {
        event.details["level_id"]: event.known_at for event in _retirements(events)
    }
    for observation in observations:
        for state in observation.market_snapshot.timeframe_states.values():
            for candidate in state.liquidity.candidates:
                clock = retired_at.get(candidate.candidate_id)
                # The retiring bar's snapshot was published before the
                # retirement was derived from it; the next bar's is without it.
                assert clock is None or observation.asof <= clock, (
                    f"{candidate.candidate_id} is still a candidate after it retired"
                )


def test_every_live_candidate_is_within_the_limits(replay) -> None:
    events, observations = replay
    retired_at = {
        event.details["level_id"]: event.known_at for event in _retirements(events)
    }
    for observation in observations:
        for timeframe, state in observation.market_snapshot.timeframe_states.items():
            for candidate in state.liquidity.candidates:
                if retired_at.get(candidate.candidate_id) == observation.asof:
                    continue  # published before its retirement was derived
                if candidate.source_kind in LIFECYCLE_OWNED:
                    continue  # pools and range boundaries retire with their entity
                assert candidate.age_bars <= MAX_NATIVE_AGE_BARS * timeframe.minutes + 1
                assert (
                    candidate.distance_atr is None
                    or candidate.distance_atr <= MAX_DISTANCE_ATR
                )


def test_retirement_is_published_on_the_candidates_own_timeframe(replay) -> None:
    events, _ = replay
    for event in _retirements(events):
        assert event.timeframe.value == event.details["source_timeframe"]
        assert event.details["level_id"] in event.source_entity_ids


def test_a_retired_item_never_reached_is_also_invalidated(replay) -> None:
    events, _ = replay
    reached = {
        event.details["level_id"] for event in events if event.kind is EventKind.LEVEL_REACHED
    }
    invalidated = {
        event.details["level_id"]
        for event in events
        if event.kind is EventKind.LEVEL_INVALIDATED
    }
    retired = {event.details["level_id"] for event in _retirements(events)}
    unreached = retired - reached
    assert unreached, "every retired candidate had been reached"
    assert unreached <= invalidated
    assert not (retired & reached & invalidated)
