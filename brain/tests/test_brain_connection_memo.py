from __future__ import annotations

import pandas as pd

from shares.core.model import to_primitive
from brain.core.playbooks import (
    _LSR_CONNECTION_MEMO,
    _whitelisted_global_connection,
)
from shares.core.scene_graph import TemporalMarketSceneGraph

from brain.tests.test_v4_typed_vertical import (
    _advance_observation,
    _brain,
    _lsr_observation,
)


def test_lsr_connection_query_is_memoized_only_within_one_brain_clock(
    monkeypatch,
) -> None:
    observation = _lsr_observation()
    graph = TemporalMarketSceneGraph()
    graph.update(observation)
    source_ids = tuple(
        sorted(
            identity
            for identity, node_ids in graph._source_index.items()
            if node_ids
        )
    )
    assert len(source_ids) >= 2
    sources = (source_ids[0],)
    targets = (source_ids[-1],)

    calls = 0
    original = graph._node_id_for_source

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(graph, "_node_id_for_source", counted)
    token = _LSR_CONNECTION_MEMO.set({})
    try:
        first = _whitelisted_global_connection(
            sources,
            targets,
            graph,
            asof=observation.asof,
        )
        first_call_count = calls
        second = _whitelisted_global_connection(
            sources,
            targets,
            graph,
            asof=observation.asof,
        )
    finally:
        _LSR_CONNECTION_MEMO.reset(token)

    assert first is second
    assert first_call_count > 0
    assert calls == first_call_count

    # A later Brain clock gets a fresh memo even when graph/asof are unchanged.
    _whitelisted_global_connection(
        sources,
        targets,
        graph,
        asof=observation.asof,
    )
    assert calls > first_call_count


def test_update_local_connection_memo_preserves_full_belief_each_bar() -> None:
    first = _lsr_observation()
    observations = (
        first,
        _advance_observation(
            first,
            first.asof + pd.Timedelta(minutes=1),
        ),
    )
    cached_brain = _brain()
    uncached_brain = _brain()
    cached_graph = TemporalMarketSceneGraph()
    uncached_graph = TemporalMarketSceneGraph()

    for observation in observations:
        cached = cached_brain.update(
            observation,
            scene_graph=cached_graph,
            scene_delta=cached_graph.update(observation),
        )
        uncached = uncached_brain._update_without_connection_memo(
            observation,
            scene_graph=uncached_graph,
            scene_delta=uncached_graph.update(observation),
        )
        assert to_primitive(cached) == to_primitive(uncached)
        assert _LSR_CONNECTION_MEMO.get() is None
