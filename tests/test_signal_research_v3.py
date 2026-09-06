from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytestmark = pytest.mark.research_orchestration

from smc_trader.model import (
    Direction,
    EventKind,
    EventOrigin,
    MarketEvent,
    Timeframe,
)
from smc_trader.signal_research import ResearchContractError
import scripts.run_semantic_signal_research as signal_runner
from scripts.run_semantic_signal_research import (
    ROOT,
    V3_HOLM_FAMILY,
    V3_MATCH_FIELDS,
    V3_NESTED_STAGE_ORDER,
    _artifact_status,
    _canonical_episode_projection,
    _output_bundle_paths,
    _preflight_output_bundle,
    _report_v3,
    _run_v3_analysis,
    _v3_enriched_rows,
    _v3_nested_deltas,
    _v3_non_nested_partition,
    _v3_non_sweep_touch_candidates,
    _v3_pseudo_touch_candidates,
    _v3_source_linked_stage,
    _v3_time_shift_stratum_exclusion,
    _validated_research_design,
    _validated_v3_link_contracts,
    run,
)


TEMPLATE = ROOT / "configs/research/semantic_event_study_v3_template.yaml"


def _clock(minute: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-02T09:30:00-05:00") + pd.Timedelta(
        minute,
        unit="min",
    )


def _event(
    event_id: str,
    minute: int,
    *,
    kind: str = "level_touched",
    lineage: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "kind": kind,
        "known_at": _clock(minute),
        "event_time": _clock(minute),
        "direction": "long",
        "timeframe": "5m",
        "symbol": "NQ",
        "instrument_id": 123,
        "origin": "semantic_atomic",
        "parent_bucket": "aligned_with_parent",
        "session_phase": "new_york_am",
        "lineage_tokens": lineage,
    }


def _row(minute: int, *, low: float = 99.0, high: float = 101.0) -> dict[str, object]:
    return {
        "asof": _clock(minute),
        "open": 100.0,
        "high": high,
        "low": low,
        "close": 100.0,
        "atr": 1.0,
        "symbol": "NQ",
        "instrument_id": 123,
        "session_phase": "new_york_am",
        "relative_volume": 1.0,
        "m1_direction": "long",
        "nearest_distance_atr": 0.5,
        "atomic_event_count": 0,
    }


def _canonical_market_event(
    event_id: str,
    minute: int,
    *,
    kind: EventKind,
    source_event_ids: tuple[str, ...] = (),
    context_event_ids: tuple[str, ...] = (),
    symbol: str | None = None,
    instrument_id: int | None = None,
    side: str | None = None,
    direction: Direction | None = Direction.LONG,
) -> MarketEvent:
    normalized = kind is EventKind.BAR_COMPLETED
    evidence = (
        {
            "symbol": symbol,
            "instrument_id": instrument_id,
            "real_completed": True,
        }
        if normalized
        else {"canonical_semantic": True, "projection_only": False}
    )
    return MarketEvent(
        event_id=event_id,
        kind=kind,
        observed_at=_clock(minute),
        timeframe=Timeframe.M5,
        side=side,
        price=100.0,
        strength=0.0,
        source_ids=source_event_ids,
        direction=None if normalized else direction,
        event_time=_clock(minute),
        known_at=_clock(minute),
        evidence=evidence,
        source_event_ids=source_event_ids,
        source_data_ids=((f"data:{event_id}",) if normalized else ()),
        context_event_ids=context_event_ids,
        origin=(
            EventOrigin.NORMALIZED_DATA
            if normalized
            else EventOrigin.SEMANTIC_ATOMIC
        ),
    )


def test_v3_template_freezes_new_semantics_chain_controls_and_inference() -> None:
    manifest = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    manifest["experiment_id"] = "contract-test-v3"
    manifest["frozen_at"] = "2026-08-21T00:00:00-04:00"

    chain, non_nested = _validated_v3_link_contracts(manifest)
    outcome, minimum = _validated_research_design(manifest)

    assert manifest["schema_version"] == 2
    assert manifest["research_protocol_version"] == 3
    assert manifest["semantic_version"] == "smc_semantics_v1.2"
    assert manifest["semantic_registry"] == "semantics/registry_v1_2.yaml"
    assert all(
        edge["previous_timeframe"] == edge["current_timeframe"] == "5m"
        for edge in chain.values()
    )
    assert chain["E1_to_E2"]["link_mode"] == "strict_source_ancestry"
    assert {
        edge["link_mode"] for name, edge in chain.items() if name != "E1_to_E2"
    } == {"cross_timeframe_constituent_bar_composition"}
    assert set(non_nested) == {
        "mss_prior_sweep",
        "mss_prior_displacement",
        "fvg_prior_displacement",
    }
    assert (
        tuple(manifest["control_definition"]["separate_control_families"])
        == V3_HOLM_FAMILY
    )
    assert outcome["horizon"] == 60
    assert minimum == 30
    assert manifest["nested_chain_metric"]["stage_order"] == list(V3_NESTED_STAGE_ORDER)
    assert manifest["nested_chain_metric"]["causal_claim"] is False
    assert manifest["control_definition"]["forward_time_shift"][
        "exact_match_fields"
    ] == list(V3_MATCH_FIELDS)
    assert (
        manifest["inference_definition"]["cross_pair_outcome_window_policy"]
        == "overlap_allowed_p_values_descriptive_unvalidated"
    )


def test_v3_contract_rejects_cross_timeframe_or_temporal_chain() -> None:
    manifest = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cross_timeframe = deepcopy(manifest)
    cross_timeframe["event_definition"]["chain_edges"]["E2_to_E3"][
        "previous_timeframe"
    ] = "1m"
    with pytest.raises(ResearchContractError, match="registered strict"):
        _validated_v3_link_contracts(cross_timeframe)

    temporal = deepcopy(manifest)
    temporal["event_definition"]["chain_edges"]["E2_to_E3"][
        "link_mode"
    ] = "registered_temporal_episode"
    with pytest.raises(ResearchContractError, match="registered strict"):
        _validated_v3_link_contracts(temporal)


def test_episode_projection_deduplicates_clock_direction_timeframe() -> None:
    first = _event("touch:a", 0)
    duplicate = _event("touch:b", 0)

    episodes, membership, by_id = _canonical_episode_projection([duplicate, first])

    assert len(episodes) == 1
    assert episodes[0]["constituent_event_ids"] == ("touch:a", "touch:b")
    assert membership["touch:a"] == membership["touch:b"]
    assert set(by_id) == {episodes[0]["event_id"]}


def test_v3_stage_requires_verified_direct_or_transitive_ancestor() -> None:
    prior = _event("sweep:1", 0, kind="sweep_confirmed")
    current = _event(
        "disp:1",
        1,
        kind="displacement_observed",
        lineage=("event:sweep:1",),
    )
    canonical = {
        "sweep:1": SimpleNamespace(
            event_id="sweep:1",
            kind="sweep_confirmed",
            timeframe="5m",
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
            known_at=_clock(1),
            source_event_ids=("sweep:1",),
            context_event_ids=(),
        ),
    }
    edge = {
        "previous_kind": "sweep_confirmed",
        "previous_timeframe": "5m",
        "current_kind": "displacement_observed",
        "current_timeframe": "5m",
        "maximum_completed_bars": 2,
        "link_mode": "strict_source_ancestry",
        "source_ancestry_required": True,
        "composition_proven": False,
    }

    raw, episodes, chains, ledger = _v3_source_linked_stage(
        [prior],
        [current],
        completed_index={_clock(0): 0, _clock(1): 1},
        edge=edge,
        stage="E3",
        prior_chains={"sweep:1": ("episode:sweep", "sweep:1")},
        event_lookup=canonical.get,
    )

    assert raw == [current]
    assert len(episodes) == 1
    assert chains["disp:1"][-1] == "disp:1"
    assert ledger[0]["semantic_event_ancestry_proven"] is True
    assert ledger[0]["composition_proven"] is False
    assert ledger[0]["ancestor_event_ids"] == ("sweep:1",)

    unlinked = {**current, "lineage_tokens": ()}
    assert (
        _v3_source_linked_stage(
            [prior],
            [unlinked],
            completed_index={_clock(0): 0, _clock(1): 1},
            edge=edge,
            stage="E3",
            prior_chains={},
            event_lookup=canonical.get,
        )[0]
        == []
    )


def test_v3_stage_accepts_real_market_event_scope_from_source_bars_only() -> None:
    """Production MarketEvent scope lives on normalized source BAR evidence."""

    canonical = {
        "bar:nq:0": _canonical_market_event(
            "bar:nq:0",
            0,
            kind=EventKind.BAR_COMPLETED,
            symbol="NQ",
            instrument_id=123,
        ),
        "bar:nq:1": _canonical_market_event(
            "bar:nq:1",
            1,
            kind=EventKind.BAR_COMPLETED,
            symbol="NQ",
            instrument_id=123,
        ),
        "bar:nq:2": _canonical_market_event(
            "bar:nq:2",
            2,
            kind=EventKind.BAR_COMPLETED,
            symbol="NQ",
            instrument_id=123,
        ),
        # Context is deliberately a different market. It may explain the
        # event, but cannot establish or contaminate typed-link market scope.
        "bar:es:2": _canonical_market_event(
            "bar:es:2",
            2,
            kind=EventKind.BAR_COMPLETED,
            symbol="ES",
            instrument_id=456,
        ),
    }
    canonical.update(
        {
            "level:1": _canonical_market_event(
                "level:1",
                0,
                kind=EventKind.LIQUIDITY_LEVEL_CREATED,
                source_event_ids=("bar:nq:0",),
                side="below",
                direction=None,
            ),
            "touch:1": _canonical_market_event(
                "touch:1",
                0,
                kind=EventKind.LEVEL_TOUCHED,
                source_event_ids=("level:1", "bar:nq:0"),
                side="below",
                direction=None,
            ),
            "penetration:1": _canonical_market_event(
                "penetration:1",
                1,
                kind=EventKind.LEVEL_PENETRATED,
                source_event_ids=("level:1", "touch:1", "bar:nq:1"),
                side="below",
                direction=None,
            ),
            "sweep:1": _canonical_market_event(
                "sweep:1",
                2,
                kind=EventKind.SWEEP_CONFIRMED,
                source_event_ids=("penetration:1", "bar:nq:2"),
                context_event_ids=("bar:es:2",),
                side="below",
                direction=Direction.LONG,
            ),
        }
    )
    prior = _event("touch:1", 0)
    current = _event(
        "sweep:1",
        2,
        kind="sweep_confirmed",
        lineage=("event:touch:1", "event:bar:es:2"),
    )
    edge = {
        "previous_kind": "level_touched",
        "previous_timeframe": "5m",
        "current_kind": "sweep_confirmed",
        "current_timeframe": "5m",
        "maximum_completed_bars": 3,
        "link_mode": "strict_source_ancestry",
        "source_ancestry_required": True,
        "composition_proven": False,
    }

    assert not hasattr(canonical["touch:1"], "symbol")
    assert not hasattr(canonical["sweep:1"], "instrument_id")
    raw, episodes, _, ledger = _v3_source_linked_stage(
        [prior],
        [current],
        completed_index={_clock(0): 0, _clock(2): 1},
        edge=edge,
        stage="E2_sweep",
        prior_chains={"touch:1": ("episode:touch", "touch:1")},
        event_lookup=canonical.get,
    )

    assert raw == [current]
    assert len(episodes) == 1
    assert ledger[0]["ancestor_event_ids"] == ("touch:1",)

    with pytest.raises(ResearchContractError, match="canonical event source scope"):
        _v3_source_linked_stage(
            [{**prior, "symbol": "ES", "instrument_id": 456}],
            [{**current, "symbol": "ES", "instrument_id": 456}],
            completed_index={_clock(0): 0, _clock(2): 1},
            edge=edge,
            stage="E2_sweep",
            prior_chains={"touch:1": ("episode:touch", "touch:1")},
            event_lookup=canonical.get,
        )

    bad_direction = dict(canonical)
    bad_direction["touch:1"] = replace(canonical["touch:1"], side=None)
    with pytest.raises(ResearchContractError, match="side must be explicit"):
        _v3_source_linked_stage(
            [prior],
            [current],
            completed_index={_clock(0): 0, _clock(2): 1},
            edge=edge,
            stage="E2_sweep",
            prior_chains={"touch:1": ("episode:touch", "touch:1")},
            event_lookup=bad_direction.get,
        )

    mismatched_direction = dict(canonical)
    mismatched_direction["sweep:1"] = replace(
        canonical["sweep:1"],
        direction=Direction.SHORT,
    )
    with pytest.raises(ResearchContractError, match="direction disagrees"):
        _v3_source_linked_stage(
            [{**prior, "direction": "short"}],
            [{**current, "direction": "short"}],
            completed_index={_clock(0): 0, _clock(2): 1},
            edge=edge,
            stage="E2_sweep",
            prior_chains={"touch:1": ("episode:touch", "touch:1")},
            event_lookup=mismatched_direction.get,
        )


def test_v3_composition_uses_exact_normalized_m5_bar_not_semantic_ancestry() -> None:
    prior = {
        **_event("sweep:1", 0, kind="sweep_confirmed"),
        "lineage_tokens": ("event:bar:1",),
        "constituent_bar_event_ids": ("bar:1",),
    }
    current = {
        **_event("disp:1", 1, kind="displacement_observed"),
        "lineage_tokens": ("event:bar:1",),
        "constituent_bar_event_ids": ("bar:1",),
    }
    bar = SimpleNamespace(
        event_id="bar:1",
        kind="bar_completed",
        timeframe="5m",
        direction=None,
        symbol="NQ",
        instrument_id=123,
        origin="normalized_data",
        real_completed=True,
        clock_only=False,
        known_at=_clock(0),
        source_event_ids=(),
        context_event_ids=(),
    )
    edge = {
        "previous_kind": "sweep_confirmed",
        "previous_timeframe": "5m",
        "current_kind": "displacement_observed",
        "current_timeframe": "5m",
        "maximum_completed_bars": 2,
        "link_mode": "cross_timeframe_constituent_bar_composition",
        "source_ancestry_required": False,
        "composition_proven": True,
        "constituent_bar_timeframe": "5m",
    }
    canonical = {
        "bar:1": bar,
        "sweep:1": SimpleNamespace(
            **prior,
            source_event_ids=("bar:1",),
            context_event_ids=(),
        ),
        "disp:1": SimpleNamespace(
            **current,
            source_event_ids=("bar:1",),
            context_event_ids=(),
        ),
    }

    raw, episodes, _, ledger = _v3_source_linked_stage(
        [prior],
        [current],
        completed_index={_clock(0): 0, _clock(1): 1},
        edge=edge,
        stage="E3",
        prior_chains={"sweep:1": ("episode:sweep", "sweep:1")},
        event_lookup=canonical.get,
    )

    assert raw == [current]
    assert len(episodes) == 1
    assert ledger[0]["semantic_event_ancestry_proven"] is False
    assert ledger[0]["source_ancestry_proven"] is False
    assert ledger[0]["composition_proven"] is True
    assert ledger[0]["shared_constituent_bar_event_ids"] == ("bar:1",)


def test_v3_nested_deltas_follow_fixed_order_and_keep_sparse_stage() -> None:
    resolved = (10, 10, 0, 10, 10, 10)
    rates = (0.4, 0.5, 0.5, 0.6, 0.7, 0.65)
    summaries = {
        stage: {
            "signals": sample,
            "resolved_n": sample,
            "laplace_success_rate": rate,
            "minimum_sample_met": False,
        }
        for stage, sample, rate in zip(
            V3_NESTED_STAGE_ORDER,
            resolved,
            rates,
            strict=True,
        )
    }

    nested = _v3_nested_deltas(summaries)

    assert tuple(nested) == V3_NESTED_STAGE_ORDER
    assert nested["E1_level_touch"]["delta_vs_prior"] is None
    assert nested["E2_sweep"]["delta_vs_prior"] == pytest.approx(0.1)
    assert nested["E3_sweep_displacement"]["delta_vs_prior"] is None
    assert nested["E4_sweep_displacement_mss"]["delta_vs_prior"] is None
    assert nested["E5_plus_fvg"]["delta_vs_prior"] == pytest.approx(0.1)
    assert nested["E6_plus_parent_alignment"]["delta_vs_prior"] == pytest.approx(-0.05)

    comparison_summary = {
        "signals": 1,
        "resolved_n": 1,
        "laplace_success_rate": 0.5,
        "minimum_sample_met": False,
    }
    report = _report_v3(
        {
            "semantic_version": "smc_semantics_v1.2",
            "status": "diagnostic_unvalidated",
            "artifact_classification": {"complete_registered_window": True},
            "nested_chain": nested,
            "non_nested_comparisons": {
                "mss_prior_sweep_composition_linked": {
                    "with": comparison_summary,
                    "without": comparison_summary,
                }
            },
            "control_comparisons": {
                name: {"requested": 0, "matched": 0} for name in V3_HOLM_FAMILY
            },
            "inference": {
                "exact_mcnemar": {name: {"p_value": 1.0} for name in V3_HOLM_FAMILY},
                "holm_fixed_family": {
                    "adjusted_p_values": {name: 1.0 for name in V3_HOLM_FAMILY}
                },
            },
        }
    )
    assert "Δ vs prior" in report
    assert "| E2_sweep | 10 | 10 | 0.500 | +0.100 |" in report
    assert "mss_prior_sweep_composition_linked" in report


def test_v3_non_nested_composition_persists_recomputable_proof() -> None:
    prior = {
        **_event("sweep:1", 0, kind="sweep_confirmed"),
        "lineage_tokens": ("event:bar:1",),
        "constituent_bar_event_ids": ("bar:1",),
    }
    linked = {
        **_event("mss:1", 1, kind="mss_core_confirmed"),
        "lineage_tokens": ("event:bar:1",),
        "constituent_bar_event_ids": ("bar:1",),
    }
    unlinked = {
        **_event("mss:2", 2, kind="mss_core_confirmed"),
        "lineage_tokens": ("event:bar:2",),
        "constituent_bar_event_ids": ("bar:2",),
    }
    bars = {
        identity: SimpleNamespace(
            event_id=identity,
            kind="bar_completed",
            timeframe="5m",
            direction=None,
            symbol="NQ",
            instrument_id=123,
            origin="normalized_data",
            real_completed=True,
            clock_only=False,
            known_at=_clock(0),
            source_event_ids=(),
            context_event_ids=(),
        )
        for identity in ("bar:1", "bar:2")
    }
    canonical = {
        **bars,
        "sweep:1": SimpleNamespace(
            **prior,
            source_event_ids=("bar:1",),
            context_event_ids=(),
        ),
        "mss:1": SimpleNamespace(
            **linked,
            source_event_ids=("bar:1",),
            context_event_ids=(),
        ),
        "mss:2": SimpleNamespace(
            **unlinked,
            source_event_ids=("bar:2",),
            context_event_ids=(),
        ),
    }
    edge = {
        "previous_kind": "sweep_confirmed",
        "previous_timeframe": "5m",
        "current_kind": "mss_core_confirmed",
        "current_timeframe": "5m",
        "maximum_completed_bars": 3,
        "link_mode": "cross_timeframe_constituent_bar_composition",
        "source_ancestry_required": False,
        "composition_proven": True,
        "constituent_bar_timeframe": "5m",
    }

    with_link, without_link, ledger = _v3_non_nested_partition(
        [prior],
        [linked, unlinked],
        completed_index={_clock(0): 0, _clock(1): 1, _clock(2): 2},
        edge=edge,
        comparison="mss_prior_sweep_composition_linked",
        event_lookup=canonical.get,
    )

    assert with_link[0]["constituent_event_ids"] == ("mss:1",)
    assert without_link[0]["constituent_event_ids"] == ("mss:2",)
    assert len(ledger) == 1
    assert ledger[0]["comparison"] == "mss_prior_sweep_composition_linked"
    assert ledger[0]["current_episode_id"] == with_link[0]["event_id"]
    assert ledger[0]["source_ancestry_proven"] is False
    assert ledger[0]["composition_proven"] is True
    assert ledger[0]["shared_constituent_bar_event_ids"] == ("bar:1",)


def test_forward_shift_excludes_changed_exact_stratum_with_reason() -> None:
    treatment = {
        "event_id": "episode:1",
        **{field: "same" for field in V3_MATCH_FIELDS},
    }
    candidate = dict(treatment)
    control = {"event_id": "shift:1", "completed_bar_offset": 90}

    assert _v3_time_shift_stratum_exclusion(treatment, candidate, control) is None

    candidate["atr_quartile"] = "changed"
    exclusion = _v3_time_shift_stratum_exclusion(treatment, candidate, control)

    assert exclusion is not None
    assert exclusion["reason"] == "shifted_exact_stratum_changed"
    assert exclusion["mismatched_fields"] == ("atr_quartile",)


def test_non_sweep_control_is_episode_level_and_same_clock_auditable() -> None:
    first = _event("touch:a", 0)
    duplicate = _event("touch:b", 0)
    later = _event("touch:c", 2)
    sweep = _event(
        "sweep:1",
        1,
        kind="sweep_confirmed",
        lineage=("event:touch:b",),
    )
    sweep["source_lineage_tokens"] = ("event:touch:b",)
    context_only_sweep = _event(
        "sweep:context",
        3,
        kind="sweep_confirmed",
        lineage=("event:touch:c",),
    )
    context_only_sweep["source_lineage_tokens"] = ()
    enriched = _v3_enriched_rows([_row(value) for value in range(5)])
    rows_by_clock = {row["asof"]: row for row in enriched}

    candidates, exclusions = _v3_non_sweep_touch_candidates(
        [first, duplicate, later],
        [sweep, context_only_sweep],
        completed_index={_clock(value): value for value in range(5)},
        maximum_sweep_window=2,
        rows_by_clock=rows_by_clock,
    )

    assert len(candidates) == 1
    assert candidates[0]["constituent_event_ids"] == ("touch:c",)
    assert {item["reason"] for item in exclusions} == {"descendant_sweep_present"}


def test_pseudo_control_requires_active_snapshot_then_real_future_touch() -> None:
    rows = _v3_enriched_rows(
        [
            _row(0, low=99.5, high=100.5),
            _row(1, low=103.5, high=104.5),
            *[_row(value) for value in range(2, 8)],
        ]
    )
    anchor = {
        "event_id": "range:active",
        "known_at": _clock(0),
        "symbol": "NQ",
        "instrument_id": 123,
        "known_real_levels": (),
        "snapshot": {
            "snapshot_id": "snapshot:0",
            "asof": _clock(0),
            "symbol": "NQ",
            "instrument_id": 123,
            "range_low": 90.0,
            "range_high": 110.0,
            "current_price": 100.0,
        },
    }
    contract = json.loads(TEMPLATE.read_text(encoding="utf-8"))["control_definition"]
    contract = deepcopy(contract)
    contract["pseudo_level_touch"]["relative_locations"] = [0.7]

    candidates, ledger = _v3_pseudo_touch_candidates(
        [anchor],
        rows,
        completed_index={_clock(value): value for value in range(8)},
        control_contract=contract,
        outcome_horizon=3,
    )

    assert len(candidates) == 1
    assert candidates[0]["known_at"] == _clock(1)
    assert candidates[0]["direction_known_at"] == _clock(0)
    assert candidates[0]["price"] == 104.0
    assert ledger[0]["status"] == "construction_input"
    assert ledger[0]["snapshot"]["snapshot_id"] == "snapshot:0"
    assert ledger[1]["status"] == "executable_touch"


def test_old_evidence_stems_and_max_bar_smokes_fail_closed(tmp_path: Path) -> None:
    assert _artifact_status("frozen", 10) == ("incomplete_smoke_not_experiment_result")
    old_output = (
        ROOT / "outputs/research/"
        "smc_semantics_v1_1_2024_01_phase5_diagnostic_v2.json"
    )
    with pytest.raises(ResearchContractError, match="immutable"):
        run(output=old_output, manifest_path=tmp_path / "never-opened.json")


def test_v3_output_bundle_preflight_requires_all_eight_files_absent(
    tmp_path: Path,
) -> None:
    output = tmp_path / "signal_research_v3.json"
    paths = _output_bundle_paths(output, research_protocol_version=3)

    assert tuple(path.name for path in paths) == (
        "signal_research_v3.json",
        "signal_research_v3.md",
        "signal_research_v3.event_study.jsonl",
        "signal_research_v3.control_pairs.jsonl",
        "signal_research_v3.control_unmatched.jsonl",
        "signal_research_v3.control_balance.jsonl",
        "signal_research_v3.source_chains.jsonl",
        "signal_research_v3.pseudo_construction.jsonl",
    )
    assert _preflight_output_bundle(
        output,
        research_protocol_version=3,
        max_bars=None,
        force=False,
    ) == paths

    for existing in paths:
        existing.write_text("occupied", encoding="utf-8")
        with pytest.raises(FileExistsError, match="output bundle already exists"):
            _preflight_output_bundle(
                output,
                research_protocol_version=3,
                max_bars=None,
                force=False,
            )
        existing.unlink()


def test_full_run_rejects_force_and_smoke_cannot_overwrite_sibling(
    tmp_path: Path,
) -> None:
    output = tmp_path / "signal_research_v3.json"
    with pytest.raises(ResearchContractError, match="--force is forbidden"):
        _preflight_output_bundle(
            output,
            research_protocol_version=3,
            max_bars=None,
            force=True,
        )

    _preflight_output_bundle(
        output,
        research_protocol_version=3,
        max_bars=100,
        force=True,
    )
    sibling = output.with_name(f"{output.stem}.control_balance.jsonl")
    sibling.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output bundle already exists"):
        _preflight_output_bundle(
            output,
            research_protocol_version=3,
            max_bars=100,
            force=True,
        )


def test_runner_invokes_bundle_preflight_before_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "signal_research_v3.json"
    sibling = output.with_name(f"{output.stem}.source_chains.jsonl")
    sibling.write_text("occupied", encoding="utf-8")
    contract = SimpleNamespace(payload={"research_protocol_version": 3})
    monkeypatch.setattr(
        signal_runner,
        "_load_contract_and_registry",
        lambda _: (contract, None, {}, {}),
    )
    monkeypatch.setattr(
        signal_runner,
        "_validated_research_design",
        lambda _: pytest.fail("analysis must not start before bundle preflight"),
    )

    with pytest.raises(FileExistsError, match="source_chains"):
        signal_runner.run(
            output=output,
            manifest_path=tmp_path / "manifest.json",
        )

    sibling.unlink()
    with pytest.raises(ResearchContractError, match="--force is forbidden"):
        signal_runner.run(
            output=output,
            manifest_path=tmp_path / "manifest.json",
            force=True,
        )


def test_v3_runner_is_one_analysis_branch_not_a_second_executable() -> None:
    # This guards the integration seam: protocol v3 is a pure analysis helper
    # called by the existing replay runner, not a second Eye/replay program.
    assert callable(_run_v3_analysis)
