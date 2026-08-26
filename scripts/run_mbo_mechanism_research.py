#!/usr/bin/env python3
"""Run the frozen Phase-6 OHLCV-to-MBO mechanism validation.

The runner constructs the production Reader and Trading Eye, but never the
Brain, playbooks, risk, execution simulator, or P&L.  It emits only compact
episode/pair ledgers and a development-association result.  The checked-in
manifest is deliberately incomplete and cannot execute until every input and
runtime identity has been reviewed and frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.artifact_stream import atomic_bytes  # noqa: E402
from smc_trader.causal import CausalMarketReader  # noqa: E402
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.market_state import MarketSnapshotAuthority  # noqa: E402
from smc_trader.mbo_mechanism_research import (  # noqa: E402
    EpisodeWindowUnavailable,
    EpisodeMatchContextUnavailable,
    EXTENSION_DISPLACEMENT_MONOTONICITY_POLICY,
    EXTENSION_WARMUP_SYNTHETIC_EXCEPTION_POLICY,
    LEGACY_SYNTHETIC_SEMANTIC_EXCEPTION_POLICY,
    MechanismEpisode,
    PHASE6_FIXED_FAMILY,
    PRIMARY_METRIC_BY_HYPOTHESIS,
    Phase6ResearchError,
    STRICT_PRIOR_CONTEXT_CENSOR_REASONS,
    SYNTHETIC_SEMANTIC_EXCEPTION_POLICY,
    aggregate_episode_feature_window,
    canonical_identity,
    canonicalize_mechanism_episodes,
    evaluate_descriptive_sensitivity,
    evaluate_fixed_mechanism_family,
    first_active_displacement_episodes,
    first_fvg_lifecycle_episodes,
    load_frozen_phase6_contract,
    pack_nonoverlapping_mechanism_controls,
    spearman_monotonicity,
    strict_prior_match_context_clock,
    validate_minute_feature_frame,
    validate_registered_synthetic_mbo_flow,
)  # noqa: E402
from smc_trader.model import (  # noqa: E402
    EventKind,
    EventOrigin,
    Timeframe,
)
from smc_trader.observation import CausalObserver, ObserverConfig  # noqa: E402
from smc_trader.scene_graph import parse_scale_specs  # noqa: E402
from smc_trader.semantics import load_semantic_selection  # noqa: E402
from smc_trader.signal_research import resolve_source_lineage_tokens  # noqa: E402


MANIFEST_PATH = (
    ROOT / "experiments/manifests/mbo_mechanism_phase6_template.yaml"
)
DEFAULT_OUTPUT = ROOT / "experiments/results/mbo_mechanism_phase6.json"
PRIMARY_COMPARISON = "primary_fixed_holm"
FVG_PSEUDO_SENSITIVITY_COMPARISON = (
    "fvg_successful_vs_pseudo_zone_descriptive_sensitivity"
)
ACTIVE_SYNTHETIC_CLOCK_SCOPE = "active_registered_window"
PRIOR_WARMUP_SYNTHETIC_CLOCK_SCOPE = "hash_bound_prior_week_warmup"


def _strict_prior_context_censor_counts(
    context_exclusions: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    registered = frozenset(STRICT_PRIOR_CONTEXT_CENSOR_REASONS)
    return dict(
        sorted(
            Counter(
                str(item.get("reason"))
                for item in context_exclusions
                if item.get("reason") in registered
            ).items()
        )
    )
TARGET_EVENT_KIND_TO_HYPOTHESIS = {
    EventKind.SWEEP_CONFIRMED: "sweep_rejection",
    EventKind.ACCEPTANCE_CONFIRMED: "acceptance_continuation",
    EventKind.DISPLACEMENT_OBSERVED: "displacement_impact",
    EventKind.MSS_CORE_CONFIRMED: "mss_flow_shift",
    EventKind.FVG_PARTIALLY_FILLED: "fvg_retest_response",
    EventKind.FVG_MIDPOINT_TOUCHED: "fvg_retest_response",
    EventKind.FVG_FULLY_FILLED: "fvg_retest_response",
    EventKind.FVG_INVALIDATED: "fvg_retest_response",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase6ResearchError(f"JSON object required: {path}")
    return value


def _extension_warmup_synthetic_registration(
    contract: Any,
) -> dict[str, Any]:
    """Resolve only exact, hash-bound week-1 warmup exceptions.

    The active Reader census remains the extension week.  This registration
    exists solely because the deterministic warmup replay crosses the prior
    week's already-audited synthetic decision clock.
    """

    prior = getattr(contract, "prior_week1_result", None)
    if prior is None:
        return {
            "registered_clocks": frozenset(),
            "expected_audits_by_event_id": {},
            "blocked_lineage_event_ids": frozenset(),
            "prior_result_identity": None,
        }
    if contract.payload.get("study_mode") != (
        "primary_plus_registered_underpowered_extension"
    ):
        raise Phase6ResearchError(
            "prior synthetic warmup registration is extension-only"
        )
    if contract.payload.get(
        "extension_warmup_synthetic_exception_policy"
    ) != EXTENSION_WARMUP_SYNTHETIC_EXCEPTION_POLICY:
        raise Phase6ResearchError(
            "extension warmup synthetic exception policy is absent"
        )
    if prior.get("result_identity") != canonical_identity(prior):
        raise Phase6ResearchError(
            "week-1 synthetic warmup registration lacks a valid result identity"
        )
    coverage = prior.get("coverage")
    reader = coverage.get("reader_active_census") if isinstance(coverage, Mapping) else None
    gate = (
        coverage.get("synthetic_semantic_exception_gate")
        if isinstance(coverage, Mapping)
        else None
    )
    if not isinstance(reader, Mapping) or not isinstance(gate, Mapping):
        raise Phase6ResearchError(
            "week-1 synthetic warmup audit coverage is incomplete"
        )
    if gate.get("policy") != LEGACY_SYNTHETIC_SEMANTIC_EXCEPTION_POLICY:
        raise Phase6ResearchError(
            "week-1 synthetic warmup audit policy changed"
        )
    allowed = gate.get("allowed")
    rejected = gate.get("rejected")
    blocked = gate.get("blocked_lineage_event_ids")
    reader_clocks = reader.get("synthetic_decision_clocks")
    if (
        not isinstance(allowed, list)
        or not isinstance(rejected, list)
        or rejected
        or gate.get("rejected_count") != 0
        or gate.get("allowed_count") != len(allowed)
        or not isinstance(blocked, list)
        or not isinstance(reader_clocks, list)
    ):
        raise Phase6ResearchError(
            "week-1 synthetic warmup audit census is invalid"
        )
    try:
        registered_reader_clocks = frozenset(
            pd.Timestamp(value).tz_convert("UTC") for value in reader_clocks
        )
    except (TypeError, ValueError) as error:
        raise Phase6ResearchError(
            "week-1 synthetic warmup Reader clocks are invalid"
        ) from error
    expected_by_id: dict[str, Mapping[str, Any]] = {}
    expected_blocked: set[str] = set()
    registered_clocks: set[pd.Timestamp] = set()
    for item in allowed:
        if not isinstance(item, Mapping):
            raise Phase6ResearchError(
                "week-1 synthetic warmup allowed audit is invalid"
            )
        event_id = item.get("event_id")
        context_id = item.get("current_synthetic_context_event_id")
        try:
            known_at = pd.Timestamp(item.get("known_at")).tz_convert("UTC")
        except (TypeError, ValueError) as error:
            raise Phase6ResearchError(
                "week-1 synthetic warmup allowed clock is invalid"
            ) from error
        if (
            not isinstance(event_id, str)
            or not event_id
            or event_id in expected_by_id
            or not isinstance(context_id, str)
            or not context_id
            or known_at not in registered_reader_clocks
            or not (
                contract.primary_window.start
                <= known_at
                < contract.primary_window.end_exclusive
            )
            or item.get("kind") != "displacement_observed"
            or item.get("timeframe") != "5m"
            or item.get("lifecycle") != "censored"
            or item.get("terminal_reason") != "synthetic_interruption"
            or item.get("disposition")
            != "allowed_but_excluded_from_all_analysis_samples"
        ):
            raise Phase6ResearchError(
                "week-1 synthetic warmup audit is outside the exact allowlist"
            )
        expected_by_id[event_id] = dict(item)
        expected_blocked.update((event_id, context_id))
        registered_clocks.add(known_at)
    if (
        registered_clocks != set(registered_reader_clocks)
        or reader.get("synthetic") != len(registered_reader_clocks)
        or gate.get("blocked_lineage_event_count") != len(expected_blocked)
        or set(str(value) for value in blocked) != expected_blocked
    ):
        raise Phase6ResearchError(
            "week-1 synthetic warmup blocked lineage census changed"
        )
    return {
        "registered_clocks": frozenset(registered_clocks),
        "expected_audits_by_event_id": expected_by_id,
        "blocked_lineage_event_ids": frozenset(expected_blocked),
        "prior_result_identity": str(prior["result_identity"]),
    }


def _bind_synthetic_exception_audit_scope(
    audit: Mapping[str, Any],
    *,
    observation_clock: pd.Timestamp,
    active_registered_clocks: frozenset[pd.Timestamp],
    prior_registration: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an allowed audit to exactly one active or prior warmup scope."""

    clock = pd.Timestamp(observation_clock).tz_convert("UTC")
    if pd.Timestamp(audit.get("known_at")).tz_convert("UTC") != clock:
        raise Phase6ResearchError(
            "synthetic semantic audit is attached to the wrong observation clock"
        )
    try:
        context_clocks = frozenset(
            pd.Timestamp(value).tz_convert("UTC")
            for value in audit.get("synthetic_context_clocks", ())
        )
    except (TypeError, ValueError) as error:
        raise Phase6ResearchError(
            "synthetic semantic audit context clocks are invalid"
        ) from error
    if not context_clocks:
        raise Phase6ResearchError(
            "synthetic semantic audit lacks registered context clocks"
        )
    prior_clocks = prior_registration["registered_clocks"]
    if context_clocks.issubset(prior_clocks):
        expected = prior_registration["expected_audits_by_event_id"].get(
            str(audit.get("event_id"))
        )
        legacy = _legacy_synthetic_exception_audit_projection(audit)
        if expected is None or _json_value(legacy) != _json_value(expected):
            raise Phase6ResearchError(
                "warmup synthetic semantic event differs from the hash-bound "
                "week-1 audit"
            )
        scope = PRIOR_WARMUP_SYNTHETIC_CLOCK_SCOPE
    elif context_clocks.issubset(active_registered_clocks):
        scope = ACTIVE_SYNTHETIC_CLOCK_SCOPE
    else:
        raise Phase6ResearchError(
            "synthetic semantic event has no registered clock scope"
        )
    return {**dict(audit), "clock_scope": scope}


def _scoped_displacement_monotonicity(
    current_week: Mapping[str, Any],
    *,
    prior_week1_result: Mapping[str, Any] | None,
    registered_policy: Any,
) -> dict[str, Any]:
    """Report extension monotonicity honestly without reconstructing raw rows."""

    if prior_week1_result is None:
        return {"scope": "primary_week_only", **dict(current_week)}
    if registered_policy != EXTENSION_DISPLACEMENT_MONOTONICITY_POLICY:
        raise Phase6ResearchError(
            "extension displacement monotonicity policy is absent"
        )
    prior = prior_week1_result.get("displacement_continuous_monotonicity")
    if not isinstance(prior, Mapping):
        raise Phase6ResearchError(
            "hash-bound week-1 displacement monotonicity is absent"
        )
    prior_n = prior.get("n")
    prior_rho = prior.get("spearman_rho")
    if (
        isinstance(prior_n, bool)
        or not isinstance(prior_n, int)
        or prior_n < 0
        or prior.get("threshold_selected") is not False
        or (
            prior_rho is not None
            and (
                isinstance(prior_rho, bool)
                or not isinstance(prior_rho, (int, float))
                or not math.isfinite(float(prior_rho))
                or not -1.0 <= float(prior_rho) <= 1.0
            )
        )
    ):
        raise Phase6ResearchError(
            "hash-bound week-1 displacement monotonicity is invalid"
        )
    return {
        "policy": EXTENSION_DISPLACEMENT_MONOTONICITY_POLICY,
        "prior_week1": dict(prior),
        "current_week2": dict(current_week),
        "combined": None,
        "combined_unavailable_reason": (
            "registered compact week-1 ledgers do not bind every displacement "
            "score/response observation"
        ),
        "threshold_selected": False,
        "holm_included": False,
        "phase7_evidence_admission": False,
    }


def _build_eye(model_path: Path) -> tuple[CausalMarketReader, CausalObserver]:
    model = _json(model_path)
    selection = load_semantic_selection(
        model.get("semantic_selection"),
        root=ROOT,
    )
    raw = model["observer"]
    specs = parse_scale_specs(model["scales"])
    minimum = raw["minimum_bars"]
    observer = CausalObserver(
        ObserverConfig(
            atr_period=int(raw["atr_period"]),
            memory_events=int(raw["memory_events"]),
            minimum_bars={
                timeframe: int(minimum[timeframe.value])
                for timeframe in (
                    Timeframe.H4,
                    Timeframe.H1,
                    Timeframe.M15,
                    Timeframe.M5,
                    Timeframe.M1,
                )
            },
            tick_size=float(model["tick_size"]),
            point_value=float(model["point_value"]),
            structure_protocol=str(ROOT / raw["structure_protocol"]),
            liquidity_protocol=str(ROOT / raw["liquidity_protocol"]),
            displacement_protocol=str(ROOT / raw["displacement_protocol"]),
            zone_protocol=str(ROOT / raw["zone_protocol"]),
            range_auction_protocol=str(ROOT / raw["range_auction_protocol"]),
            interaction_protocol=str(ROOT / raw["group5_protocol"]),
            semantic_registry=str(selection.atomic_registry.source_path),
            scale_specs=specs,
            project_scene_graph=False,
            materialize_event_view=False,
            range_auction_projection_only=False,
            eye_authority_mode=True,
            typed_transition_delta_transport=False,
            persist_state_projections=False,
        ),
        semantic_registry=selection.atomic_registry,
    )
    return CausalMarketReader(scale_specs=specs), observer


def _study_partition(
    clock: pd.Timestamp,
    *,
    primary_start: pd.Timestamp,
    primary_end: pd.Timestamp,
    extension_start: pd.Timestamp,
    extension_end: pd.Timestamp,
) -> tuple[str, str]:
    value = pd.Timestamp(clock).tz_convert("UTC")
    if primary_start <= value < primary_end:
        week, boundary = "week_1", pd.Timestamp("2024-06-05T04:00:00Z")
    elif extension_start <= value < extension_end:
        week, boundary = "week_2", pd.Timestamp("2024-06-12T04:00:00Z")
    else:
        raise Phase6ResearchError("episode lies outside the registered June weeks")
    return week, "first_half" if value < boundary else "second_half"


def _volatility_bucket(atr: float, *, tick_size: float) -> str:
    if not math.isfinite(atr) or atr <= 0.0:
        return "atr_unavailable"
    exponent = max(0, min(15, int(math.floor(math.log2(max(atr / tick_size, 1.0))))))
    return f"atr_ticks_log2_{exponent}"


def _relative_volume_bucket(value: float) -> str:
    if not math.isfinite(value) or value < 0.0:
        return "relative_volume_unavailable"
    if value < 0.5:
        return "rv_lt_0_5"
    if value < 1.0:
        return "rv_0_5_to_1"
    if value < 1.5:
        return "rv_1_to_1_5"
    if value < 2.0:
        return "rv_1_5_to_2"
    return "rv_ge_2"


def _trend_relation(direction: str, trend_direction: str | None) -> str:
    if trend_direction not in {"long", "short"}:
        return "trend_neutral"
    return "with_m5_trend" if direction == trend_direction else "against_m5_trend"


def _base_match_fields(
    *,
    study_week: str,
    half_week: str,
    direction: str,
    atr: float,
    relative_volume: float,
    trend_direction: str | None,
    tick_size: float,
) -> dict[str, Any]:
    return {
        "study_week": study_week,
        "half_week": half_week,
        "volatility_bucket": _volatility_bucket(atr, tick_size=tick_size),
        "trend_relation": _trend_relation(direction, trend_direction),
        "relative_volume_bucket": _relative_volume_bucket(relative_volume),
    }


def _strict_prior_match_context(
    source_bar_clocks: Sequence[pd.Timestamp],
    context_by_clock: Mapping[pd.Timestamp, Mapping[str, Any]],
    *,
    episode_id: str,
) -> tuple[pd.Timestamp, Mapping[str, Any]]:
    context_clock = strict_prior_match_context_clock(source_bar_clocks)
    context = context_by_clock.get(context_clock)
    if context is None:
        raise EpisodeMatchContextUnavailable(
            episode_id=episode_id,
            context_clock=context_clock,
        )
    try:
        atr = float(context.get("atr"))
    except (TypeError, ValueError):
        atr = math.nan
    if not math.isfinite(atr) or atr <= 0.0:
        raise EpisodeMatchContextUnavailable(
            episode_id=episode_id,
            context_clock=context_clock,
            reason="strict_prior_real_completed_m1_context_atr_unavailable",
        )
    try:
        relative_volume = float(context.get("relative_volume"))
    except (TypeError, ValueError):
        relative_volume = math.nan
    if not math.isfinite(relative_volume) or relative_volume < 0.0:
        raise EpisodeMatchContextUnavailable(
            episode_id=episode_id,
            context_clock=context_clock,
            reason=(
                "strict_prior_real_completed_m1_context_relative_volume_unavailable"
            ),
        )
    trend_direction = context.get("trend_direction")
    if trend_direction not in {None, "long", "short"}:
        raise Phase6ResearchError("strict-prior trend direction is invalid")
    return context_clock, {
        "atr": atr,
        "relative_volume": relative_volume,
        "trend_direction": trend_direction,
        "source": "cached_real_completed_M1_at_strict_prior_formation_clock",
    }


def _source_only_episode_m5_bars(
    event: Any,
    observer: CausalObserver,
) -> tuple[tuple[str, ...], tuple[pd.Timestamp, ...]]:
    """Resolve source-only M5 ancestry inside [event_time, known_at]."""

    tokens = resolve_source_lineage_tokens(event.event_id, observer.audit_store.get)
    bars: list[tuple[pd.Timestamp, str]] = []
    for token in tokens:
        if not token.startswith("event:"):
            continue
        ancestor = observer.audit_store.get(token.removeprefix("event:"))
        if (
            ancestor is not None
            and ancestor.origin is EventOrigin.NORMALIZED_DATA
            and ancestor.kind is EventKind.BAR_COMPLETED
            and ancestor.timeframe is Timeframe.M5
            and event.event_time <= ancestor.known_at <= event.known_at
            and ancestor.evidence.get("real_completed") is True
            and ancestor.evidence.get("clock_only") is False
        ):
            bars.append(
                (
                    pd.Timestamp(ancestor.known_at).tz_convert("UTC"),
                    str(ancestor.event_id),
                )
            )
    pairs = tuple(sorted(set(bars)))
    clocks = tuple(item[0] for item in pairs)
    event_ids = tuple(item[1] for item in pairs)
    if len(clocks) != len(set(clocks)):
        raise Phase6ResearchError(
            f"semantic event has multiple source M5 BAR identities at one clock: {event.event_id}"
        )
    if not clocks or clocks[-1] != pd.Timestamp(event.known_at).tz_convert("UTC"):
        raise Phase6ResearchError(
            f"semantic event lacks terminal source-only interval M5 BAR: {event.event_id}"
        )
    return event_ids, clocks


def _synthetic_semantic_exception_audit(
    event: Any,
    *,
    observation_clock: pd.Timestamp,
    registered_synthetic_clocks: frozenset[pd.Timestamp],
    event_get: Any,
) -> dict[str, Any]:
    """Validate a terminal against every synthetic M1 constituent root."""

    clock = pd.Timestamp(observation_clock).tz_convert("UTC")
    lifecycle = event.evidence.get("lifecycle")
    terminal_reason = event.evidence.get("terminal_reason")
    if (
        event.origin is not EventOrigin.SEMANTIC_ATOMIC
        or event.kind is not EventKind.DISPLACEMENT_OBSERVED
        or event.timeframe is not Timeframe.M5
        or lifecycle != "censored"
        or terminal_reason != "synthetic_interruption"
        or pd.Timestamp(event.known_at).tz_convert("UTC") != clock
    ):
        raise Phase6ResearchError(
            "synthetic clock emitted a semantic event outside the exact allowlist"
        )
    context_ids = tuple(str(value) for value in event.context_event_ids)
    if (
        not context_ids
        or len(context_ids) != len(set(context_ids))
        or set(context_ids).intersection(event.source_event_ids)
    ):
        raise Phase6ResearchError(
            "synthetic M1 context roots must be unique and context-only"
        )
    context_roots: list[tuple[pd.Timestamp, str, Any]] = []
    for identity in context_ids:
        root = event_get(identity)
        if root is None:
            raise Phase6ResearchError(
                "synthetic semantic exception DAG has a missing context root"
            )
        root_clock = pd.Timestamp(root.known_at).tz_convert("UTC")
        if (
            root.origin is not EventOrigin.NORMALIZED_DATA
            or root.kind is not EventKind.BAR_COMPLETED
            or root.timeframe is not Timeframe.M1
            or pd.Timestamp(root.event_time).tz_convert("UTC") != root_clock
            or root.evidence.get("clock_only") is not True
            or root.evidence.get("real_completed") is not False
            or root_clock not in registered_synthetic_clocks
            or not clock - pd.Timedelta(minutes=5) < root_clock <= clock
        ):
            raise Phase6ResearchError(
                "synthetic semantic exception has an invalid registered M1 "
                "constituent root"
            )
        context_roots.append((root_clock, identity, root))
    context_roots.sort(key=lambda item: (item[0], item[1]))
    if context_ids != tuple(item[1] for item in context_roots):
        raise Phase6ResearchError(
            "synthetic M1 context roots are not ordered by clock then event id"
        )
    context_clocks = tuple(item[0] for item in context_roots)
    expected_context_clocks = tuple(
        sorted(
            registered_clock
            for registered_clock in registered_synthetic_clocks
            if clock - pd.Timedelta(minutes=5) < registered_clock <= clock
        )
    )
    if (
        len(context_clocks) != len(set(context_clocks))
        or context_clocks != expected_context_clocks
    ):
        raise Phase6ResearchError(
            "synthetic M1 context roots do not exactly cover the registered "
            "synthetic constituent clocks"
        )

    nodes: dict[str, Any] = {str(event.event_id): event}
    edges: set[tuple[str, str, str]] = set()
    stack = [event]
    while stack:
        child = stack.pop()
        for relation, identities in (
            ("source", child.source_event_ids),
            ("context", child.context_event_ids),
        ):
            for identity in identities:
                parent = event_get(identity)
                if parent is None:
                    raise Phase6ResearchError(
                        "synthetic semantic exception DAG has a missing parent"
                    )
                edges.add((str(child.event_id), str(identity), relation))
                if identity not in nodes:
                    nodes[str(identity)] = parent
                    stack.append(parent)

    source_nodes: dict[str, Any] = {}
    stack_ids = list(event.source_event_ids)
    while stack_ids:
        identity = str(stack_ids.pop())
        if identity in source_nodes:
            continue
        parent = event_get(identity)
        if parent is None:
            raise Phase6ResearchError(
                "synthetic semantic exception source DAG has a missing parent"
            )
        source_nodes[identity] = parent
        stack_ids.extend(parent.source_event_ids)
    real_source_bars = tuple(
        parent
        for parent in source_nodes.values()
        if (
            parent.origin is EventOrigin.NORMALIZED_DATA
            and parent.kind is EventKind.BAR_COMPLETED
        )
    )
    if not real_source_bars or any(
        parent.timeframe is not Timeframe.M5
        or parent.evidence.get("real_completed") is not True
        or parent.evidence.get("clock_only") is not False
        or not isinstance(parent.evidence.get("detector_candle_id"), str)
        or not parent.evidence.get("detector_candle_id")
        for parent in real_source_bars
    ):
        raise Phase6ResearchError(
            "synthetic semantic exception has a non-real or non-M5 source BAR"
        )
    real_m5_detector_ids = tuple(
        str(parent.evidence["detector_candle_id"])
        for parent in real_source_bars
    )
    if len(real_m5_detector_ids) != len(set(real_m5_detector_ids)):
        raise Phase6ResearchError(
            "synthetic terminal recursive real M5 detector lineage has duplicates"
        )
    canonical_real_m5_detector_ids = tuple(sorted(real_m5_detector_ids))
    terminal_detector_ids_producer_order = tuple(
        str(data_id) for data_id in event.source_data_ids
    )
    if len(terminal_detector_ids_producer_order) != len(
        set(terminal_detector_ids_producer_order)
    ):
        raise Phase6ResearchError(
            "synthetic terminal detector candle identities contain duplicates"
        )
    context_rows: list[dict[str, Any]] = []
    forbidden_synthetic_identifiers: set[str] = set()
    for root_clock, identity, root in context_roots:
        source_data_ids = tuple(str(value) for value in root.source_data_ids)
        detector_id = root.evidence.get("detector_candle_id")
        if (
            not source_data_ids
            or len(source_data_ids) != len(set(source_data_ids))
            or not isinstance(detector_id, str)
            or not detector_id
        ):
            raise Phase6ResearchError(
                "synthetic context root identifiers are incomplete"
            )
        forbidden_synthetic_identifiers.update(source_data_ids)
        forbidden_synthetic_identifiers.add(detector_id)
        context_rows.append(
            {
                "event_id": identity,
                "known_at": root_clock,
                "source_data_ids": source_data_ids,
                "detector_candle_id": detector_id,
            }
        )
    if (
        set(terminal_detector_ids_producer_order).intersection(
            forbidden_synthetic_identifiers
        )
    ):
        raise Phase6ResearchError(
            "synthetic context root identifier leaked into terminal detector "
            "candle identities"
        )
    if (
        tuple(sorted(terminal_detector_ids_producer_order))
        != canonical_real_m5_detector_ids
    ):
        raise Phase6ResearchError(
            "synthetic terminal detector candle identities do not exactly equal "
            "the canonical recursive real M5 detector union"
        )
    nonreal_normalized_bars = {
        identity
        for identity, parent in nodes.items()
        if (
            parent.origin is EventOrigin.NORMALIZED_DATA
            and parent.kind is EventKind.BAR_COMPLETED
            and parent.evidence.get("real_completed") is False
            and parent.evidence.get("clock_only") is True
        )
    }
    context_id_set = set(context_ids)
    if nonreal_normalized_bars != context_id_set:
        raise Phase6ResearchError(
            "synthetic context roots are not the DAG's sole non-real BARs"
        )
    root_edges = {
        edge for edge in edges if edge[1] in context_id_set
    }
    if root_edges != {
        (str(event.event_id), identity, "context")
        for identity in context_ids
    }:
        raise Phase6ResearchError(
            "synthetic context root leaked outside terminal context edges"
        )
    last_real_source = max(
        pd.Timestamp(parent.known_at).tz_convert("UTC")
        for parent in real_source_bars
    )
    event_time = pd.Timestamp(event.event_time).tz_convert("UTC")
    if not event_time <= last_real_source < clock:
        raise Phase6ResearchError(
            "synthetic terminal event/source clocks violate causal ordering"
        )

    node_rows = sorted(
        (
            {
                "event_id": identity,
                "kind": parent.kind.value,
                "origin": parent.origin.value,
                "timeframe": parent.timeframe.value,
                "event_time": pd.Timestamp(parent.event_time).isoformat(),
                "known_at": pd.Timestamp(parent.known_at).isoformat(),
                "real_completed": parent.evidence.get("real_completed"),
                "clock_only": parent.evidence.get("clock_only"),
                "source_data_ids": list(parent.source_data_ids),
                "detector_candle_id": parent.evidence.get(
                    "detector_candle_id"
                ),
            }
            for identity, parent in nodes.items()
        ),
        key=lambda item: item["event_id"],
    )
    edge_rows = [
        {"child": child, "parent": parent, "relation": relation}
        for child, parent, relation in sorted(edges)
    ]
    dag_payload = {"nodes": node_rows, "edges": edge_rows}
    dag_sha256 = hashlib.sha256(
        json.dumps(
            dag_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "event_id": str(event.event_id),
        "known_at": clock,
        "kind": event.kind.value,
        "timeframe": event.timeframe.value,
        "lifecycle": str(lifecycle),
        "terminal_reason": str(terminal_reason),
        "source_event_ids": tuple(event.source_event_ids),
        "context_event_ids": tuple(event.context_event_ids),
        "synthetic_context_event_ids": context_ids,
        "synthetic_context_clocks": tuple(
            item["known_at"] for item in context_rows
        ),
        "synthetic_context_roots": tuple(context_rows),
        "synthetic_context_root_count": len(context_rows),
        "terminal_detector_candle_ids_producer_order": (
            terminal_detector_ids_producer_order
        ),
        "canonical_real_M5_detector_candle_id_union": (
            canonical_real_m5_detector_ids
        ),
        "terminal_detector_candle_id_count": len(
            terminal_detector_ids_producer_order
        ),
        "last_real_source_known_at": last_real_source,
        "dag_node_count": len(node_rows),
        "dag_edge_count": len(edge_rows),
        "dag_sha256": dag_sha256,
        "disposition": "allowed_but_excluded_from_all_analysis_samples",
    }


def _legacy_synthetic_exception_audit_projection(
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one aligned root to the immutable Week-1 v5 audit schema."""

    roots = audit.get("synthetic_context_roots")
    if not isinstance(roots, (tuple, list)) or len(roots) != 1:
        raise Phase6ResearchError(
            "week-1 legacy synthetic audit requires one aligned context root"
        )
    root = roots[0]
    if not isinstance(root, Mapping):
        raise Phase6ResearchError(
            "week-1 legacy synthetic audit context root is invalid"
        )
    return {
        "event_id": audit["event_id"],
        "known_at": audit["known_at"],
        "kind": audit["kind"],
        "timeframe": audit["timeframe"],
        "lifecycle": audit["lifecycle"],
        "terminal_reason": audit["terminal_reason"],
        "current_synthetic_context_event_id": root["event_id"],
        "source_event_ids": audit["source_event_ids"],
        "context_event_ids": audit["context_event_ids"],
        "terminal_detector_candle_ids_producer_order": audit[
            "terminal_detector_candle_ids_producer_order"
        ],
        "canonical_real_M5_detector_candle_id_union": audit[
            "canonical_real_M5_detector_candle_id_union"
        ],
        "synthetic_context_root_source_data_ids": root["source_data_ids"],
        "synthetic_context_root_detector_candle_id": root[
            "detector_candle_id"
        ],
        "terminal_detector_candle_id_count": audit[
            "terminal_detector_candle_id_count"
        ],
        "last_real_source_known_at": audit["last_real_source_known_at"],
        "dag_node_count": audit["dag_node_count"],
        "dag_edge_count": audit["dag_edge_count"],
        "dag_sha256": audit["dag_sha256"],
        "disposition": audit["disposition"],
    }


def _blocked_synthetic_exception_lineage(
    event: Any,
    *,
    event_get: Any,
    blocked_event_ids: frozenset[str] | set[str],
) -> tuple[str, ...]:
    """Find blocked IDs in explicit semantic-atomic source/context ancestry."""

    blocked = frozenset(str(value) for value in blocked_event_ids)
    if not blocked:
        return ()
    visited: set[str] = set()
    hits: set[str] = set()
    stack = [event]
    while stack:
        current = stack.pop()
        identity = str(current.event_id)
        if identity in visited:
            continue
        visited.add(identity)
        if identity in blocked:
            hits.add(identity)
        if current.origin is not EventOrigin.SEMANTIC_ATOMIC:
            # Only the canonical semantic namespace guarantees that these
            # fields contain event IDs.  Legacy transports reuse them for
            # opaque entity/candle tokens; normalized data and projections
            # are terminal roots for this sample-admission scan.
            continue
        for parent_id in (
            *current.source_event_ids,
            *current.context_event_ids,
        ):
            parent = event_get(parent_id)
            if parent is None:
                raise Phase6ResearchError(
                    "sample-admission lineage has a missing source/context parent"
                )
            stack.append(parent)
    return tuple(sorted(hits))


def _event_variant(kind: EventKind) -> str | None:
    if kind is EventKind.FVG_INVALIDATED:
        return "failed_retest"
    if kind in {
        EventKind.FVG_PARTIALLY_FILLED,
        EventKind.FVG_MIDPOINT_TOUCHED,
        EventKind.FVG_FULLY_FILLED,
    }:
        return "successful_retest"
    return None


def _episode_from_event(
    event: Any,
    *,
    hypothesis: str,
    observer: CausalObserver,
    snapshot: Any,
    contract: Any,
    tick_size: float,
    context_by_clock: Mapping[pd.Timestamp, Mapping[str, Any]],
    control_kind: str | None = None,
) -> MechanismEpisode:
    if event.origin is not EventOrigin.SEMANTIC_ATOMIC:
        raise Phase6ResearchError("Phase-6 treatment is not semantic_atomic")
    if event.timeframe is not Timeframe.M5 or event.direction is None:
        raise Phase6ResearchError("Phase-6 target event must be directional M5")
    known_at = pd.Timestamp(event.known_at).tz_convert("UTC")
    week, half = _study_partition(
        known_at,
        primary_start=contract.primary_window.start,
        primary_end=contract.primary_window.end_exclusive,
        extension_start=contract.extension_window.start,
        extension_end=contract.extension_window.end_exclusive,
    )
    direction = event.direction.value
    source_m5_bar_event_ids, source_bar_clocks = _source_only_episode_m5_bars(
        event,
        observer,
    )
    context_clock, context = _strict_prior_match_context(
        source_bar_clocks,
        context_by_clock,
        episode_id=str(event.event_id),
    )
    entity_id = event.entity_id
    if entity_id is None and hypothesis == "fvg_retest_response":
        entity_id = event.evidence.get("fvg_id")
    if entity_id is None and event.source_entity_ids:
        entity_id = event.source_entity_ids[0]
    return MechanismEpisode(
        episode_id=str(event.event_id),
        hypothesis=hypothesis,
        event_kind=event.kind.value,
        entity_id=None if entity_id is None else str(entity_id),
        known_at=known_at,
        event_time=pd.Timestamp(event.event_time).tz_convert("UTC"),
        source_bar_clocks=source_bar_clocks,
        source_m5_bar_event_ids=source_m5_bar_event_ids,
        symbol=str(snapshot.symbol),
        instrument_id=int(snapshot.instrument_id),
        timeframe="5m",
        direction=direction,
        session_phase=str(snapshot.session.phase),
        half_week=half,
        match_fields=_base_match_fields(
            study_week=week,
            half_week=half,
            direction=direction,
            atr=float(context["atr"]),
            relative_volume=float(context["relative_volume"]),
            trend_direction=context.get("trend_direction"),
            tick_size=tick_size,
        ),
        score=float(event.strength) if hypothesis == "displacement_impact" else None,
        control_kind=control_kind,
        event_variant=_event_variant(event.kind),
        match_context_clock=context_clock,
    )


def _candidate_episode(
    base: Mapping[str, Any],
    *,
    hypothesis: str,
    direction: str,
    candidate_id: str,
    event_kind: str,
    control_kind: str,
    event_variant: str | None = None,
) -> MechanismEpisode:
    match_fields = _base_match_fields(
        study_week=str(base["study_week"]),
        half_week=str(base["half_week"]),
        direction=direction,
        atr=float(base["atr"]),
        relative_volume=float(base["relative_volume"]),
        trend_direction=base.get("trend_direction"),
        tick_size=float(base["tick_size"]),
    )
    return MechanismEpisode(
        episode_id=candidate_id,
        hypothesis=hypothesis,
        event_kind=event_kind,
        entity_id=None,
        known_at=base["known_at"],
        event_time=base["known_at"],
        source_bar_clocks=(base["known_at"],),
        source_m5_bar_event_ids=(str(base["source_m5_bar_event_id"]),),
        symbol=str(base["symbol"]),
        instrument_id=int(base["instrument_id"]),
        timeframe="5m",
        direction=direction,
        session_phase=str(base["session_phase"]),
        half_week=str(base["half_week"]),
        match_fields=match_fields,
        control_kind=control_kind,
        event_variant=event_variant,
        match_context_clock=base["match_context_clock"],
        match_context_source=str(base["match_context_source"]),
    )


def _validate_reader_census(
    *,
    registered: Mapping[str, Any],
    active_window_id: str,
    feature_clocks: frozenset[pd.Timestamp],
    observed_clocks: set[pd.Timestamp],
    completed: int,
    real: int,
    synthetic: int,
    synthetic_clocks: set[pd.Timestamp],
    first: pd.Timestamp | None,
    last: pd.Timestamp | None,
    contracts: set[tuple[str, int]],
    expected_contract: tuple[str, int],
    data_gap_resets: int,
    contract_changes: int,
) -> None:
    expected_synthetic_clocks = {
        pd.Timestamp(value).tz_convert("UTC")
        for value in registered["synthetic_decision_clocks"]
    }
    if (
        registered.get("window_id") != active_window_id
        or observed_clocks != set(feature_clocks)
        or completed != int(registered["completed_clocks"])
        or real != int(registered["real_completed"])
        or synthetic != int(registered["synthetic_no_trade"])
        or synthetic_clocks != expected_synthetic_clocks
        or first != min(feature_clocks)
        or last != max(feature_clocks)
        or contracts != {expected_contract}
        or data_gap_resets != 0
        or contract_changes != 0
    ):
        raise Phase6ResearchError(
            "Trading Eye replay census disagrees with the exact MBO completed clock"
        )


def _collect_eye_inputs(
    *,
    loaded: Any,
    feature_clocks: frozenset[pd.Timestamp],
    contract: Any,
    tick_size: float,
) -> dict[str, Any]:
    reader, observer = _build_eye(contract.model_path)
    reader_contract = contract.payload.get("reader_census_contract")
    if not isinstance(reader_contract, Mapping):
        raise Phase6ResearchError("frozen Reader census contract is absent")
    active_registered_synthetic_clocks = frozenset(
        pd.Timestamp(value).tz_convert("UTC")
        for value in reader_contract["synthetic_decision_clocks"]
    )
    if contract.payload.get("synthetic_semantic_exception_policy") != (
        SYNTHETIC_SEMANTIC_EXCEPTION_POLICY
    ):
        raise Phase6ResearchError(
            "frozen synthetic semantic exception policy is absent"
        )
    prior_warmup_registration = _extension_warmup_synthetic_registration(
        contract
    )
    prior_warmup_synthetic_clocks = prior_warmup_registration[
        "registered_clocks"
    ]
    if active_registered_synthetic_clocks.intersection(
        prior_warmup_synthetic_clocks
    ):
        raise Phase6ResearchError(
            "active and prior-warmup synthetic clock scopes overlap"
        )
    registered_synthetic_clocks = frozenset(
        active_registered_synthetic_clocks.union(
            prior_warmup_synthetic_clocks
        )
    )
    expected_prior_audits = prior_warmup_registration[
        "expected_audits_by_event_id"
    ]
    consumed_prior_audit_ids: set[str] = set()
    targets: dict[str, list[MechanismEpisode]] = defaultdict(list)
    raw_break_episodes: list[MechanismEpisode] = []
    mss_source_raw_ids: set[str] = set()
    fvg_creations: list[dict[str, Any]] = []
    m5_bases: list[dict[str, Any]] = []
    active_displacement_clocks: set[pd.Timestamp] = set()
    seen_event_ids: set[str] = set()
    context_by_clock: dict[pd.Timestamp, dict[str, Any]] = {}
    real_context_clocks_seen: set[pd.Timestamp] = set()
    context_exclusions: list[dict[str, Any]] = []
    synthetic_semantic_allowed: list[dict[str, Any]] = []
    synthetic_semantic_rejected: list[dict[str, Any]] = []
    synthetic_semantic_allowed_ids: set[str] = set()
    synthetic_semantic_blocked_lineage_ids: set[str] = set(
        prior_warmup_registration["blocked_lineage_event_ids"]
    )
    emitted = 0
    active_completed = 0
    active_real = 0
    active_synthetic = 0
    active_clocks: set[pd.Timestamp] = set()
    active_synthetic_clocks: set[pd.Timestamp] = set()
    active_first: pd.Timestamp | None = None
    active_last: pd.Timestamp | None = None
    active_contracts: set[tuple[str, int]] = set()
    active_data_gap_resets = 0
    active_contract_changes = 0

    def exclude_blocked_descendant(
        event: Any,
        *,
        hypothesis: str,
        role: str,
    ) -> bool:
        blocked_ids = _blocked_synthetic_exception_lineage(
            event,
            event_get=observer.audit_store.get,
            blocked_event_ids=synthetic_semantic_blocked_lineage_ids,
        )
        if not blocked_ids:
            return False
        context_exclusions.append(
            {
                "hypothesis": hypothesis,
                "role": role,
                "episode_id": str(event.event_id),
                "known_at": pd.Timestamp(event.known_at).tz_convert("UTC"),
                "blocked_lineage_event_ids": blocked_ids,
                "blocked_lineage_event_count": len(blocked_ids),
                "blocked_lineage_event_ids_sha256": hashlib.sha256(
                    json.dumps(
                        list(blocked_ids),
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "reason": (
                    "synthetic_semantic_exception_descendant_excluded_from_samples"
                ),
            }
        )
        return True

    for bar in iter_completed_bars(loaded.frame, allow_data_gap_reset=True):
        update = reader.on_bar(bar)
        observation = observer.observe(update)
        emitted += 1
        snapshot = observation.market_snapshot
        if snapshot is None:
            raise Phase6ResearchError("Trading Eye did not publish a market snapshot")
        if snapshot.authority is not MarketSnapshotAuthority.ATOMIC_EVENT_REDUCER:
            raise Phase6ResearchError(
                "Phase-6 requires atomic_event_reducer MarketSnapshot authority"
            )
        observation_clock = observation.asof.tz_convert("UTC")
        m5_state = snapshot.timeframe_states[Timeframe.M5]
        current_trend = (
            None
            if m5_state.structure.internal_direction is None
            else m5_state.structure.internal_direction.value
        )
        if update.completed_1m.real_completed:
            if observation_clock in real_context_clocks_seen:
                raise Phase6ResearchError("duplicate real completed M1 context clock")
            real_context_clocks_seen.add(observation_clock)
            context_by_clock[observation_clock] = {
                "atr": observation.frame(Timeframe.M1).metrics.get("atr"),
                "relative_volume": snapshot.session.relative_volume,
                "trend_direction": current_trend,
                "source": (
                    "cached_real_completed_M1_at_strict_prior_formation_clock"
                ),
            }
        normalized_m5_bar_ids = {
            pd.Timestamp(event.known_at).tz_convert("UTC"): str(event.event_id)
            for event in observation.semantic_events_this_update
            if (
                event.origin is EventOrigin.NORMALIZED_DATA
                and event.kind is EventKind.BAR_COMPLETED
                and event.timeframe is Timeframe.M5
                and event.evidence.get("real_completed") is True
                and event.evidence.get("clock_only") is False
            )
        }
        semantic_atomic_events = tuple(
            event
            for event in observation.semantic_events_this_update
            if event.origin is EventOrigin.SEMANTIC_ATOMIC
        )
        synthetic_gate_events = (
            semantic_atomic_events
            if not update.completed_1m.real_completed
            else tuple(
                event
                for event in semantic_atomic_events
                if (
                    event.kind is EventKind.DISPLACEMENT_OBSERVED
                    and event.evidence.get("lifecycle") == "censored"
                    and event.evidence.get("terminal_reason")
                    == "synthetic_interruption"
                )
            )
        )
        for event in synthetic_gate_events:
            try:
                audit = _synthetic_semantic_exception_audit(
                    event,
                    observation_clock=observation_clock,
                    registered_synthetic_clocks=registered_synthetic_clocks,
                    event_get=observer.audit_store.get,
                )
                audit = _bind_synthetic_exception_audit_scope(
                    audit,
                    observation_clock=observation_clock,
                    active_registered_clocks=(
                        active_registered_synthetic_clocks
                    ),
                    prior_registration=prior_warmup_registration,
                )
            except Phase6ResearchError as error:
                synthetic_semantic_rejected.append(
                    {
                        "event_id": str(event.event_id),
                        "known_at": observation_clock,
                        "kind": event.kind.value,
                        "timeframe": event.timeframe.value,
                        "reason": str(error),
                    }
                )
                raise Phase6ResearchError(
                    "synthetic semantic event failed the frozen exception "
                    f"gate: {event.event_id}: {error}"
                ) from error
            event_id = str(event.event_id)
            if audit["clock_scope"] == PRIOR_WARMUP_SYNTHETIC_CLOCK_SCOPE:
                consumed_prior_audit_ids.add(event_id)
            synthetic_semantic_allowed.append(audit)
            synthetic_semantic_allowed_ids.add(event_id)
            synthetic_semantic_blocked_lineage_ids.update(
                {event_id, *audit["synthetic_context_event_ids"]}
            )
        active_clock = observation_clock in feature_clocks
        if active_clock:
            if observation_clock in active_clocks:
                raise Phase6ResearchError("Reader repeated an active completed clock")
            active_clocks.add(observation_clock)
            active_completed += 1
            active_first = (
                observation_clock
                if active_first is None
                else active_first
            )
            active_last = observation_clock
            active_contracts.add(
                (
                    str(update.completed_1m.symbol),
                    int(update.completed_1m.instrument_id),
                )
            )
            if update.completed_1m.real_completed:
                active_real += 1
            else:
                active_synthetic += 1
                active_synthetic_clocks.add(observation_clock)
            if "data_gap_history_reset" in update.anomalies:
                active_data_gap_resets += 1
                raise Phase6ResearchError(
                    "data-gap history reset inside the registered Phase-6 window"
                )
            if "contract_change_history_reset" in update.anomalies:
                active_contract_changes += 1
                raise Phase6ResearchError(
                    "contract change inside the registered Phase-6 window"
                )
            for candle in update.newly_completed.get(Timeframe.M5, ()):
                clock = candle.end.tz_convert("UTC")
                if clock not in feature_clocks or not candle.real_completed:
                    continue
                source_m5_bar_event_id = normalized_m5_bar_ids.get(clock)
                if source_m5_bar_event_id is None:
                    raise Phase6ResearchError(
                        "real completed M5 control base lacks normalized source identity"
                    )
                source_m5_bar_event = observer.audit_store.get(
                    source_m5_bar_event_id
                )
                if source_m5_bar_event is None:
                    raise Phase6ResearchError(
                        "real completed M5 control base source is absent from audit"
                    )
                if exclude_blocked_descendant(
                    source_m5_bar_event,
                    hypothesis="shared_m5_control_base",
                    role="shared_m5_candidate_control_base",
                ):
                    continue
                try:
                    context_clock, context = _strict_prior_match_context(
                        (clock,),
                        context_by_clock,
                        episode_id=f"completed_m5:{clock.isoformat()}",
                    )
                except EpisodeMatchContextUnavailable as error:
                    context_exclusions.append(
                        {
                            "hypothesis": "shared_m5_control_base",
                            "applies_to_hypotheses": (
                                "displacement_impact",
                                "fvg_retest_response",
                            ),
                            "role": "shared_m5_candidate_control_base",
                            "episode_id": error.episode_id,
                            "context_clock": error.context_clock,
                            "context_source": (
                                "cached_real_completed_M1_at_strict_prior_formation_clock"
                            ),
                            "reason": error.reason,
                        }
                    )
                    continue
                direction = (
                    "long"
                    if candle.close > candle.open
                    else "short" if candle.close < candle.open else None
                )
                if direction is None:
                    direction = context.get("trend_direction")
                if direction not in {"long", "short"}:
                    continue
                week, half = _study_partition(
                    clock,
                    primary_start=contract.primary_window.start,
                    primary_end=contract.primary_window.end_exclusive,
                    extension_start=contract.extension_window.start,
                    extension_end=contract.extension_window.end_exclusive,
                )
                m5_bases.append(
                    {
                        "known_at": clock,
                        "symbol": candle.symbol,
                        "instrument_id": candle.instrument_id,
                        "open": float(candle.open),
                        "high": float(candle.high),
                        "low": float(candle.low),
                        "close": float(candle.close),
                        "candle_direction": direction,
                        "trend_direction": context.get("trend_direction"),
                        "session_phase": snapshot.session.phase,
                        "relative_volume": float(context["relative_volume"]),
                        "atr": float(context["atr"]),
                        "tick_size": tick_size,
                        "study_week": week,
                        "half_week": half,
                        "match_context_clock": context_clock,
                        "match_context_source": context["source"],
                        "source_m5_bar_event_id": source_m5_bar_event_id,
                    }
                )

        for event in observation.semantic_events_this_update:
            if event.event_id in seen_event_ids:
                raise Phase6ResearchError("immutable semantic event repeated")
            seen_event_ids.add(event.event_id)
            if (
                event.origin is EventOrigin.SEMANTIC_ATOMIC
                and event.known_at != observation.asof
            ):
                raise Phase6ResearchError(
                    "semantic event was attached to a snapshot other than its known_at"
                )
            if event.kind is EventKind.FVG_TOUCHED:
                raise Phase6ResearchError(
                    "FVG_TOUCHED compatibility alias cannot enter Phase 6"
                )
            if str(event.event_id) in synthetic_semantic_allowed_ids:
                continue
            if event.kind is EventKind.FVG_CREATED:
                if exclude_blocked_descendant(
                    event,
                    hypothesis="fvg_retest_response",
                    role="fvg_creation",
                ):
                    continue
                if event.zone is not None and event.direction is not None:
                    fvg_creations.append(
                        {
                            "event_id": event.event_id,
                            "entity_id": (
                                event.entity_id
                                or event.evidence.get("fvg_id")
                                or (
                                    event.source_entity_ids[0]
                                    if event.source_entity_ids
                                    else None
                                )
                            ),
                            "known_at": event.known_at.tz_convert("UTC"),
                            "direction": event.direction.value,
                            "zone": tuple(float(value) for value in event.zone),
                        }
                    )
                continue
            if not active_clock or event.timeframe is not Timeframe.M5:
                continue
            if event.kind is EventKind.RAW_BOUNDARY_BREAK:
                if exclude_blocked_descendant(
                    event,
                    hypothesis="mss_flow_shift",
                    role="raw_break_control",
                ):
                    continue
                try:
                    raw_break_episodes.append(
                        _episode_from_event(
                            event,
                            hypothesis="mss_flow_shift",
                            observer=observer,
                            snapshot=snapshot,
                            contract=contract,
                            tick_size=tick_size,
                            context_by_clock=context_by_clock,
                            control_kind="raw_break_without_mss",
                        )
                    )
                except EpisodeMatchContextUnavailable as error:
                    context_exclusions.append(
                        {
                            "hypothesis": "mss_flow_shift",
                            "role": "raw_break_control",
                            "episode_id": error.episode_id,
                            "context_clock": error.context_clock,
                            "context_source": (
                                "cached_real_completed_M1_at_strict_prior_formation_clock"
                            ),
                            "reason": error.reason,
                        }
                    )
                continue
            hypothesis = TARGET_EVENT_KIND_TO_HYPOTHESIS.get(event.kind)
            if hypothesis is None:
                continue
            if exclude_blocked_descendant(
                event,
                hypothesis=hypothesis,
                role="treatment",
            ):
                continue
            if (
                event.kind is EventKind.DISPLACEMENT_OBSERVED
                and event.evidence.get("lifecycle") != "active"
            ):
                continue
            if hypothesis == "displacement_impact":
                # Every ACTIVE clock excludes the corresponding M5 control,
                # even when a later entity-first or context gate removes it.
                active_displacement_clocks.add(
                    pd.Timestamp(event.known_at).tz_convert("UTC")
                )
            elif hypothesis == "mss_flow_shift":
                # A context-censored MSS still disqualifies its source raw
                # break from the non-MSS control population.
                mss_source_raw_ids.update(event.source_event_ids)
            try:
                targets[hypothesis].append(
                    _episode_from_event(
                        event,
                        hypothesis=hypothesis,
                        observer=observer,
                        snapshot=snapshot,
                        contract=contract,
                        tick_size=tick_size,
                        context_by_clock=context_by_clock,
                    )
                )
            except EpisodeMatchContextUnavailable as error:
                context_exclusions.append(
                    {
                        "hypothesis": hypothesis,
                        "role": "treatment",
                        "episode_id": error.episode_id,
                        "context_clock": error.context_clock,
                        "context_source": (
                            "cached_real_completed_M1_at_strict_prior_formation_clock"
                        ),
                        "reason": error.reason,
                    }
                )
        if emitted % 2500 == 0:
            print(f"Phase6 Eye replay: {emitted:,} completed bars", flush=True)

    missing_prior_audit_ids = set(expected_prior_audits).difference(
        consumed_prior_audit_ids
    )
    if missing_prior_audit_ids:
        raise Phase6ResearchError(
            "hash-bound week-1 synthetic warmup exception was not reproduced: "
            + ",".join(sorted(missing_prior_audit_ids))
        )

    raw_controls = [
        episode
        for episode in raw_break_episodes
        if episode.episode_id not in mss_source_raw_ids
    ]
    _validate_reader_census(
        registered=reader_contract,
        active_window_id=contract.active_window.window_id,
        feature_clocks=feature_clocks,
        observed_clocks=active_clocks,
        completed=active_completed,
        real=active_real,
        synthetic=active_synthetic,
        synthetic_clocks=active_synthetic_clocks,
        first=active_first,
        last=active_last,
        contracts=active_contracts,
        expected_contract=(contract.symbol, contract.instrument_id),
        data_gap_resets=active_data_gap_resets,
        contract_changes=active_contract_changes,
    )
    return {
        "targets": {name: list(targets.get(name, ())) for name in PHASE6_FIXED_FAMILY},
        "raw_break_controls": raw_controls,
        "m5_bases": sorted(m5_bases, key=lambda item: item["known_at"]),
        "active_displacement_clocks": frozenset(active_displacement_clocks),
        "fvg_creations": sorted(fvg_creations, key=lambda item: item["known_at"]),
        "context_exclusions": context_exclusions,
        "synthetic_semantic_exception_audit": {
            "allowed": synthetic_semantic_allowed,
            "rejected": synthetic_semantic_rejected,
            "allowed_by_clock_scope": dict(
                sorted(
                    Counter(
                        item["clock_scope"]
                        for item in synthetic_semantic_allowed
                    ).items()
                )
            ),
            "prior_warmup_registration": {
                "policy": EXTENSION_WARMUP_SYNTHETIC_EXCEPTION_POLICY,
                "prior_result_identity": prior_warmup_registration[
                    "prior_result_identity"
                ],
                "registered_clocks": tuple(
                    sorted(prior_warmup_synthetic_clocks)
                ),
                "expected_audit_count": len(expected_prior_audits),
                "replayed_verified_count": len(consumed_prior_audit_ids),
            },
            "blocked_lineage_event_ids": tuple(
                sorted(synthetic_semantic_blocked_lineage_ids)
            ),
        },
        "emitted_bars": emitted,
        "semantic_event_ids_seen": len(seen_event_ids),
        "reader_census": {
            "completed": active_completed,
            "real": active_real,
            "synthetic": active_synthetic,
            "synthetic_decision_clocks": tuple(sorted(active_synthetic_clocks)),
            "first": active_first,
            "last": active_last,
            "contracts": tuple(sorted(active_contracts)),
            "data_gap_resets": active_data_gap_resets,
            "contract_changes": active_contract_changes,
        },
    }


def _pseudo_fvg_controls(
    creations: Sequence[Mapping[str, Any]],
    m5_bases: Sequence[Mapping[str, Any]],
    *,
    maximum_future_m5_bars: int = 120,
) -> tuple[list[MechanismEpisode], list[dict[str, Any]]]:
    """Construct outcome-blind, same-width pseudo zones known at FVG creation."""

    base_by_clock = {item["known_at"]: item for item in m5_bases}
    ordered_clocks = [item["known_at"] for item in m5_bases]
    index_by_clock = {clock: index for index, clock in enumerate(ordered_clocks)}
    controls: list[MechanismEpisode] = []
    exclusions: list[dict[str, Any]] = []
    known_real_zones: list[tuple[pd.Timestamp, float, float]] = []
    for creation in creations:
        clock = pd.Timestamp(creation["known_at"]).tz_convert("UTC")
        zone = tuple(float(value) for value in creation["zone"])
        lower, upper = min(zone), max(zone)
        width = upper - lower
        base = base_by_clock.get(clock)
        known_real_zones.append((clock, lower, upper))
        if base is None or width <= 0.0:
            exclusions.append(
                {"creation_event_id": creation["event_id"], "reason": "creation_clock_or_width_unavailable"}
            )
            continue
        close = float(base["close"])
        tick = float(base["tick_size"])
        if creation["direction"] == "long":
            gap = max(tick, close - upper)
            pseudo_upper = lower - gap
            pseudo_lower = pseudo_upper - width
        else:
            gap = max(tick, lower - close)
            pseudo_lower = upper + gap
            pseudo_upper = pseudo_lower + width
        if any(
            known_at <= clock
            and not (pseudo_upper < real_lower or pseudo_lower > real_upper)
            for known_at, real_lower, real_upper in known_real_zones
        ):
            exclusions.append(
                {"creation_event_id": creation["event_id"], "reason": "pseudo_zone_overlaps_known_real_fvg"}
            )
            continue
        start_index = index_by_clock[clock]
        touched: Mapping[str, Any] | None = None
        for candidate in m5_bases[
            start_index + 1 : start_index + 1 + maximum_future_m5_bars
        ]:
            if float(candidate["low"]) <= pseudo_upper and float(candidate["high"]) >= pseudo_lower:
                touched = candidate
                break
        if touched is None:
            exclusions.append(
                {"creation_event_id": creation["event_id"], "reason": "pseudo_zone_not_touched_in_registered_window"}
            )
            continue
        identity = hashlib.sha256(
            json.dumps(
                [
                    creation["event_id"],
                    pseudo_lower,
                    pseudo_upper,
                    touched["known_at"].isoformat(),
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        controls.append(
            _candidate_episode(
                touched,
                hypothesis="fvg_retest_response",
                direction=str(creation["direction"]),
                candidate_id=f"pseudo_fvg:{identity}",
                event_kind="pseudo_fvg_zone_retest",
                control_kind="pseudo_zone_retest",
                event_variant="pseudo_zone",
            )
        )
    return controls, exclusions


def _load_feature_artifact(contract: Any) -> tuple[pd.DataFrame, str]:
    from smc_trader.mbo_mechanism import (  # noqa: PLC0415
        MBO_MECHANISM_PROTOCOL_SHA256,
        load_mbo_mechanism_artifact,
    )

    frame = load_mbo_mechanism_artifact(
        contract.feature_artifact_path,
        manifest_path=contract.feature_manifest_path,
        verify_lineage=True,
        expected_manifest_sha256=hashlib.sha256(
            contract.feature_manifest_path.read_bytes()
        ).hexdigest(),
        expected_start=contract.active_window.start,
        expected_end=contract.active_window.end_exclusive,
        expected_symbol=contract.symbol,
        expected_instrument_id=contract.instrument_id,
        expected_rows=contract.active_window.expected_rows,
    )
    return frame, str(MBO_MECHANISM_PROTOCOL_SHA256)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(child) for child in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_bytes(
        path,
        (
            json.dumps(
                _json_value(value),
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(
            _json_value(row),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    atomic_bytes(path, payload.encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise Phase6ResearchError(
                    f"invalid prior ledger JSON at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise Phase6ResearchError("prior ledger row must be an object")
            rows.append(value)
    return rows


def _report(result: Mapping[str, Any]) -> str:
    if result.get("comparison_validation_only") is True:
        evidence_lines = [
            "## Comparison containment",
            "",
            (
                "No mechanism is admitted from this fixed historical "
                "comparison. Statistical support remains diagnostic only."
            ),
            "",
            (
                "Diagnostic support not admitted: `"
                + ", ".join(
                    result.get(
                        "comparison_supported_mechanisms_not_admitted",
                        (),
                    )
                )
                + "`"
            ),
        ]
    else:
        evidence_lines = [
            "## Phase-7 evidence admission",
            "",
            (
                "Only statistically supported mechanisms are admitted; raw "
                "post-event window values remain retrospective and are never "
                "live evidence."
            ),
            "",
            (
                "Allowlist: `"
                + (
                    ", ".join(result["phase7_evidence_allowlist"])
                    or "empty"
                )
                + "`"
            ),
        ]
    lines = [
        "# Phase 6 — MBO mechanism validation",
        "",
        "> Development association only; not causal, OOS, model-fit, or trading authority.",
        "> Display classification: `non_authoritative_derived_display`.",
        "",
        f"- Engineering/data status: `{result['engineering_status']}`",
        f"- Experiment: `{result['experiment_id']}`",
        f"- Study mode: `{result['study_mode']}`",
        f"- Manifest: `{result['manifest_path']}` / `{result['manifest_sha256']}`",
        f"- Raw partition hashes verified: `{result['raw_partition_hashes_verified']}`",
        f"- MBO feature artifact: `{result['feature_artifact']}` / `{result['feature_artifact_sha256']}`",
        f"- Result identity: `{result['result_identity']}`",
        f"- MBO response rows / real OHLCV source rows / registered synthetic no-trade rows: {result['coverage']['feature_rows']:,} / {result['coverage']['ohlcv_real_source_rows']:,} / {result['coverage']['registered_synthetic_mbo_response_rows']:,}",
        f"- Synthetic semantic exception gate allowed/rejected: {result['coverage']['synthetic_semantic_exception_gate']['allowed_count']} / {result['coverage']['synthetic_semantic_exception_gate']['rejected_count']}",
        f"- Synthetic exception clock scopes: `{json.dumps(result['coverage']['synthetic_semantic_exception_gate']['allowed_by_clock_scope'], sort_keys=True)}`",
        f"- Week-2 extension required: `{result['extension_required']}`",
        f"- Registered extension consumed: `{result['registered_extension_consumed']}`; further extension authorized: `{result['further_extension_authorized']}`",
        f"- Displacement monotonicity reporting: `{json.dumps(_json_value(result['displacement_continuous_monotonicity']), sort_keys=True)}`",
        "",
        "| Fixed hypothesis | Status | Matched n | Mean effect | Holm p | Stable |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in PHASE6_FIXED_FAMILY:
        value = result["mechanisms"][name]
        effect = value["primary_effect"]["mean_effect"]
        lines.append(
            f"| {name} | {value['status']} | {value['primary_matched_n']} | "
            f"{('n/a' if effect is None else f'{effect:.6f}')} | "
            f"{value['holm_adjusted_p_value']:.6f} | "
            f"{value['stability_complete'] and not value['systematic_sign_reversal']} |"
        )
    lines.extend(
        [
            "",
            "## FVG pseudo-zone descriptive sensitivity",
            "",
            "This comparison is independently matched and descriptive only. It is excluded from Holm correction and cannot enter the Phase-7 allowlist.",
            "",
            (
                f"- Packed n: {result['fvg_pseudo_zone_descriptive_sensitivity']['paired_n']}; "
                f"mean effect: {result['fvg_pseudo_zone_descriptive_sensitivity']['mean_effect']}; "
                f"descriptive CI: [{result['fvg_pseudo_zone_descriptive_sensitivity']['descriptive_bootstrap_ci_low']}, "
                f"{result['fvg_pseudo_zone_descriptive_sensitivity']['descriptive_bootstrap_ci_high']}]"
            ),
            "",
            "## Compact audit ledgers",
            "",
            *[
                f"- {name}: `{value['path']}` — {value['rows']:,} rows — `{value['sha256']}`"
                for name, value in result["ledgers"].items()
            ],
            "",
            *evidence_lines,
            "",
            "The displayed-defense net-add metric is an all-book A−C−passive-F proxy. It is not proof of same-level queue replenishment or absorption.",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in result["limitations"]],
        ]
    )
    return "\n".join(lines)


def run(
    *,
    manifest_path: Path = MANIFEST_PATH,
    output: Path = DEFAULT_OUTPUT,
    verify_raw_partition_hashes: bool = True,
    comparison_validation_only: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    if type(comparison_validation_only) is not bool:
        raise Phase6ResearchError("comparison_validation_only must be boolean")
    if verify_raw_partition_hashes is not True:
        raise Phase6ResearchError(
            "formal Phase-6 research requires raw partition hash verification"
        )
    output = output.resolve()
    try:
        output.relative_to(ROOT)
    except ValueError as error:
        raise Phase6ResearchError("Phase-6 output must remain inside repository") from error
    paths = {
        output,
        output.with_suffix(".md"),
        output.with_name(f"{output.stem}.episodes.jsonl"),
        output.with_name(f"{output.stem}.matched_pairs.jsonl"),
        output.with_name(f"{output.stem}.unmatched.jsonl"),
    }
    if any(path.exists() for path in paths):
        raise FileExistsError("Phase-6 result or ledger already exists")
    contract = load_frozen_phase6_contract(
        manifest_path,
        root=ROOT,
        verify_raw_partition_hashes=verify_raw_partition_hashes,
        comparison_validation_only=comparison_validation_only,
    )
    manifest = contract.payload
    feature_raw, feature_protocol_sha = _load_feature_artifact(contract)
    features = validate_minute_feature_frame(
        feature_raw,
        start=contract.active_window.start,
        end_exclusive=contract.active_window.end_exclusive,
        symbol=contract.symbol,
        instrument_id=contract.instrument_id,
        expected_rows=contract.active_window.expected_rows,
    )
    feature_clocks = frozenset(features["decision_time"])
    registered_synthetic_clocks = {
        pd.Timestamp(value).tz_convert("UTC")
        for value in manifest["reader_census_contract"][
            "synthetic_decision_clocks"
        ]
    }
    validate_registered_synthetic_mbo_flow(
        features,
        tuple(sorted(registered_synthetic_clocks)),
    )
    loaded = load_ohlcv(
        contract.ohlcv_path,
        start=contract.warmup_start,
        end=contract.active_window.end_exclusive,
    )
    if loaded.warnings or not loaded.contract_selection_causal:
        raise Phase6ResearchError("Phase-6 requires the causal processed OHLCV front")
    ohlcv_decision_clocks = {
        pd.Timestamp(clock).tz_convert("UTC") + pd.Timedelta(minutes=1)
        for clock, row in loaded.frame.iterrows()
        if contract.active_window.start
        <= pd.Timestamp(clock).tz_convert("UTC") + pd.Timedelta(minutes=1)
        < contract.active_window.end_exclusive
        and str(row["symbol"]) == contract.symbol
        and int(row["instrument_id"]) == contract.instrument_id
    }
    expected_real_source_clocks = set(feature_clocks) - registered_synthetic_clocks
    if (
        ohlcv_decision_clocks != expected_real_source_clocks
        or registered_synthetic_clocks.intersection(ohlcv_decision_clocks)
    ):
        raise Phase6ResearchError(
            "OHLCV real completed-clock preflight disagrees with the registered "
            "synthetic no-trade census"
        )

    eye = _collect_eye_inputs(
        loaded=loaded,
        feature_clocks=feature_clocks,
        contract=contract,
        tick_size=float(_json(contract.model_path)["tick_size"]),
    )
    raw_targets = eye["targets"]
    displacement_entity_first = first_active_displacement_episodes(
        raw_targets["displacement_impact"]
    )
    fvg_entity_first = first_fvg_lifecycle_episodes(
        raw_targets["fvg_retest_response"]
    )
    successful_fvg_entity_first = [
        item
        for item in fvg_entity_first
        if item.event_variant == "successful_retest"
    ]
    failed_fvg_entity_first = [
        item
        for item in fvg_entity_first
        if item.event_variant == "failed_retest"
    ]
    pseudo_fvg_raw, pseudo_exclusions = _pseudo_fvg_controls(
        eye["fvg_creations"],
        eye["m5_bases"],
        maximum_future_m5_bars=int(
            manifest["hypothesis_design"]["fvg_retest_response"][
                "pseudo_zone_protocol"
            ]["maximum_future_m5_bars"]
        ),
    )

    displacement_controls_raw = [
        _candidate_episode(
            base,
            hypothesis="displacement_impact",
            direction=str(base["candle_direction"]),
            candidate_id=f"non_displacement:{base['known_at'].isoformat()}",
            event_kind="completed_m5_clock",
            control_kind="non_active_displacement_clock",
        )
        for base in eye["m5_bases"]
        if base["known_at"] not in eye["active_displacement_clocks"]
    ]
    unit_exclusions: list[dict[str, Any]] = []
    selected_displacement_ids = {
        item.episode_id for item in displacement_entity_first
    }
    unit_exclusions.extend(
        {
            "hypothesis": "displacement_impact",
            "role": "treatment",
            "episode_id": item.episode_id,
            "entity_id": item.entity_id,
            "known_at": item.known_at,
            "reason": "later_active_clock_same_displacement_entity_not_statistical_unit",
        }
        for item in raw_targets["displacement_impact"]
        if item.episode_id not in selected_displacement_ids
    )
    selected_fvg_ids = {item.episode_id for item in fvg_entity_first}
    unit_exclusions.extend(
        {
            "hypothesis": "fvg_retest_response",
            "role": "lifecycle",
            "episode_id": item.episode_id,
            "entity_id": item.entity_id,
            "known_at": item.known_at,
            "reason": "later_fvg_lifecycle_transition_not_statistical_unit",
        }
        for item in raw_targets["fvg_retest_response"]
        if item.episode_id not in selected_fvg_ids
    )

    def canonical(
        episodes: Sequence[MechanismEpisode],
        *,
        hypothesis: str,
        control_kind: str | None,
        role: str,
    ) -> tuple[MechanismEpisode, ...]:
        projected, exclusions = canonicalize_mechanism_episodes(
            episodes,
            analysis_hypothesis=hypothesis,
            control_kind=control_kind,
        )
        unit_exclusions.extend({**item, "role": role} for item in exclusions)
        return projected

    treatment_population: dict[str, Sequence[MechanismEpisode]] = {
        "sweep_rejection": canonical(
            raw_targets["sweep_rejection"],
            hypothesis="sweep_rejection",
            control_kind=None,
            role="treatment",
        ),
        "acceptance_continuation": canonical(
            raw_targets["acceptance_continuation"],
            hypothesis="acceptance_continuation",
            control_kind=None,
            role="treatment",
        ),
        "displacement_impact": canonical(
            displacement_entity_first,
            hypothesis="displacement_impact",
            control_kind=None,
            role="treatment",
        ),
        "mss_flow_shift": canonical(
            raw_targets["mss_flow_shift"],
            hypothesis="mss_flow_shift",
            control_kind=None,
            role="treatment",
        ),
        "fvg_retest_response": canonical(
            successful_fvg_entity_first,
            hypothesis="fvg_retest_response",
            control_kind=None,
            role="treatment",
        ),
    }
    control_population: dict[str, Sequence[MechanismEpisode]] = {
        "sweep_rejection": canonical(
            raw_targets["acceptance_continuation"],
            hypothesis="sweep_rejection",
            control_kind="acceptance_crossing_primary_control",
            role="primary_control",
        ),
        "acceptance_continuation": canonical(
            raw_targets["sweep_rejection"],
            hypothesis="acceptance_continuation",
            control_kind="sweep_crossing_primary_control",
            role="primary_control",
        ),
        "displacement_impact": canonical(
            displacement_controls_raw,
            hypothesis="displacement_impact",
            control_kind="non_active_displacement_clock",
            role="primary_control",
        ),
        "mss_flow_shift": canonical(
            eye["raw_break_controls"],
            hypothesis="mss_flow_shift",
            control_kind="raw_break_without_mss",
            role="primary_control",
        ),
        # The sole primary FVG control is a failed first retest. Pseudo zones
        # are matched below as a separate descriptive sensitivity population.
        "fvg_retest_response": canonical(
            failed_fvg_entity_first,
            hypothesis="fvg_retest_response",
            control_kind="failed_retest_primary_control",
            role="primary_control",
        ),
    }
    pseudo_fvg = canonical(
        pseudo_fvg_raw,
        hypothesis="fvg_retest_response",
        control_kind="pseudo_zone_retest",
        role="descriptive_sensitivity_control",
    )

    all_episodes: dict[str, MechanismEpisode] = {}
    windows: dict[str, Any] = {}
    window_exclusions: list[dict[str, Any]] = []
    for role, populations in (
        ("treatment", treatment_population),
        ("control", control_population),
        (
            "descriptive_sensitivity_control",
            {
                **{name: () for name in PHASE6_FIXED_FAMILY},
                "fvg_retest_response": pseudo_fvg,
            },
        ),
    ):
        for hypothesis in PHASE6_FIXED_FAMILY:
            for episode in populations[hypothesis]:
                prior = all_episodes.get(episode.episode_id)
                if prior is not None and prior != episode:
                    raise Phase6ResearchError("episode identity collision")
                all_episodes[episode.episode_id] = episode
                if episode.episode_id in windows:
                    continue
                try:
                    windows[episode.episode_id] = aggregate_episode_feature_window(
                        features,
                        episode,
                    )
                except EpisodeWindowUnavailable as error:
                    window_exclusions.append(
                        {
                            "hypothesis": hypothesis,
                            "role": role,
                            "episode_id": episode.episode_id,
                            "reason": error.reason,
                        }
                    )

    matching = manifest["matching"]
    completed_index = {
        clock: index for index, clock in enumerate(features["decision_time"])
    }
    prior_episode_ledger: list[dict[str, Any]] = []
    prior_pair_ledger: list[dict[str, Any]] = []
    prior_unmatched_ledger: list[dict[str, Any]] = []
    if (
        contract.prior_week1_ledger_paths is not None
        and not comparison_validation_only
    ):
        prior_episode_ledger = _read_jsonl(
            contract.prior_week1_ledger_paths["episodes"]
        )
        prior_pair_ledger = _read_jsonl(
            contract.prior_week1_ledger_paths["matched_pairs"]
        )
        prior_unmatched_ledger = _read_jsonl(
            contract.prior_week1_ledger_paths["unmatched"]
        )
    paired_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in PHASE6_FIXED_FAMILY
    }
    pseudo_sensitivity_rows: list[dict[str, Any]] = []
    for item in prior_pair_ledger:
        hypothesis = item.get("hypothesis")
        if hypothesis not in PHASE6_FIXED_FAMILY:
            raise Phase6ResearchError("prior pair ledger has an unknown hypothesis")
        if (
            item.get("treatment_study_week") != "week_1"
            or item.get("control_study_week") != "week_1"
            or item.get("post_features_retrospective_only") is not True
            or not isinstance(item.get("treatment_metrics"), Mapping)
            or not isinstance(item.get("control_metrics"), Mapping)
            or item.get("treatment_match_context_source")
            != "cached_real_completed_M1_at_strict_prior_formation_clock"
            or item.get("control_match_context_source")
            != "cached_real_completed_M1_at_strict_prior_formation_clock"
        ):
            raise Phase6ResearchError("prior pair ledger violates the week-1 contract")
        comparison = item.get("comparison")
        if comparison == PRIMARY_COMPARISON:
            if (
                item.get("holm_included") is not True
                or item.get("phase7_evidence_candidate") is not True
            ):
                raise Phase6ResearchError(
                    "prior primary pair is outside the fixed-Holm contract"
                )
            paired_rows[str(hypothesis)].append(item)
        elif comparison == FVG_PSEUDO_SENSITIVITY_COMPARISON:
            if (
                hypothesis != "fvg_retest_response"
                or item.get("holm_included") is not False
                or item.get("phase7_evidence_candidate") is not False
            ):
                raise Phase6ResearchError(
                    "prior pseudo-zone pair entered primary inference"
                )
            pseudo_sensitivity_rows.append(item)
        else:
            raise Phase6ResearchError("prior pair comparison is not registered")
    pair_ledger: list[dict[str, Any]] = list(prior_pair_ledger)
    unmatched_ledger: list[dict[str, Any]] = [
        *prior_unmatched_ledger,
        *eye["context_exclusions"],
        *(
            {
                **item,
                "hypothesis": "displacement_impact",
                "role": "registered_synthetic_terminal_censor",
                "comparison": "excluded_from_all_analysis_samples",
                "reason": (
                    "registered_synthetic_terminal_censor_not_a_statistical_unit"
                ),
            }
            for item in eye["synthetic_semantic_exception_audit"]["allowed"]
            if item.get("clock_scope") == ACTIVE_SYNTHETIC_CLOCK_SCOPE
        ),
        *(
            {
                **item,
                "hypothesis": "synthetic_semantic_gate",
                "role": "rejected",
                "comparison": "fail_closed_before_analysis",
            }
            for item in eye["synthetic_semantic_exception_audit"]["rejected"]
        ),
        *unit_exclusions,
        *window_exclusions,
        *(
            {
                **item,
                "hypothesis": "fvg_retest_response",
                "role": "descriptive_sensitivity_control",
                "comparison": FVG_PSEUDO_SENSITIVITY_COMPARISON,
            }
            for item in pseudo_exclusions
        ),
    ]
    match_coverage: dict[str, Any] = {}

    def pair_row(
        *,
        hypothesis: str,
        pair: Any,
        comparison: str,
        holm_included: bool,
    ) -> dict[str, Any]:
        treatment_episode = all_episodes[pair.treatment_id]
        control_episode = all_episodes[pair.candidate_id]
        treatment_window = windows[pair.treatment_id]
        control_window = windows[pair.candidate_id]

        def source_hash(episode: MechanismEpisode) -> str:
            return hashlib.sha256(
                json.dumps(
                    list(episode.source_m5_bar_event_ids),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()

        return {
            "comparison": comparison,
            "holm_included": holm_included,
            "phase7_evidence_candidate": holm_included,
            "hypothesis": hypothesis,
            "treatment_episode_id": pair.treatment_id,
            "control_episode_id": pair.candidate_id,
            "treatment_metrics": dict(treatment_window.metrics),
            "control_metrics": dict(control_window.metrics),
            "treatment_half_week": treatment_episode.half_week,
            "control_half_week": control_episode.half_week,
            "treatment_study_week": treatment_episode.match_fields["study_week"],
            "control_study_week": control_episode.match_fields["study_week"],
            "treatment_session_phase": treatment_episode.session_phase,
            "control_session_phase": control_episode.session_phase,
            "completed_minute_offset": pair.completed_bar_offset,
            "treatment_known_at": treatment_episode.known_at,
            "control_known_at": control_episode.known_at,
            "treatment_match_context_clock": treatment_episode.match_context_clock,
            "control_match_context_clock": control_episode.match_context_clock,
            "treatment_match_context_source": treatment_episode.match_context_source,
            "control_match_context_source": control_episode.match_context_source,
            "treatment_source_bar_clocks": treatment_episode.source_bar_clocks,
            "control_source_bar_clocks": control_episode.source_bar_clocks,
            "treatment_source_m5_bar_event_ids": (
                treatment_episode.source_m5_bar_event_ids
            ),
            "control_source_m5_bar_event_ids": (
                control_episode.source_m5_bar_event_ids
            ),
            "treatment_source_m5_bar_count": len(
                treatment_episode.source_m5_bar_event_ids
            ),
            "control_source_m5_bar_count": len(
                control_episode.source_m5_bar_event_ids
            ),
            "treatment_source_m5_bar_ids_sha256": source_hash(treatment_episode),
            "control_source_m5_bar_ids_sha256": source_hash(control_episode),
            "source_m5_lineage_definition": (
                treatment_episode.source_m5_lineage_definition
            ),
            "source_m5_bar_event_id_order": "sorted_by_clock_then_event_id",
            "treatment_constituent_event_ids": (
                treatment_episode.constituent_event_ids
            ),
            "control_constituent_event_ids": control_episode.constituent_event_ids,
            "treatment_formation_feature_clocks": treatment_window.formation_clocks,
            "control_formation_feature_clocks": control_window.formation_clocks,
            "treatment_post_feature_clocks": treatment_window.post_clocks,
            "control_post_feature_clocks": control_window.post_clocks,
            "treatment_inference_feature_clocks": tuple(
                sorted(
                    set(
                        (*treatment_window.formation_clocks, *treatment_window.post_clocks)
                    )
                )
            ),
            "control_inference_feature_clocks": tuple(
                sorted(
                    set((*control_window.formation_clocks, *control_window.post_clocks))
                )
            ),
            "post_features_retrospective_only": True,
        }

    for hypothesis in PHASE6_FIXED_FAMILY:
        treatments = [
            item
            for item in treatment_population[hypothesis]
            if item.episode_id in windows
        ]
        controls = [
            item
            for item in control_population[hypothesis]
            if item.episode_id in windows
        ]
        match = pack_nonoverlapping_mechanism_controls(
            treatments,
            controls,
            windows=windows,
            completed_index=completed_index,
            exact_fields=tuple(matching["exact_fields"]),
            maximum_completed_minute_offset=int(
                matching["maximum_completed_minute_offset"]
            ),
            embargo_minutes=int(matching["embargo_minutes"]),
        )
        for treatment_id, reason in match.unmatched.items():
            unmatched_ledger.append(
                {
                    "hypothesis": hypothesis,
                    "role": "treatment",
                    "comparison": PRIMARY_COMPARISON,
                    "holm_included": True,
                    "episode_id": treatment_id,
                    "reason": reason,
                }
            )
        for pair in match.pairs:
            row = pair_row(
                hypothesis=hypothesis,
                pair=pair,
                comparison=PRIMARY_COMPARISON,
                holm_included=True,
            )
            paired_rows[hypothesis].append(row)
            pair_ledger.append(row)
        match_coverage[hypothesis] = {
            "canonical_requested": len(treatment_population[hypothesis]),
            "feature_window_eligible": match.requested,
            "edge_window_eligible": match.window_eligible,
            "eligible_controls": match.eligible_candidates,
            "packed": match.matched,
            "overlap_exclusions": match.overlap_exclusions,
            "packed_coverage": (
                0.0 if match.requested == 0 else match.matched / match.requested
            ),
        }

    sensitivity_treatments = [
        item
        for item in treatment_population["fvg_retest_response"]
        if item.episode_id in windows
    ]
    sensitivity_controls = [
        item for item in pseudo_fvg if item.episode_id in windows
    ]
    sensitivity_match = pack_nonoverlapping_mechanism_controls(
        sensitivity_treatments,
        sensitivity_controls,
        windows=windows,
        completed_index=completed_index,
        exact_fields=tuple(matching["exact_fields"]),
        maximum_completed_minute_offset=int(
            matching["maximum_completed_minute_offset"]
        ),
        embargo_minutes=int(matching["embargo_minutes"]),
    )
    for treatment_id, reason in sensitivity_match.unmatched.items():
        unmatched_ledger.append(
            {
                "hypothesis": "fvg_retest_response",
                "role": "treatment",
                "comparison": FVG_PSEUDO_SENSITIVITY_COMPARISON,
                "holm_included": False,
                "episode_id": treatment_id,
                "reason": reason,
            }
        )
    for pair in sensitivity_match.pairs:
        row = pair_row(
            hypothesis="fvg_retest_response",
            pair=pair,
            comparison=FVG_PSEUDO_SENSITIVITY_COMPARISON,
            holm_included=False,
        )
        pseudo_sensitivity_rows.append(row)
        pair_ledger.append(row)
    sensitivity_coverage = {
        "comparison": FVG_PSEUDO_SENSITIVITY_COMPARISON,
        "canonical_requested": len(
            treatment_population["fvg_retest_response"]
        ),
        "feature_window_eligible": sensitivity_match.requested,
        "edge_window_eligible": sensitivity_match.window_eligible,
        "eligible_controls": sensitivity_match.eligible_candidates,
        "packed": sensitivity_match.matched,
        "overlap_exclusions": sensitivity_match.overlap_exclusions,
        "packed_coverage": (
            0.0
            if sensitivity_match.requested == 0
            else sensitivity_match.matched / sensitivity_match.requested
        ),
        "holm_included": False,
        "phase7_evidence_admission": False,
    }

    evaluated = evaluate_fixed_mechanism_family(
        paired_rows,
        minimum_matched=int(manifest["minimum_matched_episodes"]),
        alpha=float(manifest["inference"]["alpha"]),
        bootstrap_replicates=int(manifest["inference"]["bootstrap_replicates"]),
        bootstrap_seed=int(manifest["inference"]["bootstrap_seed"]),
        stability_stratum_minimum_n=int(
            manifest["support_rule"]["stability_stratum_minimum_n"]
        ),
    )
    if comparison_validation_only:
        observed_support = tuple(evaluated["phase7_evidence_allowlist"])
        evaluated = {
            **evaluated,
            "phase7_evidence_allowlist": [],
            "phase7_excluded_mechanisms": list(PHASE6_FIXED_FAMILY),
            "comparison_supported_mechanisms_not_admitted": list(
                observed_support
            ),
        }
    fvg_pseudo_sensitivity = evaluate_descriptive_sensitivity(
        pseudo_sensitivity_rows,
        hypothesis="fvg_retest_response",
        comparison=FVG_PSEUDO_SENSITIVITY_COMPARISON,
        bootstrap_replicates=int(manifest["inference"]["bootstrap_replicates"]),
        bootstrap_seed=int(manifest["inference"]["bootstrap_seed"]) + 900,
    )
    displacement_metric = PRIMARY_METRIC_BY_HYPOTHESIS["displacement_impact"]
    current_displacement_monotonicity = spearman_monotonicity(
        [item.score for item in treatment_population["displacement_impact"] if item.episode_id in windows],
        [
            windows[item.episode_id].metrics.get(displacement_metric)
            for item in treatment_population["displacement_impact"]
            if item.episode_id in windows
        ],
    )
    displacement_monotonicity = _scoped_displacement_monotonicity(
        current_displacement_monotonicity,
        prior_week1_result=(
            None if comparison_validation_only else contract.prior_week1_result
        ),
        registered_policy=manifest.get(
            "extension_displacement_monotonicity_policy"
        ),
    )

    def episode_ledger_row(item: MechanismEpisode) -> dict[str, Any]:
        source_ids_hash = hashlib.sha256(
            json.dumps(
                list(item.source_m5_bar_event_ids),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        window = windows.get(item.episode_id)
        return {
            "episode_id": item.episode_id,
            "hypothesis": item.hypothesis,
            "event_kind": item.event_kind,
            "event_variant": item.event_variant,
            "control_kind": item.control_kind,
            "entity_id": item.entity_id,
            "event_time": item.event_time,
            "known_at": item.known_at,
            "source_bar_clocks": item.source_bar_clocks,
            "source_m5_bar_event_ids": item.source_m5_bar_event_ids,
            "source_m5_bar_count": len(item.source_m5_bar_event_ids),
            "source_m5_bar_ids_sha256": source_ids_hash,
            "source_m5_bar_event_id_order": "sorted_by_clock_then_event_id",
            "source_m5_lineage_definition": item.source_m5_lineage_definition,
            "episode_local_formation_clocks": (
                None if window is None else window.formation_clocks
            ),
            "symbol": item.symbol,
            "instrument_id": item.instrument_id,
            "timeframe": item.timeframe,
            "direction": item.direction,
            "session_phase": item.session_phase,
            "half_week": item.half_week,
            "match_fields": item.match_fields,
            "match_context_clock": item.match_context_clock,
            "match_context_source": item.match_context_source,
            "score": item.score,
            "statistical_unit": item.statistical_unit,
            "constituent_event_ids": item.constituent_event_ids,
            "constituent_event_count": len(item.constituent_event_ids),
            "constituent_entity_ids": item.constituent_entity_ids,
            "constituent_entity_count": len(item.constituent_entity_ids),
            "displacement_first_active_entity_unit": (
                item.hypothesis == "displacement_impact"
                and item.control_kind is None
            ),
            "feature_window_status": (
                "included" if window is not None else "censored"
            ),
            "formation_available_at": (
                None if window is None else window.formation_available_at
            ),
            "post_available_at": (
                None if window is None else window.post_available_at
            ),
            "post_features_retrospective_only": True,
        }

    current_episode_ledger = [
        episode_ledger_row(item)
        for item in sorted(
            all_episodes.values(),
            key=lambda value: (value.known_at, value.episode_id),
        )
    ]
    episode_ledger = [*prior_episode_ledger, *current_episode_ledger]
    episode_path = output.with_name(f"{output.stem}.episodes.jsonl")
    pair_path = output.with_name(f"{output.stem}.matched_pairs.jsonl")
    unmatched_path = output.with_name(f"{output.stem}.unmatched.jsonl")
    _write_jsonl(episode_path, episode_ledger)
    _write_jsonl(pair_path, pair_ledger)
    _write_jsonl(unmatched_path, unmatched_ledger)

    result = {
        **evaluated,
        "status": (
            "phase6_foundation_v2_comparison_complete_no_admission"
            if comparison_validation_only
            else "phase6_engineering_complete_mechanism_support_reported_separately"
        ),
        "study_mode": manifest["study_mode"],
        "experiment_id": manifest["experiment_id"],
        "semantic_version": manifest["semantic_version"],
        "manifest_path": str(contract.manifest_path.relative_to(ROOT)),
        "manifest_sha256": contract.manifest_sha256,
        "raw_partition_hashes_verified": True,
        "feature_artifact": str(contract.feature_artifact_path.relative_to(ROOT)),
        "feature_artifact_sha256": hashlib.sha256(
            contract.feature_artifact_path.read_bytes()
        ).hexdigest(),
        "feature_manifest": str(contract.feature_manifest_path.relative_to(ROOT)),
        "feature_protocol_sha256": feature_protocol_sha,
        "active_window": {
            "id": contract.active_window.window_id,
            "start": contract.active_window.start,
            "end_exclusive": contract.active_window.end_exclusive,
        },
        "week1_gate_result": (
            None
            if contract.prior_week1_result_path is None
            else {
                "path": str(contract.prior_week1_result_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(
                    contract.prior_week1_result_path.read_bytes()
                ).hexdigest(),
                "result_identity": (contract.prior_week1_result or {}).get(
                    "result_identity"
                ),
            }
        ),
        "registered_extension_consumed": (
            manifest["study_mode"]
            == "primary_plus_registered_underpowered_extension"
        ),
        "further_extension_authorized": False,
        **(
            {
                "comparison_contract": dict(manifest["comparison_contract"]),
                "comparison_validation_only": True,
            }
            if comparison_validation_only
            else {}
        ),
        "fvg_pseudo_zone_descriptive_sensitivity": fvg_pseudo_sensitivity,
        "coverage": {
            "feature_rows": len(features),
            "ohlcv_real_source_rows": len(ohlcv_decision_clocks),
            "registered_synthetic_mbo_response_rows": len(
                registered_synthetic_clocks
            ),
            "eye_replayed_bars_including_warmup": eye["emitted_bars"],
            "reader_active_census": eye["reader_census"],
            "synthetic_semantic_exception_gate": {
                "policy": SYNTHETIC_SEMANTIC_EXCEPTION_POLICY,
                "extension_warmup_policy": (
                    EXTENSION_WARMUP_SYNTHETIC_EXCEPTION_POLICY
                ),
                "allowed_count": len(
                    eye["synthetic_semantic_exception_audit"]["allowed"]
                ),
                "rejected_count": len(
                    eye["synthetic_semantic_exception_audit"]["rejected"]
                ),
                "blocked_lineage_event_ids": eye[
                    "synthetic_semantic_exception_audit"
                ]["blocked_lineage_event_ids"],
                "blocked_lineage_event_count": len(
                    eye["synthetic_semantic_exception_audit"][
                        "blocked_lineage_event_ids"
                    ]
                ),
                "descendant_sample_exclusion_count": sum(
                    item.get("reason")
                    == "synthetic_semantic_exception_descendant_excluded_from_samples"
                    for item in eye["context_exclusions"]
                ),
                "allowed": eye["synthetic_semantic_exception_audit"]["allowed"],
                "rejected": eye["synthetic_semantic_exception_audit"]["rejected"],
                "allowed_by_clock_scope": eye[
                    "synthetic_semantic_exception_audit"
                ]["allowed_by_clock_scope"],
                "prior_warmup_registration": eye[
                    "synthetic_semantic_exception_audit"
                ]["prior_warmup_registration"],
            },
            "semantic_event_ids_seen": eye["semantic_event_ids_seen"],
            "episode_rows": len(episode_ledger),
            "matched_pair_rows": len(pair_ledger),
            "primary_fixed_holm_pair_rows": sum(
                len(value) for value in paired_rows.values()
            ),
            "pseudo_zone_descriptive_pair_rows": len(pseudo_sensitivity_rows),
            "prior_week1_episode_rows": len(prior_episode_ledger),
            "prior_week1_matched_pair_rows": len(prior_pair_ledger),
            "current_week_episode_rows": len(current_episode_ledger),
            "current_week_matched_pair_rows": len(pair_ledger) - len(prior_pair_ledger),
            "window_censor_reasons": dict(
                sorted(Counter(item["reason"] for item in window_exclusions).items())
            ),
            "strict_prior_context_censor_reasons": (
                _strict_prior_context_censor_counts(
                    eye["context_exclusions"]
                )
            ),
            "primary_matching": match_coverage,
            "fvg_pseudo_zone_descriptive_matching": sensitivity_coverage,
            "fvg_variants": {
                "successful_entity_first_retests": len(
                    successful_fvg_entity_first
                ),
                "failed_entity_first_retests": len(failed_fvg_entity_first),
                "successful_canonical_clock_units": len(
                    treatment_population["fvg_retest_response"]
                ),
                "failed_primary_control_canonical_clock_units": len(
                    control_population["fvg_retest_response"]
                ),
                "pseudo_zone_canonical_controls": len(pseudo_fvg),
                "primary_comparison": "successful_retest_vs_failed_retest",
                "pseudo_zone_comparison": (
                    "descriptive_sensitivity_only_excluded_from_holm_and_phase7"
                ),
                "pooling": "forbidden",
            },
            "displacement_units": {
                "active_semantic_events": len(
                    raw_targets["displacement_impact"]
                ),
                "first_active_entity_units": len(displacement_entity_first),
                "canonical_clock_units": len(
                    treatment_population["displacement_impact"]
                ),
                "all_active_clocks_excluded_from_controls": len(
                    eye["active_displacement_clocks"]
                ),
            },
        },
        "displacement_continuous_monotonicity": displacement_monotonicity,
        "ledgers": {
            "episodes": {
                "path": str(episode_path.relative_to(ROOT)),
                "rows": len(episode_ledger),
                "sha256": hashlib.sha256(episode_path.read_bytes()).hexdigest(),
            },
            "matched_pairs": {
                "path": str(pair_path.relative_to(ROOT)),
                "rows": len(pair_ledger),
                "sha256": hashlib.sha256(pair_path.read_bytes()).hexdigest(),
            },
            "unmatched": {
                "path": str(unmatched_path.relative_to(ROOT)),
                "rows": len(unmatched_ledger),
                "sha256": hashlib.sha256(unmatched_path.read_bytes()).hexdigest(),
            },
        },
        "limitations": [
            "development association only; no causal claim",
            *(
                [
                    "fixed Foundation v2 historical comparison only; no OOF, sealed OOS, model admission, or trading authority",
                    "the W2 prior result authorizes the registered historical window and warmup audit only; prior W1 pairs are excluded from W2 comparison inference",
                ]
                if comparison_validation_only
                else []
            ),
            "post-event windows are retrospective mechanism validation and are not live evidence",
            "registered OHLCV synthetic no-trade decision clocks have real MBO/BBO rows with zero trade/fill flow and may appear in retrospective MBO response windows; they cannot become M5 sources/control bases or emit any sample-eligible semantic event",
            "the sole registered synthetic-context semantic exception is a displacement CENSORED/synthetic_interruption terminal; every exact clock-only M1 root in its M5 constituent interval is context-only, the terminal may settle on the later M5 boundary, and it plus all descendants are excluded from every analysis sample",
            "FVG pseudo-zone results are descriptive sensitivity only and are excluded from the five-test Holm family and Phase-7 evidence admission",
            "primary inference uses deterministic earliest-first hypothesis-local packing; formation/post MBO clocks are not reused within a hypothesis and the packing is not maximum-cardinality",
            "displayed-defense net-add is an all-book A/C/F proxy, not same-level queue replenishment or absorption",
            "unsupported and underpowered mechanisms are excluded from Phase 7",
            "in an extension result displacement score monotonicity is reported separately for hash-bound week 1 and current week 2; no combined value is reconstructed from compact ledgers",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    result["result_identity"] = canonical_identity(result)
    _write_json(output, result)
    atomic_bytes(output.with_suffix(".md"), (_report(result) + "\n").encode("utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(
        manifest_path=args.manifest.resolve(),
        output=args.output.resolve(),
        verify_raw_partition_hashes=True,
    )
    print(
        json.dumps(
            {
                "engineering_status": result["engineering_status"],
                "extension_required": result["extension_required"],
                "phase7_evidence_allowlist": result["phase7_evidence_allowlist"],
                "result_identity": result["result_identity"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
