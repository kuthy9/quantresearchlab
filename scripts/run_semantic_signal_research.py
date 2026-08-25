#!/usr/bin/env python3
"""Run the frozen January-2024 structural Signal Research diagnostic.

This runner constructs only the production causal Reader and Eye.  It does
not construct the Brain, Decision, Risk, execution simulation, MBO, or P&L.
January 2024 is registered as a calibration-row diagnostic window and is
therefore never allowed to fit or authorize a model artifact.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import gc
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.artifact_stream import atomic_bytes  # noqa: E402
from smc_trader.causal import CausalMarketReader  # noqa: E402
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.market_state import MarketSnapshotAuthority  # noqa: E402
from smc_trader.model import (  # noqa: E402
    Direction,
    EventOrigin,
    EventKind,
    MarketEvent,
    Timeframe,
    to_primitive,
)
from smc_trader.observation import CausalObserver, ObserverConfig  # noqa: E402
from smc_trader.scene_graph import parse_scale_specs  # noqa: E402
from smc_trader.semantics import (  # noqa: E402
    SemanticRegistry,
    load_semantic_selection,
)
from smc_trader.signal_research import (  # noqa: E402
    ControlDirectionPolicy,
    MatchResult,
    MatchSpec,
    PseudoLevelSpec,
    ResearchContractError,
    ResearchLinkMode,
    TimeShiftSpec,
    TypedLinkSpec,
    build_forward_time_shift_controls,
    canonical_treatment_episodes,
    canonical_result_identity,
    construct_pseudo_levels,
    deterministic_maximum_cardinality_match,
    exact_mcnemar,
    find_prior_typed_link,
    find_prior_source_link,
    holm_adjust_fixed_family,
    load_frozen_research_contract,
    resolve_lineage_tokens,
    resolve_source_lineage_tokens,
    validate_split_authority,
)
from smc_trader.structural_outcome import (  # noqa: E402
    OutcomeBar,
    OutcomeTerminal,
    StructuralOutcomeEngine,
    StructuralOutcomeSpec,
)
from smc_trader.validation import load_validation_protocol  # noqa: E402


MANIFEST_PATH = ROOT / "experiments/manifests/" "semantic_event_study_v3_template.yaml"
DEFAULT_OUTPUT = ROOT / "experiments/results/" "smc_semantic_v3_signal_diagnostic.json"
MODEL_PATH = ROOT / "configs/model.json"
# Production-emitted Phase-2/3 semantic atoms only. Normalized BAR/reset
# infrastructure, state projections, compatibility transports, aliases, and
# reserved-but-unemitted lifecycle kinds are deliberately outside this study.
ATOMIC_KINDS = frozenset(
    {
        EventKind.SWING_CONFIRMED,
        EventKind.STRUCTURAL_LEG_CREATED,
        EventKind.LIQUIDITY_LEVEL_CREATED,
        EventKind.LEVEL_TOUCHED,
        EventKind.LEVEL_PENETRATED,
        EventKind.SWEEP_CONFIRMED,
        EventKind.ACCEPTANCE_CONFIRMED,
        EventKind.DISPLACEMENT_OBSERVED,
        EventKind.RAW_BOUNDARY_BREAK,
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        EventKind.QUALIFIED_BOS,
        EventKind.PROTECTED_SWING_ASSIGNED,
        EventKind.MSS_CORE_CONFIRMED,
        EventKind.FVG_CREATED,
        EventKind.FVG_PARTIALLY_FILLED,
        EventKind.FVG_MIDPOINT_TOUCHED,
        EventKind.FVG_FULLY_FILLED,
        EventKind.FVG_INVALIDATED,
        EventKind.DEALING_RANGE_CREATED,
        EventKind.DEALING_RANGE_ACTIVATED,
        EventKind.DEALING_RANGE_INVALIDATED,
        EventKind.DEALING_RANGE_REPLACED,
        EventKind.ORIGIN_ZONE_CREATED,
        EventKind.ORIGIN_ZONE_MITIGATED,
        EventKind.ORIGIN_ZONE_INVALIDATED,
    }
)
REGISTERED_ATOMIC_POPULATION = frozenset(kind.value for kind in ATOMIC_KINDS)
RESEARCH_SELECTION = {
    EventKind.DISPLACEMENT_OBSERVED.value: {
        "predicate": "evidence.lifecycle == 'active'",
    }
}
RANGE_INVALIDATION_VARIANT_CONTRACT = {
    "forming_close_before_activation": {
        "transition_reason": "close_beyond_frozen_range_before_activation",
        "required_source_kinds": [
            EventKind.DEALING_RANGE_CREATED.value,
            EventKind.BAR_COMPLETED.value,
        ],
    },
    "forming_source_invalidated": {
        "transition_reason": "forming_source_invalidated",
        "required_source_kinds": [
            EventKind.DEALING_RANGE_CREATED.value,
            EventKind.BAR_COMPLETED.value,
        ],
    },
    "forming_maturity_deadline_elapsed": {
        "transition_reason": "maturity_deadline_elapsed",
        "required_source_kinds": [
            EventKind.DEALING_RANGE_CREATED.value,
            EventKind.BAR_COMPLETED.value,
        ],
    },
    "active_acceptance": {
        "transition_reason": "close_beyond_frozen_range",
        "required_source_kinds": [
            EventKind.DEALING_RANGE_CREATED.value,
            EventKind.DEALING_RANGE_ACTIVATED.value,
            EventKind.BAR_COMPLETED.value,
            EventKind.ACCEPTANCE_CONFIRMED.value,
        ],
    },
}
RANGE_INVALIDATION_POOLING = "forbidden_summarize_each_variant_separately"
DIRECTIONAL_STRUCTURAL_KINDS = frozenset(
    {
        EventKind.RAW_BOUNDARY_BREAK,
        EventKind.QUALIFIED_BOS,
        EventKind.MSS_CORE_CONFIRMED,
    }
)
FVG_LIFECYCLE_KINDS = frozenset(
    kind.value for kind in ATOMIC_KINDS if kind.value.startswith("fvg_")
)
SIGNAL_DIRECTION_ASSIGNMENT = {
    "explicit_event_direction": "use_event_direction",
    "level_touched_above": "short",
    "level_touched_below": "long",
    "otherwise": "not_outcome_eligible",
}
SECONDARY_OUTCOME_DEFINITIONS = {
    "mfe_atr": (
        "maximum favorable intrabar excursion from the known_at close across "
        "the full same-contract completed-bar horizon, divided by known_at ATR"
    ),
    "mae_atr": (
        "maximum adverse intrabar excursion from the known_at close across "
        "the full same-contract completed-bar horizon, divided by known_at ATR"
    ),
    "mfe_over_mae": "mfe_atr divided by mae_atr when mae_atr is positive",
    "continuation_distance_atr": (
        "maximum favorable completed-bar close displacement from the known_at "
        "close across the full same-contract horizon, divided by known_at ATR"
    ),
    "retracement_depth_atr": (
        "maximum completed-bar close pullback from the running directionally "
        "favorable close watermark, divided by known_at ATR"
    ),
    "range_extension_atr": (
        "maximum directional extension beyond the known_at bar extreme across "
        "the full same-contract horizon, divided by known_at ATR"
    ),
    "time_to_target_completed_bars": (
        "first later completed real 1m bar touching the frozen ATR target"
    ),
    "time_to_invalidation_completed_bars": (
        "first later completed real 1m bar touching the frozen ATR invalidation"
    ),
    "time_to_first_retest_completed_bars": (
        "first later completed real 1m bar overlapping the event zone; null "
        "when no event zone exists or no retest occurs"
    ),
    "time_to_fvg_midpoint_touch_completed_bars": (
        "first later completed real 1m bar touching the frozen FVG-zone midpoint; "
        "null for non-FVG events, missing zones, or no touch"
    ),
    "next_structural_event_id": (
        "identity of the first strictly later same-instrument raw boundary "
        "break, qualified BOS, or MSS Core"
    ),
    "next_structural_event_kind": (
        "kind of the first strictly later same-instrument raw boundary break, "
        "qualified BOS, or MSS Core"
    ),
    "next_structural_event_direction": (
        "direction of the first strictly later same-instrument raw boundary "
        "break, qualified BOS, or MSS Core"
    ),
    "next_structural_event_time": (
        "event_time of the first strictly later same-instrument structural event"
    ),
    "next_structural_event_known_at": (
        "known_at of the first strictly later same-instrument structural event"
    ),
    "next_structural_direction_match": (
        "whether the next structural event direction equals the assigned signal "
        "direction"
    ),
    "next_qualified_bos_direction": (
        "direction of the first strictly later same-instrument qualified BOS"
    ),
    "next_qualified_bos_direction_match": (
        "whether the next qualified BOS direction equals the assigned signal "
        "direction"
    ),
    "signal_half_life_completed_bars": (
        "cohort median of the first target-or-invalidation completed-bar offset "
        "over resolved non-ambiguous samples"
    ),
}
PRIMARY_PATH_SCAN_CONTRACT = {
    "path_scan": "full_same_contract_completed_bar_horizon",
    "first_hit_rule": "first_target_or_invalidation_determines_primary",
    "incomplete_horizon_policy": (
        "retain_primary_only_if_resolved_before_censoring_and_exclude_"
        "full_horizon_path_metrics"
    ),
}
PARENT_RELATION_PRIORITY = {
    "1m": ["5m"],
    "5m": ["1h", "15m"],
    "15m": ["1h"],
    "1h": ["4h"],
    "4h": [],
}
STATE_PROJECTION_PERSISTENCE = (
    "disabled_in_diagnostic_redundant_non_authoritative_transport"
)
SNAPSHOT_AUTHORITY = MarketSnapshotAuthority.ATOMIC_EVENT_REDUCER.value
DIAGNOSTIC_DATA_GAP_POLICY = "fail_closed_on_diagnostic_data_gap_history_reset"
REQUIRED_RESEARCH_LIMITATIONS = (
    "protected_swing_survival requires a separate preregistered "
    "survival/censoring protocol and is not computed by this diagnostic",
    "origin-zone/Order Block reaction requires a separate first-retest "
    "reaction and matched-zone control protocol and is not computed by this "
    "diagnostic",
)

RESEARCH_PROTOCOL_V3 = 3
V3_EPISODE_IDENTITY_FIELDS = (
    "kind",
    "symbol",
    "instrument_id",
    "known_at",
    "direction",
    "timeframe",
)
V3_MATCH_FIELDS = (
    "symbol",
    "instrument_id",
    "session_phase",
    "m1_direction",
    "atr_quartile",
    "nearest_distance_quartile",
    "relative_volume_quartile",
)
V3_HOLM_FAMILY = (
    "quiet_zero_event",
    "same_session_non_sweep_touch",
    "pseudo_level_touch",
    "forward_time_shift",
)
V3_CHAIN_EDGE_CONTRACT = {
    "E1_to_E2": (
        EventKind.LEVEL_TOUCHED.value,
        Timeframe.M5.value,
        EventKind.SWEEP_CONFIRMED.value,
        Timeframe.M5.value,
    ),
    "E2_to_E3": (
        EventKind.SWEEP_CONFIRMED.value,
        Timeframe.M5.value,
        EventKind.DISPLACEMENT_OBSERVED.value,
        Timeframe.M5.value,
    ),
    "E3_to_E4": (
        EventKind.DISPLACEMENT_OBSERVED.value,
        Timeframe.M5.value,
        EventKind.MSS_CORE_CONFIRMED.value,
        Timeframe.M5.value,
    ),
    "E4_to_E5": (
        EventKind.MSS_CORE_CONFIRMED.value,
        Timeframe.M5.value,
        EventKind.FVG_CREATED.value,
        Timeframe.M5.value,
    ),
}
V3_CHAIN_EDGE_MODES = {
    "E1_to_E2": ResearchLinkMode.STRICT_SOURCE_ANCESTRY,
    "E2_to_E3": ResearchLinkMode.CROSS_TIMEFRAME_CONSTITUENT_BAR,
    "E3_to_E4": ResearchLinkMode.CROSS_TIMEFRAME_CONSTITUENT_BAR,
    "E4_to_E5": ResearchLinkMode.CROSS_TIMEFRAME_CONSTITUENT_BAR,
}
V3_NESTED_STAGE_ORDER = (
    "E1_level_touch",
    "E2_sweep",
    "E3_sweep_displacement",
    "E4_sweep_displacement_mss",
    "E5_plus_fvg",
    "E6_plus_parent_alignment",
)
V3_NESTED_METRIC_CONTRACT = {
    "stage_order": list(V3_NESTED_STAGE_ORDER),
    "rate": "laplace_success_rate",
    "delta": "rate(E_i)-rate(E_i-1)",
    "missing_rate_policy": (
        "delta_null_if_current_or_immediate_prior_resolved_n_is_zero"
    ),
    "causal_claim": False,
}
V3_NON_NESTED_LINK_CONTRACT = {
    "mss_prior_sweep": (
        EventKind.SWEEP_CONFIRMED.value,
        Timeframe.M5.value,
        EventKind.MSS_CORE_CONFIRMED.value,
        Timeframe.M5.value,
    ),
    "mss_prior_displacement": (
        EventKind.DISPLACEMENT_OBSERVED.value,
        Timeframe.M5.value,
        EventKind.MSS_CORE_CONFIRMED.value,
        Timeframe.M5.value,
    ),
    "fvg_prior_displacement": (
        EventKind.DISPLACEMENT_OBSERVED.value,
        Timeframe.M5.value,
        EventKind.FVG_CREATED.value,
        Timeframe.M5.value,
    ),
}
V3_NON_NESTED_LINK_MODES = {
    name: ResearchLinkMode.CROSS_TIMEFRAME_CONSTITUENT_BAR
    for name in V3_NON_NESTED_LINK_CONTRACT
}
_IMMUTABLE_RESULT_STEMS = frozenset(
    {
        "smc_semantic_v1_2024_01_signal_diagnostic",
        "smc_semantics_v1_1_2024_01_phase5_diagnostic_v2",
    }
)
_V2_LEDGER_LABELS = (
    "event_study",
    "control_pairs",
    "source_chains",
)
_V3_LEDGER_LABELS = (
    "event_study",
    "control_pairs",
    "control_unmatched",
    "control_balance",
    "source_chains",
    "pseudo_construction",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _build_eye(
    model_path: Path = MODEL_PATH,
) -> tuple[CausalMarketReader, CausalObserver]:
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
            persist_state_projections=False,
            semantic_registry=str(selection.atomic_registry.source_path),
            scale_specs=specs,
            project_scene_graph=False,
            materialize_event_view=False,
            range_auction_projection_only=False,
            eye_authority_mode=True,
            canonical_foundation_enabled=True,
        ),
        semantic_registry=selection.atomic_registry,
    )
    return CausalMarketReader(scale_specs=specs), observer


def _event_direction(event: MarketEvent) -> Direction | None:
    if event.direction is not None:
        return event.direction
    if event.kind is EventKind.LEVEL_TOUCHED:
        if event.side == "above":
            return Direction(SIGNAL_DIRECTION_ASSIGNMENT["level_touched_above"])
        if event.side == "below":
            return Direction(SIGNAL_DIRECTION_ASSIGNMENT["level_touched_below"])
    return None


def _parent_bucket(
    snapshot: Any,
    timeframe: Timeframe,
    event_direction: Direction | None,
) -> str:
    child_label = timeframe.value.lower()
    priority = PARENT_RELATION_PRIORITY.get(child_label)
    if priority is None:
        raise ResearchContractError(
            f"parent relation priority is not registered for {timeframe.value}"
        )
    by_parent: dict[str, Any] = {}
    for item in snapshot.relations.values():
        if item.child_tf is not timeframe:
            continue
        parent = item.parent_tf.value.lower()
        if parent in by_parent:
            raise ResearchContractError(
                f"duplicate parent relation for {parent}->{child_label}"
            )
        by_parent[parent] = item
    unexpected = set(by_parent).difference(priority)
    if unexpected:
        raise ResearchContractError(
            f"unregistered parent relation(s) for {child_label}: "
            f"{sorted(unexpected)}"
        )
    relation = next(
        (by_parent[parent] for parent in priority if parent in by_parent),
        None,
    )
    if relation is None:
        return "parent_neutral"
    if relation.parent_state_invalidated:
        return "after_parent_invalidation"
    if relation.parent_direction is None or event_direction is None:
        return "parent_neutral"
    if event_direction == relation.parent_direction:
        return "aligned_with_parent"
    if relation.parent_protected_swing_intact is True:
        return "against_parent_but_parent_intact"
    return "parent_unresolved"


def _nearest_candidate_distance(snapshot: Any, price: float, atr: float) -> float:
    prices = [
        candidate
        for state in snapshot.timeframe_states.values()
        for candidate in (
            *state.liquidity.unswept_bsl,
            *state.liquidity.unswept_ssl,
        )
    ]
    return (
        math.inf
        if not prices
        else min(abs(float(candidate) - price) for candidate in prices)
        / max(atr, 1e-12)
    )


def _range_invalidation_variant(
    event: MarketEvent,
    source_event_kinds: Mapping[str, str] | None,
) -> str | None:
    """Resolve and verify the non-poolable range invalidation identity."""

    if event.kind is not EventKind.DEALING_RANGE_INVALIDATED:
        return None
    transition_reason = event.evidence.get("transition_reason")
    matches = tuple(
        (name, definition)
        for name, definition in RANGE_INVALIDATION_VARIANT_CONTRACT.items()
        if definition["transition_reason"] == transition_reason
    )
    if len(matches) != 1:
        raise ResearchContractError(
            "dealing-range invalidation has an unregistered transition variant: "
            f"{event.event_id}={transition_reason!r}"
        )
    if source_event_kinds is None:
        raise ResearchContractError(
            "dealing-range invalidation requires exact source-kind ancestry"
        )
    unresolved = tuple(
        event_id
        for event_id in event.source_event_ids
        if event_id not in source_event_kinds
    )
    if unresolved:
        raise ResearchContractError(
            "dealing-range invalidation has unresolved source events: "
            + ", ".join(unresolved)
        )
    name, definition = matches[0]
    actual = Counter(
        source_event_kinds[event_id] for event_id in event.source_event_ids
    )
    expected = Counter(definition["required_source_kinds"])
    if actual != expected:
        raise ResearchContractError(
            "dealing-range invalidation ancestry disagrees with its registered "
            f"variant: {event.event_id} expected={dict(expected)} "
            f"actual={dict(actual)}"
        )
    return name


def _passes_research_selection(event: MarketEvent) -> bool:
    """Apply the small, frozen Phase-5 selection predicate registry."""

    selection = RESEARCH_SELECTION.get(event.kind.value)
    if selection is None:
        return True
    if selection != {"predicate": "evidence.lifecycle == 'active'"}:
        raise ResearchContractError("unsupported runtime research selection")
    return event.evidence.get("lifecycle") == "active"


def _event_record(
    event: MarketEvent,
    snapshot: Any,
    *,
    source_event_kinds: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if event.origin is not EventOrigin.SEMANTIC_ATOMIC:
        raise ResearchContractError(
            f"research event is not semantic_atomic: {event.event_id}"
        )
    if event.known_at != snapshot.asof:
        raise ResearchContractError(
            "semantic event must be recorded against its exact known_at snapshot: "
            f"{event.event_id} known_at={event.known_at.isoformat()} "
            f"snapshot={snapshot.asof.isoformat()}"
        )
    direction = _event_direction(event)
    event_variant = _range_invalidation_variant(event, source_event_kinds)
    return {
        "event_id": event.event_id,
        "entity_id": event.entity_id,
        "kind": event.kind.value,
        "known_at": event.known_at,
        "event_time": event.event_time,
        "semantic_version": event.semantic_version,
        "origin": event.origin.value,
        "timeframe": event.timeframe.value,
        "symbol": snapshot.symbol,
        "instrument_id": snapshot.instrument_id,
        "direction": None if direction is None else direction.value,
        "side": event.side,
        "price": event.price,
        "zone": event.zone,
        "strength": float(event.strength),
        "parent_bucket": _parent_bucket(snapshot, event.timeframe, direction),
        "session_phase": snapshot.session.phase,
        "lifecycle": event.evidence.get("lifecycle"),
        "event_variant": event_variant,
        "source_event_ids": tuple(event.source_event_ids),
        "source_data_ids": tuple(event.source_data_ids),
        "source_entity_ids": tuple(event.source_entity_ids),
        "context_event_ids": tuple(event.context_event_ids),
        "evidence": to_primitive(event.evidence),
    }


def _outcome(
    signal: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    row_index: Mapping[pd.Timestamp, int],
    *,
    horizon: int = 60,
    target_atr: float = 1.0,
    invalidation_atr: float = 1.0,
    _allow_test_compatibility_rows: bool = False,
) -> dict[str, Any] | None:
    direction = signal.get("direction")
    start_index = row_index.get(signal["known_at"])
    if direction not in {"long", "short"} or start_index is None:
        return None
    start = rows[start_index]
    signal_symbol = signal.get("symbol")
    signal_instrument_id = signal.get("instrument_id")
    source_event_id = signal.get("event_id")
    if _allow_test_compatibility_rows:
        if signal_symbol is None:
            signal_symbol = start.get("symbol")
        if signal_instrument_id is None:
            signal_instrument_id = start.get("instrument_id")
        if source_event_id is None:
            payload = json.dumps(
                to_primitive(dict(signal)),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            source_event_id = (
                "research-signal:" + hashlib.sha256(payload).hexdigest()
            )
    if (
        not isinstance(source_event_id, str)
        or not source_event_id.strip()
        or not isinstance(signal_symbol, str)
        or not signal_symbol.strip()
        or type(signal_instrument_id) is not int
        or signal_instrument_id < 0
    ):
        raise ResearchContractError(
            "structural outcome signal lacks exact event/contract identity"
        )
    if (
        start.get("symbol") != signal_symbol
        or start.get("instrument_id") != signal_instrument_id
    ):
        raise ResearchContractError(
            "signal identity does not match its exact known_at entry row"
        )
    if type(horizon) is not int or horizon < 1:
        raise ResearchContractError("structural outcome horizon must be positive")
    try:
        target_multiple = float(target_atr)
        invalidation_multiple = float(invalidation_atr)
    except (TypeError, ValueError) as error:
        raise ResearchContractError(
            "structural outcome ATR barriers must be numeric"
        ) from error
    if (
        not math.isfinite(target_multiple)
        or target_multiple <= 0.0
        or not math.isfinite(invalidation_multiple)
        or invalidation_multiple <= 0.0
    ):
        raise ResearchContractError(
            "structural outcome ATR barriers must be finite and positive"
        )
    atr = float(start["atr"])
    if not math.isfinite(atr) or atr <= 0.0:
        return None
    entry = float(start["close"])
    sign = 1.0 if direction == "long" else -1.0
    try:
        tick_size = float(start["tick_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ResearchContractError(
            "structural outcome entry row lacks its frozen tick size"
        ) from error
    if not math.isfinite(tick_size) or tick_size <= 0.0:
        raise ResearchContractError(
            "structural outcome entry tick size must be finite and positive"
        )

    def outward_tick_price(raw_price: float, *, upward: bool) -> float:
        tick = Decimal(str(tick_size))
        coordinate = (Decimal(str(raw_price)) / tick).to_integral_value(
            rounding=ROUND_CEILING if upward else ROUND_FLOOR
        )
        return float(coordinate * tick)

    raw_target = entry + sign * target_multiple * atr
    raw_invalidation = entry - sign * invalidation_multiple * atr
    target = outward_tick_price(raw_target, upward=sign > 0.0)
    invalidation = outward_tick_price(
        raw_invalidation,
        upward=sign < 0.0,
    )
    path_rows = tuple(rows[start_index + 1 : start_index + 1 + horizon])
    if not rows:
        raise ResearchContractError("structural outcome row census is empty")
    start_clock = pd.Timestamp(signal["known_at"])
    final_clock = pd.Timestamp(rows[-1]["asof"])
    if start_clock.tzinfo is None or final_clock.tzinfo is None:
        raise ResearchContractError(
            "structural outcome clocks must be timezone aware"
        )
    window_end_exclusive = final_clock + pd.Timedelta(1, unit="min")
    horizon_seconds = int(
        (window_end_exclusive - start_clock).total_seconds()
    )
    if horizon_seconds < 1:
        raise ResearchContractError(
            "structural outcome observation window is empty"
        )
    outcome_bars: list[OutcomeBar] = []
    row_by_bar_event_id: dict[str, Mapping[str, Any]] = {}
    for row in path_rows:
        bar_event_id = row.get("bar_event_id")
        if (
            not isinstance(bar_event_id, str)
            or not bar_event_id.strip()
            or (
                bar_event_id.startswith("research-row:")
                and not _allow_test_compatibility_rows
            )
        ):
            raise ResearchContractError(
                "formal structural outcome row lacks a normalized BAR event ID"
            )
        if bar_event_id in row_by_bar_event_id:
            raise ResearchContractError(
                "structural outcome row repeats a BAR event ID"
            )
        same_contract = (
            row.get("symbol") == signal_symbol
            and row.get("instrument_id") == signal_instrument_id
        )
        if same_contract and row.get("tick_size") != tick_size:
            raise ResearchContractError(
                "structural outcome row tick size changed inside its window"
            )
        try:
            bar = OutcomeBar(
                bar_event_id=bar_event_id,
                symbol=str(row["symbol"]),
                instrument_id=row["instrument_id"],
                timeframe=Timeframe(str(row["timeframe"])),
                known_at=pd.Timestamp(row["asof"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ResearchContractError(
                "structural outcome row is not an exact normalized BAR fact"
            ) from error
        outcome_bars.append(bar)
        row_by_bar_event_id[bar_event_id] = row
    start_bar_event_id = start.get("bar_event_id")
    if (
        not isinstance(start_bar_event_id, str)
        or not start_bar_event_id.strip()
        or (
            start_bar_event_id.startswith("research-row:")
            and not _allow_test_compatibility_rows
        )
        or start.get("timeframe") != Timeframe.M1.value
        or pd.Timestamp(start.get("asof")) != start_clock
    ):
        raise ResearchContractError(
            "formal structural outcome entry row lacks exact M1 BAR identity"
        )
    try:
        OutcomeBar(
            bar_event_id=start_bar_event_id,
            symbol=str(start["symbol"]),
            instrument_id=start["instrument_id"],
            timeframe=Timeframe(str(start["timeframe"])),
            known_at=pd.Timestamp(start["asof"]),
            open=float(start["open"]),
            high=float(start["high"]),
            low=float(start["low"]),
            close=float(start["close"]),
        )
        spec = StructuralOutcomeSpec(
            source_event_id=source_event_id,
            symbol=signal_symbol,
            instrument_id=signal_instrument_id,
            timeframe=Timeframe.M1,
            direction=Direction(direction),
            observation_start_known_at=start_clock,
            observation_window_end_exclusive=window_end_exclusive,
            reference_price=entry,
            target_price=target,
            invalidation_price=invalidation,
            target_definition=(
                "directional_target_"
                f"{target_multiple:g}_atr_outward_to_first_tradable_tick"
            ),
            invalidation_definition=(
                "opposite_direction_invalidation_"
                f"{invalidation_multiple:g}_atr_outward_to_first_tradable_tick"
            ),
            atr_at_start=atr,
            tick_size=tick_size,
            horizon_bars=horizon,
            horizon_seconds=horizon_seconds,
        )
        engine_outcome = StructuralOutcomeEngine.evaluate(spec, outcome_bars)
    except (TypeError, ValueError) as error:
        raise ResearchContractError(
            "unified structural outcome evaluation failed closed"
        ) from error

    observed_rows = tuple(
        row_by_bar_event_id[event_id]
        for event_id in engine_outcome.source_bar_event_ids
    )
    offset_by_clock = {
        pd.Timestamp(row["asof"]): offset
        for offset, row in enumerate(observed_rows, start=1)
    }
    target_offset = (
        None
        if engine_outcome.target_hit_at is None
        else offset_by_clock[engine_outcome.target_hit_at]
    )
    invalidation_offset = (
        None
        if engine_outcome.invalidation_hit_at is None
        else offset_by_clock[engine_outcome.invalidation_hit_at]
    )
    favorable_closes: list[float] = []
    retracement_depth = 0.0
    running_favorable_close = entry
    zone = signal.get("zone")
    zone_bounds: tuple[float, float] | None = None
    if (
        isinstance(zone, Sequence)
        and not isinstance(zone, (str, bytes))
        and len(zone) == 2
    ):
        try:
            lower, upper = sorted((float(zone[0]), float(zone[1])))
        except (TypeError, ValueError):
            lower = upper = math.nan
        if math.isfinite(lower) and math.isfinite(upper):
            zone_bounds = (lower, upper)
    first_retest_offset = None
    fvg_midpoint_offset = None
    for offset, row in enumerate(observed_rows, start=1):
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        directional_close = sign * (close - entry)
        favorable_closes.append(directional_close)
        if sign > 0:
            running_favorable_close = max(running_favorable_close, close)
            retracement_depth = max(
                retracement_depth,
                running_favorable_close - close,
            )
        else:
            running_favorable_close = min(running_favorable_close, close)
            retracement_depth = max(
                retracement_depth,
                close - running_favorable_close,
            )
        if zone_bounds is not None:
            lower, upper = zone_bounds
            if first_retest_offset is None and high >= lower and low <= upper:
                first_retest_offset = offset
            midpoint = (lower + upper) / 2.0
            if (
                signal.get("kind") in FVG_LIFECYCLE_KINDS
                and fvg_midpoint_offset is None
                and low <= midpoint <= high
            ):
                fvg_midpoint_offset = offset
    ambiguous = (
        engine_outcome.terminal is OutcomeTerminal.AMBIGUOUS_SAME_BAR
    )
    resolved = engine_outcome.terminal in {
        OutcomeTerminal.TARGET_FIRST,
        OutcomeTerminal.INVALIDATION_FIRST,
    }
    success = (
        engine_outcome.terminal is OutcomeTerminal.TARGET_FIRST
        if resolved
        else None
    )
    continuation_distance = max(0.0, max(favorable_closes, default=0.0)) / atr
    event_high = float(start.get("high", entry))
    event_low = float(start.get("low", entry))
    range_extension = (
        max(
            0.0,
            max(
                (float(row["high"]) for row in observed_rows),
                default=event_high,
            )
            - event_high,
        )
        if sign > 0
        else max(
            0.0,
            event_low
            - min(
                (float(row["low"]) for row in observed_rows),
                default=event_low,
            ),
        )
    ) / atr
    observed_completed_bars = engine_outcome.observed_bars
    censored_by_contract_change = (
        engine_outcome.path_censor_reason == "contract_change"
    )
    censored_by_window_end = (
        not censored_by_contract_change and observed_completed_bars < horizon
    )
    full_horizon_observed = engine_outcome.full_horizon_observed
    if full_horizon_observed != (
        observed_completed_bars == horizon
        and not censored_by_contract_change
        and not censored_by_window_end
    ):
        raise ResearchContractError(
            "unified structural outcome censoring disagrees with research schema"
        )
    # Historical compatibility proxy only.  It is not the canonical
    # first_retest_event introduced by the semantic foundation.
    return {
        "resolved": resolved,
        "ambiguous": ambiguous,
        "success": success,
        "mfe_atr": engine_outcome.mfe_atr,
        "mae_atr": engine_outcome.mae_atr,
        "mfe_over_mae": (
            engine_outcome.mfe_atr / engine_outcome.mae_atr
            if engine_outcome.mfe_atr is not None
            and engine_outcome.mae_atr is not None
            and engine_outcome.mae_atr > 0.0
            else None
        ),
        "continuation_distance_atr": (
            continuation_distance if full_horizon_observed else None
        ),
        "retracement_depth_atr": (
            retracement_depth / atr if full_horizon_observed else None
        ),
        "range_extension_atr": (range_extension if full_horizon_observed else None),
        "time_to_target_completed_bars": target_offset,
        "time_to_invalidation_completed_bars": invalidation_offset,
        "time_to_first_retest_completed_bars": first_retest_offset,
        "time_to_fvg_midpoint_touch_completed_bars": fvg_midpoint_offset,
        "observed_completed_bars": observed_completed_bars,
        "censored_by_contract_change": censored_by_contract_change,
        "censored_by_window_end": censored_by_window_end,
        "full_horizon_observed": full_horizon_observed,
        "next_structural_event_id": signal.get("next_structural_event_id"),
        "next_structural_event_kind": signal.get("next_structural_event_kind"),
        "next_structural_event_direction": signal.get(
            "next_structural_event_direction"
        ),
        "next_structural_event_time": signal.get("next_structural_event_time"),
        "next_structural_event_known_at": signal.get("next_structural_event_known_at"),
        "next_structural_direction_match": signal.get(
            "next_structural_direction_match"
        ),
        "next_qualified_bos_direction": signal.get("next_qualified_bos_direction"),
        "next_qualified_bos_direction_match": signal.get(
            "next_qualified_bos_direction_match"
        ),
    }


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return None if not finite else sum(finite) / len(finite)


def _median(values: Iterable[int | float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return None if not finite else float(statistics.median(finite))


def _summary(
    signals: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    row_index: Mapping[pd.Timestamp, int],
    *,
    outcome_parameters: Mapping[str, Any] | None = None,
    _allow_test_compatibility_rows: bool = False,
) -> dict[str, Any]:
    parameters = {} if outcome_parameters is None else dict(outcome_parameters)
    outcomes = [
        value
        for signal in signals
        if (
            value := _outcome(
                signal,
                rows,
                row_index,
                _allow_test_compatibility_rows=(
                    _allow_test_compatibility_rows
                ),
                **parameters,
            )
        )
        is not None
    ]
    full_horizon_outcomes = [
        value for value in outcomes if value["full_horizon_observed"]
    ]
    resolved = [value for value in outcomes if value["resolved"]]
    successes = sum(value["success"] is True for value in resolved)
    resolved_n = len(resolved)
    resolution_times = [
        min(
            value
            for value in (
                outcome["time_to_target_completed_bars"],
                outcome["time_to_invalidation_completed_bars"],
            )
            if value is not None
        )
        for outcome in resolved
    ]
    structural_kind_counts = Counter(
        value["next_structural_event_kind"]
        for value in outcomes
        if value["next_structural_event_kind"] is not None
    )
    structural_direction_counts = Counter(
        value["next_structural_event_direction"]
        for value in outcomes
        if value["next_structural_event_direction"] is not None
    )
    qualified_bos_direction_counts = Counter(
        value["next_qualified_bos_direction"]
        for value in outcomes
        if value["next_qualified_bos_direction"] is not None
    )
    return {
        "signals": len(signals),
        "outcome_eligible": len(outcomes),
        "resolved_n": resolved_n,
        "ambiguous_n": sum(value["ambiguous"] for value in outcomes),
        "censored_n": sum(
            not value["resolved"] and not value["ambiguous"] for value in outcomes
        ),
        "full_horizon_outcome_n": len(full_horizon_outcomes),
        "contract_change_censored_n": sum(
            value["censored_by_contract_change"] for value in outcomes
        ),
        "window_end_censored_n": sum(
            value["censored_by_window_end"] for value in outcomes
        ),
        "successes": successes,
        "raw_success_rate": (None if resolved_n == 0 else successes / resolved_n),
        "laplace_success_rate": (successes + 1) / (resolved_n + 2),
        "mean_mfe_atr": _mean(value["mfe_atr"] for value in full_horizon_outcomes),
        "mean_mae_atr": _mean(value["mae_atr"] for value in full_horizon_outcomes),
        "mean_mfe_over_mae": _mean(
            value["mfe_over_mae"] for value in full_horizon_outcomes
        ),
        "mean_continuation_distance_atr": _mean(
            value["continuation_distance_atr"] for value in full_horizon_outcomes
        ),
        "mean_retracement_depth_atr": _mean(
            value["retracement_depth_atr"] for value in full_horizon_outcomes
        ),
        "mean_range_extension_atr": _mean(
            value["range_extension_atr"] for value in full_horizon_outcomes
        ),
        "next_structural_direction_match_rate": _mean(
            (1.0 if value["next_structural_direction_match"] else 0.0)
            for value in outcomes
            if value["next_structural_direction_match"] is not None
        ),
        "next_qualified_bos_direction_match_rate": _mean(
            (1.0 if value["next_qualified_bos_direction_match"] else 0.0)
            for value in outcomes
            if value["next_qualified_bos_direction_match"] is not None
        ),
        "next_structural_event_kind_counts": dict(
            sorted(structural_kind_counts.items())
        ),
        "next_structural_event_direction_counts": dict(
            sorted(structural_direction_counts.items())
        ),
        "next_qualified_bos_direction_counts": dict(
            sorted(qualified_bos_direction_counts.items())
        ),
        "median_time_to_target_completed_bars": _median(
            value["time_to_target_completed_bars"] for value in outcomes
        ),
        "median_time_to_invalidation_completed_bars": _median(
            value["time_to_invalidation_completed_bars"] for value in outcomes
        ),
        "median_time_to_first_retest_completed_bars": _median(
            value["time_to_first_retest_completed_bars"] for value in outcomes
        ),
        "median_time_to_fvg_midpoint_touch_completed_bars": _median(
            value["time_to_fvg_midpoint_touch_completed_bars"] for value in outcomes
        ),
        "signal_half_life_completed_bars": _median(resolution_times),
    }


def _validate_directional_entry_rows(
    events: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    row_index: Mapping[pd.Timestamp, int],
) -> None:
    """Every directional study event needs one exact real entry row."""

    for event in events:
        if event.get("direction") is None:
            continue
        index = row_index.get(event["known_at"])
        if index is None:
            raise ResearchContractError(
                "directional semantic event lacks an exact real known_at row: "
                f"{event['event_id']}"
            )
        row = rows[index]
        if row.get("symbol") != event.get("symbol") or row.get(
            "instrument_id"
        ) != event.get("instrument_id"):
            raise ResearchContractError(
                "directional semantic event entry identity mismatch: "
                f"{event['event_id']}"
            )


def _artifact_status(manifest_status: str, max_bars: int | None) -> str:
    return (
        "incomplete_smoke_not_experiment_result"
        if max_bars is not None
        else manifest_status
    )


def _enrich_next_structural_context(
    records: Sequence[dict[str, Any]],
    semantic_events: Sequence[Mapping[str, Any]],
) -> None:
    """Persist strictly later same-contract structural identities on signals."""

    structural_kinds = {kind.value for kind in DIRECTIONAL_STRUCTURAL_KINDS}
    structural_by_instrument: dict[tuple[str, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    qualified_bos_by_instrument: dict[tuple[str, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for item in semantic_events:
        symbol = item.get("symbol")
        instrument_id = item.get("instrument_id")
        if (
            not isinstance(symbol, str)
            or isinstance(instrument_id, bool)
            or not isinstance(instrument_id, Integral)
            or item.get("direction") is None
        ):
            continue
        identity = (symbol, int(instrument_id))
        if item.get("kind") in structural_kinds:
            structural_by_instrument[identity].append(item)
        if item.get("kind") == EventKind.QUALIFIED_BOS.value:
            qualified_bos_by_instrument[identity].append(item)
    for population in (
        structural_by_instrument,
        qualified_bos_by_instrument,
    ):
        for items in population.values():
            items.sort(key=lambda item: (item["known_at"], item["event_id"]))

    def strictly_later(
        items: Sequence[Mapping[str, Any]],
        known_at: pd.Timestamp,
    ) -> Mapping[str, Any] | None:
        clocks = [item["known_at"] for item in items]
        index = bisect_right(clocks, known_at)
        return None if index >= len(items) else items[index]

    for record in records:
        symbol = record.get("symbol")
        instrument_id = record.get("instrument_id")
        known_at = record.get("known_at")
        identity = (
            (symbol, instrument_id)
            if isinstance(symbol, str)
            and isinstance(instrument_id, Integral)
            and not isinstance(instrument_id, bool)
            and isinstance(known_at, pd.Timestamp)
            else None
        )
        if identity is not None:
            identity = (identity[0], int(identity[1]))
        next_event = (
            None
            if identity is None
            else strictly_later(
                structural_by_instrument.get(identity, ()),
                known_at,
            )
        )
        next_bos = (
            None
            if identity is None
            else strictly_later(
                qualified_bos_by_instrument.get(identity, ()),
                known_at,
            )
        )
        record.update(
            {
                "next_structural_event_id": (
                    None if next_event is None else next_event["event_id"]
                ),
                "next_structural_event_kind": (
                    None if next_event is None else next_event["kind"]
                ),
                "next_structural_event_direction": (
                    None if next_event is None else next_event["direction"]
                ),
                "next_structural_event_time": (
                    None if next_event is None else next_event["event_time"]
                ),
                "next_structural_event_known_at": (
                    None if next_event is None else next_event["known_at"]
                ),
                "next_structural_direction_match": (
                    None
                    if next_event is None or record.get("direction") is None
                    else next_event["direction"] == record["direction"]
                ),
                "next_qualified_bos_direction": (
                    None if next_bos is None else next_bos["direction"]
                ),
                "next_qualified_bos_direction_match": (
                    None
                    if next_bos is None or record.get("direction") is None
                    else next_bos["direction"] == record["direction"]
                ),
            }
        )


def _quartile_cutoffs(values: Sequence[float]) -> tuple[float, float, float] | None:
    finite = sorted(item for item in values if math.isfinite(item))
    if not finite:
        return None
    return (
        finite[min(len(finite) - 1, int(len(finite) * 0.25))],
        finite[min(len(finite) - 1, int(len(finite) * 0.50))],
        finite[min(len(finite) - 1, int(len(finite) * 0.75))],
    )


def _quartile(
    cutoffs: tuple[float, float, float] | None,
    value: float,
) -> int:
    if cutoffs is None or not math.isfinite(value):
        return -1
    return sum(value > cutoff for cutoff in cutoffs)


def _matched_controls(
    touches: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[Mapping[str, Any]], list[dict[str, Any]]]:
    atr_values = [float(row["atr"]) for row in rows]
    distance_values = [float(row["nearest_distance_atr"]) for row in rows]
    volume_values = [
        float(row["relative_volume"])
        for row in rows
        if row["relative_volume"] is not None
    ]
    atr_cutoffs = _quartile_cutoffs(atr_values)
    distance_cutoffs = _quartile_cutoffs(distance_values)
    volume_cutoffs = _quartile_cutoffs(volume_values)
    enriched = []
    by_clock = {row["asof"]: row for row in rows}
    index_by_clock = {row["asof"]: index for index, row in enumerate(rows)}
    for index, row in enumerate(rows):
        enriched.append(
            {
                **row,
                "index": index,
                "atr_q": _quartile(atr_cutoffs, float(row["atr"])),
                "distance_q": _quartile(
                    distance_cutoffs,
                    float(row["nearest_distance_atr"]),
                ),
                "volume_q": _quartile(
                    volume_cutoffs,
                    (
                        math.inf
                        if row["relative_volume"] is None
                        else float(row["relative_volume"])
                    ),
                ),
            }
        )
    strata: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        if row["atomic_event_count"]:
            continue
        key = (
            row["symbol"],
            row["instrument_id"],
            row["session_phase"],
            row["m1_direction"],
            row["atr_q"],
            row["distance_q"],
            row["volume_q"],
        )
        strata[key].append(row)
    used: set[pd.Timestamp] = set()
    controls: list[dict[str, Any]] = []
    matched_treatments: list[Mapping[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for touch in touches:
        treatment_row = by_clock.get(touch["known_at"])
        if treatment_row is None:
            continue
        if (
            touch.get("symbol") != treatment_row["symbol"]
            or touch.get("instrument_id") != treatment_row["instrument_id"]
        ):
            raise ResearchContractError(
                "treatment event and completed-bar instrument identity disagree"
            )
        key = (
            treatment_row["symbol"],
            treatment_row["instrument_id"],
            treatment_row["session_phase"],
            treatment_row["m1_direction"],
            _quartile(atr_cutoffs, float(treatment_row["atr"])),
            _quartile(
                distance_cutoffs,
                float(treatment_row["nearest_distance_atr"]),
            ),
            _quartile(
                volume_cutoffs,
                (
                    math.inf
                    if treatment_row["relative_volume"] is None
                    else float(treatment_row["relative_volume"])
                ),
            ),
        )
        candidates = [item for item in strata.get(key, ()) if item["asof"] not in used]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda item: (
                abs(item["index"] - index_by_clock[touch["known_at"]]),
                item["asof"],
            ),
        )
        used.add(selected["asof"])
        control = {
            "event_id": f"control:{touch['event_id']}",
            "kind": "matched_control",
            "known_at": selected["asof"],
            "symbol": selected["symbol"],
            "instrument_id": selected["instrument_id"],
            "direction": touch["direction"],
            "timeframe": touch["timeframe"],
        }
        controls.append(control)
        matched_treatments.append(touch)
        pairs.append(
            {
                "pair_id": f"pair:{touch['event_id']}",
                "treatment_event_id": touch["event_id"],
                "treatment_known_at": touch["known_at"],
                "control_event_id": control["event_id"],
                "control_known_at": control["known_at"],
                "completed_bar_offset": (
                    selected["index"] - index_by_clock[touch["known_at"]]
                ),
                "stratum": {
                    "symbol": key[0],
                    "instrument_id": key[1],
                    "session_phase": key[2],
                    "m1_direction": key[3],
                    "atr_quartile": key[4],
                    "nearest_distance_quartile": key[5],
                    "relative_volume_quartile": key[6],
                },
            }
        )
    return controls, matched_treatments, pairs


def _source_linked_stage(
    previous: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    *,
    completed_index: Mapping[pd.Timestamp, int],
    maximum_completed_bars: int,
    previous_kind: str,
    previous_timeframe: str,
    current_kind: str,
    current_timeframe: str,
    stage: str,
    prior_chains: Mapping[str, tuple[str, ...]],
) -> tuple[list[Mapping[str, Any]], dict[str, tuple[str, ...]], list[dict[str, Any]]]:
    """Advance one registered edge using completed bars and source ancestry."""

    admitted: list[Mapping[str, Any]] = []
    chains: dict[str, tuple[str, ...]] = {}
    ledger: list[dict[str, Any]] = []
    for event in current:
        if (
            event.get("kind") != current_kind
            or event.get("timeframe") != current_timeframe
        ):
            raise ResearchContractError(
                f"{stage} received an event outside its registered population"
            )
        link = find_prior_source_link(
            previous,
            event,
            completed_index=completed_index,
            kinds=frozenset({previous_kind}),
            maximum_completed_bars=maximum_completed_bars,
            timeframe=previous_timeframe,
        )
        if link is None:
            continue
        prior_id = str(link.prior["event_id"])
        chain = (*prior_chains.get(prior_id, (prior_id,)), str(event["event_id"]))
        admitted.append(event)
        chains[str(event["event_id"])] = chain
        ledger.append(
            {
                "stage": stage,
                "prior_event_id": prior_id,
                "current_event_id": event["event_id"],
                "prior_known_at": link.prior["known_at"],
                "current_known_at": event["known_at"],
                "completed_bars": link.completed_bars,
                "shared_lineage_tokens": link.shared_tokens,
                "chain_event_ids": chain,
            }
        )
    return admitted, chains, ledger


def _canonical_episode_projection(
    events: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, str],
    dict[str, Mapping[str, Any]],
]:
    """Project duplicate semantic observations into auditable analysis units."""

    episodes = canonical_treatment_episodes(
        events,
        identity_fields=V3_EPISODE_IDENTITY_FIELDS,
    )
    by_event_id = {str(event["event_id"]): event for event in events}
    projections: list[dict[str, Any]] = []
    event_to_episode: dict[str, str] = {}
    episode_by_id: dict[str, Mapping[str, Any]] = {}
    for episode in episodes:
        constituent_ids = tuple(
            str(value) for value in episode["constituent_event_ids"]
        )
        representative_id = min(constituent_ids)
        representative = by_event_id[representative_id]
        episode_id = str(episode["event_id"])
        projection = {
            **representative,
            **episode,
            "event_id": episode_id,
            "representative_event_id": representative_id,
            "constituent_event_ids": constituent_ids,
        }
        projections.append(projection)
        episode_by_id[episode_id] = projection
        for event_id in constituent_ids:
            event_to_episode[event_id] = episode_id
    projections.sort(key=lambda item: (item["known_at"], item["event_id"]))
    return projections, event_to_episode, episode_by_id


def _v3_enriched_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Freeze exact-match strata from contemporaneous completed-row features."""

    atr_cutoffs = _quartile_cutoffs([float(row["atr"]) for row in rows])
    distance_cutoffs = _quartile_cutoffs(
        [float(row["nearest_distance_atr"]) for row in rows]
    )
    volume_cutoffs = _quartile_cutoffs(
        [
            float(row["relative_volume"])
            for row in rows
            if row.get("relative_volume") is not None
        ]
    )
    enriched: list[dict[str, Any]] = []
    for row in rows:
        volume = row.get("relative_volume")
        enriched.append(
            {
                **row,
                "atr_quartile": _quartile(atr_cutoffs, float(row["atr"])),
                "nearest_distance_quartile": _quartile(
                    distance_cutoffs,
                    float(row["nearest_distance_atr"]),
                ),
                "relative_volume_quartile": _quartile(
                    volume_cutoffs,
                    math.inf if volume is None else float(volume),
                ),
            }
        )
    return enriched


def _v3_attach_match_strata(
    treatments: Sequence[Mapping[str, Any]],
    rows_by_clock: Mapping[pd.Timestamp, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    admitted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for treatment in treatments:
        row = rows_by_clock.get(pd.Timestamp(treatment["known_at"]))
        if row is None:
            excluded.append(
                {
                    "treatment_id": treatment["event_id"],
                    "reason": "treatment_clock_not_completed",
                }
            )
            continue
        if row.get("symbol") != treatment.get("symbol") or row.get(
            "instrument_id"
        ) != treatment.get("instrument_id"):
            raise ResearchContractError(
                "v3 treatment and exact completed row identity disagree"
            )
        if any(
            row.get(field) is None or row.get(field) == "" for field in V3_MATCH_FIELDS
        ):
            excluded.append(
                {
                    "treatment_id": treatment["event_id"],
                    "reason": "incomplete_exact_match_stratum",
                }
            )
            continue
        admitted.append(
            {
                **treatment,
                **{field: row[field] for field in V3_MATCH_FIELDS},
            }
        )
    return admitted, excluded


def _v3_match_spec(
    controls: Mapping[str, Any],
    *,
    direction_policy: ControlDirectionPolicy,
) -> MatchSpec:
    matching = controls["matching"]
    return MatchSpec(
        exact_fields=V3_MATCH_FIELDS,
        maximum_completed_bar_offset=int(matching["maximum_completed_bar_offset"]),
        outcome_horizon_completed_bars=int(matching["outcome_horizon_completed_bars"]),
        embargo_completed_bars=int(matching["embargo_completed_bars"]),
        forward_only=True,
        replacement_limit=int(matching["replacement_limit"]),
        direction_policy=direction_policy,
    )


def _v3_exact_stratum_mismatches(
    treatment: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return registered v3 strata that differ at the two entry clocks."""

    return tuple(
        field
        for field in V3_MATCH_FIELDS
        if treatment.get(field) != candidate.get(field)
    )


def _v3_time_shift_stratum_exclusion(
    treatment: Mapping[str, Any],
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Materialize an auditable exclusion for a shifted stratum change."""

    mismatched_fields = _v3_exact_stratum_mismatches(treatment, candidate)
    if not mismatched_fields:
        return None
    return {
        "control_type": "forward_time_shift",
        "treatment_id": treatment["event_id"],
        "candidate_id": control["event_id"],
        "completed_bar_offset": control["completed_bar_offset"],
        "reason": "shifted_exact_stratum_changed",
        "mismatched_fields": mismatched_fields,
    }


def _v3_quiet_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if int(row["atomic_event_count"]) != 0 or any(
            row.get(field) is None or row.get(field) == "" for field in V3_MATCH_FIELDS
        ):
            continue
        raw_id = (
            f"quiet|{row['symbol']}|{row['instrument_id']}|"
            f"{pd.Timestamp(row['asof']).isoformat()}"
        )
        candidates.append(
            {
                **row,
                "candidate_id": (
                    "quiet:" + hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]
                ),
                "known_at": row["asof"],
            }
        )
    return candidates


def _v3_non_sweep_touch_candidates(
    raw_touches: Sequence[Mapping[str, Any]],
    sweeps: Sequence[Mapping[str, Any]],
    *,
    completed_index: Mapping[pd.Timestamp, int],
    maximum_sweep_window: int,
    rows_by_clock: Mapping[pd.Timestamp, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes, event_to_episode, _ = _canonical_episode_projection(raw_touches)
    swept_episode_ids: set[str] = set()
    touch_ids = set(event_to_episode)
    touch_by_id = {str(item["event_id"]): item for item in raw_touches}
    for sweep in sweeps:
        for token in sweep.get("source_lineage_tokens", ()):
            if not isinstance(token, str) or not token.startswith("event:"):
                continue
            event_id = token.removeprefix("event:")
            if event_id not in touch_ids:
                continue
            touch = touch_by_id[event_id]
            distance = completed_index.get(pd.Timestamp(sweep["known_at"]))
            start = completed_index.get(pd.Timestamp(touch["known_at"]))
            if (
                start is not None
                and distance is not None
                and 0 < distance - start <= maximum_sweep_window
            ):
                swept_episode_ids.add(event_to_episode[event_id])
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = str(episode["event_id"])
        if episode_id in swept_episode_ids:
            exclusions.append(
                {"candidate_id": episode_id, "reason": "descendant_sweep_present"}
            )
            continue
        row = rows_by_clock.get(pd.Timestamp(episode["known_at"]))
        if row is None:
            exclusions.append(
                {"candidate_id": episode_id, "reason": "touch_clock_not_completed"}
            )
            continue
        if any(
            row.get(field) is None or row.get(field) == "" for field in V3_MATCH_FIELDS
        ):
            exclusions.append(
                {"candidate_id": episode_id, "reason": "incomplete_exact_match_stratum"}
            )
            continue
        candidates.append(
            {
                **episode,
                **{field: row[field] for field in V3_MATCH_FIELDS},
                "candidate_id": episode_id,
                "direction_known_at": episode["known_at"],
            }
        )
    return candidates, exclusions


def _v3_pseudo_touch_candidates(
    anchors: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    completed_index: Mapping[pd.Timestamp, int],
    control_contract: Mapping[str, Any],
    outcome_horizon: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    definition = control_contract["pseudo_level_touch"]
    spec = PseudoLevelSpec(
        protocol_id="smc_signal_research_v3_pseudo_level",
        relative_locations=tuple(definition["relative_locations"]),
        tick_size=float(definition["tick_size"]),
        minimum_separation_ticks=int(definition["minimum_separation_ticks"]),
    )
    rows_by_index = {
        completed_index[pd.Timestamp(row["asof"])]: row
        for row in rows
        if pd.Timestamp(row["asof"]) in completed_index
    }
    maximum_index = max(rows_by_index, default=-1)
    maximum_touch_window = int(definition["maximum_touch_window_completed_bars"])
    candidates: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for anchor in anchors:
        snapshot = anchor["snapshot"]
        ledger.append(
            {
                "anchor_event_id": anchor["event_id"],
                "construction_known_at": anchor["known_at"],
                "status": "construction_input",
                "snapshot": snapshot,
                "known_real_levels": anchor["known_real_levels"],
                "protocol": {
                    "protocol_id": spec.protocol_id,
                    "relative_locations": spec.relative_locations,
                    "tick_size": spec.tick_size,
                    "minimum_separation_ticks": spec.minimum_separation_ticks,
                },
            }
        )
        build = construct_pseudo_levels(
            anchor,
            snapshot,
            anchor["known_real_levels"],
            spec=spec,
        )
        for exclusion in build.exclusions:
            ledger.append(
                {
                    "anchor_event_id": anchor["event_id"],
                    "construction_known_at": anchor["known_at"],
                    "status": "construction_excluded",
                    **dict(exclusion),
                }
            )
        anchor_index = completed_index.get(pd.Timestamp(anchor["known_at"]))
        for level in build.levels:
            touched_row: Mapping[str, Any] | None = None
            touched_offset: int | None = None
            reason = None
            if anchor_index is None:
                reason = "construction_clock_not_completed"
            else:
                stop = min(maximum_index, anchor_index + maximum_touch_window)
                for index in range(anchor_index + 1, stop + 1):
                    row = rows_by_index.get(index)
                    if row is None:
                        continue
                    if (
                        row.get("symbol") != level.symbol
                        or row.get("instrument_id") != level.instrument_id
                    ):
                        break
                    if float(row["low"]) <= level.price <= float(row["high"]):
                        touched_row = row
                        touched_offset = index - anchor_index
                        break
                if touched_row is None:
                    reason = "no_executable_touch_inside_registered_window"
                elif (
                    completed_index[pd.Timestamp(touched_row["asof"])] + outcome_horizon
                    > maximum_index
                ):
                    reason = "touch_outcome_horizon_incomplete"
            if reason is not None:
                ledger.append(
                    {
                        "anchor_event_id": anchor["event_id"],
                        "pseudo_level_id": level.pseudo_level_id,
                        "construction_known_at": level.construction_known_at,
                        "price": level.price,
                        "normalized_location": level.normalized_location,
                        "status": "not_executable",
                        "reason": reason,
                    }
                )
                continue
            assert touched_row is not None and touched_offset is not None
            if any(
                touched_row.get(field) is None or touched_row.get(field) == ""
                for field in V3_MATCH_FIELDS
            ):
                ledger.append(
                    {
                        "anchor_event_id": anchor["event_id"],
                        "pseudo_level_id": level.pseudo_level_id,
                        "construction_known_at": level.construction_known_at,
                        "touch_known_at": touched_row["asof"],
                        "status": "not_executable",
                        "reason": "incomplete_exact_match_stratum",
                    }
                )
                continue
            candidate_id = f"{level.pseudo_level_id}:touch"
            candidates.append(
                {
                    **touched_row,
                    "candidate_id": candidate_id,
                    "event_id": candidate_id,
                    "kind": "pseudo_level_touch_control",
                    "known_at": touched_row["asof"],
                    "direction": level.direction,
                    "direction_known_at": level.construction_known_at,
                    "timeframe": Timeframe.M5.value,
                    "pseudo_level_id": level.pseudo_level_id,
                    "price": level.price,
                    "construction_known_at": level.construction_known_at,
                    "anchor_event_id": level.anchor_event_id,
                }
            )
            ledger.append(
                {
                    "anchor_event_id": anchor["event_id"],
                    "pseudo_level_id": level.pseudo_level_id,
                    "construction_known_at": level.construction_known_at,
                    "touch_known_at": touched_row["asof"],
                    "touch_completed_bars": touched_offset,
                    "price": level.price,
                    "normalized_location": level.normalized_location,
                    "direction": level.direction,
                    "status": "executable_touch",
                }
            )
    candidates.sort(key=lambda item: (item["known_at"], item["candidate_id"]))
    return candidates, ledger


def _v3_source_linked_stage(
    previous: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    *,
    completed_index: Mapping[pd.Timestamp, int],
    edge: Mapping[str, Any],
    stage: str,
    prior_chains: Mapping[str, tuple[str, ...]],
    event_lookup,
) -> tuple[
    list[Mapping[str, Any]],
    list[dict[str, Any]],
    dict[str, tuple[str, ...]],
    list[dict[str, Any]],
]:
    """Advance one M5 edge using only its registered typed proof."""

    spec = TypedLinkSpec(
        previous_kind=str(edge["previous_kind"]),
        previous_timeframe=str(edge["previous_timeframe"]),
        current_kind=str(edge["current_kind"]),
        current_timeframe=str(edge["current_timeframe"]),
        maximum_completed_bars=int(edge["maximum_completed_bars"]),
        mode=ResearchLinkMode(str(edge["link_mode"])),
        constituent_bar_timeframe=edge.get("constituent_bar_timeframe"),
    )
    admitted_raw: list[Mapping[str, Any]] = []
    chains: dict[str, tuple[str, ...]] = {}
    ledger: list[dict[str, Any]] = []
    for event in sorted(
        current,
        key=lambda item: (pd.Timestamp(item["known_at"]), str(item["event_id"])),
    ):
        link = find_prior_typed_link(
            previous,
            event,
            completed_index=completed_index,
            spec=spec,
            event_lookup=event_lookup,
        )
        if link is None:
            continue
        if not link.shared_event_ids:
            raise ResearchContractError(
                "v3 nested chain cannot admit unproven temporal co-occurrence"
            )
        semantic_ancestry = link.source_ancestry_proven
        composition = link.composition_proven
        if semantic_ancestry != bool(edge["source_ancestry_required"]) or (
            composition != bool(edge["composition_proven"])
        ):
            raise ResearchContractError("v3 link proof type changed at runtime")
        prior_id = str(link.prior["event_id"])
        current_id = str(event["event_id"])
        chain = (*prior_chains.get(prior_id, (prior_id,)), current_id)
        admitted_raw.append(event)
        chains[current_id] = chain
        ledger.append(
            {
                "stage": stage,
                "prior_event_id": prior_id,
                "current_event_id": current_id,
                "prior_known_at": link.prior["known_at"],
                "current_known_at": event["known_at"],
                "completed_bars": link.completed_bars,
                "link_mode": link.mode.value,
                "semantic_event_ancestry_proven": semantic_ancestry,
                "source_ancestry_proven": semantic_ancestry,
                "composition_proven": composition,
                "shared_constituent_bar_event_ids": (
                    link.shared_event_ids if composition else ()
                ),
                "ancestor_event_ids": (
                    link.shared_event_ids if semantic_ancestry else ()
                ),
                "chain_event_ids": chain,
            }
        )
    episodes, event_to_episode, _ = _canonical_episode_projection(admitted_raw)
    for row in ledger:
        row["current_episode_id"] = event_to_episode[str(row["current_event_id"])]
    return admitted_raw, episodes, chains, ledger


def _v3_non_nested_partition(
    previous: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    *,
    completed_index: Mapping[pd.Timestamp, int],
    edge: Mapping[str, Any],
    comparison: str,
    event_lookup,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Partition canonical current episodes by one registered typed edge."""

    spec = TypedLinkSpec(
        previous_kind=str(edge["previous_kind"]),
        previous_timeframe=str(edge["previous_timeframe"]),
        current_kind=str(edge["current_kind"]),
        current_timeframe=str(edge["current_timeframe"]),
        maximum_completed_bars=int(edge["maximum_completed_bars"]),
        mode=ResearchLinkMode(str(edge["link_mode"])),
        constituent_bar_timeframe=edge.get("constituent_bar_timeframe"),
    )
    linked_event_ids: set[str] = set()
    proof_ledger: list[dict[str, Any]] = []
    for event in sorted(
        current,
        key=lambda item: (pd.Timestamp(item["known_at"]), str(item["event_id"])),
    ):
        link = find_prior_typed_link(
            previous,
            event,
            completed_index=completed_index,
            spec=spec,
            event_lookup=event_lookup,
        )
        if link is None:
            continue
        current_id = str(event["event_id"])
        linked_event_ids.add(current_id)
        proof_ledger.append(
            {
                "stage": "non_nested_typed_link",
                "comparison": comparison,
                "prior_event_id": str(link.prior["event_id"]),
                "current_event_id": current_id,
                "prior_known_at": link.prior["known_at"],
                "current_known_at": event["known_at"],
                "completed_bars": link.completed_bars,
                "link_mode": link.mode.value,
                "source_ancestry_proven": link.source_ancestry_proven,
                "composition_proven": link.composition_proven,
                "shared_constituent_bar_event_ids": (
                    link.shared_event_ids if link.composition_proven else ()
                ),
                "ancestor_event_ids": (
                    link.shared_event_ids if link.source_ancestry_proven else ()
                ),
            }
        )
    episodes, membership, _ = _canonical_episode_projection(current)
    for row in proof_ledger:
        row["current_episode_id"] = membership[str(row["current_event_id"])]
    with_link: list[dict[str, Any]] = []
    without_link: list[dict[str, Any]] = []
    for episode in episodes:
        target = (
            with_link
            if linked_event_ids.intersection(episode["constituent_event_ids"])
            else without_link
        )
        target.append(episode)
    return with_link, without_link, proof_ledger


def _v3_nested_deltas(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Attach adjacent registered-stage deltas without imputing sparse rates."""

    if set(summaries) != set(V3_NESTED_STAGE_ORDER):
        raise ResearchContractError("v3 nested stages are incomplete or changed")
    result: dict[str, dict[str, Any]] = {}
    prior_rate: float | None = None
    for stage in V3_NESTED_STAGE_ORDER:
        value = dict(summaries[stage])
        current_rate = (
            float(value["laplace_success_rate"])
            if int(value["resolved_n"]) > 0
            else None
        )
        value["delta_vs_prior"] = (
            None
            if prior_rate is None or current_rate is None
            else current_rate - prior_rate
        )
        result[stage] = value
        prior_rate = current_rate
    return result


def _v3_materialize_match(
    control_type: str,
    result: MatchResult,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    controls: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    unmatched = [
        {
            "control_type": control_type,
            "treatment_id": treatment_id,
            "reason": reason,
        }
        for treatment_id, reason in result.unmatched.items()
    ]
    for pair in result.pairs:
        candidate = pair.candidate
        control_id = f"control:{control_type}:{pair.candidate_id}"
        control = {
            **candidate,
            "event_id": control_id,
            "kind": f"{control_type}_control",
            "known_at": candidate["known_at"],
            "direction": pair.control_direction,
            "direction_known_at": pair.direction_known_at,
            "timeframe": Timeframe.M5.value,
        }
        controls.append(control)
        pairs.append(
            {
                "control_type": control_type,
                "pair_id": f"pair:{control_type}:{pair.treatment_id}",
                "treatment_event_id": pair.treatment_id,
                "treatment_known_at": pair.treatment["known_at"],
                "control_event_id": control_id,
                "control_known_at": control["known_at"],
                "control_candidate_id": pair.candidate_id,
                "completed_bar_offset": pair.completed_bar_offset,
                "control_direction": pair.control_direction,
                "direction_known_at": pair.direction_known_at,
                "stratum": {field: pair.treatment[field] for field in V3_MATCH_FIELDS},
            }
        )
    return controls, pairs, unmatched


def _v3_balance_record(
    control_type: str,
    pairs: Sequence[Mapping[str, Any]],
    *,
    treatments: Mapping[str, Mapping[str, Any]],
    controls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    mismatch_counts = Counter({field: 0 for field in V3_MATCH_FIELDS})
    offsets: list[int] = []
    stratum_counts: Counter[str] = Counter()
    for pair in pairs:
        treatment = treatments[str(pair["treatment_event_id"])]
        control = controls[str(pair["control_event_id"])]
        for field in V3_MATCH_FIELDS:
            mismatch_counts[field] += treatment[field] != control[field]
        offsets.append(int(pair["completed_bar_offset"]))
        stratum_counts[
            json.dumps(pair["stratum"], sort_keys=True, separators=(",", ":"))
        ] += 1
    return {
        "control_type": control_type,
        "matched_pairs": len(pairs),
        "exact_field_mismatch_counts": dict(mismatch_counts),
        "minimum_completed_bar_offset": min(offsets) if offsets else None,
        "median_completed_bar_offset": _median(offsets),
        "maximum_completed_bar_offset": max(offsets) if offsets else None,
        "matched_stratum_counts": dict(sorted(stratum_counts.items())),
    }


def _publish_bytes(path: Path, payload: bytes, *, no_clobber: bool) -> None:
    """Publish small bundle members without weakening legacy replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not no_clobber:
        atomic_bytes(path, payload)
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"research output already exists: {_display_path(path)}") from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # The destination hard-link is the publication linearization
            # point; best-effort temp cleanup must not reverse that outcome.
            pass


def _write_json(path: Path, value: Any, *, no_clobber: bool = False) -> None:
    payload = (
        json.dumps(
            to_primitive(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _publish_bytes(path, payload, no_clobber=no_clobber)


def _write_jsonl(
    path: Path,
    records: Iterable[Mapping[str, Any]],
    *,
    no_clobber: bool = False,
) -> dict[str, int | str]:
    """Stream canonical JSONL once and atomically publish the completed file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    rows = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                line = (
                    json.dumps(
                        to_primitive(record),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
                handle.write(line)
                digest.update(line)
                rows += 1
            handle.flush()
            os.fsync(handle.fileno())
        if no_clobber:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"research output already exists: {_display_path(path)}"
                ) from error
        else:
            os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # os.link/os.replace is authoritative once it succeeds.  A stale
            # private temp is preferable to reporting a committed file failed.
            pass
    return {"sha256": digest.hexdigest(), "rows": rows}


def _write_report_then_result(
    output: Path,
    result: Mapping[str, Any],
    report: str,
    *,
    no_clobber: bool,
) -> None:
    """Publish the result JSON last so it remains the bundle commit marker."""

    report_path = output.with_suffix(".md")
    _publish_bytes(
        report_path,
        (report + "\n").encode("utf-8"),
        no_clobber=no_clobber,
    )
    _write_json(output, result, no_clobber=no_clobber)


def _report(result: Mapping[str, Any]) -> str:
    minimum_sample = int(result["minimum_sample_requirement"])
    lines = [
        f"# {result['semantic_version']} — 2024-01 Signal Research Diagnostic",
        "",
        "> Diagnostic only. This window is not OOS, cannot fit a Brain artifact, and cannot authorize trading.",
        "",
        f"- Artifact status: `{result['status']}`",
        f"- Snapshot authority: `{result['snapshot_authority']}`",
        f"- Replayed real 1m rows: {result['coverage']['diagnostic_real_rows']:,}",
        f"- Canonical atomic events: {result['coverage']['atomic_events']:,}",
        f"- Audit events retained: {result['coverage']['audit_events']:,}",
        f"- Deterministic audit fingerprint: `{result['coverage']['audit_fingerprint']}`",
        "",
        "## Nested event chain",
        "",
        f"| Stage | Signals | Resolved | Laplace rate | Δ vs prior | Min n={minimum_sample} |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if result["status"] == "incomplete_smoke_not_experiment_result":
        lines[4:4] = [
            "> **INCOMPLETE SMOKE PREFIX — NOT AN EXPERIMENT RESULT.**",
            "",
        ]
    for name, value in result["nested_chain"].items():
        rate = value["laplace_success_rate"]
        delta = value["delta_vs_prior"]
        lines.append(
            f"| {name} | {value['signals']} | {value['resolved_n']} | "
            f"{rate:.3f} | {'—' if delta is None else f'{delta:+.3f}'} | "
            f"{'yes' if value['minimum_sample_met'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Atomic event studies",
            "",
            "| Event | Signals | Resolved | Laplace rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, value in result["atomic_event_studies"].items():
        if value.get("pooling") == RANGE_INVALIDATION_POOLING:
            for variant, variant_value in value["variants"].items():
                lines.append(
                    f"| {name}:{variant} | {variant_value['signals']} | "
                    f"{variant_value['resolved_n']} | "
                    f"{variant_value['laplace_success_rate']:.3f} |"
                )
            continue
        lines.append(
            f"| {name} | {value['signals']} | {value['resolved_n']} | "
            f"{value['laplace_success_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Non-nested comparisons",
            "",
            f"| Comparison | Population | Signals | Resolved | Laplace rate | Min n={minimum_sample} |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name, comparison in result["non_nested_comparisons"].items():
        for bucket, value in comparison.items():
            rate = (
                "—"
                if value["resolved_n"] == 0
                else f"{value['laplace_success_rate']:.3f}"
            )
            lines.append(
                f"| {name} | {bucket} | {value['signals']} | "
                f"{value['resolved_n']} | "
                f"{rate} | "
                f"{'yes' if value['signals'] >= minimum_sample else 'no'} |"
            )
    matched = result["matched_control_coverage"]
    lines.extend(
        [
            "",
            "## Matched controls",
            "",
            f"- Requested: {matched['requested']:,}",
            f"- Matched: {matched['matched']:,}",
            f"- Unmatched: {matched['unmatched']:,}",
            "",
            "## Interpretation guardrails",
            "",
            "- Rates are structural one-ATR diagnostics before execution costs, fills, stops, or P&L.",
            f"- Sparse cells below the preregistered minimum n={minimum_sample} are retained and marked; they are not evidence of absence or permission to retune.",
            "- Nested stages occur at later confirmation clocks, so deltas measure registered conditional populations, not a causal treatment effect.",
            "- No pseudo-level or time-shifted control was silently added; those remain explicit follow-up studies.",
            "",
        ]
    )
    return "\n".join(lines)


_CHAIN_EDGE_CONTRACT = {
    "E1_to_E2": (
        EventKind.LEVEL_TOUCHED.value,
        Timeframe.M1.value,
        EventKind.SWEEP_CONFIRMED.value,
        Timeframe.M1.value,
    ),
    "E2_to_E3": (
        EventKind.SWEEP_CONFIRMED.value,
        Timeframe.M1.value,
        EventKind.DISPLACEMENT_OBSERVED.value,
        Timeframe.M5.value,
    ),
    "E3_to_E4": (
        EventKind.DISPLACEMENT_OBSERVED.value,
        Timeframe.M5.value,
        EventKind.MSS_CORE_CONFIRMED.value,
        Timeframe.M5.value,
    ),
    "E4_to_E5": (
        EventKind.MSS_CORE_CONFIRMED.value,
        Timeframe.M5.value,
        EventKind.FVG_CREATED.value,
        Timeframe.M5.value,
    ),
}
_NON_NESTED_LINK_CONTRACT = {
    "mss_prior_sweep": (
        EventKind.SWEEP_CONFIRMED.value,
        Timeframe.M1.value,
        EventKind.MSS_CORE_CONFIRMED.value,
        Timeframe.M5.value,
    ),
    "mss_prior_displacement": (
        EventKind.DISPLACEMENT_OBSERVED.value,
        Timeframe.M5.value,
        EventKind.MSS_CORE_CONFIRMED.value,
        Timeframe.M5.value,
    ),
    "fvg_prior_displacement": (
        EventKind.DISPLACEMENT_OBSERVED.value,
        Timeframe.M5.value,
        EventKind.FVG_CREATED.value,
        Timeframe.M5.value,
    ),
}


def _validated_link_contracts(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]:
    event_definition = manifest.get("event_definition")
    if not isinstance(event_definition, Mapping):
        raise ResearchContractError("event_definition must be preregistered")
    chain_edges = event_definition.get("chain_edges")
    non_nested = event_definition.get("non_nested_links")
    if not isinstance(chain_edges, Mapping) or set(chain_edges) != set(
        _CHAIN_EDGE_CONTRACT
    ):
        raise ResearchContractError("chain_edges are incomplete or changed")
    if not isinstance(non_nested, Mapping) or set(non_nested) != set(
        _NON_NESTED_LINK_CONTRACT
    ):
        raise ResearchContractError("non_nested_links are incomplete or changed")

    def validate(
        values: Mapping[str, Any],
        expected: Mapping[str, tuple[str, str, str, str]],
    ) -> None:
        for name, identity in expected.items():
            raw = values[name]
            if not isinstance(raw, Mapping):
                raise ResearchContractError(f"research link {name} is invalid")
            actual = (
                raw.get("previous_kind"),
                raw.get("previous_timeframe"),
                raw.get("current_kind"),
                raw.get("current_timeframe"),
            )
            maximum = raw.get("maximum_completed_bars")
            if (
                actual != identity
                or raw.get("source_lineage_required") is not True
                or isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or maximum < 1
            ):
                raise ResearchContractError(
                    f"research link {name} must use registered source lineage "
                    "and completed-bar distance"
                )

    validate(chain_edges, _CHAIN_EDGE_CONTRACT)
    validate(non_nested, _NON_NESTED_LINK_CONTRACT)
    return chain_edges, non_nested


def _validated_v3_link_contracts(
    manifest: Mapping[str, Any],
) -> tuple[
    Mapping[str, Mapping[str, Any]],
    Mapping[str, Mapping[str, Any]],
]:
    """Validate the M5-only typed evidence chain registered by protocol v3."""

    if manifest.get("research_protocol_version") != RESEARCH_PROTOCOL_V3:
        raise ResearchContractError("research protocol v3 is not registered")
    event_definition = manifest.get("event_definition")
    chain_edges = (
        event_definition.get("chain_edges")
        if isinstance(event_definition, Mapping)
        else None
    )
    if not isinstance(chain_edges, Mapping) or set(chain_edges) != set(
        V3_CHAIN_EDGE_CONTRACT
    ):
        raise ResearchContractError("v3 chain_edges are incomplete or changed")
    non_nested = event_definition.get("non_nested_links")
    if not isinstance(non_nested, Mapping) or set(non_nested) != set(
        V3_NON_NESTED_LINK_CONTRACT
    ):
        raise ResearchContractError("v3 non_nested_links are incomplete or changed")

    def validate(
        values: Mapping[str, Any],
        expected: Mapping[str, tuple[str, str, str, str]],
        modes: Mapping[str, ResearchLinkMode],
    ) -> None:
        for name, identity in expected.items():
            raw = values[name]
            actual = (
                raw.get("previous_kind") if isinstance(raw, Mapping) else None,
                raw.get("previous_timeframe") if isinstance(raw, Mapping) else None,
                raw.get("current_kind") if isinstance(raw, Mapping) else None,
                raw.get("current_timeframe") if isinstance(raw, Mapping) else None,
            )
            maximum = (
                raw.get("maximum_completed_bars") if isinstance(raw, Mapping) else None
            )
            mode = modes[name]
            strict = mode is ResearchLinkMode.STRICT_SOURCE_ANCESTRY
            if (
                actual != identity
                or raw.get("link_mode") != mode.value
                or raw.get("source_ancestry_required") is not strict
                or raw.get("composition_proven") is not (not strict)
                or (
                    raw.get("constituent_bar_timeframe")
                    != (None if strict else Timeframe.M5.value)
                )
                or isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or maximum < 1
            ):
                raise ResearchContractError(
                    f"v3 research link {name} changed its registered strict-"
                    "ancestry/shared-M5-BAR mode or completed-bar distance"
                )

    validate(chain_edges, V3_CHAIN_EDGE_CONTRACT, V3_CHAIN_EDGE_MODES)
    validate(
        non_nested,
        V3_NON_NESTED_LINK_CONTRACT,
        V3_NON_NESTED_LINK_MODES,
    )
    return chain_edges, non_nested


def _validated_v3_control_contract(
    manifest: Mapping[str, Any],
    *,
    horizon: int,
) -> Mapping[str, Any]:
    controls = manifest.get("control_definition")
    if not isinstance(controls, Mapping):
        raise ResearchContractError("v3 control_definition is required")
    if tuple(controls.get("separate_control_families", ())) != V3_HOLM_FAMILY:
        raise ResearchContractError("v3 control families must remain separate")
    if tuple(controls.get("treatment_episode_identity", ())) != (
        V3_EPISODE_IDENTITY_FIELDS
    ):
        raise ResearchContractError("v3 treatment episode identity changed")
    if tuple(controls.get("exact_match_fields", ())) != V3_MATCH_FIELDS:
        raise ResearchContractError("v3 exact matching strata changed")

    matching = controls.get("matching")
    if not isinstance(matching, Mapping) or matching != {
        "algorithm": "deterministic_maximum_cardinality_sparse_bipartite",
        "forward_only": True,
        "maximum_completed_bar_offset": 780,
        "outcome_horizon_completed_bars": horizon,
        "embargo_completed_bars": 5,
        "replacement_limit": 1,
    }:
        raise ResearchContractError("v3 causal matching contract changed")
    if controls.get("quiet_zero_event") != {
        "candidate": "completed real 1m row with zero canonical atomic events",
        "direction_policy": (
            ControlDirectionPolicy.INHERIT_TREATMENT_AFTER_KNOWN_AT.value
        ),
    }:
        raise ResearchContractError("v3 quiet-control definition changed")
    if controls.get("same_session_non_sweep_touch") != {
        "candidate": (
            "canonical M5 level-touch episode with no descendant M5 sweep "
            "inside 25 completed real 1m bars"
        ),
        "maximum_sweep_window_completed_bars": 25,
        "direction_policy": ControlDirectionPolicy.CANDIDATE_LOCAL.value,
    }:
        raise ResearchContractError("v3 non-sweep-touch definition changed")
    if controls.get("pseudo_level_touch") != {
        "anchor": "H1 dealing_range_activated exact-known_at snapshot",
        "relative_locations": [0.2, 0.35, 0.65, 0.8],
        "tick_size": 0.25,
        "minimum_separation_ticks": 4,
        "maximum_touch_window_completed_bars": 120,
        "direction_policy": ControlDirectionPolicy.CANDIDATE_LOCAL.value,
        "eye_publication": False,
    }:
        raise ResearchContractError("v3 pseudo-level definition changed")
    if controls.get("forward_time_shift") != {
        "forward_offsets_completed_bars": [90],
        "outcome_horizon_completed_bars": horizon,
        "embargo_completed_bars": 5,
        "require_same_session_phase": True,
        "exclude_other_treatment_windows": True,
        "exact_match_fields": list(V3_MATCH_FIELDS),
    }:
        raise ResearchContractError("v3 time-shift definition changed")
    return controls


def _validated_v3_inference_contract(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    inference = manifest.get("inference_definition")
    if not isinstance(inference, Mapping) or inference != {
        "paired_test": "two_sided_exact_mcnemar",
        "family_order": list(V3_HOLM_FAMILY),
        "multiple_testing_method": "holm_fixed_family",
        "alpha": 0.05,
        "missing_or_below_minimum_sample_p": 1.0,
        "claim_authority": "diagnostic_unvalidated_no_acceptance_claim",
        "cross_pair_outcome_window_policy": (
            "overlap_allowed_p_values_descriptive_unvalidated"
        ),
    }:
        raise ResearchContractError("v3 fixed-family inference contract changed")
    return inference


def _validated_research_design(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    experiment_id = manifest.get("experiment_id")
    frozen_at = pd.Timestamp(manifest.get("frozen_at"))
    if (
        not isinstance(experiment_id, str)
        or not experiment_id
        or frozen_at.tzinfo is None
    ):
        raise ResearchContractError("experiment identity and frozen_at are required")
    event_definition = manifest.get("event_definition")
    atomic_population = (
        event_definition.get("atomic_population")
        if isinstance(event_definition, Mapping)
        else None
    )
    if (
        not isinstance(atomic_population, Sequence)
        or isinstance(atomic_population, (str, bytes))
        or any(not isinstance(value, str) for value in atomic_population)
        or len(atomic_population) != len(set(atomic_population))
        or frozenset(atomic_population) != REGISTERED_ATOMIC_POPULATION
    ):
        raise ResearchContractError(
            "manifest atomic population does not match the runtime lifecycle contract"
        )
    if event_definition.get("research_selection") != RESEARCH_SELECTION:
        raise ResearchContractError(
            "research selection predicates changed from the runtime contract"
        )
    if (
        event_definition.get("range_invalidation_variants")
        != RANGE_INVALIDATION_VARIANT_CONTRACT
        or event_definition.get("range_invalidation_pooling")
        != RANGE_INVALIDATION_POOLING
    ):
        raise ResearchContractError(
            "range invalidation variant contract is incomplete or changed"
        )
    if (
        not isinstance(event_definition, Mapping)
        or event_definition.get("signal_direction_assignment")
        != SIGNAL_DIRECTION_ASSIGNMENT
    ):
        raise ResearchContractError(
            "event-to-signal direction assignment changed from the runtime contract"
        )
    if (
        event_definition.get("state_projection_persistence")
        != STATE_PROJECTION_PERSISTENCE
    ):
        raise ResearchContractError(
            "state-projection persistence changed from the runtime contract"
        )
    if event_definition.get("snapshot_authority") != SNAPSHOT_AUTHORITY:
        raise ResearchContractError(
            "snapshot authority changed from the runtime contract"
        )
    if event_definition.get("parent_relation_priority") != PARENT_RELATION_PRIORITY:
        raise ResearchContractError(
            "parent relation priority changed from the runtime contract"
        )
    if manifest.get("secondary_outcomes") != SECONDARY_OUTCOME_DEFINITIONS:
        raise ResearchContractError(
            "secondary outcome definitions changed from the runtime contract"
        )
    limitations = manifest.get("limitations")
    if (
        not isinstance(limitations, Sequence)
        or isinstance(limitations, (str, bytes))
        or tuple(limitations) != REQUIRED_RESEARCH_LIMITATIONS
    ):
        raise ResearchContractError(
            "deferred structural outcome limitations changed from the runtime contract"
        )
    primary = manifest.get("primary_outcome")
    if (
        not isinstance(primary, Mapping)
        or primary.get("name") != "target_before_invalidation"
    ):
        raise ResearchContractError("primary structural outcome is not registered")
    horizon = primary.get("horizon_completed_bars")
    target_atr = primary.get("target_atr")
    invalidation_atr = primary.get("invalidation_atr")
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon < 1
        or isinstance(target_atr, bool)
        or not isinstance(target_atr, (int, float))
        or not math.isfinite(float(target_atr))
        or float(target_atr) <= 0.0
        or isinstance(invalidation_atr, bool)
        or not isinstance(invalidation_atr, (int, float))
        or not math.isfinite(float(invalidation_atr))
        or float(invalidation_atr) <= 0.0
        or primary.get("same_bar_tie") != "ambiguous_and_excluded"
        or any(
            primary.get(name) != value
            for name, value in PRIMARY_PATH_SCAN_CONTRACT.items()
        )
    ):
        raise ResearchContractError("primary outcome parameters are incomplete")
    minimum = manifest.get("minimum_sample_requirement")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ResearchContractError("minimum_sample_requirement must be positive")
    protocol_version = manifest.get("research_protocol_version", 2)
    if protocol_version == RESEARCH_PROTOCOL_V3:
        _validated_v3_control_contract(manifest, horizon=horizon)
        _validated_v3_inference_contract(manifest)
        if manifest.get("nested_chain_metric") != V3_NESTED_METRIC_CONTRACT:
            raise ResearchContractError("v3 nested-chain delta contract changed")
        ledgers = manifest.get("ledger_definition")
        if ledgers != {
            "event_study": "canonical_json_lines_v1",
            "control_pairs": "canonical_json_lines_v1",
            "control_unmatched": "canonical_json_lines_v1",
            "control_balance": "canonical_json_lines_v1",
            "source_chains": "canonical_json_lines_v1",
            "pseudo_construction": "canonical_json_lines_v1",
        }:
            raise ResearchContractError(
                "persistent v3 research ledgers are not registered"
            )
    else:
        controls = manifest.get("control_definition")
        if not isinstance(controls, Mapping) or any(
            controls.get(name) is not True
            for name in (
                "candidate_rows_require_zero_atomic_events",
                "same_matched_cohort_required",
                "exact_instrument_identity_required",
            )
        ):
            raise ResearchContractError("matched-control safeguards are not registered")
        ledgers = manifest.get("ledger_definition")
        if ledgers != {
            "event_study": "canonical_json_lines_v1",
            "control_pairs": "canonical_json_lines_v1",
            "source_chains": "canonical_json_lines_v1",
        }:
            raise ResearchContractError(
                "persistent research ledgers are not registered"
            )
    census = manifest.get("input_census")
    required_census_counts = (
        "expected_emitted_bars_including_warmup",
        "expected_diagnostic_completed_bars",
        "expected_diagnostic_real_rows",
        "expected_diagnostic_ready_real_rows",
        "expected_diagnostic_synthetic_bars",
        "expected_warmup_data_gap_resets",
        "expected_diagnostic_data_gap_resets",
        "expected_contract_changes",
    )
    if (
        not isinstance(census, Mapping)
        or census.get("diagnostic_data_gap_policy") != DIAGNOSTIC_DATA_GAP_POLICY
        or any(
            isinstance(census.get(name), bool)
            or not isinstance(census.get(name), int)
            or int(census[name]) < 0
            for name in required_census_counts
        )
    ):
        raise ResearchContractError("input census contract is incomplete")
    for name in (
        "expected_last_processed_asof",
        "expected_last_diagnostic_asof",
    ):
        raw_clock = census.get(name)
        if not isinstance(raw_clock, str) or not raw_clock:
            raise ResearchContractError(f"input census {name} is required")
        value = pd.Timestamp(raw_clock)
        if value.tzinfo is None:
            raise ResearchContractError(f"input census {name} must be timezone aware")
    return {
        "horizon": horizon,
        "target_atr": float(target_atr),
        "invalidation_atr": float(invalidation_atr),
    }, minimum


def _assert_full_input_census(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    """Fail before artifact writes when the frozen input window is incomplete."""

    for name in (
        "emitted_bars_including_warmup",
        "diagnostic_completed_bars",
        "diagnostic_real_rows",
        "diagnostic_ready_real_rows",
        "diagnostic_synthetic_bars",
        "warmup_data_gap_resets",
        "diagnostic_data_gap_resets",
        "contract_changes",
    ):
        expected_name = f"expected_{name}"
        if int(actual[name]) != int(expected[expected_name]):
            raise ResearchContractError(
                f"full input census mismatch for {name}: "
                f"expected={expected[expected_name]} actual={actual[name]}"
            )
    for name in ("last_processed_asof", "last_diagnostic_asof"):
        expected_clock = pd.Timestamp(expected[f"expected_{name}"])
        actual_clock = actual[name]
        if actual_clock is None or pd.Timestamp(actual_clock) != expected_clock:
            raise ResearchContractError(
                f"full input census mismatch for {name}: "
                f"expected={expected_clock.isoformat()} actual={actual_clock}"
            )
    if "expected_first_diagnostic_asof" in expected:
        expected_clock = pd.Timestamp(expected["expected_first_diagnostic_asof"])
        actual_clock = actual.get("first_diagnostic_asof")
        if actual_clock is None or pd.Timestamp(actual_clock) != expected_clock:
            raise ResearchContractError(
                "full input census mismatch for first_diagnostic_asof: "
                f"expected={expected_clock.isoformat()} actual={actual_clock}"
            )
    if "expected_synthetic_clocks" in expected:
        expected_clocks = [
            pd.Timestamp(value) for value in expected["expected_synthetic_clocks"]
        ]
        actual_clocks = [
            pd.Timestamp(value) for value in actual.get("synthetic_clocks", ())
        ]
        if actual_clocks != expected_clocks:
            raise ResearchContractError(
                "full input census mismatch for synthetic_clocks: "
                f"expected={expected_clocks} actual={actual_clocks}"
            )
    if "expected_contracts" in expected:
        if actual.get("contracts") != expected["expected_contracts"]:
            raise ResearchContractError(
                "full input census mismatch for contracts: "
                f"expected={expected['expected_contracts']} "
                f"actual={actual.get('contracts')}"
            )


def _validate_registry_atomic_population(registry: SemanticRegistry) -> None:
    """Require the runtime study surface to equal the versioned registry."""

    registered_emitted = registry.canonical_emitted_event_kinds
    if registered_emitted == ATOMIC_KINDS:
        return
    missing_runtime = sorted(kind.value for kind in registered_emitted - ATOMIC_KINDS)
    unregistered_runtime = sorted(
        kind.value for kind in ATOMIC_KINDS - registered_emitted
    )
    raise ResearchContractError(
        "registry canonical_emitted union disagrees with Phase5 runtime "
        f"ATOMIC_KINDS: missing_runtime={missing_runtime} "
        f"unregistered_runtime={unregistered_runtime}"
    )


def _load_contract_and_registry(
    manifest_path: Path,
    *,
    contract_loader: Callable[..., Any] = load_frozen_research_contract,
    split_validator: Callable[[Any, Any], None] = validate_split_authority,
) -> tuple[
    Any,
    SemanticRegistry,
    Mapping[str, Mapping[str, Any]],
    Mapping[str, Mapping[str, Any]],
]:
    preview = _json(manifest_path)
    manifest_semantic_version = preview.get("semantic_version")
    if not isinstance(manifest_semantic_version, str) or not manifest_semantic_version:
        raise ResearchContractError("manifest semantic_version is required")
    registry_label = preview.get("semantic_registry")
    if not isinstance(registry_label, str) or not registry_label:
        raise ResearchContractError("semantic_registry path is required")
    registry_path = (ROOT / registry_label).resolve(strict=False)
    try:
        registry_path.relative_to(ROOT)
    except ValueError as error:
        raise ResearchContractError(
            "semantic_registry path escapes repository"
        ) from error
    if not registry_path.is_file() or registry_path.is_symlink():
        raise ResearchContractError("semantic_registry is not a regular file")
    registry = SemanticRegistry.from_file(
        registry_path,
        required_version=manifest_semantic_version,
    )
    _validate_registry_atomic_population(registry)
    contract = contract_loader(
        manifest_path,
        root=ROOT,
        actual_semantic_registry_identity=registry.identity,
    )
    if (
        registry.semantic_version != contract.payload.get("semantic_version")
        or registry.identity != contract.semantic_registry_identity
    ):
        raise ResearchContractError("semantic registry version or identity changed")
    validation = load_validation_protocol(contract.split_registry_path)
    split_validator(contract, validation)

    fixed_bindings = {
        "runner": Path(__file__).resolve(),
        "pyproject": (ROOT / "pyproject.toml").resolve(),
        "lockfile": (ROOT / "uv.lock").resolve(),
    }
    for name, expected_path in fixed_bindings.items():
        if contract.identity_paths[name] != expected_path:
            raise ResearchContractError(f"{name} does not bind the executed project")

    model = _json(contract.model_path)
    observer = model.get("observer")
    if not isinstance(observer, Mapping):
        raise ResearchContractError("bound model observer configuration is invalid")
    try:
        selection = load_semantic_selection(
            model.get("semantic_selection"),
            root=ROOT,
        )
    except ValueError as error:
        raise ResearchContractError(
            "bound model semantic_selection is invalid"
        ) from error
    if (
        selection.atomic_registry.source_path.resolve()
        != contract.identity_paths["semantic_registry"]
        or selection.atomic_definition_identity != registry.identity
    ):
        raise ResearchContractError(
            "model atomic semantic selection disagrees with exact identity binding"
        )
    model_bindings = {
        "structure_protocol": "structure_protocol",
        "liquidity_protocol": "liquidity_protocol",
        "displacement_protocol": "displacement_protocol",
        "zone_protocol": "group3_protocol",
        "range_auction_protocol": "group4_protocol",
        "group5_protocol": "group5_protocol",
    }
    for model_field, binding_name in model_bindings.items():
        label = observer.get(model_field)
        if (
            not isinstance(label, str)
            or (ROOT / label).resolve(strict=False)
            != contract.identity_paths[binding_name]
        ):
            raise ResearchContractError(
                f"model {model_field} disagrees with exact identity binding"
            )
    parameters_label = json.loads(registry_path.read_text(encoding="utf-8")).get(
        "parameters_file"
    )
    if (
        not isinstance(parameters_label, str)
        or (ROOT / parameters_label).resolve(strict=False)
        != contract.identity_paths["semantic_parameters"]
    ):
        raise ResearchContractError(
            "semantic registry parameters disagree with exact identity binding"
        )
    if contract.payload.get("research_protocol_version") == RESEARCH_PROTOCOL_V3:
        chain_edges, non_nested = _validated_v3_link_contracts(contract.payload)
    else:
        chain_edges, non_nested = _validated_link_contracts(contract.payload)
    return contract, registry, chain_edges, non_nested


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _ledger_path(output: Path, label: str) -> Path:
    return output.with_name(f"{output.stem}.{label}.jsonl")


def _output_bundle_paths(
    output: Path,
    *,
    research_protocol_version: int,
) -> tuple[Path, ...]:
    """Return every file one runner invocation can publish.

    The preflight treats the result, report, and ledgers as one immutable
    evidence bundle.  Checking only the result JSON would allow an interrupted
    or forced retry to replace a sibling ledger without first noticing it.
    """

    labels = (
        _V3_LEDGER_LABELS
        if research_protocol_version == RESEARCH_PROTOCOL_V3
        else _V2_LEDGER_LABELS
    )
    paths = (
        output,
        output.with_suffix(".md"),
        *(_ledger_path(output, label) for label in labels),
    )
    if len(paths) != len(set(paths)):
        raise ResearchContractError(
            "research output path aliases another evidence-bundle member"
        )
    return paths


def _preflight_output_bundle(
    output: Path,
    *,
    research_protocol_version: int,
    max_bars: int | None,
    force: bool,
) -> tuple[Path, ...]:
    """Require a fresh output bundle before any replay or artifact write."""

    paths = _output_bundle_paths(
        output,
        research_protocol_version=research_protocol_version,
    )
    if force and max_bars is None:
        raise ResearchContractError(
            "--force is forbidden for a frozen full-window research run"
        )
    existing = tuple(path for path in paths if path.exists() or path.is_symlink())
    if existing:
        labels = ", ".join(_display_path(path) for path in existing)
        raise FileExistsError(f"research output bundle already exists: {labels}")
    return paths


def _report_v3(result: Mapping[str, Any]) -> str:
    comparison_mode = (
        result.get("validation_state") == "fixed_historical_comparison_unvalidated"
    )
    title = (
        f"# {result['semantic_version']} — Fixed Phase-4/5 comparison"
        if comparison_mode
        else f"# {result['semantic_version']} — Signal Research protocol v3"
    )
    scope_note = (
        "> Fixed historical comparison only: unvalidated and without model-action authority."
        if comparison_mode
        else "> Development diagnostic only: unvalidated, not OOS, not a model, and not trading authority."
    )
    lines = [
        title,
        "",
        scope_note,
        "",
        f"- Status: `{result['status']}`",
        f"- Complete registered window: `{result['artifact_classification']['complete_registered_window']}`",
        f"- Canonical M5 touch episodes: {result['nested_chain']['E1_level_touch']['signals']}",
        "",
        "## Registered M5 evidence chain",
        "",
        "| Stage | Episodes | Resolved | Laplace rate | Δ vs prior | Min sample met |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in result["nested_chain"].items():
        delta = value["delta_vs_prior"]
        lines.append(
            f"| {name} | {value['signals']} | {value['resolved_n']} | "
            f"{value['laplace_success_rate']:.3f} | "
            f"{'—' if delta is None else f'{delta:+.3f}'} | "
            f"{'yes' if value['minimum_sample_met'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Non-nested typed comparisons",
            "",
            "| Comparison | Population | Episodes | Resolved | Laplace rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for name, comparison in result["non_nested_comparisons"].items():
        for bucket, value in comparison.items():
            lines.append(
                f"| {name} | {bucket} | {value['signals']} | "
                f"{value['resolved_n']} | {value['laplace_success_rate']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Separate control families",
            "",
            "| Control | Requested | Matched | Exact McNemar p | Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    holm = result["inference"]["holm_fixed_family"]
    for name in V3_HOLM_FAMILY:
        value = result["control_comparisons"][name]
        test = result["inference"]["exact_mcnemar"][name]
        lines.append(
            f"| {name} | {value['requested']} | {value['matched']} | "
            f"{test['p_value']:.4f} | {holm['adjusted_p_values'][name]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- E1→E2 requires semantic-event ancestry; later edges require a shared exact canonical M5 BAR and are composition evidence, not causal ancestry.",
            "- Quiet, non-sweep-touch, pseudo-level, and forward-time-shift controls are separate populations and are never pooled.",
            "- Missing or underpowered fixed-family tests enter Holm with p=1.",
            "- Cross-pair outcome windows may overlap; McNemar/Holm p-values remain descriptive and unvalidated.",
            "- Any `--max-bars` run remains an incomplete smoke artifact permanently.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_v3_analysis(
    *,
    output: Path,
    contract: Any,
    registry: SemanticRegistry,
    manifest: Mapping[str, Any],
    chain_edges: Mapping[str, Mapping[str, Any]],
    non_nested_links: Mapping[str, Mapping[str, Any]],
    events: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    completed_index: Mapping[pd.Timestamp, int],
    observer: CausalObserver,
    outcome_parameters: Mapping[str, Any],
    minimum_sample: int,
    pseudo_anchors: Sequence[Mapping[str, Any]],
    pseudo_anchor_exclusions: Sequence[Mapping[str, Any]],
    actual_input_census: Mapping[str, Any],
    max_bars: int | None,
    truncated_by_max_bars: bool,
    started: float,
) -> dict[str, Any]:
    """Finish protocol-v3 analysis without granting validation authority."""

    enriched_rows = _v3_enriched_rows(rows)
    rows_by_clock = {pd.Timestamp(row["asof"]): row for row in enriched_rows}
    row_index = {
        pd.Timestamp(row["asof"]): index for index, row in enumerate(enriched_rows)
    }
    _enrich_next_structural_context(events, events)

    raw_by_kind = {
        kind: [
            event
            for event in events
            if event["kind"] == kind
            and event["timeframe"] == Timeframe.M5.value
            and event["direction"] is not None
        ]
        for kind in (
            EventKind.LEVEL_TOUCHED.value,
            EventKind.SWEEP_CONFIRMED.value,
            EventKind.DISPLACEMENT_OBSERVED.value,
            EventKind.MSS_CORE_CONFIRMED.value,
            EventKind.FVG_CREATED.value,
        )
    }
    e1_raw = raw_by_kind[EventKind.LEVEL_TOUCHED.value]
    e1, e1_member_to_episode, _ = _canonical_episode_projection(e1_raw)
    prior_chains = {
        str(event["event_id"]): (
            e1_member_to_episode[str(event["event_id"])],
            str(event["event_id"]),
        )
        for event in e1_raw
    }
    e2_raw, e2, e2_chains, e2_ledger = _v3_source_linked_stage(
        e1_raw,
        raw_by_kind[EventKind.SWEEP_CONFIRMED.value],
        completed_index=completed_index,
        edge=chain_edges["E1_to_E2"],
        stage="E2_sweep",
        prior_chains=prior_chains,
        event_lookup=observer.audit_store.get,
    )
    e3_raw, e3, e3_chains, e3_ledger = _v3_source_linked_stage(
        e2_raw,
        raw_by_kind[EventKind.DISPLACEMENT_OBSERVED.value],
        completed_index=completed_index,
        edge=chain_edges["E2_to_E3"],
        stage="E3_sweep_displacement",
        prior_chains=e2_chains,
        event_lookup=observer.audit_store.get,
    )
    e4_raw, e4, e4_chains, e4_ledger = _v3_source_linked_stage(
        e3_raw,
        raw_by_kind[EventKind.MSS_CORE_CONFIRMED.value],
        completed_index=completed_index,
        edge=chain_edges["E3_to_E4"],
        stage="E4_sweep_displacement_mss",
        prior_chains=e3_chains,
        event_lookup=observer.audit_store.get,
    )
    e5_raw, e5, e5_chains, e5_ledger = _v3_source_linked_stage(
        e4_raw,
        raw_by_kind[EventKind.FVG_CREATED.value],
        completed_index=completed_index,
        edge=chain_edges["E4_to_E5"],
        stage="E5_plus_fvg",
        prior_chains=e4_chains,
        event_lookup=observer.audit_store.get,
    )
    e6 = [event for event in e5 if event["parent_bucket"] == "aligned_with_parent"]
    e5_raw_to_episode = _canonical_episode_projection(e5_raw)[1]
    chain_ledger = [*e2_ledger, *e3_ledger, *e4_ledger, *e5_ledger]
    chain_ledger.extend(
        {
            "stage": "E6_plus_parent_alignment",
            "current_event_id": event["representative_event_id"],
            "current_episode_id": event["event_id"],
            "current_known_at": event["known_at"],
            "parent_bucket": event["parent_bucket"],
            "chain_event_ids": e5_chains.get(
                str(event["representative_event_id"]),
                (e5_raw_to_episode.get(str(event["representative_event_id"])),),
            ),
        }
        for event in e6
    )

    def summarize(population: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        value = _summary(
            population,
            enriched_rows,
            row_index,
            outcome_parameters=outcome_parameters,
        )
        value["minimum_sample_met"] = value["signals"] >= minimum_sample
        return value

    nested_populations = {
        "E1_level_touch": e1,
        "E2_sweep": e2,
        "E3_sweep_displacement": e3,
        "E4_sweep_displacement_mss": e4,
        "E5_plus_fvg": e5,
        "E6_plus_parent_alignment": e6,
    }
    nested = _v3_nested_deltas(
        {name: summarize(value) for name, value in nested_populations.items()}
    )
    non_nested_populations = {
        "mss_prior_sweep_composition_linked": _v3_non_nested_partition(
            raw_by_kind[EventKind.SWEEP_CONFIRMED.value],
            raw_by_kind[EventKind.MSS_CORE_CONFIRMED.value],
            completed_index=completed_index,
            edge=non_nested_links["mss_prior_sweep"],
            comparison="mss_prior_sweep_composition_linked",
            event_lookup=observer.audit_store.get,
        ),
        "mss_prior_displacement_composition_linked": _v3_non_nested_partition(
            raw_by_kind[EventKind.DISPLACEMENT_OBSERVED.value],
            raw_by_kind[EventKind.MSS_CORE_CONFIRMED.value],
            completed_index=completed_index,
            edge=non_nested_links["mss_prior_displacement"],
            comparison="mss_prior_displacement_composition_linked",
            event_lookup=observer.audit_store.get,
        ),
        "fvg_prior_displacement_composition_linked": _v3_non_nested_partition(
            raw_by_kind[EventKind.DISPLACEMENT_OBSERVED.value],
            raw_by_kind[EventKind.FVG_CREATED.value],
            completed_index=completed_index,
            edge=non_nested_links["fvg_prior_displacement"],
            comparison="fvg_prior_displacement_composition_linked",
            event_lookup=observer.audit_store.get,
        ),
    }
    non_nested = {
        name: {
            "with": summarize(partition[0]),
            "without": summarize(partition[1]),
        }
        for name, partition in non_nested_populations.items()
    }
    chain_ledger.extend(
        row for partition in non_nested_populations.values() for row in partition[2]
    )

    controls_contract = manifest["control_definition"]
    treatments, treatment_exclusions = _v3_attach_match_strata(
        e1,
        rows_by_clock,
    )
    treatment_by_id = {str(item["event_id"]): item for item in treatments}
    quiet_candidates = _v3_quiet_candidates(enriched_rows)
    quiet_match = deterministic_maximum_cardinality_match(
        treatments,
        quiet_candidates,
        completed_index=completed_index,
        spec=_v3_match_spec(
            controls_contract,
            direction_policy=(ControlDirectionPolicy.INHERIT_TREATMENT_AFTER_KNOWN_AT),
        ),
    )
    non_sweep_candidates, non_sweep_exclusions = _v3_non_sweep_touch_candidates(
        e1_raw,
        raw_by_kind[EventKind.SWEEP_CONFIRMED.value],
        completed_index=completed_index,
        maximum_sweep_window=int(
            controls_contract["same_session_non_sweep_touch"][
                "maximum_sweep_window_completed_bars"
            ]
        ),
        rows_by_clock=rows_by_clock,
    )
    non_sweep_match = deterministic_maximum_cardinality_match(
        treatments,
        non_sweep_candidates,
        completed_index=completed_index,
        spec=_v3_match_spec(
            controls_contract,
            direction_policy=ControlDirectionPolicy.CANDIDATE_LOCAL,
        ),
    )
    pseudo_candidates, pseudo_ledger = _v3_pseudo_touch_candidates(
        pseudo_anchors,
        enriched_rows,
        completed_index=completed_index,
        control_contract=controls_contract,
        outcome_horizon=int(outcome_parameters["horizon"]),
    )
    pseudo_match = deterministic_maximum_cardinality_match(
        treatments,
        pseudo_candidates,
        completed_index=completed_index,
        spec=_v3_match_spec(
            controls_contract,
            direction_policy=ControlDirectionPolicy.CANDIDATE_LOCAL,
        ),
    )

    control_signals: dict[str, list[dict[str, Any]]] = {}
    control_pairs: dict[str, list[dict[str, Any]]] = {}
    unmatched_ledger: list[dict[str, Any]] = [
        {"control_type": "all", **item} for item in treatment_exclusions
    ]
    for name, match in (
        ("quiet_zero_event", quiet_match),
        ("same_session_non_sweep_touch", non_sweep_match),
        ("pseudo_level_touch", pseudo_match),
    ):
        signals, pairs, unmatched = _v3_materialize_match(name, match)
        control_signals[name] = signals
        control_pairs[name] = pairs
        unmatched_ledger.extend(unmatched)
    unmatched_ledger.extend(
        {"control_type": "same_session_non_sweep_touch", **item}
        for item in non_sweep_exclusions
    )

    time_shift_definition = controls_contract["forward_time_shift"]
    shift_build = build_forward_time_shift_controls(
        treatments,
        enriched_rows,
        completed_index=completed_index,
        spec=TimeShiftSpec(
            protocol_id="smc_signal_research_v3_forward_shift",
            forward_offsets_completed_bars=tuple(
                time_shift_definition["forward_offsets_completed_bars"]
            ),
            outcome_horizon_completed_bars=int(
                time_shift_definition["outcome_horizon_completed_bars"]
            ),
            embargo_completed_bars=int(time_shift_definition["embargo_completed_bars"]),
            require_same_session_phase=bool(
                time_shift_definition["require_same_session_phase"]
            ),
            exclude_other_treatment_windows=bool(
                time_shift_definition["exclude_other_treatment_windows"]
            ),
        ),
    )
    shift_signals: list[dict[str, Any]] = []
    shift_pairs: list[dict[str, Any]] = []
    for control in shift_build.controls:
        row = rows_by_clock[pd.Timestamp(control["known_at"])]
        treatment = treatment_by_id[str(control["source_event_id"])]
        stratum_exclusion = _v3_time_shift_stratum_exclusion(treatment, row, control)
        if stratum_exclusion is not None:
            unmatched_ledger.append(stratum_exclusion)
            continue
        signal = {
            **dict(control),
            **{field: row[field] for field in V3_MATCH_FIELDS},
            "timeframe": Timeframe.M5.value,
        }
        shift_signals.append(signal)
        shift_pairs.append(
            {
                "control_type": "forward_time_shift",
                "pair_id": f"pair:forward_time_shift:{treatment['event_id']}",
                "treatment_event_id": treatment["event_id"],
                "treatment_known_at": treatment["known_at"],
                "control_event_id": signal["event_id"],
                "control_known_at": signal["known_at"],
                "completed_bar_offset": signal["completed_bar_offset"],
                "control_direction": signal["direction"],
                "direction_known_at": signal["direction_known_at"],
                "stratum": {field: treatment[field] for field in V3_MATCH_FIELDS},
            }
        )
    control_signals["forward_time_shift"] = shift_signals
    control_pairs["forward_time_shift"] = shift_pairs
    unmatched_ledger.extend(
        {"control_type": "forward_time_shift", **dict(item)}
        for item in shift_build.exclusions
    )

    all_controls = [
        signal for name in V3_HOLM_FAMILY for signal in control_signals[name]
    ]
    _enrich_next_structural_context(all_controls, events)
    control_by_id = {str(item["event_id"]): item for item in all_controls}
    paired_outcome_ledger: list[dict[str, Any]] = []
    mcnemar_pairs: dict[str, list[tuple[bool | None, bool | None]]] = {
        name: [] for name in V3_HOLM_FAMILY
    }
    for name in V3_HOLM_FAMILY:
        for pair in control_pairs[name]:
            treatment = treatment_by_id[str(pair["treatment_event_id"])]
            control = control_by_id[str(pair["control_event_id"])]
            treatment_outcome = _outcome(
                treatment,
                enriched_rows,
                row_index,
                **outcome_parameters,
            )
            control_outcome = _outcome(
                control,
                enriched_rows,
                row_index,
                **outcome_parameters,
            )
            treatment_success = (
                None if treatment_outcome is None else treatment_outcome["success"]
            )
            control_success = (
                None if control_outcome is None else control_outcome["success"]
            )
            mcnemar_pairs[name].append((treatment_success, control_success))
            paired_outcome_ledger.append(
                {
                    **pair,
                    "treatment_outcome": treatment_outcome,
                    "control_outcome": control_outcome,
                }
            )

    mcnemar = {name: exact_mcnemar(mcnemar_pairs[name]) for name in V3_HOLM_FAMILY}
    holm_input = {
        name: (
            mcnemar[name].p_value if mcnemar[name].paired_n >= minimum_sample else None
        )
        for name in V3_HOLM_FAMILY
    }
    holm = holm_adjust_fixed_family(
        holm_input,
        family_order=V3_HOLM_FAMILY,
        alpha=float(manifest["inference_definition"]["alpha"]),
    )

    control_comparisons: dict[str, Any] = {}
    balance_ledger: list[dict[str, Any]] = []
    for name in V3_HOLM_FAMILY:
        pairs = control_pairs[name]
        paired_treatment_ids = {str(pair["treatment_event_id"]) for pair in pairs}
        requested = len(e1)
        control_comparisons[name] = {
            "requested": requested,
            "matched": len(pairs),
            "unmatched": requested - len(paired_treatment_ids),
            "treatment": summarize(
                [treatment_by_id[value] for value in sorted(paired_treatment_ids)]
            ),
            "control": summarize(control_signals[name]),
            "pooled_with_other_controls": False,
        }
        balance_ledger.append(
            _v3_balance_record(
                name,
                pairs,
                treatments=treatment_by_id,
                controls=control_by_id,
            )
        )

    stage_membership: dict[str, list[str]] = defaultdict(list)
    for stage, population in (
        ("E1_level_touch", e1_raw),
        ("E2_sweep", e2_raw),
        ("E3_sweep_displacement", e3_raw),
        ("E4_sweep_displacement_mss", e4_raw),
        ("E5_plus_fvg", e5_raw),
    ):
        for event in population:
            stage_membership[str(event["event_id"])].append(stage)
    event_ledger = (
        {
            **event,
            "stage_membership": tuple(stage_membership.get(str(event["event_id"]), ())),
            "outcome": _outcome(
                event,
                enriched_rows,
                row_index,
                **outcome_parameters,
            ),
        }
        for event in events
    )
    pseudo_construction_ledger = [
        {"status": "anchor_excluded", **item} for item in pseudo_anchor_exclusions
    ] + pseudo_ledger
    ledger_records = {
        "event_study": event_ledger,
        "control_pairs": paired_outcome_ledger,
        "control_unmatched": unmatched_ledger,
        "control_balance": balance_ledger,
        "source_chains": chain_ledger,
        "pseudo_construction": pseudo_construction_ledger,
    }
    if tuple(ledger_records) != _V3_LEDGER_LABELS:
        raise ResearchContractError("v3 output bundle and ledger writers disagree")
    comparison_contract = manifest.get("comparison_contract")
    comparison_mode = isinstance(comparison_contract, Mapping)
    ledger_metadata: dict[str, dict[str, Any]] = {}
    for label, records in ledger_records.items():
        path = _ledger_path(output, label)
        receipt = _write_jsonl(path, records, no_clobber=comparison_mode)
        ledger_metadata[label] = {
            "path": _display_path(path),
            "format": "canonical_json_lines_v1",
            **receipt,
        }

    status = _artifact_status(str(manifest["status"]), max_bars)
    if comparison_mode:
        artifact_classification = {
            "complete_registered_window": max_bars is None,
            "max_bars_smoke_limit": max_bars,
            "truncated_by_max_bars": truncated_by_max_bars,
            "inference_allowed": False,
            "parameter_change_allowed": False,
            "model_action_allowed": False,
            "max_bars_artifact_is_permanently_incomplete": max_bars is not None,
        }
        split_authority = {
            "comparison_role": contract.split_role,
            "window_id": comparison_contract.get("window_id"),
        }
        limitations = [
            "fixed historical comparison only; not independently validated",
            "strict ancestry populations may remain sparse and are not retuned",
            "shared-constituent-BAR composition is not a causal effect or semantic-event ancestry claim",
            "cross-pair outcome windows may overlap; exact McNemar/Holm p-values are descriptive and unvalidated",
            "OHLCV price geometry only; order-book mechanism is not evaluated",
            "no execution, fills, costs, stops, P&L, or model-action authority",
            *manifest.get("limitations", ()),
        ]
        validation_state = "fixed_historical_comparison_unvalidated"
    else:
        artifact_classification = {
            "complete_registered_window": max_bars is None,
            "max_bars_smoke_limit": max_bars,
            "truncated_by_max_bars": truncated_by_max_bars,
            "inference_allowed": False,
            "semantic_acceptance_allowed": False,
            "artifact_fit_allowed": False,
            "trading_authority": False,
            "out_of_sample_opened": False,
            "max_bars_artifact_is_permanently_incomplete": max_bars is not None,
        }
        split_authority = {
            "role": contract.split_role,
            "out_of_sample_opened": False,
        }
        limitations = [
            "development diagnostic only; not OOS and not validated",
            "strict ancestry populations may remain sparse and are not retuned",
            "shared-constituent-BAR composition is not a causal effect or semantic-event ancestry claim",
            "cross-pair outcome windows may overlap; exact McNemar/Holm p-values are descriptive and unvalidated",
            "OHLCV price geometry only; MBO mechanism not evaluated",
            "no execution, fills, costs, stops, P&L, or trading authority",
            *manifest.get("limitations", ()),
        ]
        validation_state = "development_diagnostic_unvalidated"
    result = {
        "schema_version": 3,
        "research_protocol_version": RESEARCH_PROTOCOL_V3,
        "status": status,
        "validation_state": validation_state,
        "artifact_classification": artifact_classification,
        "authority": manifest["authority"],
        "experiment_id": manifest["experiment_id"],
        "semantic_version": registry.semantic_version,
        "semantic_registry_identity": registry.identity,
        "manifest_path": _display_path(contract.manifest_path),
        "manifest_sha256": contract.manifest_sha256,
        "dataset": manifest["dataset_version"],
        "split_authority": split_authority,
        "coverage": {
            **actual_input_census,
            "diagnostic_real_rows": len(enriched_rows),
            "atomic_events": len(events),
            "atomic_event_counts": dict(Counter(item["kind"] for item in events)),
            "audit_events": len(observer.audit_store),
            "audit_fingerprint": observer.audit_store.fingerprint(),
        },
        "run_metadata": {
            "elapsed_seconds": time.monotonic() - started,
            "max_bars_smoke_limit": max_bars,
            "truncated_by_max_bars": truncated_by_max_bars,
        },
        "primary_outcome": manifest["primary_outcome"],
        "minimum_sample_requirement": minimum_sample,
        "nested_chain": nested,
        "non_nested_comparisons": non_nested,
        "control_comparisons": control_comparisons,
        "inference": {
            "claim_authority": "none_diagnostic_unvalidated",
            "cross_pair_outcome_window_policy": manifest["inference_definition"][
                "cross_pair_outcome_window_policy"
            ],
            "exact_mcnemar": {
                name: {
                    **to_primitive(mcnemar[name]),
                    "descriptive_exact_p_value": mcnemar[name].p_value,
                    # Missing/underpowered registered tests are p=1.
                    "p_value": holm.raw_p_values[name],
                    "minimum_sample_met": (mcnemar[name].paired_n >= minimum_sample),
                    "holm_input_p": holm.raw_p_values[name],
                }
                for name in V3_HOLM_FAMILY
            },
            "holm_fixed_family": to_primitive(holm),
        },
        "ledgers": ledger_metadata,
        "limitations": limitations,
    }
    result["result_identity"] = canonical_result_identity(result)
    _write_report_then_result(
        output,
        result,
        _report_v3(result),
        no_clobber=comparison_mode,
    )
    return result


def run(
    *,
    output: Path,
    max_bars: int | None = None,
    manifest_path: Path = MANIFEST_PATH,
    force: bool = False,
    contract_loader: Callable[..., Any] = load_frozen_research_contract,
    split_validator: Callable[[Any, Any], None] = validate_split_authority,
) -> dict[str, Any]:
    output = output.resolve()
    if any(
        output.name == stem or output.name.startswith(f"{stem}.")
        for stem in _IMMUTABLE_RESULT_STEMS
    ):
        raise ResearchContractError(
            "historical schema-1/protocol-v2 evidence is immutable"
        )
    historical = {
        (
            ROOT / "experiments/results/smc_semantic_v1_2024_01_signal_diagnostic.json"
        ).resolve(),
        (
            ROOT / "experiments/results/smc_semantic_v1_2024_01_signal_diagnostic.md"
        ).resolve(),
    }
    if output in historical or output.with_suffix(".md") in historical:
        raise ResearchContractError("historical schema-1 evidence is immutable")
    if (
        contract_loader is load_frozen_research_contract
        and split_validator is validate_split_authority
    ):
        # Preserve the original one-argument seam used by historical runner
        # tests and callers; the additive comparison wrapper is the only path
        # that supplies alternate validators.
        contract, registry, chain_edges, non_nested_links = (
            _load_contract_and_registry(manifest_path.resolve())
        )
    else:
        contract, registry, chain_edges, non_nested_links = (
            _load_contract_and_registry(
                manifest_path.resolve(),
                contract_loader=contract_loader,
                split_validator=split_validator,
            )
        )
    manifest = contract.payload
    _preflight_output_bundle(
        output,
        research_protocol_version=int(manifest.get("research_protocol_version", 2)),
        max_bars=max_bars,
        force=force,
    )
    outcome_parameters, minimum_sample = _validated_research_design(manifest)
    source = contract.dataset_path
    warmup = contract.warmup_start
    start = contract.diagnostic_start
    end = contract.diagnostic_end
    loaded = load_ohlcv(source, start=warmup, end=end)
    if loaded.warnings or not loaded.contract_selection_causal:
        raise RuntimeError("research requires the causal processed front")
    reader, observer = _build_eye(contract.model_path)
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    pseudo_anchors: list[dict[str, Any]] = []
    pseudo_anchor_exclusions: list[dict[str, Any]] = []
    completed_index: dict[pd.Timestamp, int] = {}
    seen_event_ids: set[str] = set()
    started = time.monotonic()
    emitted = 0
    truncated_by_max_bars = False
    last_observation_asof: pd.Timestamp | None = None
    first_diagnostic_asof: pd.Timestamp | None = None
    last_diagnostic_asof: pd.Timestamp | None = None
    diagnostic_completed_bars = 0
    diagnostic_real_rows = 0
    diagnostic_synthetic_bars = 0
    warmup_data_gap_resets = 0
    diagnostic_data_gap_resets = 0
    contract_changes = 0
    contracts_seen: set[tuple[str, int]] = set()
    synthetic_clocks: list[pd.Timestamp] = []
    for bar in iter_completed_bars(
        loaded.frame,
        allow_data_gap_reset=True,
    ):
        if max_bars is not None and emitted >= max_bars:
            truncated_by_max_bars = True
            break
        update = reader.on_bar(bar)
        observation = observer.observe(update)
        emitted += 1
        last_observation_asof = observation.asof
        snapshot = observation.market_snapshot
        if snapshot is None:
            raise RuntimeError("Eye did not publish hierarchical state")
        if snapshot.authority is not MarketSnapshotAuthority.ATOMIC_EVENT_REDUCER:
            raise ResearchContractError(
                "Phase5 requires atomic_event_reducer snapshot authority: "
                f"{observation.asof.isoformat()}={snapshot.authority.value}"
            )
        contracts_seen.add((str(snapshot.symbol), int(snapshot.instrument_id)))
        in_diagnostic = start <= observation.asof < end
        if "contract_change_history_reset" in update.anomalies:
            contract_changes += 1
        if "data_gap_history_reset" in update.anomalies:
            if in_diagnostic:
                diagnostic_data_gap_resets += 1
                raise ResearchContractError(
                    "diagnostic data-gap reset is forbidden by the frozen census"
                )
            if observation.asof < start:
                warmup_data_gap_resets += 1
        if not in_diagnostic:
            continue
        if first_diagnostic_asof is None:
            first_diagnostic_asof = observation.asof
        last_diagnostic_asof = observation.asof
        diagnostic_completed_bars += 1
        completed = update.completed_1m
        if completed.real_completed:
            diagnostic_real_rows += 1
        else:
            diagnostic_synthetic_bars += 1
            synthetic_clocks.append(observation.asof)
        if completed.real_completed and observation.asof not in completed_index:
            completed_index[observation.asof] = len(completed_index)
        canonical_atomic = tuple(
            semantic_event
            for semantic_event in observation.semantic_events_this_update
            if semantic_event.kind in ATOMIC_KINDS
        )
        atomic = tuple(
            semantic_event
            for semantic_event in canonical_atomic
            if _passes_research_selection(semantic_event)
        )
        for semantic_event in atomic:
            if semantic_event.event_id in seen_event_ids:
                raise RuntimeError("canonical semantic event repeated")
            seen_event_ids.add(semantic_event.event_id)
            source_event_kinds = {
                source_event_id: source_event.kind.value
                for source_event_id in semantic_event.source_event_ids
                if (source_event := observer.audit_store.get(source_event_id))
                is not None
            }
            events.append(
                _event_record(
                    semantic_event,
                    snapshot,
                    source_event_kinds=source_event_kinds,
                )
            )
        if not observation.frame(Timeframe.M1).ready:
            continue
        m1 = snapshot.timeframe_states[Timeframe.M1]
        atr = float(observation.frame(Timeframe.M1).metrics.get("atr", 0.0))
        if not completed.real_completed:
            continue
        normalized_bar_roots = tuple(
            event
            for event in observation.semantic_events_this_update
            if event.kind is EventKind.BAR_COMPLETED
            and event.origin is EventOrigin.NORMALIZED_DATA
            and event.timeframe is Timeframe.M1
            and event.known_at == observation.asof
            and event.event_time == observation.asof
            and event.evidence.get("real_completed") is True
            and event.evidence.get("clock_only") is False
        )
        if len(normalized_bar_roots) != 1:
            raise ResearchContractError(
                "ready real research row lacks one exact normalized M1 BAR root"
            )
        normalized_bar_root = normalized_bar_roots[0]
        normalized_evidence = normalized_bar_root.evidence
        expected_bar_facts = {
            "open": float(completed.open),
            "high": float(completed.high),
            "low": float(completed.low),
            "close": float(completed.close),
            "atr": atr,
        }
        if (
            normalized_evidence.get("symbol") != completed.symbol
            or normalized_evidence.get("instrument_id")
            != completed.instrument_id
            or any(
                normalized_evidence.get(name) != value
                for name, value in expected_bar_facts.items()
            )
        ):
            raise ResearchContractError(
                "normalized M1 BAR root disagrees with the completed research row"
            )
        row = {
            "asof": observation.asof,
            "bar_event_id": normalized_bar_root.event_id,
            "timeframe": Timeframe.M1.value,
            "tick_size": float(observer.config.tick_size),
            "open": float(completed.open),
            "high": float(completed.high),
            "low": float(completed.low),
            "close": float(completed.close),
            "atr": atr,
            "symbol": completed.symbol,
            "instrument_id": completed.instrument_id,
            "session_phase": snapshot.session.phase,
            "relative_volume": snapshot.session.relative_volume,
            "m1_direction": (
                None
                if m1.structure.internal_direction is None
                else m1.structure.internal_direction.value
            ),
            "nearest_distance_atr": _nearest_candidate_distance(
                snapshot,
                float(completed.close),
                atr,
            ),
            # Quiet controls exclude every canonical atom, including inactive
            # displacement observations that are outside the study selection.
            "atomic_event_count": len(canonical_atomic),
        }
        rows.append(row)
        for activation in atomic:
            if (
                activation.kind is not EventKind.DEALING_RANGE_ACTIVATED
                or activation.timeframe is not Timeframe.H1
            ):
                continue
            anchor_state = snapshot.timeframe_states[Timeframe.H1]
            active_range = anchor_state.range
            if (
                active_range.range_kind != "active_dealing_range"
                or active_range.range_id != activation.entity_id
                or active_range.low is None
                or active_range.high is None
                or not active_range.low <= snapshot.price <= active_range.high
            ):
                pseudo_anchor_exclusions.append(
                    {
                        "anchor_event_id": activation.event_id,
                        "known_at": activation.known_at,
                        "reason": "range_not_active_at_exact_post_event_snapshot",
                    }
                )
                continue
            known_real_levels: list[dict[str, Any]] = []
            unresolved_real_level_source = False
            for candidate in anchor_state.liquidity.candidates:
                source = (
                    None
                    if candidate.source_event_id is None
                    else observer.audit_store.get(candidate.source_event_id)
                )
                if source is None or source.known_at > activation.known_at:
                    unresolved_real_level_source = True
                    break
                known_real_levels.append(
                    {
                        "event_id": source.event_id,
                        "known_at": source.known_at,
                        "symbol": snapshot.symbol,
                        "instrument_id": snapshot.instrument_id,
                        "price": candidate.price,
                    }
                )
            if unresolved_real_level_source:
                pseudo_anchor_exclusions.append(
                    {
                        "anchor_event_id": activation.event_id,
                        "known_at": activation.known_at,
                        "reason": "known_real_level_source_unresolved",
                    }
                )
                continue
            pseudo_anchors.append(
                {
                    "event_id": activation.event_id,
                    "known_at": activation.known_at,
                    "symbol": snapshot.symbol,
                    "instrument_id": snapshot.instrument_id,
                    "known_real_levels": tuple(known_real_levels),
                    "snapshot": {
                        "snapshot_id": snapshot.fingerprint,
                        "asof": snapshot.asof,
                        "symbol": snapshot.symbol,
                        "instrument_id": snapshot.instrument_id,
                        "range_low": active_range.low,
                        "range_high": active_range.high,
                        "current_price": snapshot.price,
                    },
                }
            )
        if emitted % 5000 == 0:
            elapsed = max(1e-9, time.monotonic() - started)
            print(
                f"replay {emitted:,} bars; {emitted / elapsed:.1f} bars/s; "
                f"{len(events):,} atomic events",
                flush=True,
            )
    actual_input_census = {
        "emitted_bars_including_warmup": emitted,
        "diagnostic_completed_bars": diagnostic_completed_bars,
        "diagnostic_real_rows": diagnostic_real_rows,
        "diagnostic_ready_real_rows": len(rows),
        "diagnostic_synthetic_bars": diagnostic_synthetic_bars,
        "warmup_data_gap_resets": warmup_data_gap_resets,
        "diagnostic_data_gap_resets": diagnostic_data_gap_resets,
        "contract_changes": contract_changes,
        "first_diagnostic_asof": first_diagnostic_asof,
        "last_processed_asof": last_observation_asof,
        "last_diagnostic_asof": last_diagnostic_asof,
        "synthetic_clocks": synthetic_clocks,
        "contracts": [
            {"symbol": symbol, "instrument_id": instrument_id}
            for symbol, instrument_id in sorted(contracts_seen)
        ],
    }
    if max_bars is None:
        _assert_full_input_census(manifest["input_census"], actual_input_census)
    if not rows:
        raise RuntimeError("diagnostic replay produced no ready real 1m rows")
    rows.sort(key=lambda item: item["asof"])
    events.sort(key=lambda item: (item["known_at"], item["event_id"]))
    row_index = {row["asof"]: index for index, row in enumerate(rows)}
    _validate_directional_entry_rows(events, rows, row_index)

    def summarize(population: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return _summary(
            population,
            rows,
            row_index,
            outcome_parameters=outcome_parameters,
        )

    def structural_outcome(signal: Mapping[str, Any]) -> dict[str, Any] | None:
        return _outcome(signal, rows, row_index, **outcome_parameters)

    lineage_cache: dict[str, frozenset[str]] = {}
    source_lineage_cache: dict[str, frozenset[str]] = {}
    for event_record in events:
        lineage_tokens = tuple(
            sorted(
                resolve_lineage_tokens(
                    event_record["event_id"],
                    observer.audit_store.get,
                    memo=lineage_cache,
                )
            )
        )
        event_record["lineage_tokens"] = lineage_tokens
        source_lineage_tokens = resolve_source_lineage_tokens(
            event_record["event_id"],
            observer.audit_store.get,
            memo=source_lineage_cache,
        )
        event_record["source_lineage_tokens"] = tuple(sorted(source_lineage_tokens))
        event_timeframe = Timeframe(str(event_record["timeframe"]))
        event_record["constituent_bar_event_ids"] = tuple(
            sorted(
                ancestor.event_id
                for token in source_lineage_tokens
                if token.startswith("event:")
                and (ancestor := observer.audit_store.get(token.removeprefix("event:")))
                is not None
                and ancestor.origin is EventOrigin.NORMALIZED_DATA
                and ancestor.kind is EventKind.BAR_COMPLETED
                and ancestor.timeframe is event_timeframe
                and ancestor.evidence.get("real_completed") is True
                and ancestor.evidence.get("clock_only") is False
            )
        )
    if manifest.get("research_protocol_version") == RESEARCH_PROTOCOL_V3:
        return _run_v3_analysis(
            output=output,
            contract=contract,
            registry=registry,
            manifest=manifest,
            chain_edges=chain_edges,
            non_nested_links=non_nested_links,
            events=events,
            rows=rows,
            completed_index=completed_index,
            observer=observer,
            outcome_parameters=outcome_parameters,
            minimum_sample=minimum_sample,
            pseudo_anchors=pseudo_anchors,
            pseudo_anchor_exclusions=pseudo_anchor_exclusions,
            actual_input_census=actual_input_census,
            max_bars=max_bars,
            truncated_by_max_bars=truncated_by_max_bars,
            started=started,
        )
    touches = [
        event
        for event in events
        if event["kind"] == EventKind.LEVEL_TOUCHED.value
        and event["timeframe"] == Timeframe.M1.value
        and event["direction"] is not None
    ]
    sweeps = [
        event
        for event in events
        if event["kind"] == EventKind.SWEEP_CONFIRMED.value
        and event["timeframe"] == Timeframe.M1.value
        and event["direction"] is not None
    ]
    active_displacements = [
        event
        for event in events
        if event["kind"] == EventKind.DISPLACEMENT_OBSERVED.value
        and event["timeframe"] == Timeframe.M5.value
        and event["direction"] is not None
    ]
    mss = [
        event
        for event in events
        if event["kind"] == EventKind.MSS_CORE_CONFIRMED.value
        and event["timeframe"] == Timeframe.M5.value
        and event["direction"] is not None
    ]
    fvgs = [
        event
        for event in events
        if event["kind"] == EventKind.FVG_CREATED.value
        and event["timeframe"] == Timeframe.M5.value
        and event["direction"] is not None
    ]
    controls, matched_touches, control_pairs = _matched_controls(touches, rows)
    _enrich_next_structural_context([*events, *controls], events)
    e1_chains = {
        str(event["event_id"]): (str(event["event_id"]),) for event in matched_touches
    }
    e2, e2_chains, e2_ledger = _source_linked_stage(
        matched_touches,
        sweeps,
        completed_index=completed_index,
        maximum_completed_bars=chain_edges["E1_to_E2"]["maximum_completed_bars"],
        previous_kind=chain_edges["E1_to_E2"]["previous_kind"],
        previous_timeframe=chain_edges["E1_to_E2"]["previous_timeframe"],
        current_kind=chain_edges["E1_to_E2"]["current_kind"],
        current_timeframe=chain_edges["E1_to_E2"]["current_timeframe"],
        stage="E2_sweep",
        prior_chains=e1_chains,
    )
    e3, e3_chains, e3_ledger = _source_linked_stage(
        e2,
        active_displacements,
        completed_index=completed_index,
        maximum_completed_bars=chain_edges["E2_to_E3"]["maximum_completed_bars"],
        previous_kind=chain_edges["E2_to_E3"]["previous_kind"],
        previous_timeframe=chain_edges["E2_to_E3"]["previous_timeframe"],
        current_kind=chain_edges["E2_to_E3"]["current_kind"],
        current_timeframe=chain_edges["E2_to_E3"]["current_timeframe"],
        stage="E3_sweep_displacement",
        prior_chains=e2_chains,
    )
    e4, e4_chains, e4_ledger = _source_linked_stage(
        e3,
        mss,
        completed_index=completed_index,
        maximum_completed_bars=chain_edges["E3_to_E4"]["maximum_completed_bars"],
        previous_kind=chain_edges["E3_to_E4"]["previous_kind"],
        previous_timeframe=chain_edges["E3_to_E4"]["previous_timeframe"],
        current_kind=chain_edges["E3_to_E4"]["current_kind"],
        current_timeframe=chain_edges["E3_to_E4"]["current_timeframe"],
        stage="E4_sweep_displacement_mss",
        prior_chains=e3_chains,
    )
    e5, e5_chains, e5_ledger = _source_linked_stage(
        e4,
        fvgs,
        completed_index=completed_index,
        maximum_completed_bars=chain_edges["E4_to_E5"]["maximum_completed_bars"],
        previous_kind=chain_edges["E4_to_E5"]["previous_kind"],
        previous_timeframe=chain_edges["E4_to_E5"]["previous_timeframe"],
        current_kind=chain_edges["E4_to_E5"]["current_kind"],
        current_timeframe=chain_edges["E4_to_E5"]["current_timeframe"],
        stage="E5_plus_fvg",
        prior_chains=e4_chains,
    )
    e6 = [event for event in e5 if event["parent_bucket"] == "aligned_with_parent"]
    chain_ledger = [*e2_ledger, *e3_ledger, *e4_ledger, *e5_ledger]
    chain_ledger.extend(
        {
            "stage": "E6_plus_parent_alignment",
            "current_event_id": event["event_id"],
            "current_known_at": event["known_at"],
            "parent_bucket": event["parent_bucket"],
            "chain_event_ids": e5_chains[str(event["event_id"])],
        }
        for event in e6
    )
    stages: dict[str, Sequence[Mapping[str, Any]]] = {
        "C0_matched_control": controls,
        "E1_level_touch": matched_touches,
        "E2_sweep": e2,
        "E3_sweep_displacement": e3,
        "E4_sweep_displacement_mss": e4,
        "E5_plus_fvg": e5,
        "E6_plus_parent_alignment": e6,
    }
    nested: dict[str, dict[str, Any]] = {}
    prior_rate = None
    for name, stage_population in stages.items():
        value = summarize(stage_population)
        rate = value["laplace_success_rate"]
        value["delta_vs_prior"] = None if prior_rate is None else rate - prior_rate
        value["minimum_sample_met"] = value["signals"] >= minimum_sample
        nested[name] = value
        prior_rate = rate

    def split_summary(
        population: Sequence[Mapping[str, Any]],
        predicate,
    ) -> dict[str, Any]:
        yes = [item for item in population if predicate(item)]
        no = [item for item in population if not predicate(item)]
        return {
            "with": summarize(yes),
            "without": summarize(no),
        }

    def source_linked(
        prior: Sequence[Mapping[str, Any]],
        current: Mapping[str, Any],
        link: Mapping[str, Any],
    ) -> bool:
        return (
            find_prior_source_link(
                prior,
                current,
                completed_index=completed_index,
                kinds=frozenset({str(link["previous_kind"])}),
                maximum_completed_bars=int(link["maximum_completed_bars"]),
                timeframe=str(link["previous_timeframe"]),
            )
            is not None
        )

    non_nested = {
        "mss_prior_sweep_source_linked": split_summary(
            mss,
            lambda item: source_linked(
                sweeps, item, non_nested_links["mss_prior_sweep"]
            ),
        ),
        "mss_prior_displacement_source_linked": split_summary(
            mss,
            lambda item: source_linked(
                active_displacements,
                item,
                non_nested_links["mss_prior_displacement"],
            ),
        ),
        "fvg_prior_displacement_source_linked": split_summary(
            fvgs,
            lambda item: source_linked(
                active_displacements,
                item,
                non_nested_links["fvg_prior_displacement"],
            ),
        ),
    }
    directional_events = [event for event in events if event["direction"] is not None]
    event_strata = sorted(
        {(event["kind"], event["timeframe"]) for event in directional_events}
    )
    parent_relation_stratified = {
        f"{kind}|{timeframe}": {
            bucket: summarize(
                [
                    event
                    for event in directional_events
                    if event["kind"] == kind
                    and event["timeframe"] == timeframe
                    and event["parent_bucket"] == bucket
                ],
            )
            for bucket in (
                "aligned_with_parent",
                "against_parent_but_parent_intact",
                "parent_neutral",
                "after_parent_invalidation",
                "parent_unresolved",
            )
        }
        for kind, timeframe in event_strata
    }
    session_context_stratified = {
        f"{kind}|{timeframe}": {
            phase: summarize(
                [
                    event
                    for event in directional_events
                    if event["kind"] == kind
                    and event["timeframe"] == timeframe
                    and event["session_phase"] == phase
                ],
            )
            for phase in sorted(
                {
                    event["session_phase"]
                    for event in directional_events
                    if event["kind"] == kind and event["timeframe"] == timeframe
                }
            )
        }
        for kind, timeframe in event_strata
    }
    range_invalidation_variant_studies = {
        variant: summarize(
            [
                item
                for item in events
                if item["kind"] == EventKind.DEALING_RANGE_INVALIDATED.value
                and item["event_variant"] == variant
            ]
        )
        for variant in RANGE_INVALIDATION_VARIANT_CONTRACT
    }
    atomic_studies: dict[str, Any] = {}
    for kind in sorted(ATOMIC_KINDS, key=lambda item: item.value):
        if kind is EventKind.DEALING_RANGE_INVALIDATED:
            atomic_studies[kind.value] = {
                "pooling": RANGE_INVALIDATION_POOLING,
                "variants": range_invalidation_variant_studies,
            }
            continue
        atomic_studies[kind.value] = summarize(
            [item for item in events if item["kind"] == kind.value]
        )
    stage_membership: dict[str, list[str]] = defaultdict(list)
    for stage, stage_population in stages.items():
        for stage_record in stage_population:
            stage_membership[str(stage_record["event_id"])].append(stage)
    event_ledger = (
        {
            **event,
            "stage_membership": tuple(stage_membership.get(event["event_id"], ())),
            "outcome": structural_outcome(event),
        }
        for event in events
    )
    treatment_by_id = {str(event["event_id"]): event for event in matched_touches}
    control_by_id = {str(event["event_id"]): event for event in controls}
    control_pair_ledger = [
        {
            **pair,
            "treatment_outcome": structural_outcome(
                treatment_by_id[str(pair["treatment_event_id"])]
            ),
            "control_outcome": structural_outcome(
                control_by_id[str(pair["control_event_id"])]
            ),
        }
        for pair in control_pairs
    ]
    ledger_records = {
        "event_study": event_ledger,
        "control_pairs": control_pair_ledger,
        "source_chains": chain_ledger,
    }
    if tuple(ledger_records) != _V2_LEDGER_LABELS:
        raise ResearchContractError("v2 output bundle and ledger writers disagree")
    ledger_metadata: dict[str, dict[str, Any]] = {}
    for label, records in ledger_records.items():
        path = _ledger_path(output, label)
        receipt = _write_jsonl(path, records)
        ledger_metadata[label] = {
            "path": _display_path(path),
            "format": "canonical_json_lines_v1",
            **receipt,
        }

    result = {
        "schema_version": 2,
        "status": _artifact_status(manifest["status"], max_bars),
        "manifest_status": manifest["status"],
        "authority": manifest["authority"],
        "experiment_id": manifest["experiment_id"],
        "semantic_version": registry.semantic_version,
        "semantic_registry_identity": registry.identity,
        "manifest_path": _display_path(contract.manifest_path),
        "manifest_sha256": contract.manifest_sha256,
        "identity_bindings": manifest["identity_bindings"],
        "dataset": manifest["dataset_version"],
        "split_authority": {
            "role": contract.split_role,
            "split_registry_path": _display_path(contract.split_registry_path),
            "split_registry_sha256": contract.split_registry_sha256,
            "out_of_sample_opened": False,
        },
        "coverage": {
            "warmup_start": warmup,
            "diagnostic_start": start,
            "diagnostic_end_exclusive": end,
            "emitted_rows_including_warmup": emitted,
            "diagnostic_real_rows": len(rows),
            "atomic_events": len(events),
            "atomic_event_counts": dict(Counter(item["kind"] for item in events)),
            "audit_events": len(observer.audit_store),
            "audit_fingerprint": observer.audit_store.fingerprint(),
            "first_ready_asof": rows[0]["asof"],
            "last_ready_asof": rows[-1]["asof"],
            "input_census": actual_input_census,
            "registered_input_census": manifest["input_census"],
        },
        "run_metadata": {
            "elapsed_seconds": time.monotonic() - started,
            "max_bars_smoke_limit": max_bars,
            "truncated_by_max_bars": truncated_by_max_bars,
            "cyclic_gc_enabled_during_run": gc.isenabled(),
        },
        "primary_outcome": manifest["primary_outcome"],
        "secondary_outcomes": manifest["secondary_outcomes"],
        "registered_atomic_population": manifest["event_definition"][
            "atomic_population"
        ],
        "research_selection": manifest["event_definition"]["research_selection"],
        "range_invalidation_variants": manifest["event_definition"][
            "range_invalidation_variants"
        ],
        "range_invalidation_pooling": manifest["event_definition"][
            "range_invalidation_pooling"
        ],
        "signal_direction_assignment": manifest["event_definition"][
            "signal_direction_assignment"
        ],
        "state_projection_persistence": manifest["event_definition"][
            "state_projection_persistence"
        ],
        "snapshot_authority": manifest["event_definition"]["snapshot_authority"],
        "parent_relation_priority": manifest["event_definition"][
            "parent_relation_priority"
        ],
        "minimum_sample_requirement": minimum_sample,
        "atomic_event_studies": atomic_studies,
        "nested_chain": nested,
        "non_nested_comparisons": non_nested,
        "parent_relation_stratified": parent_relation_stratified,
        "session_context_stratified": session_context_stratified,
        "matched_control_coverage": {
            "requested": len(touches),
            "matched": len(controls),
            "unmatched": len(touches) - len(controls),
            "treatment_cohort": len(matched_touches),
            "same_cohort_enforced": len(controls) == len(matched_touches),
        },
        "ledgers": ledger_metadata,
        "limitations": [
            "diagnostic split, not out-of-sample",
            "descriptive estimates only; Holm inference not run",
            "no pseudo-level or time-shifted control in this first pass",
            "OHLCV price geometry only; MBO mechanism not evaluated",
            "no execution, fills, costs, stops, P&L, or trading authority",
            *manifest.get("limitations", ()),
        ],
    }
    result["result_identity"] = canonical_result_identity(result)
    _write_report_then_result(
        output,
        result,
        _report(result),
        no_clobber=False,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-bars", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.max_bars is not None and args.max_bars < 1:
        raise ValueError("--max-bars must be positive")
    result = run(
        output=output,
        max_bars=args.max_bars,
        manifest_path=args.manifest.resolve(),
        force=args.force,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": result["coverage"]["diagnostic_real_rows"],
                "events": result["coverage"]["atomic_events"],
                "status": result["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
