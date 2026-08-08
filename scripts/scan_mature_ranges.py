#!/usr/bin/env python3
"""Lightweight, outcome-blind coverage scan for natural Group 4 ranges.

The scan reuses the causal reader and observer but deliberately omits the
Brain, actions, PnL, MBO and per-minute trace output.  A completed window is
one resumable unit and produces one small JSON result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.artifact_stream import atomic_bytes  # noqa: E402
from smc_trader.causal import CausalMarketReader  # noqa: E402
from smc_trader.group4 import Group4Protocol  # noqa: E402
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.model import (  # noqa: E402
    DealingRangeLifecycle,
    DealingRangeState,
    GROUP4_HARD_BOUNDARY_REASONS,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    ManipulationLifecycle,
    ManipulationState,
    SupportResistanceLifecycle,
    SupportResistanceState,
    Timeframe,
    to_primitive,
)
from smc_trader.observation import (  # noqa: E402
    CausalObserver,
    ObserverConfig,
)
from smc_trader.scene_graph import parse_scale_specs  # noqa: E402
from smc_trader.validation import load_validation_protocol  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/data_splits.json"
DEFAULT_OUTPUT = ROOT / "outputs/development/mature_range_coverage"
CANONICAL_CALIBRATION_PROFILE = (
    "group4_natural_authority_2023_full_year"
)
LIVE_ZONE_STATES = {
    SupportResistanceLifecycle.ACTIVE,
    SupportResistanceLifecycle.TESTED,
}
GATE_NAMES = (
    "duration",
    "bilateral_touches",
    "midpoint_crossing",
    "inside_close_fraction",
    "width",
    "compression",
)
STRATA = (
    "mature_recognized",
    "obvious_immature_rejected",
    "near_mature_single_gate",
    "forming_reasonable_broken",
    "natural_mature_range_manipulation",
)
ELIGIBLE_MANIPULATION_SOURCE_KINDS = {
    "equal_highs",
    "equal_lows",
    "range_boundary",
}
MANIPULATION_FUNNEL_FIELDS = (
    "visible_eligible_sources",
    "crossed_sources",
    "swept_created",
    "reaccepted",
    "accepted_outside",
    "deadline_censored",
    "hard_boundary_censored",
    "right_censored",
    "ambiguous_dual_side",
    "atr_unready",
)
MANIPULATION_OUTCOMES = (
    "reaccepted",
    "accepted_outside",
    "deadline_censored",
    "hard_boundary_censored",
    "right_censored",
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _coverage_payload(
    payload: Mapping[str, Any],
    config: Path,
    *,
    profile: str,
) -> dict[str, Any]:
    """Resolve one exact registered authority profile; dates are never CLI input."""

    sources = payload.get("sources")
    profiles = payload.get("authority_scan_profiles")
    if not isinstance(sources, Mapping) or not isinstance(profiles, Mapping):
        raise ValueError("data splits lack source or authority-scan profiles")
    ohlcv = sources.get("ohlcv")
    coverage = profiles.get(profile)
    if not isinstance(ohlcv, Mapping):
        raise ValueError("data splits lack the registered OHLCV source")
    if not isinstance(coverage, Mapping):
        raise ValueError(f"unknown registered authority profile: {profile}")
    protocols = coverage.get("protocols")
    if not isinstance(protocols, Mapping):
        raise ValueError(f"authority profile lacks protocol bindings: {profile}")
    windows = coverage.get("windows")
    fixed_window_set = coverage.get("fixed_window_set")
    if windows is None and fixed_window_set is not None:
        fixed = payload.get("fixed_development_windows")
        registered = (
            fixed.get(fixed_window_set)
            if isinstance(fixed, Mapping)
            else None
        )
        if not isinstance(registered, Mapping):
            raise ValueError(
                f"authority profile references an unknown window set: {profile}"
            )
        windows = registered.get("windows")
    return {
        "schema_version": payload.get("schema_version"),
        "profile": profile,
        "purpose": coverage.get("purpose"),
        "source": ohlcv.get("path"),
        "source_sha256": ohlcv.get("sha256"),
        "validation_protocol": str(config.relative_to(ROOT)),
        "model_config": coverage.get("model_config"),
        "protocols": dict(protocols),
        "warmup_calendar_days": coverage.get("warmup_calendar_days"),
        "timezone": coverage.get("timezone"),
        "allowed_ohlcv_role": coverage.get("allowed_ohlcv_role"),
        "registered_calibration_exception": coverage.get(
            "registered_calibration_exception"
        ),
        "allow_data_gap_reset": coverage.get("allow_data_gap_reset"),
        "threshold_search": coverage.get("threshold_search"),
        "outcome_fields_used": coverage.get("outcome_fields_used"),
        "pnl_used": coverage.get("pnl_used"),
        "mbo_used": coverage.get("mbo_used"),
        "include_all_pool_source_timeframes": coverage.get(
            "include_all_pool_source_timeframes"
        ),
        "pool_source_timeframes": coverage.get(
            "pool_source_timeframes"
        ),
        "permanent_result_path": coverage.get("permanent_result_path"),
        "windows": windows,
    }


def _write_json(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(
            to_primitive(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    atomic_bytes(path, encoded)


def _aware(value: Any, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError(f"{name} must be timezone aware")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise RuntimeError("authority scan could not resolve one Git commit")
    return value


def _calendar_warmup_start(
    start: pd.Timestamp,
    *,
    days: int,
    timezone: str,
) -> pd.Timestamp:
    return start.tz_convert(timezone) - pd.DateOffset(days=days)


def _source_kind(item: LiquidityInventoryItem) -> str:
    if item.kind in {"equal_highs", "equal_lows"}:
        return "formed_liquidity_pool"
    if item.kind == "range_boundary":
        return "mature_range_boundary"
    raise ValueError("inventory item is not a Group 4 manipulation source")


def _source_key_from_item(
    item: LiquidityInventoryItem,
) -> tuple[str, str, str]:
    return (_source_kind(item), item.timeframe.value, item.side)


def _source_key_from_state(
    state: ManipulationState,
) -> tuple[str, str, str]:
    return (
        state.source_kind,
        state.source_timeframe.value,
        state.side,
    )


def _eligible_sources(
    items: Iterable[LiquidityInventoryItem],
    *,
    candle: Any,
) -> tuple[LiquidityInventoryItem, ...]:
    """Mirror Group4's public prior-inventory admission predicate."""

    return tuple(
        item
        for item in items
        if (
            item.lifecycle is LiquidityInventoryLifecycle.VISIBLE
            and item.kind in ELIGIBLE_MANIPULATION_SOURCE_KINDS
            and item.confirmed_at <= candle.start
        )
    )


def _crossed_sources(
    items: Iterable[LiquidityInventoryItem],
    *,
    candle: Any,
) -> tuple[LiquidityInventoryItem, ...]:
    return tuple(
        item
        for item in items
        if (
            candle.high > item.upper_bound
            if item.side == "above"
            else candle.low < item.lower_bound
        )
    )


def _build_observer(
    model: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[CausalMarketReader, CausalObserver]:
    """Build the production Observer path, limited to Groups 1-2 and 4."""

    scale_specs = parse_scale_specs(model.get("scales"))
    observer_raw = model.get("observer")
    if not isinstance(observer_raw, Mapping):
        raise ValueError("model observer configuration is missing")
    minimum = observer_raw.get("minimum_bars", {})
    if not isinstance(minimum, Mapping):
        raise ValueError("model observer minimum bars are invalid")
    protocols = payload["protocols"]
    observer = CausalObserver(
        ObserverConfig(
            atr_period=int(observer_raw.get("atr_period", 14)),
            memory_events=int(observer_raw.get("memory_events", 512)),
            minimum_bars={
                timeframe: int(minimum.get(timeframe.value, default))
                for timeframe, default in {
                    Timeframe.H4: 16,
                    Timeframe.H1: 24,
                    Timeframe.M15: 24,
                    Timeframe.M5: 24,
                    Timeframe.M1: 30,
                }.items()
                if any(
                    spec.enabled
                    and spec.native_timeframe is timeframe
                    for spec in scale_specs
                )
            },
            tick_size=float(model.get("tick_size", 0.25)),
            point_value=float(model.get("point_value", 20.0)),
            structure_protocol=str(ROOT / str(protocols["group12"])),
            liquidity_protocol=str(ROOT / str(protocols["group12"])),
            group4_protocol=str(ROOT / str(protocols["group4"])),
            scale_specs=scale_specs,
            project_scene_graph=False,
            materialize_event_view=False,
        )
    )
    return CausalMarketReader(scale_specs=scale_specs), observer


def maturity_gate_margins(
    state: DealingRangeState,
    protocol: Group4Protocol,
) -> dict[str, float]:
    """Return signed, dimensionless margins for the frozen maturity gates."""

    return {
        "duration": (
            state.candidate_real_h1_bars
            - protocol.minimum_candidate_real_h1_bars
        )
        / protocol.minimum_candidate_real_h1_bars,
        "bilateral_touches": (
            min(state.lower_touch_count, state.upper_touch_count)
            - protocol.minimum_boundary_touches_each
        )
        / protocol.minimum_boundary_touches_each,
        "midpoint_crossing": (
            state.midpoint_crossings - protocol.minimum_midpoint_crossings
        )
        / protocol.minimum_midpoint_crossings,
        "inside_close_fraction": (
            state.inside_close_fraction
            - protocol.minimum_inside_close_fraction
        )
        / protocol.minimum_inside_close_fraction,
        "width": (
            protocol.maximum_width_atr_at_formation
            - state.width_atr_at_formation
        )
        / protocol.maximum_width_atr_at_formation,
        "compression": (
            protocol.maximum_compression_ratio - state.compression_ratio
        )
        / protocol.maximum_compression_ratio,
    }


def unmet_maturity_gates(
    state: DealingRangeState,
    protocol: Group4Protocol,
) -> tuple[str, ...]:
    margins = maturity_gate_margins(state, protocol)
    return tuple(name for name in GATE_NAMES if margins[name] < 0.0)


def geometrically_valid_source_pairs(
    zones: Sequence[SupportResistanceState],
    close: float,
) -> tuple[tuple[str, str], ...]:
    """Describe live H1 source-pair coverage without imitating range identity."""

    supports = tuple(
        zone
        for zone in zones
        if zone.side == "support" and zone.lifecycle in LIVE_ZONE_STATES
    )
    resistances = tuple(
        zone
        for zone in zones
        if zone.side == "resistance" and zone.lifecycle in LIVE_ZONE_STATES
    )
    return tuple(
        sorted(
            {
                (lower.zone_id, upper.zone_id)
                for lower in supports
                for upper in resistances
                if lower.upper_bound < upper.lower_bound
                and lower.lower_bound <= close <= upper.upper_bound
            }
        )
    )


def _case_key(record: Mapping[str, Any]) -> str:
    raw = "|".join(
        (
            str(record["entity_kind"]),
            str(record["entity_id"]),
            str(record["focus_clock"]),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _opaque_case_id(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        f"group4-coverage-blind-v1|{_case_key(record)}".encode("utf-8")
    ).hexdigest()[:16]


def select_stratified_cases(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    """Select deterministic outcome-blind representatives; empty is valid."""

    values = [dict(record) for record in records]
    selected: dict[str, dict[str, Any] | None] = {}
    mature = [item for item in values if item.get("case_class") == "mature"]
    obvious = [
        item
        for item in values
        if item.get("case_class") == "forming_terminal"
        and len(item.get("unmet_gates", ())) >= 3
    ]
    near = [
        item
        for item in values
        if item.get("case_class") == "forming_terminal"
        and int(item.get("candidate_real_h1_bars", 0)) >= 8
        and len(item.get("unmet_gates", ())) == 1
    ]
    reasonable = [
        item
        for item in values
        if item.get("case_class") == "forming_terminal"
        and item.get("transition_reason") == "close_beyond_frozen_range"
    ]
    manipulations = [
        item
        for item in values
        if item.get("case_class") == "range_boundary_manipulation"
    ]

    def pick(candidates: Sequence[dict[str, Any]], key: Any) -> dict[str, Any] | None:
        if not candidates:
            return None
        record = dict(sorted(candidates, key=key)[0])
        record["opaque_case_id"] = _opaque_case_id(record)
        return record

    selected["mature_recognized"] = pick(
        mature,
        lambda item: (-float(item.get("minimum_gate_margin", -1e9)), _case_key(item)),
    )
    selected["obvious_immature_rejected"] = pick(
        obvious,
        lambda item: (-len(item.get("unmet_gates", ())), _case_key(item)),
    )
    selected["near_mature_single_gate"] = pick(
        near,
        lambda item: (float(item.get("total_gate_shortfall", 1e9)), _case_key(item)),
    )
    selected["forming_reasonable_broken"] = pick(
        reasonable,
        lambda item: _case_key(item),
    )
    selected["natural_mature_range_manipulation"] = pick(
        manipulations,
        lambda item: (
            0 if item.get("lifecycle") == "reaccepted" else 1,
            _case_key(item),
        ),
    )
    if tuple(selected) != STRATA:
        raise AssertionError("stratified case schema drifted")
    return selected


def _compact_selected_cases(
    selected: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, dict[str, Any] | None]:
    fields = (
        "opaque_case_id",
        "window_id",
        "entity_kind",
        "entity_id",
        "focus_clock",
        "lifecycle",
        "source_kind",
        "source_id",
        "side",
    )
    return {
        stratum: (
            None
            if record is None
            else {
                field: record[field]
                for field in fields
                if field in record
            }
        )
        for stratum, record in selected.items()
    }


def _range_record(
    state: DealingRangeState,
    protocol: Group4Protocol,
    *,
    window_id: str,
    case_class: str,
    focus_clock: pd.Timestamp,
) -> dict[str, Any]:
    margins = maturity_gate_margins(state, protocol)
    unmet = unmet_maturity_gates(state, protocol)
    return {
        "window_id": window_id,
        "case_class": case_class,
        "entity_kind": "dealing_range",
        "entity_id": state.range_id,
        "focus_clock": focus_clock,
        "lifecycle": state.lifecycle.value,
        "transition_reason": state.transition_reason,
        "formed_at": state.formed_at,
        "mature_at": state.mature_at,
        "broken_at": state.broken_at,
        "lower_bound": state.lower_bound,
        "upper_bound": state.upper_bound,
        "midpoint": state.midpoint,
        "lower_source_zone_id": state.lower_source_zone_id,
        "upper_source_zone_id": state.upper_source_zone_id,
        "candidate_real_h1_bars": state.candidate_real_h1_bars,
        "lower_touch_count": state.lower_touch_count,
        "upper_touch_count": state.upper_touch_count,
        "midpoint_crossings": state.midpoint_crossings,
        "inside_close_fraction": state.inside_close_fraction,
        "width_atr_at_formation": state.width_atr_at_formation,
        "compression_ratio": state.compression_ratio,
        "gate_margins": margins,
        "minimum_gate_margin": min(margins.values()),
        "unmet_gates": unmet,
        "total_gate_shortfall": sum(-value for value in margins.values() if value < 0.0),
    }


def _manipulation_record(
    state: ManipulationState,
    *,
    window_id: str,
    focus_clock: pd.Timestamp,
) -> dict[str, Any]:
    return {
        "window_id": window_id,
        "case_class": "range_boundary_manipulation",
        "entity_kind": "manipulation",
        "entity_id": state.manipulation_id,
        "focus_clock": focus_clock,
        "lifecycle": state.lifecycle.value,
        "transition_reason": state.transition_reason,
        "source_kind": state.source_kind,
        "source_id": state.source_id,
        "source_inventory_item_id": state.source_inventory_item_id,
        "side": state.side,
        "swept_at": state.swept_at,
        "reaccepted_at": state.reaccepted_at,
        "accepted_outside_at": state.accepted_outside_at,
        "sweep_extreme": state.sweep_extreme,
        "reentry_price": state.reentry_price,
    }


@dataclass
class CoverageAccumulator:
    window_id: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    protocol: Group4Protocol
    coverage_start: pd.Timestamp | None = None
    unique_pairs: set[tuple[str, str]] = field(default_factory=set)
    pair_clock_count: int = 0
    formed_ids: set[str] = field(default_factory=set)
    mature_ids: set[str] = field(default_factory=set)
    warmup_mature_ids: set[str] = field(default_factory=set)
    broken_ids: set[str] = field(default_factory=set)
    latest_ranges: dict[str, DealingRangeState] = field(default_factory=dict)
    terminal_reasons: Counter[str] = field(default_factory=Counter)
    forming_terminal_reasons: Counter[str] = field(default_factory=Counter)
    mature_terminal_reasons: Counter[str] = field(default_factory=Counter)
    unmet_gate_counts: Counter[str] = field(default_factory=Counter)
    inventory_ids_observed: set[str] = field(default_factory=set)
    inventory_ids_formed: set[str] = field(default_factory=set)
    inventory_lifecycles: dict[str, str] = field(default_factory=dict)
    latest_manipulations: dict[str, ManipulationState] = field(
        default_factory=dict
    )
    source_metric_ids: dict[
        tuple[str, str, str],
        dict[str, set[str]],
    ] = field(
        default_factory=dict
    )
    source_items: dict[str, LiquidityInventoryItem] = field(default_factory=dict)
    real_completed_bars: int = 0
    synthetic_completed_bars: int = 0
    data_gap_resets: int = 0
    contract_resets: int = 0
    case_records: list[dict[str, Any]] = field(default_factory=list)
    _seen_range_events: set[tuple[str, str, pd.Timestamp]] = field(default_factory=set)
    _seen_manipulation_events: set[tuple[str, str, pd.Timestamp]] = field(default_factory=set)

    def _in_window(self, clock: pd.Timestamp) -> bool:
        return self.start <= clock < self.end_exclusive

    def _in_coverage(self, clock: pd.Timestamp) -> bool:
        lower = self.start if self.coverage_start is None else self.coverage_start
        return lower <= clock < self.end_exclusive

    def observe_source_pairs(
        self,
        zones: Sequence[SupportResistanceState],
        close: float,
    ) -> None:
        pairs = geometrically_valid_source_pairs(zones, close)
        self.unique_pairs.update(pairs)
        self.pair_clock_count += len(pairs)

    def observe_range(self, state: DealingRangeState) -> None:
        self.latest_ranges[state.range_id] = state
        if self._in_window(state.formed_at):
            event = (state.range_id, "forming", state.formed_at)
            if event not in self._seen_range_events:
                self._seen_range_events.add(event)
                self.formed_ids.add(state.range_id)
        if state.mature_at is not None and self._in_window(state.mature_at):
            event = (state.range_id, "mature", state.mature_at)
            if event not in self._seen_range_events:
                self._seen_range_events.add(event)
                self.mature_ids.add(state.range_id)
                self.case_records.append(
                    _range_record(
                        state,
                        self.protocol,
                        window_id=self.window_id,
                        case_class="mature",
                        focus_clock=state.mature_at,
                    )
                )
        elif (
            state.mature_at is not None
            and self._in_coverage(state.mature_at)
            and state.mature_at < self.start
        ):
            event = (state.range_id, "warmup_mature", state.mature_at)
            if event not in self._seen_range_events:
                self._seen_range_events.add(event)
                self.warmup_mature_ids.add(state.range_id)
                record = _range_record(
                    state,
                    self.protocol,
                    window_id=self.window_id,
                    case_class="mature",
                    focus_clock=state.mature_at,
                )
                record["coverage_phase"] = "warmup"
                self.case_records.append(record)
        if state.broken_at is None or not self._in_window(state.broken_at):
            return
        event = (state.range_id, "broken", state.broken_at)
        if event in self._seen_range_events:
            return
        self._seen_range_events.add(event)
        self.broken_ids.add(state.range_id)
        reason = str(state.transition_reason or "unknown")
        self.terminal_reasons[reason] += 1
        if state.mature_at is None:
            self.forming_terminal_reasons[reason] += 1
            unmet = unmet_maturity_gates(state, self.protocol)
            self.unmet_gate_counts.update(unmet)
            self.case_records.append(
                _range_record(
                    state,
                    self.protocol,
                    window_id=self.window_id,
                    case_class="forming_terminal",
                    focus_clock=state.broken_at,
                )
            )
        else:
            self.mature_terminal_reasons[reason] += 1

    def observe_inventory(self, items: Iterable[Any]) -> None:
        for item in items:
            if item.kind != "range_boundary":
                continue
            self.inventory_ids_observed.add(item.item_id)
            if self._in_window(item.confirmed_at):
                self.inventory_ids_formed.add(item.item_id)
            self.inventory_lifecycles[item.item_id] = item.lifecycle.value

    def _metric_set(
        self,
        key: tuple[str, str, str],
        metric: str,
    ) -> set[str]:
        values = self.source_metric_ids.setdefault(
            key,
            {name: set() for name in MANIPULATION_FUNNEL_FIELDS},
        )
        return values[metric]

    def observe_source_inputs(
        self,
        items: Iterable[LiquidityInventoryItem],
        *,
        candle: Any,
    ) -> None:
        eligible = _eligible_sources(items, candle=candle)
        crossed = _crossed_sources(eligible, candle=candle)
        for item in eligible:
            self.source_items[item.item_id] = item
            self._metric_set(
                _source_key_from_item(item),
                "visible_eligible_sources",
            ).add(item.item_id)
        for item in crossed:
            self._metric_set(
                _source_key_from_item(item),
                "crossed_sources",
            ).add(item.item_id)

    def observe_unclassified_sources(
        self,
        item_ids: Iterable[str],
        *,
        metric: str,
        inventory: Mapping[str, LiquidityInventoryItem],
    ) -> None:
        if metric not in {"ambiguous_dual_side", "atr_unready"}:
            raise ValueError("invalid unclassified source metric")
        for item_id in item_ids:
            item = inventory.get(item_id) or self.source_items.get(item_id)
            if item is None:
                raise RuntimeError(
                    "Group 4 unclassified source is absent from the "
                    "production inventory"
                )
            self.source_items[item.item_id] = item
            self._metric_set(
                _source_key_from_item(item),
                metric,
            ).add(item.item_id)

    def observe_manipulation(self, state: ManipulationState) -> None:
        self.latest_manipulations[state.manipulation_id] = state
        event_clocks = (
            ("swept", state.swept_at),
            ("reaccepted", state.reaccepted_at),
            ("accepted_outside", state.accepted_outside_at),
        )
        for lifecycle, clock in event_clocks:
            if clock is None or not self._in_window(clock):
                continue
            event = (state.manipulation_id, lifecycle, clock)
            if event in self._seen_manipulation_events:
                continue
            self._seen_manipulation_events.add(event)
            if (
                state.source_kind == "mature_range_boundary"
                and lifecycle in {"swept", "reaccepted"}
            ):
                self.case_records.append(
                    _manipulation_record(
                        state,
                        window_id=self.window_id,
                        focus_clock=clock,
                    )
                )

    def observe_bar(self, *, real_completed: bool) -> None:
        if real_completed:
            self.real_completed_bars += 1
        else:
            self.synthetic_completed_bars += 1

    def observe_resets(self, anomalies: Iterable[str]) -> None:
        values = set(anomalies)
        self.data_gap_resets += int("data_gap_history_reset" in values)
        self.contract_resets += int(
            "contract_change_history_reset" in values
        )

    def _manipulation_result(
        self,
    ) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, Any]]:
        created = {
            identity
            for identity, state in self.latest_manipulations.items()
            if self._in_window(state.swept_at)
        }
        outcomes = {name: set() for name in MANIPULATION_OUTCOMES}
        for identity in created:
            state = self.latest_manipulations[identity]
            if state.lifecycle is ManipulationLifecycle.REACCEPTED:
                outcome = "reaccepted"
            elif state.lifecycle is ManipulationLifecycle.ACCEPTED_OUTSIDE:
                outcome = "accepted_outside"
            elif state.censored_at is not None and state.deadline_elapsed:
                outcome = "deadline_censored"
            elif (
                state.censored_at is not None
                and state.transition_reason in GROUP4_HARD_BOUNDARY_REASONS
            ):
                outcome = "hard_boundary_censored"
            else:
                outcome = "right_censored"
            outcomes[outcome].add(identity)

        union = set().union(*outcomes.values())
        overlap = sum(len(values) for values in outcomes.values()) - len(union)
        if created != union or overlap:
            raise AssertionError(
                "manipulation created/outcome cohort is not conserved"
            )
        unclassified_source_ids = set().union(
            *(
                values[metric]
                for values in self.source_metric_ids.values()
                for metric in ("ambiguous_dual_side", "atr_unready")
            ),
        )
        created_source_ids = {
            self.latest_manipulations[identity].source_inventory_item_id
            for identity in created
        }
        unclassified_created_overlap = (
            unclassified_source_ids & created_source_ids
        )
        if unclassified_created_overlap:
            raise AssertionError(
                "ambiguous or ATR-unready source created a manipulation"
            )

        grouped: dict[
            tuple[str, str, str],
            dict[str, set[str]],
        ] = {
            key: {
                name: set(values.get(name, set()))
                for name in MANIPULATION_FUNNEL_FIELDS
            }
            for key, values in self.source_metric_ids.items()
        }
        for identity in created:
            state = self.latest_manipulations[identity]
            key = _source_key_from_state(state)
            metrics = grouped.setdefault(
                key,
                {name: set() for name in MANIPULATION_FUNNEL_FIELDS},
            )
            metrics["swept_created"].add(identity)
            for outcome, identities in outcomes.items():
                if identity in identities:
                    metrics[outcome].add(identity)
                    break

        rows: list[dict[str, Any]] = []
        for (source_kind, timeframe, side), metrics in sorted(grouped.items()):
            created_row = metrics["swept_created"]
            outcome_total = sum(
                len(metrics[name]) for name in MANIPULATION_OUTCOMES
            )
            if len(created_row) != outcome_total:
                raise AssertionError(
                    "stratified manipulation outcome conservation failed"
                )
            rows.append(
                {
                    "source_kind": source_kind,
                    "source_timeframe": timeframe,
                    "side": side,
                    **{
                        name: len(metrics[name])
                        for name in MANIPULATION_FUNNEL_FIELDS
                    },
                }
            )

        global_counts = {
            name: sum(int(row[name]) for row in rows)
            for name in MANIPULATION_FUNNEL_FIELDS
        }
        conservation = {
            "swept_created": len(created),
            "classified_outcomes": sum(
                len(values) for values in outcomes.values()
            ),
            "balanced": len(created) == len(union) and overlap == 0,
            "ambiguous_and_atr_unready_excluded_from_created": (
                not unclassified_created_overlap
            ),
        }
        return global_counts, rows, conservation

    def result(self, *, source_rows: int, observed_updates: int) -> dict[str, Any]:
        manipulation_funnel, by_source, conservation = (
            self._manipulation_result()
        )
        forming_censored = sorted(
            range_id
            for range_id, state in self.latest_ranges.items()
            if state.lifecycle is DealingRangeLifecycle.FORMING
            and self._in_window(state.formed_at)
        )
        mature_censored = sorted(
            range_id
            for range_id, state in self.latest_ranges.items()
            if state.lifecycle is DealingRangeLifecycle.MATURE
            and state.mature_at is not None
            and self._in_coverage(state.mature_at)
        )
        warmup_mature_censored = [
            range_id
            for range_id in mature_censored
            if self.latest_ranges[range_id].mature_at < self.start
        ]
        return {
            "window_id": self.window_id,
            "start": self.start,
            "end_exclusive": self.end_exclusive,
            "coverage_start": (
                self.start if self.coverage_start is None else self.coverage_start
            ),
            "source_rows_with_warmup": source_rows,
            "observed_updates_in_window": observed_updates,
            "bar_counts": {
                "real_completed": self.real_completed_bars,
                "synthetic_completed": self.synthetic_completed_bars,
                "data_gap_resets": self.data_gap_resets,
                "contract_resets": self.contract_resets,
            },
            "source_pair_funnel": {
                "unique_geometrically_valid_live_pairs": len(self.unique_pairs),
                "geometrically_valid_pair_clocks": self.pair_clock_count,
            },
            "range_funnel": {
                "forming": len(self.formed_ids),
                "mature": len(self.mature_ids),
                "mature_during_warmup": len(self.warmup_mature_ids),
                "mature_observed_within_coverage": len(
                    self.mature_ids | self.warmup_mature_ids
                ),
                "broken": len(self.broken_ids),
                "right_censored": len(forming_censored) + len(mature_censored),
                "forming_right_censored": len(forming_censored),
                "mature_right_censored": len(mature_censored),
                "warmup_mature_right_censored": len(
                    warmup_mature_censored
                ),
                "terminal_reason_counts": dict(sorted(self.terminal_reasons.items())),
                "forming_terminal_reason_counts": dict(
                    sorted(self.forming_terminal_reasons.items())
                ),
                "mature_terminal_reason_counts": dict(
                    sorted(self.mature_terminal_reasons.items())
                ),
                "forming_terminal_unmet_gate_counts": {
                    name: int(self.unmet_gate_counts[name]) for name in GATE_NAMES
                },
            },
            "range_boundary_inventory": {
                "unique_items_observed": len(self.inventory_ids_observed),
                "unique_items_formed": len(self.inventory_ids_formed),
                "latest_lifecycle_counts": dict(
                    sorted(Counter(self.inventory_lifecycles.values()).items())
                ),
            },
            "manipulation_funnel": manipulation_funnel,
            "manipulation_by_source": by_source,
            "manipulation_conservation": conservation,
            "right_censored_range_ids": {
                "forming": forming_censored,
                "mature": mature_censored,
            },
            "case_candidates": self.case_records,
            "selected_cases": select_stratified_cases(self.case_records),
        }


def _validate_config(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("threshold_search") is not False
        or payload.get("outcome_fields_used") is not False
        or payload.get("pnl_used") is not False
        or payload.get("mbo_used") is not False
        or payload.get("allow_data_gap_reset") is not True
        or payload.get("include_all_pool_source_timeframes") is not True
        or int(payload.get("warmup_calendar_days", 0)) != 7
    ):
        raise ValueError(
            "authority scan must remain registered, outcome blind and "
            "all-scale"
        )
    if payload.get("allowed_ohlcv_role") not in {
        "development",
        "calibration",
    }:
        raise ValueError("authority profile has an invalid OHLCV role")
    if (
        payload.get("allowed_ohlcv_role") == "calibration"
        and payload.get("registered_calibration_exception")
        != "outcome_blind_natural_authority_only"
    ):
        raise ValueError(
            "calibration data requires the registered outcome-blind "
            "authority exception"
        )
    if payload.get("allowed_ohlcv_role") == "calibration":
        if (
            payload.get("validation_protocol")
            != str(DEFAULT_CONFIG.relative_to(ROOT))
            or payload.get("profile") != CANONICAL_CALIBRATION_PROFILE
        ):
            raise ValueError(
                "calibration authority scan requires the canonical "
                "registered profile"
            )
        canonical = _coverage_payload(
            _json(DEFAULT_CONFIG),
            DEFAULT_CONFIG,
            profile=CANONICAL_CALIBRATION_PROFILE,
        )
        frozen_fields = (
            "source",
            "source_sha256",
            "model_config",
            "protocols",
            "timezone",
            "warmup_calendar_days",
            "pool_source_timeframes",
            "permanent_result_path",
            "windows",
        )
        if any(
            payload.get(field) != canonical.get(field)
            for field in frozen_fields
        ):
            raise ValueError(
                "calibration authority scan differs from the canonical "
                "registered profile"
            )
    expected_timeframes = {"4H", "1H", "15m", "5m", "1m"}
    if set(payload.get("pool_source_timeframes") or ()) != expected_timeframes:
        raise ValueError("authority scan must include every enabled pool scale")
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("authority profile requires frozen windows")
    ids = [str(item.get("id", "")) for item in windows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("coverage window identities are invalid")
    intervals = []
    for item in windows:
        start = _aware(item.get("start"), name="profile window start")
        end = _aware(
            item.get("end_exclusive"),
            name="profile window end",
        )
        if end <= start:
            raise ValueError("authority profile window is not positive")
        intervals.append((start, end))
    ordered = sorted(intervals)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("authority profile windows overlap")


def _window_result_path(output: Path, window_id: str) -> Path:
    return output / "windows" / f"{window_id}.json"


def _scan_window(
    *,
    window: Mapping[str, Any],
    payload: Mapping[str, Any],
    output: Path,
    force: bool,
    scan_identity: str,
) -> dict[str, Any]:
    window_id = str(window["id"])
    destination = _window_result_path(output, window_id)
    if destination.is_file() and not force:
        prior = _json(destination)
        if (
            prior.get("window_id") == window_id
            and prior.get("start") == str(window["start"])
            and prior.get("end_exclusive") == str(window["end_exclusive"])
            and prior.get("scan_identity") == scan_identity
        ):
            print(f"[{window_id}] resume: using completed window result", flush=True)
            return prior

    start = _aware(window["start"], name=f"{window_id}.start")
    end = _aware(window["end_exclusive"], name=f"{window_id}.end_exclusive")
    if end <= start:
        raise ValueError(f"coverage window is not positive: {window_id}")
    warmup_start = _calendar_warmup_start(
        start,
        days=int(payload["warmup_calendar_days"]),
        timezone=str(payload["timezone"]),
    )
    validation = load_validation_protocol(ROOT / str(payload["validation_protocol"]))
    role = validation.classify_ohlcv(start, end)
    warmup_role = validation.classify_ohlcv(warmup_start, end)
    allowed_role = str(payload["allowed_ohlcv_role"])
    if role.role != allowed_role or warmup_role.role != allowed_role:
        raise ValueError(
            "authority profile interval differs from its registered OHLCV role"
        )

    source = ROOT / str(payload["source"])
    loaded = load_ohlcv(source, start=warmup_start, end=end)
    group4_path = ROOT / str(payload["protocols"]["group4"])
    protocol = Group4Protocol.from_file(group4_path)
    model = _json(ROOT / str(payload["model_config"]))
    reader, observer = _build_observer(model, payload)
    prior_observation = None
    accumulator = CoverageAccumulator(
        window_id,
        start,
        end,
        protocol,
        coverage_start=warmup_start,
    )
    bars = iter_completed_bars(
        loaded.frame,
        allow_data_gap_reset=True,
    )
    total = len(loaded.frame)
    if total == 0:
        raise RuntimeError(f"coverage window has no completed bars: {window_id}")
    next_progress = 10
    observed_updates = 0
    for index, bar in enumerate(bars, start=1):
        update = reader.on_bar(bar)
        in_window = start <= update.asof < end
        reset_anomalies = {
            value
            for value in update.anomalies
            if value
            in {
                "contract_change_history_reset",
                "data_gap_history_reset",
            }
        }
        if in_window and prior_observation is not None and not reset_anomalies:
            accumulator.observe_source_inputs(
                prior_observation.liquidity_inventory,
                candle=update.completed_1m,
            )
        observation = observer.observe(update)
        if in_window:
            observed_updates += 1
            accumulator.observe_bar(
                real_completed=bool(update.completed_1m.real_completed)
            )
            accumulator.observe_resets(observation.anomalies)
            completed_h1_values = update.newly_completed.get(
                Timeframe.H1,
                (),
            )
            if len(completed_h1_values) > 1:
                raise RuntimeError("one 1m clock emitted multiple H1 candles")
            if completed_h1_values:
                accumulator.observe_source_pairs(
                    observation.frames[
                        Timeframe.H1
                    ].support_resistance,
                    completed_h1_values[0].close,
                )
            h1_frame = observation.frames[Timeframe.H1]
            for state in h1_frame.dealing_ranges:
                accumulator.observe_range(state)
            for state in observation.group4_boundary_range_transitions:
                accumulator.observe_range(state)
            accumulator.observe_inventory(observation.liquidity_inventory)
            if (
                observation.group4_ambiguous_sweep_item_ids
                or observation.group4_atr_unready_sweep_item_ids
            ):
                inventory = {
                    item.item_id: item
                    for item in (
                        *(
                            ()
                            if prior_observation is None
                            else prior_observation.liquidity_inventory
                        ),
                        *observation.liquidity_inventory,
                    )
                }
                accumulator.observe_unclassified_sources(
                    observation.group4_ambiguous_sweep_item_ids,
                    metric="ambiguous_dual_side",
                    inventory=inventory,
                )
                accumulator.observe_unclassified_sources(
                    observation.group4_atr_unready_sweep_item_ids,
                    metric="atr_unready",
                    inventory=inventory,
                )
            for state in observation.manipulations:
                accumulator.observe_manipulation(state)
            for state in observation.group4_boundary_manipulation_transitions:
                accumulator.observe_manipulation(state)
        prior_observation = observation
        progress = index * 100 // total
        if progress >= next_progress:
            print(f"[{window_id}] {next_progress}%", flush=True)
            next_progress += 10

    result = accumulator.result(
        source_rows=len(loaded.frame),
        observed_updates=observed_updates,
    )
    result["scan_identity"] = scan_identity
    _write_json(destination, result)
    return result


def _add_counts(target: Counter[str], values: Mapping[str, Any]) -> None:
    target.update({str(key): int(value) for key, value in values.items()})


def _run_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = ROOT / str(payload["source"])
    source_hash = _sha256_file(source)
    if source_hash != payload.get("source_sha256"):
        raise RuntimeError("authority scan OHLCV source hash mismatch")
    protocol_identity = {}
    for name, value in sorted(payload["protocols"].items()):
        path = ROOT / str(value)
        protocol = _json(path)
        protocol_identity[str(name)] = {
            "path": str(value),
            "protocol_version": protocol.get("protocol_version"),
            "sha256": _sha256_file(path),
        }
    validation_path = ROOT / str(payload["validation_protocol"])
    model_path = ROOT / str(payload["model_config"])
    return {
        "git_commit": _git_commit(),
        "ohlcv_source": {
            "path": str(payload["source"]),
            "sha256": source_hash,
        },
        "data_splits": {
            "path": str(payload["validation_protocol"]),
            "sha256": _sha256_file(validation_path),
        },
        "model_config": {
            "path": str(payload["model_config"]),
            "sha256": _sha256_file(model_path),
        },
        "protocols": protocol_identity,
    }


def _scan_identity(
    run_identity: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        to_primitive(
            {
                "identity": dict(run_identity),
                "profile": payload.get("profile"),
                "windows": payload.get("windows"),
                "warmup_calendar_days": payload.get(
                    "warmup_calendar_days"
                ),
            }
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate_results(
    results: Sequence[Mapping[str, Any]],
    *,
    run_identity: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pairs = Counter()
    ranges = Counter()
    bars = Counter()
    inventory = Counter()
    inventory_lifecycles = Counter()
    manipulations = Counter()
    manipulation_by_source: dict[
        tuple[str, str, str],
        Counter[str],
    ] = {}
    terminal = Counter()
    forming_terminal = Counter()
    mature_terminal = Counter()
    gates = Counter()
    candidates: list[dict[str, Any]] = []
    for result in results:
        _add_counts(pairs, result["source_pair_funnel"])
        _add_counts(bars, result["bar_counts"])
        funnel = result["range_funnel"]
        _add_counts(
            ranges,
            {
                key: funnel[key]
                for key in (
                    "forming",
                    "mature",
                    "mature_during_warmup",
                    "mature_observed_within_coverage",
                    "broken",
                    "right_censored",
                    "forming_right_censored",
                    "mature_right_censored",
                    "warmup_mature_right_censored",
                )
            },
        )
        _add_counts(terminal, funnel["terminal_reason_counts"])
        _add_counts(forming_terminal, funnel["forming_terminal_reason_counts"])
        _add_counts(mature_terminal, funnel["mature_terminal_reason_counts"])
        _add_counts(gates, funnel["forming_terminal_unmet_gate_counts"])
        inventory_result = result["range_boundary_inventory"]
        _add_counts(
            inventory,
            {
                "unique_items_observed": inventory_result[
                    "unique_items_observed"
                ],
                "unique_items_formed": inventory_result[
                    "unique_items_formed"
                ],
            },
        )
        _add_counts(
            inventory_lifecycles,
            inventory_result["latest_lifecycle_counts"],
        )
        _add_counts(manipulations, result["manipulation_funnel"])
        for row in result["manipulation_by_source"]:
            key = (
                str(row["source_kind"]),
                str(row["source_timeframe"]),
                str(row["side"]),
            )
            target = manipulation_by_source.setdefault(key, Counter())
            _add_counts(
                target,
                {
                    name: row[name]
                    for name in MANIPULATION_FUNNEL_FIELDS
                },
            )
        candidates.extend(dict(item) for item in result["case_candidates"])
    created = int(manipulations["swept_created"])
    classified = sum(
        int(manipulations[name]) for name in MANIPULATION_OUTCOMES
    )
    if created != classified:
        raise AssertionError(
            "aggregate manipulation created/outcome conservation failed"
        )
    source_rows = sum(
        int(result["source_rows_with_warmup"])
        for result in results
    )
    observed_updates = sum(
        int(result["observed_updates_in_window"])
        for result in results
    )
    selected = select_stratified_cases(candidates)
    return {
        "schema_version": 1,
        "profile": None if profile is None else profile.get("profile"),
        "run_context": {
            "window_count": len(results),
            "timezone": None if profile is None else profile.get("timezone"),
            "warmup_calendar_days": (
                None
                if profile is None
                else profile.get("warmup_calendar_days")
            ),
            "pool_source_timeframes": (
                []
                if profile is None
                else list(profile.get("pool_source_timeframes", ()))
            ),
            "threshold_search": False,
            "outcome_fields_used": False,
            "pnl_used": False,
            "brain_used": False,
            "mbo_used": False,
            "decision_used": False,
            "risk_used": False,
            "observer_scope": "production_multiscale_group12_group4",
            "executed_protocols": ["group12", "group4"],
            "context_identity_only_protocols": [
                "displacement",
                "group3",
                "group5",
            ],
            "all_pool_source_timeframes": True,
            "scene_graph_projection_used": False,
            "event_view_materialized": False,
        },
        "identity": dict(run_identity or {}),
        "bar_counts": {
            "source_rows_with_warmup": source_rows,
            "observed_updates": observed_updates,
            **dict(sorted(bars.items())),
        },
        "source_pair_funnel": dict(sorted(pairs.items())),
        "range_funnel": {
            **{
                key: int(ranges[key])
                for key in (
                    "forming",
                    "mature",
                    "mature_during_warmup",
                    "mature_observed_within_coverage",
                    "broken",
                    "right_censored",
                    "forming_right_censored",
                    "mature_right_censored",
                    "warmup_mature_right_censored",
                )
            },
            "terminal_reason_counts": dict(sorted(terminal.items())),
            "forming_terminal_reason_counts": dict(
                sorted(forming_terminal.items())
            ),
            "mature_terminal_reason_counts": dict(
                sorted(mature_terminal.items())
            ),
            "forming_terminal_unmet_gate_counts": {
                name: int(gates[name]) for name in GATE_NAMES
            },
        },
        "range_boundary_inventory": {
            **dict(sorted(inventory.items())),
            "latest_lifecycle_counts": dict(
                sorted(inventory_lifecycles.items())
            ),
        },
        "manipulation_funnel": {
            name: int(manipulations[name])
            for name in MANIPULATION_FUNNEL_FIELDS
        },
        "manipulation_by_source": [
            {
                "source_kind": key[0],
                "source_timeframe": key[1],
                "side": key[2],
                **{
                    name: int(values[name])
                    for name in MANIPULATION_FUNNEL_FIELDS
                },
            }
            for key, values in sorted(manipulation_by_source.items())
        ],
        "manipulation_conservation": {
            "formula": (
                "swept_created = reaccepted + accepted_outside + "
                "deadline_censored + hard_boundary_censored + "
                "right_censored"
            ),
            "swept_created": created,
            "classified_outcomes": classified,
            "balanced": created == classified,
            "ambiguous_and_atr_unready_excluded_from_created": True,
        },
        "selected_cases": _compact_selected_cases(selected),
        "case_candidate_count": len(candidates),
        "windows": [
            {
                "window_id": result["window_id"],
                "start": result["start"],
                "end_exclusive": result["end_exclusive"],
                "coverage_start": result["coverage_start"],
                "warmup_start": result["coverage_start"],
            }
            for result in results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output_argument = args.output or (DEFAULT_OUTPUT / args.profile)
    output = (
        output_argument
        if output_argument.is_absolute()
        else ROOT / output_argument
    )
    payload = _coverage_payload(
        _json(config),
        config,
        profile=args.profile,
    )
    _validate_config(payload)
    identity = _run_identity(payload)
    scan_identity = _scan_identity(identity, payload)
    output.mkdir(parents=True, exist_ok=True)
    results = [
        _scan_window(
            window=window,
            payload=payload,
            output=output,
            force=args.force,
            scan_identity=scan_identity,
        )
        for window in payload["windows"]
    ]
    summary = aggregate_results(
        results,
        run_identity={**identity, "scan_identity": scan_identity},
        profile=payload,
    )
    _write_json(output / "summary.json", summary)
    permanent_path = payload.get("permanent_result_path")
    if permanent_path:
        destination = (ROOT / str(permanent_path)).resolve()
        if ROOT.resolve() not in destination.parents:
            raise ValueError("permanent result path escapes the repository")
        _write_json(destination, summary)
    print(json.dumps(to_primitive(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
