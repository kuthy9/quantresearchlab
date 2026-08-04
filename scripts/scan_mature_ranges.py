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
import sys
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.artifact_stream import atomic_bytes  # noqa: E402
from smc_trader.causal import CausalMarketReader  # noqa: E402
from smc_trader.group4 import CausalGroup4Tracker, Group4Protocol  # noqa: E402
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.liquidity import CausalLiquidityTracker, LiquidityConfig  # noqa: E402
from smc_trader.model import (  # noqa: E402
    DealingRangeLifecycle,
    DealingRangeState,
    ManipulationLifecycle,
    ManipulationState,
    SupportResistanceLifecycle,
    SupportResistanceState,
    Timeframe,
    to_primitive,
)
from smc_trader.structure import StructureConfig, StructureTracker  # noqa: E402
from smc_trader.validation import load_validation_protocol  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/group4_mature_range_coverage_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/development/group4_mature_range_coverage_v1"
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


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _aware(value: Any, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError(f"{name} must be timezone aware")
    return result


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
    manipulation_ids: set[str] = field(default_factory=set)
    manipulation_lifecycle_ids: dict[str, set[str]] = field(
        default_factory=lambda: {
            "swept": set(),
            "reaccepted": set(),
            "accepted_outside": set(),
        }
    )
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

    def observe_manipulation(self, state: ManipulationState) -> None:
        if state.source_kind != "mature_range_boundary":
            return
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
            self.manipulation_ids.add(state.manipulation_id)
            self.manipulation_lifecycle_ids[lifecycle].add(state.manipulation_id)
            if lifecycle in {"swept", "reaccepted"}:
                self.case_records.append(
                    _manipulation_record(
                        state,
                        window_id=self.window_id,
                        focus_clock=clock,
                    )
                )

    def result(self, *, source_rows: int, observed_updates: int) -> dict[str, Any]:
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
            "range_boundary_manipulation": {
                "unique_entities": len(self.manipulation_ids),
                **{
                    lifecycle: len(identities)
                    for lifecycle, identities in self.manipulation_lifecycle_ids.items()
                },
            },
            "right_censored_range_ids": {
                "forming": forming_censored,
                "mature": mature_censored,
            },
            "case_candidates": self.case_records,
            "selected_cases": select_stratified_cases(self.case_records),
        }


def _validate_config(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("protocol_version") != "group4-mature-range-coverage.1"
        or payload.get("threshold_search") is not False
        or payload.get("allow_data_gap_reset") is not True
        or int(payload.get("warmup_calendar_days", 0)) != 7
    ):
        raise ValueError("mature-range coverage config differs from its frozen contract")
    forbidden = {str(item).lower() for item in payload.get("forbidden_inputs", ())}
    if not {"pnl", "mbo", "future path", "action labels"}.issubset(forbidden):
        raise ValueError("coverage config does not fail closed on outcome inputs")
    windows = payload.get("windows")
    if not isinstance(windows, list) or len(windows) != 5:
        raise ValueError("coverage requires five frozen development windows")
    ids = [str(item.get("id", "")) for item in windows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("coverage window identities are invalid")


def _window_result_path(output: Path, window_id: str) -> Path:
    return output / "windows" / f"{window_id}.json"


def _scan_window(
    *,
    window: Mapping[str, Any],
    payload: Mapping[str, Any],
    config_hash: str,
    output: Path,
    force: bool,
) -> dict[str, Any]:
    window_id = str(window["id"])
    destination = _window_result_path(output, window_id)
    if destination.is_file() and not force:
        prior = _json(destination)
        if prior.get("bindings", {}).get("coverage_config_sha256") == config_hash:
            print(f"[{window_id}] resume: using completed window result", flush=True)
            return prior

    start = _aware(window["start"], name=f"{window_id}.start")
    end = _aware(window["end_exclusive"], name=f"{window_id}.end_exclusive")
    if end <= start:
        raise ValueError(f"coverage window is not positive: {window_id}")
    warmup_start = start - pd.Timedelta(
        days=int(payload["warmup_calendar_days"])
    )
    validation = load_validation_protocol(ROOT / str(payload["validation_protocol"]))
    role = validation.classify_ohlcv(start, end)
    warmup_role = validation.classify_ohlcv(warmup_start, end)
    if role.role != "development" or warmup_role.role != "development":
        raise ValueError("mature-range coverage may use development OHLCV only")

    source = ROOT / str(payload["source"])
    loaded = load_ohlcv(source, start=warmup_start, end=end)
    group12_path = ROOT / str(payload["group12_protocol"])
    group4_path = ROOT / str(payload["group4_protocol"])
    protocol = Group4Protocol.from_file(group4_path)
    structure_config = StructureConfig.from_file(
        group12_path,
        atr_period=protocol.h1_atr_period,
        tick_size=protocol.tick_size,
    )
    liquidity_config = LiquidityConfig.from_file(
        group12_path,
        tick_size=protocol.tick_size,
        atr_period=protocol.h1_atr_period,
    )
    reader = CausalMarketReader()
    structure = StructureTracker(Timeframe.H1, structure_config)
    liquidity = CausalLiquidityTracker(Timeframe.H1, liquidity_config)
    group4 = CausalGroup4Tracker(protocol)
    h1_zones: tuple[SupportResistanceState, ...] = ()
    prior_range_inventory: tuple[Any, ...] = ()
    accumulator = CoverageAccumulator(
        window_id,
        start,
        end,
        protocol,
        coverage_start=warmup_start,
    )
    bars = tuple(
        iter_completed_bars(
            loaded.frame,
            allow_data_gap_reset=True,
        )
    )
    total = len(bars)
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
        if reset_anomalies:
            reason = (
                "contract_change_reset"
                if "contract_change_history_reset" in reset_anomalies
                else "data_gap_reset"
            )
            boundary = group4.on_boundary(reason, update.asof)
            structure.reset_for_boundary(reason=reason, observed_at=update.asof)
            liquidity.reset()
            h1_zones = ()
            prior_range_inventory = ()
            if in_window:
                observed_updates += 1
                for state in boundary.range_transitions:
                    accumulator.observe_range(state)
                for state in boundary.manipulation_transitions:
                    accumulator.observe_manipulation(state)
            if update.newly_completed.get(Timeframe.H1, ()):
                raise RuntimeError(
                    "a hard-boundary clock unexpectedly completed H1"
                )
            progress = index * 100 // total
            if progress >= next_progress:
                print(f"[{window_id}] {next_progress}%", flush=True)
                next_progress += 10
            continue

        completed_h1_values = update.newly_completed.get(Timeframe.H1, ())
        if len(completed_h1_values) > 1:
            raise RuntimeError("one 1m clock emitted multiple H1 candles")
        completed_h1 = completed_h1_values[0] if completed_h1_values else None
        if completed_h1 is not None:
            structure.on_candle(completed_h1)
            swings, _, _ = structure.snapshot()
            liquidity.on_candle(completed_h1, swings)
            h1_zones, _, _ = liquidity.snapshot()

        group4_update = group4.on_completed_update(
            update.completed_1m,
            prior_inventory=prior_range_inventory,
            liquidity_pools=(),
            completed_h1=completed_h1,
            h1_support_resistance=h1_zones,
        )
        prior_range_inventory = group4_update.range_boundary_inventory
        for state in group4_update.dealing_ranges:
            accumulator.observe_range(state)
        for state in group4_update.range_transitions:
            accumulator.observe_range(state)
        if in_window:
            observed_updates += 1
            if completed_h1 is not None:
                accumulator.observe_source_pairs(
                    h1_zones,
                    completed_h1.close,
                )
            accumulator.observe_inventory(group4_update.range_boundary_inventory)
            for state in group4_update.manipulations:
                accumulator.observe_manipulation(state)
            for state in group4_update.manipulation_transitions:
                accumulator.observe_manipulation(state)
        progress = index * 100 // total
        if progress >= next_progress:
            print(f"[{window_id}] {next_progress}%", flush=True)
            next_progress += 10

    result = accumulator.result(
        source_rows=len(loaded.frame),
        observed_updates=observed_updates,
    )
    result["bindings"] = {
        "coverage_config_sha256": config_hash,
        "source": str(source.relative_to(ROOT)),
        "source_role": loaded.source_role,
        "contract_selection_causal": loaded.contract_selection_causal,
        "replay_mode": "causal_reader_group12_h1_and_group4_reducers_only",
        "range_semantics_authoritative": True,
        "range_boundary_manipulation_requires_targeted_full_observer_confirmation": True,
        "validation_protocol_hash": validation.fingerprint,
        "validation_window_role": role.role,
        "group12_protocol_sha256": _sha256(
            group12_path
        ),
        "group4_protocol_sha256": protocol.protocol_hash,
        "threshold_search": False,
        "future_or_outcome_fields_used": False,
        "brain_used": False,
        "mbo_used": False,
        "data_gap_policy": "hard_reset_without_synthetic_fill",
    }
    _write_json(destination, result)
    return result


def _add_counts(target: Counter[str], values: Mapping[str, Any]) -> None:
    target.update({str(key): int(value) for key, value in values.items()})


def aggregate_results(
    results: Sequence[Mapping[str, Any]],
    *,
    config_hash: str,
) -> dict[str, Any]:
    pairs = Counter()
    ranges = Counter()
    inventory = Counter()
    inventory_lifecycles = Counter()
    manipulations = Counter()
    terminal = Counter()
    forming_terminal = Counter()
    mature_terminal = Counter()
    gates = Counter()
    candidates: list[dict[str, Any]] = []
    for result in results:
        _add_counts(pairs, result["source_pair_funnel"])
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
        _add_counts(manipulations, result["range_boundary_manipulation"])
        candidates.extend(dict(item) for item in result["case_candidates"])
    return {
        "protocol_version": "group4-mature-range-coverage.1",
        "bindings": {
            "coverage_config_sha256": config_hash,
            "window_count": len(results),
            "threshold_search": False,
            "future_or_outcome_fields_used": False,
            "brain_used": False,
            "mbo_used": False,
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
        "range_boundary_manipulation": dict(sorted(manipulations.items())),
        "selected_cases": select_stratified_cases(candidates),
        "case_candidate_count": len(candidates),
        "windows": [
            {
                "window_id": result["window_id"],
                "start": result["start"],
                "end_exclusive": result["end_exclusive"],
                "result_path": f"windows/{result['window_id']}.json",
            }
            for result in results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    payload = _json(config)
    _validate_config(payload)
    config_hash = _sha256(config)
    output.mkdir(parents=True, exist_ok=True)
    results = [
        _scan_window(
            window=window,
            payload=payload,
            config_hash=config_hash,
            output=output,
            force=args.force,
        )
        for window in payload["windows"]
    ]
    summary = aggregate_results(results, config_hash=config_hash)
    _write_json(output / "summary.json", summary)
    print(json.dumps(to_primitive(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
