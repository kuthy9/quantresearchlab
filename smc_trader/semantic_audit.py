"""Outcome-free two-pass semantic audit infrastructure for v3 structure/BOS.

This module deliberately has no trading-brain, decision, risk, execution,
shadow-path, or PnL dependency.  Pass one selects typed structure cases.
Pass two must replay an exact physical source prefix and creates a raw-candle
blind packet whose authority labels live in a separate directory.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .artifact_stream import atomic_bytes, canonical_json, sha256_file
from .model import (
    BOS_FAILURE_REASONS,
    BOS_SAME_CLOCK_FAILURE_REASONS,
    BOSLifecycle,
    BOSScope,
    BreakOfStructureState,
    Candle,
    CORE_TIMEFRAMES,
    Direction,
    MarketObservation,
    SwingLifecycle,
    SwingRelation,
    SwingSide,
    Timeframe,
    content_hash,
    to_primitive,
)


DEFAULT_AUDIT_CONTRACT = Path(
    "configs/experiments/"
    "EXP-SMC-3.0.1-001-BLIND-SEMANTIC-AUDIT-R3.json"
)
ROOT = Path(__file__).resolve().parents[1]
BOUND_FILES = {
    "parent_experiment_sha256": (
        "configs/experiments/"
        "EXP-SMC-3.0.1-001-STRUCTURE-BOS-AUDIT-CLOSURE-R3.json"
    ),
    "primitive_protocol_sha256": (
        "configs/"
        "smc_primitives_v3_0_1_structure_bos_audit_closure_r3.json"
    ),
    "playbook_registry_sha256": "configs/playbooks_v3.json",
    "validation_protocol_sha256": (
        "configs/validation_protocol_v3_0_1_exp001.json"
    ),
    "model_config_sha256": (
        "configs/model_v3_0_1_exp001_structure_bos_identity_r3.json"
    ),
    "structure_code_sha256": "smc_trader/structure.py",
    "model_contracts_sha256": "smc_trader/model.py",
    "observer_code_sha256": "smc_trader/observation.py",
    "causal_reader_code_sha256": "smc_trader/causal.py",
    "playbook_code_sha256": "smc_trader/playbooks.py",
}
CASE_CLASSES = (
    "confirmed_bos",
    "wick_only_no_close",
    "broken_or_opposed",
)
REVIEW_DIRECTIONS = ("long", "short", "uncertain")
REVIEW_LIFECYCLES = ("pending", "confirmed", "failed", "uncertain")
HEX64 = re.compile(r"[0-9a-f]{64}")
REQUIRED_DISCOVERY_IMPLEMENTATION_HASHES = frozenset(
    {
        "semantic_discovery_runner_sha256",
        "semantic_audit_sha256",
        "io_sha256",
        "causal_sha256",
        "market_clock_sha256",
        "observation_sha256",
        "structure_sha256",
    }
)
REQUIRED_DISCOVERY_ITERATOR_BINDINGS = frozenset(
    {
        "maximum_history",
        "maximum_no_trade_gap_minutes",
        "allow_data_gap_reset",
        "source_batch_rows",
    }
)
ACTION_LANGUAGE = re.compile(
    r"\b(?:buy|sell|enter|wait|abstain|hold|protect|exit|"
    r"profit|pnl|best\s+entry|best\s+action|should\s+enter|"
    r"should\s+exit)\b",
    flags=re.IGNORECASE,
)
CLOCK_KEY = re.compile(
    r"(?:^|_)(?:at|clock|cutoff|start|end|time)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SemanticCase:
    semantic_event_id: str
    bos_id: str
    timeframe: Timeframe
    direction: Direction
    case_class: str
    calendar_year: int
    case_clock: pd.Timestamp
    selection_score: str
    semantic_state_hash: str
    target_swing_id: str
    source_structure_id: str | None

    @property
    def bucket_key(self) -> str:
        return "|".join(
            (
                self.timeframe.value,
                self.direction.value,
                self.case_class,
                str(self.calendar_year),
            )
        )


@dataclass(frozen=True)
class SemanticImageArtifact:
    path: Path
    sha256: str
    opaque_case_id: str
    case_clock: pd.Timestamp
    maximum_market_time: pd.Timestamp
    kind: str


def load_audit_contract(
    path: str | Path = DEFAULT_AUDIT_CONTRACT,
    *,
    verify_bound_files: bool = True,
) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "implementation_contract_frozen":
        raise ValueError("semantic audit contract is not frozen")
    selection = payload.get("selection", {})
    if (
        selection.get("bucket_count") != 120
        or selection.get("total_cases") != 240
        or selection.get("cases_per_bucket") != 2
    ):
        raise ValueError("semantic audit sample dimensions changed")
    source_binding = payload.get("source", {})
    if (
        source_binding.get("window_role") != "semantic_discovery"
        or source_binding.get("sealed_data_allowed") is not False
    ):
        raise ValueError("semantic audit contract does not fail closed on data role")
    if type(source_binding.get("access_authorized")) is not bool:
        raise ValueError(
            "semantic audit source access authority is ambiguous"
        )
    for name, digest in payload.get("bindings", {}).items():
        if HEX64.fullmatch(str(digest)) is None:
            raise ValueError(f"semantic audit binding is not a SHA-256: {name}")
    if verify_bound_files:
        bound_files = payload.get("bound_files", BOUND_FILES)
        if (
            not isinstance(bound_files, Mapping)
            or set(bound_files) != set(BOUND_FILES)
        ):
            raise ValueError(
                "semantic audit bound-file family changed"
            )
        for name, relative in bound_files.items():
            expected = payload["bindings"].get(name)
            actual = sha256_file(ROOT / relative)
            if expected != actual:
                raise ValueError(
                    f"semantic audit frozen binding changed: {name}"
                )
        if source_binding.get("access_authorized") is True:
            source_path = ROOT / str(source_binding["path"])
            if (
                payload["bindings"].get("causal_source_sha256")
                != sha256_file(source_path)
            ):
                raise ValueError("semantic audit causal source hash changed")
    return payload


def _aware(value: Any, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone aware")
    return timestamp


def selection_score(
    primitive_protocol_sha256: str,
    semantic_event_id: str,
) -> str:
    if HEX64.fullmatch(primitive_protocol_sha256) is None:
        raise ValueError("primitive protocol digest must be SHA-256")
    if not semantic_event_id:
        raise ValueError("semantic event id is required")
    raw = f"{primitive_protocol_sha256}|{semantic_event_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def classify_bos_case(
    item: BreakOfStructureState,
    *,
    case_clock: pd.Timestamp,
) -> str | None:
    clock = _aware(case_clock, name="case_clock")
    if (
        item.lifecycle is BOSLifecycle.CONFIRMED
        and item.resolved_at == clock
    ):
        return (
            "broken_or_opposed"
            if item.scope is BOSScope.OPPOSED
            else "confirmed_bos"
        )
    if (
        item.lifecycle is BOSLifecycle.PENDING
        and item.last_attempt_at == clock
    ):
        return "wick_only_no_close"
    return None


def semantic_event_id(
    item: BreakOfStructureState,
    *,
    case_class: str,
    case_clock: pd.Timestamp,
) -> str:
    if case_class not in CASE_CLASSES:
        raise ValueError("unregistered semantic case class")
    return "|".join(
        (
            item.bos_id,
            case_class,
            _aware(case_clock, name="case_clock").isoformat(),
        )
    )


class FixedSemanticCaseSelector:
    """Order-independent bounded selector for the preregistered 120 buckets."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        self.contract = dict(contract)
        self.primitive_hash = str(
            self.contract["bindings"]["primitive_protocol_sha256"]
        )
        self.allowed_years = frozenset(
            int(value)
            for value in self.contract["selection"]["calendar_years"]
        )
        self.cases_per_bucket = int(
            self.contract["selection"]["cases_per_bucket"]
        )
        self._seen: set[str] = set()
        self._buckets: dict[str, list[SemanticCase]] = {}

    def observe(self, observation: MarketObservation) -> None:
        case_clock = observation.asof
        year = int(
            case_clock.tz_convert("America/New_York").year
        )
        if year not in self.allowed_years:
            return
        for timeframe in CORE_TIMEFRAMES:
            frame = observation.frame(timeframe)
            for item in frame.structure_breaks:
                case_class = classify_bos_case(
                    item,
                    case_clock=case_clock,
                )
                if case_class is None:
                    continue
                event_id = semantic_event_id(
                    item,
                    case_class=case_class,
                    case_clock=case_clock,
                )
                if event_id in self._seen:
                    continue
                self._seen.add(event_id)
                candidate = SemanticCase(
                    semantic_event_id=event_id,
                    bos_id=item.bos_id,
                    timeframe=timeframe,
                    direction=item.direction,
                    case_class=case_class,
                    calendar_year=year,
                    case_clock=case_clock,
                    selection_score=selection_score(
                        self.primitive_hash,
                        event_id,
                    ),
                    semantic_state_hash=content_hash(item),
                    target_swing_id=item.target_swing_id,
                    source_structure_id=item.source_structure_id,
                )
                bucket = self._buckets.setdefault(candidate.bucket_key, [])
                bucket.append(candidate)
                bucket.sort(
                    key=lambda value: (
                        value.selection_score,
                        value.semantic_event_id,
                    )
                )
                del bucket[self.cases_per_bucket :]

    def expected_bucket_keys(self) -> tuple[str, ...]:
        selection = self.contract["selection"]
        return tuple(
            "|".join((timeframe, direction, case_class, str(year)))
            for timeframe in selection["timeframes"]
            for direction in selection["directions"]
            for case_class in selection["case_classes"]
            for year in selection["calendar_years"]
        )

    def manifest(
        self,
        *,
        source_sha256: str,
        source_start: pd.Timestamp,
        source_end_exclusive: pd.Timestamp,
        implementation_hashes: Mapping[str, str],
    ) -> dict[str, Any]:
        if source_sha256 != self.contract["bindings"]["causal_source_sha256"]:
            raise ValueError("semantic selector source differs from frozen source")
        frozen_source = self.contract["source"]
        if _aware(source_start, name="source_start") != _aware(
            frozen_source["start"],
            name="frozen_source_start",
        ):
            raise ValueError("semantic selector start differs from frozen window")
        if _aware(
            source_end_exclusive,
            name="source_end_exclusive",
        ) != _aware(
            frozen_source["end_exclusive"],
            name="frozen_source_end",
        ):
            raise ValueError("semantic selector end differs from frozen window")
        expected = self.expected_bucket_keys()
        missing = [
            key
            for key in expected
            if len(self._buckets.get(key, ())) != self.cases_per_bucket
        ]
        cases = [
            to_primitive(item)
            for key in expected
            for item in self._buckets.get(key, ())
        ]
        return {
            "format_version": 1,
            "artifact": "v3_structure_bos_selection_authority",
            "audit_id": self.contract["audit_id"],
            "status": "unavailable" if missing else "complete",
            "source_sha256": source_sha256,
            "source_start": source_start,
            "source_end_exclusive": source_end_exclusive,
            "selection_contract_sha256": content_hash(self.contract),
            "implementation_hashes": dict(
                sorted(implementation_hashes.items())
            ),
            "expected_bucket_count": len(expected),
            "cases_per_bucket": self.cases_per_bucket,
            "expected_case_count": len(expected) * self.cases_per_bucket,
            "selected_case_count": len(cases),
            "missing_buckets": missing,
            "cases": cases,
        }

    def write_manifest(
        self,
        destination: str | Path,
        **kwargs: Any,
    ) -> Path:
        target = Path(destination)
        if target.exists():
            raise FileExistsError("selection authority manifest is immutable")
        payload = self.manifest(**kwargs)
        atomic_bytes(target, canonical_json(to_primitive(payload)))
        return target


def _record_clock(
    records: list[dict[str, str]],
    *,
    path: str,
    value: pd.Timestamp | None,
    ceiling: pd.Timestamp,
) -> None:
    if value is None:
        return
    clock = _aware(value, name=path)
    if clock > ceiling:
        raise ValueError(f"semantic clock exceeds its causal ceiling: {path}")
    records.append({"path": path, "clock": clock.isoformat()})


def _record_nested_clocks(
    records: list[dict[str, str]],
    value: Any,
    *,
    path: str,
    ceiling: pd.Timestamp,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            child_path = f"{path}.{name}"
            if (
                child is not None
                and re.search(
                    r"(?:^|_)clocks$",
                    name,
                    flags=re.IGNORECASE,
                )
            ):
                if not isinstance(child, (list, tuple)):
                    raise ValueError(
                        "semantic clock collection is not a sequence: "
                        f"{child_path}"
                    )
                for index, item in enumerate(child):
                    _record_clock(
                        records,
                        path=f"{child_path}[{index}]",
                        value=item,
                        ceiling=ceiling,
                    )
            elif child is not None and CLOCK_KEY.search(name):
                try:
                    timestamp = pd.Timestamp(child)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"semantic clock field is not a timestamp: {child_path}"
                    ) from exc
                if timestamp.tzinfo is None:
                    raise ValueError(
                        f"semantic clock field is timezone-naive: {child_path}"
                    )
                _record_clock(
                    records,
                    path=child_path,
                    value=timestamp,
                    ceiling=ceiling,
                )
            else:
                _record_nested_clocks(
                    records,
                    child,
                    path=child_path,
                    ceiling=ceiling,
                )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _record_nested_clocks(
                records,
                child,
                path=f"{path}[{index}]",
                ceiling=ceiling,
            )


def semantic_clock_certificate(
    observation: MarketObservation,
    histories: Mapping[Timeframe, Sequence[Candle]],
    *,
    case_clock: pd.Timestamp,
    max_source_time_loaded: pd.Timestamp,
    max_engine_time_processed: pd.Timestamp,
    prefix_source_sha256: str,
) -> dict[str, Any]:
    """Recursively certify every typed semantic clock visible in a case."""

    clock = _aware(case_clock, name="case_clock")
    source_cutoff = _aware(
        max_source_time_loaded,
        name="max_source_time_loaded",
    )
    engine_cutoff = _aware(
        max_engine_time_processed,
        name="max_engine_time_processed",
    )
    if observation.asof != clock or engine_cutoff != clock:
        raise ValueError("case, observation and engine clocks must be identical")
    if source_cutoff >= clock:
        raise ValueError(
            "physical source row start must be strictly before the case clock"
        )
    if HEX64.fullmatch(prefix_source_sha256) is None:
        raise ValueError("prefix source commitment must be SHA-256")

    records: list[dict[str, str]] = []
    maximum_market_time: pd.Timestamp | None = None
    for timeframe in CORE_TIMEFRAMES:
        values = tuple(histories.get(timeframe, ()))
        if not values:
            raise ValueError(
                f"semantic audit requires a visible {timeframe.value} history"
            )
        if any(
            candle.timeframe is not timeframe
            or not candle.complete
            or candle.end > clock
            for candle in values
        ):
            raise ValueError("blind history contains invalid or future candles")
        maximum_market_time = max(
            maximum_market_time or values[-1].end,
            max(candle.end for candle in values),
        )
        frame = observation.frame(timeframe)
        _record_clock(
            records,
            path=f"frame.{timeframe.value}.cutoff",
            value=frame.cutoff,
            ceiling=clock,
        )
        for level in frame.liquidity:
            _record_clock(
                records,
                path=f"liquidity.{level.level_id}.formed_at",
                value=level.formed_at,
                ceiling=frame.cutoff,
            )
            _record_clock(
                records,
                path=f"liquidity.{level.level_id}.confirmed_at",
                value=level.confirmed_at,
                ceiling=frame.cutoff,
            )
        for swing in frame.swings:
            for name in (
                "pivot_start",
                "pivot_end",
                "observed_at",
                "confirmed_at",
                "broken_at",
            ):
                _record_clock(
                    records,
                    path=f"swing.{swing.swing_id}.{name}",
                    value=getattr(swing, name),
                    ceiling=frame.cutoff,
                )
        for structure in frame.structures:
            identity = structure.structure_id or structure.direction.value
            for name in ("formed_at", "confirmed_at", "broken_at"):
                _record_clock(
                    records,
                    path=f"structure.{identity}.{name}",
                    value=getattr(structure, name),
                    ceiling=frame.cutoff,
                )
        for item in frame.structure_breaks:
            for name in ("pending_at", "last_attempt_at", "resolved_at"):
                _record_clock(
                    records,
                    path=f"bos.{item.bos_id}.{name}",
                    value=getattr(item, name),
                    ceiling=frame.cutoff,
                )
            for index, attempt_clock in enumerate(item.attempt_clocks):
                _record_clock(
                    records,
                    path=f"bos.{item.bos_id}.attempt_clocks[{index}]",
                    value=attempt_clock,
                    ceiling=frame.cutoff,
                )
    for event in observation.recent_events:
        _record_clock(
            records,
            path=f"event.{event.event_id}.observed_at",
            value=event.observed_at,
            ceiling=clock,
        )
        _record_nested_clocks(
            records,
            event.details,
            path=f"event.{event.event_id}.details",
            ceiling=clock,
        )
    if maximum_market_time is None or maximum_market_time > clock:
        raise ValueError("blind image market clock exceeds the case clock")
    maximum_semantic = max(
        (pd.Timestamp(item["clock"]) for item in records),
        default=clock,
    )
    if maximum_semantic > clock:
        raise ValueError("semantic known-at clock exceeds the case clock")
    payload = {
        "format_version": 1,
        "artifact": "v3_semantic_clock_certificate",
        "case_clock": clock,
        "observation_hash": content_hash(observation),
        "prefix_source_sha256": prefix_source_sha256,
        "max_source_time_loaded": source_cutoff,
        "max_engine_time_processed": engine_cutoff,
        "maximum_market_time": maximum_market_time,
        "maximum_semantic_known_at": maximum_semantic,
        "clock_records": sorted(
            records,
            key=lambda item: (item["clock"], item["path"]),
        ),
    }
    payload["certificate_sha256"] = content_hash(payload)
    return payload


class SemanticCaseVisualizer:
    """Raw-candle blind renderer; truth overlays require a later locked phase."""

    PANEL_BARS = {
        Timeframe.H4: 20,
        Timeframe.H1: 48,
        Timeframe.M5: 48,
        Timeframe.M1: 80,
    }

    @staticmethod
    def _candles(
        axis: Any,
        candles: Sequence[Candle],
        *,
        first_history_index: int,
        tick_size: float,
    ) -> None:
        from matplotlib.ticker import MultipleLocator
        from matplotlib.patches import Rectangle

        for index, candle in enumerate(candles):
            up = candle.close >= candle.open
            synthetic = candle.synthetic_minutes > 0
            color = (
                "#64748b"
                if synthetic
                else "#0f766e"
                if up
                else "#b91c1c"
            )
            axis.vlines(
                index,
                candle.low,
                candle.high,
                color="#334155",
                linewidth=0.7,
            )
            bottom = min(candle.open, candle.close)
            height = abs(candle.close - candle.open)
            if height < 1e-12:
                axis.hlines(
                    candle.open,
                    index - 0.30,
                    index + 0.30,
                    color=color,
                    linewidth=1,
                )
            else:
                axis.add_patch(
                    Rectangle(
                        (index - 0.30, bottom),
                        0.60,
                        height,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=0.4,
                        hatch="///" if synthetic else None,
                    )
                )
        tick_count = min(8, len(candles))
        ticks = (
            [0]
            if tick_count == 1
            else sorted(
                {
                    int(
                        round(
                            index
                            * (len(candles) - 1)
                            / (tick_count - 1)
                        )
                    )
                    for index in range(tick_count)
                }
            )
        )
        axis.set_xticks(ticks)
        axis.set_xticklabels(
            [
                (
                    f"#{first_history_index + index}\n"
                    f"{candles[index].start:%m-%d %H:%M}"
                )
                for index in ticks
            ],
            fontsize=7,
        )
        span_ticks = max(
            1,
            int(
                math.ceil(
                    (
                        max(candle.high for candle in candles)
                        - min(candle.low for candle in candles)
                    )
                    / tick_size
                )
            ),
        )
        grid_step_ticks = max(1, int(math.ceil(span_ticks / 12)))
        axis.yaxis.set_major_locator(
            MultipleLocator(grid_step_ticks * tick_size)
        )
        axis.grid(
            True,
            color="#dbe4ee",
            linewidth=0.4,
            alpha=0.7,
        )

    @staticmethod
    def _panels(
        histories: Mapping[Timeframe, Sequence[Candle]],
        case_clock: pd.Timestamp,
    ) -> dict[Timeframe, tuple[int, tuple[Candle, ...]]]:
        panels: dict[
            Timeframe,
            tuple[int, tuple[Candle, ...]],
        ] = {}
        for timeframe, count in SemanticCaseVisualizer.PANEL_BARS.items():
            values = tuple(histories.get(timeframe, ()))
            if not values or any(
                candle.timeframe is not timeframe
                or not candle.complete
                or candle.end > case_clock
                for candle in values
            ):
                raise ValueError("blind renderer requires causal complete histories")
            panel = values[-count:]
            panels[timeframe] = (len(values) - len(panel), panel)
        return panels

    def render_blind(
        self,
        observation: MarketObservation,
        histories: Mapping[Timeframe, Sequence[Candle]],
        destination: str | Path,
        *,
        opaque_case_id: str,
        tick_size: float,
    ) -> SemanticImageArtifact:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not opaque_case_id:
            raise ValueError("blind case requires an opaque identity")
        if not math.isfinite(float(tick_size)) or tick_size <= 0:
            raise ValueError("blind chart tick size is invalid")
        panels = self._panels(histories, observation.asof)
        figure, axes = plt.subplots(
            4,
            1,
            figsize=(15, 12),
            dpi=110,
            constrained_layout=True,
        )
        for axis, timeframe in zip(axes, CORE_TIMEFRAMES):
            first_history_index, values = panels[timeframe]
            self._candles(
                axis,
                values,
                first_history_index=first_history_index,
                tick_size=tick_size,
            )
            axis.axvline(
                len(values) - 0.5,
                color="#111827",
                linewidth=1.0,
            )
            axis.set_title(
                f"{timeframe.value} · raw completed candles through "
                f"{values[-1].end:%Y-%m-%d %H:%M %Z}",
                loc="left",
                fontsize=9,
            )
        figure.suptitle(
            "BLIND STRUCTURE CASE — model labels and later market path absent",
            fontsize=13,
            weight="bold",
        )
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, bbox_inches="tight")
        plt.close(figure)
        maximum = max(
            candle.end
            for _, values in panels.values()
            for candle in values
        )
        if maximum > observation.asof:
            raise AssertionError("blind image contains a future candle")
        return SemanticImageArtifact(
            path=target,
            sha256=sha256_file(target),
            opaque_case_id=opaque_case_id,
            case_clock=observation.asof,
            maximum_market_time=maximum,
            kind="blind_raw_candles",
        )


def _blind_raw_evidence(
    histories: Mapping[Timeframe, Sequence[Candle]],
    *,
    opaque_case_id: str,
    case_clock: pd.Timestamp,
    tick_size: float,
    prefix_source_sha256: str,
    history_capacity: int,
    reset_epoch: int,
    last_reset_at: pd.Timestamp | None,
    last_reset_reason: str | None,
) -> dict[str, Any]:
    case_clock = _aware(case_clock, name="blind.case_clock")
    if (
        history_capacity <= 0
        or reset_epoch < 0
        or not math.isfinite(float(tick_size))
        or tick_size <= 0
    ):
        raise ValueError("blind evidence bounds are invalid")
    if (last_reset_at is None) != (last_reset_reason is None):
        raise ValueError("blind evidence reset identity is incomplete")
    if (reset_epoch == 0) != (last_reset_at is None):
        raise ValueError("blind evidence reset epoch/identity disagree")
    if last_reset_at is not None:
        last_reset_at = _aware(
            last_reset_at,
            name="blind.last_reset_at",
        )
        if last_reset_at > case_clock:
            raise ValueError(
                "blind evidence reset clock exceeds case clock"
            )
        if last_reset_reason not in {
            "data_gap_reset",
            "contract_change_reset",
        }:
            raise ValueError("blind evidence reset reason is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", prefix_source_sha256):
        raise ValueError("blind evidence source prefix is invalid")
    panels: dict[str, Any] = {}
    for timeframe in CORE_TIMEFRAMES:
        values = tuple(histories.get(timeframe, ()))
        if not values or len(values) > history_capacity:
            raise ValueError("blind evidence requires all four histories")
        rows: list[dict[str, Any]] = []
        for index, candle in enumerate(values):
            if (
                candle.timeframe is not timeframe
                or not candle.complete
                or candle.end > case_clock
            ):
                raise ValueError(
                    "blind evidence contains invalid or future candles"
                )
            prices = {
                name: float(getattr(candle, name))
                for name in ("open", "high", "low", "close")
            }
            rows.append(
                {
                    "bar_index": index,
                    "bar_id": content_hash(
                        {
                            "timeframe": timeframe,
                            "history_index": index,
                            "start": candle.start,
                            "end": candle.end,
                            "symbol": candle.symbol,
                            "instrument_id": candle.instrument_id,
                            "prices": prices,
                            "volume": candle.volume,
                            "real_minutes": candle.real_minutes,
                            "synthetic_minutes": (
                                candle.synthetic_minutes
                            ),
                        }
                    ),
                    "start": candle.start,
                    "end": candle.end,
                    **prices,
                    "open_ticks": int(
                        round(candle.open / tick_size)
                    ),
                    "high_ticks": int(
                        round(candle.high / tick_size)
                    ),
                    "low_ticks": int(
                        round(candle.low / tick_size)
                    ),
                    "close_ticks": int(
                        round(candle.close / tick_size)
                    ),
                    "volume": float(candle.volume),
                    "observed_minutes": candle.observed_minutes,
                    "expected_minutes": candle.expected_minutes,
                    "real_minutes": candle.real_minutes,
                    "synthetic_minutes": candle.synthetic_minutes,
                    "complete": candle.complete,
                    "real_completed": candle.real_completed,
                    "symbol": candle.symbol,
                    "instrument_id": candle.instrument_id,
                }
            )
        panels[timeframe.value] = {
            "history_count": len(values),
            "history_capacity": history_capacity,
            "context_start_truncated": (
                len(values) >= history_capacity
            ),
            "first_start": values[0].start,
            "last_end": values[-1].end,
            "bars": rows,
        }
    return {
        "format_version": 1,
        "artifact": "v3_structure_bos_blind_raw_evidence",
        "opaque_case_id": opaque_case_id,
        "case_clock": case_clock,
        "tick_size": float(tick_size),
        "prefix_source_sha256": prefix_source_sha256,
        "reset_epoch": int(reset_epoch),
        "last_reset_at": last_reset_at,
        "last_reset_reason": last_reset_reason,
        "panels": panels,
    }


def semantic_case_context_complete(
    observation: MarketObservation,
    histories: Mapping[Timeframe, Sequence[Candle]],
    focus_bos: BreakOfStructureState,
) -> bool:
    """Return whether Pass2 can reproduce a fully reviewable blind packet.

    Eligibility is decided at the case clock from causal histories only.  It
    deliberately excludes an otherwise valid BOS when any registered panel is
    absent after a reset, contract provenance differs, or the focal BOS/swing
    clocks have already fallen outside the retained raw context.
    """

    if set(histories) != set(CORE_TIMEFRAMES):
        return False
    for timeframe in CORE_TIMEFRAMES:
        values = tuple(histories.get(timeframe, ()))
        if not values:
            return False
        if any(
            candle.timeframe is not timeframe
            or not candle.complete
            or candle.end > observation.asof
            or candle.symbol != observation.symbol
            or int(candle.instrument_id) != int(observation.instrument_id)
            for candle in values
        ):
            return False
    frame = observation.frame(focus_bos.timeframe)
    target = next(
        (
            item
            for item in frame.swings
            if item.swing_id == focus_bos.target_swing_id
        ),
        None,
    )
    if target is None or target.confirmed_at is None:
        return False
    rows = tuple(histories[focus_bos.timeframe])
    starts = {item.start for item in rows}
    ends = {item.end for item in rows}
    if target.pivot_start not in starts or target.confirmed_at not in ends:
        return False
    if target.prior_same_side_id is not None:
        prior = next(
            (
                item
                for item in frame.swings
                if item.swing_id == target.prior_same_side_id
            ),
            None,
        )
        if (
            prior is None
            or prior.confirmed_at is None
            or prior.pivot_start not in starts
            or prior.confirmed_at not in ends
        ):
            return False
    if (
        focus_bos.pending_at not in ends
        or any(value not in ends for value in focus_bos.attempt_clocks)
        or (
            focus_bos.resolved_at is not None
            and focus_bos.resolved_at not in ends
        )
    ):
        return False
    if focus_bos.source_structure_id is not None and not any(
        item.structure_id == focus_bos.source_structure_id
        for item in frame.structures
    ):
        return False
    return True


def _enumerated_structure_truth(
    observation: MarketObservation,
    raw_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Map every typed, raw-recoverable swing/BOS state to review indices."""

    panels = raw_evidence.get("panels")
    if not isinstance(panels, Mapping):
        raise ValueError("semantic truth requires raw evidence panels")

    def panel_rows(timeframe: Timeframe) -> list[Mapping[str, Any]]:
        panel = panels.get(timeframe.value)
        if not isinstance(panel, Mapping):
            raise ValueError("semantic truth panel is absent")
        rows = panel.get("bars")
        if not isinstance(rows, list) or not rows:
            raise ValueError("semantic truth panel has no bars")
        return rows

    def index_at(
        timeframe: Timeframe,
        clock: pd.Timestamp | None,
        *,
        field: str,
    ) -> int | None:
        if clock is None:
            return None
        wanted = _aware(clock, name="semantic truth clock")
        matches = [
            int(row["bar_index"])
            for row in panel_rows(timeframe)
            if _aware(row[field], name=f"raw.{field}") == wanted
        ]
        if len(matches) > 1:
            raise ValueError("semantic truth raw clock is ambiguous")
        return None if not matches else matches[0]

    swings: list[dict[str, Any]] = []
    bos_candidates: list[dict[str, Any]] = []
    unavailable_prior_same_side_count = 0
    nonrecoverable_swing_count = 0
    nonrecoverable_bos_count = 0
    timeframe_order = {
        timeframe: index
        for index, timeframe in enumerate(CORE_TIMEFRAMES)
    }
    for timeframe in CORE_TIMEFRAMES:
        frame = observation.frame(timeframe)
        by_swing_id = {item.swing_id: item for item in frame.swings}
        recoverable_bos_targets: set[str] = set()
        for swing in frame.swings:
            pivot = index_at(
                timeframe,
                swing.pivot_start,
                field="start",
            )
            observed = index_at(
                timeframe,
                swing.observed_at,
                field="end",
            )
            confirmation = index_at(
                timeframe,
                swing.confirmed_at,
                field="end",
            )
            broken = index_at(
                timeframe,
                swing.broken_at,
                field="end",
            )
            prior_index: int | None = None
            if swing.prior_same_side_id is not None:
                prior = by_swing_id.get(swing.prior_same_side_id)
                if prior is not None:
                    prior_index = index_at(
                        timeframe,
                        prior.pivot_start,
                        field="start",
                    )
                if prior_index is None:
                    unavailable_prior_same_side_count += 1
            if (
                pivot is None
                or observed is None
                or (
                    swing.confirmed_at is not None
                    and confirmation is None
                )
                or (swing.broken_at is not None and broken is None)
            ):
                nonrecoverable_swing_count += 1
                continue
            swings.append(
                {
                    "timeframe": timeframe.value,
                    "side": swing.side.value,
                    "pivot_index": pivot,
                    "prior_same_side_index": prior_index,
                    "relation": swing.relation.value,
                    "observed_index": observed,
                    "confirmation_index": confirmation,
                    "broken_index": broken,
                    "lifecycle": swing.lifecycle.value,
                    "failure_reason": swing.failure_reason,
                }
            )
            if swing.lifecycle in {
                SwingLifecycle.CONFIRMED,
                SwingLifecycle.BROKEN,
            }:
                recoverable_bos_targets.add(swing.swing_id)
        for item in frame.structure_breaks:
            target = by_swing_id.get(item.target_swing_id)
            if (
                target is None
                or target.swing_id not in recoverable_bos_targets
            ):
                nonrecoverable_bos_count += 1
                continue
            target_index = index_at(
                timeframe,
                target.pivot_start,
                field="start",
            )
            pending = index_at(
                timeframe,
                item.pending_at,
                field="end",
            )
            resolved = index_at(
                timeframe,
                item.resolved_at,
                field="end",
            )
            attempts = [
                index_at(timeframe, value, field="end")
                for value in item.attempt_clocks
            ]
            if (
                target_index is None
                or pending is None
                or any(value is None for value in attempts)
                or (item.resolved_at is not None and resolved is None)
            ):
                nonrecoverable_bos_count += 1
                continue
            bos_candidates.append(
                {
                    "timeframe": timeframe.value,
                    "direction": item.direction.value,
                    "target_pivot_index": target_index,
                    "target_price_ticks": item.target_ticks,
                    "pending_index": pending,
                    "wick_attempt_indices": [
                        int(value) for value in attempts if value is not None
                    ],
                    "resolved_index": resolved,
                    "lifecycle": item.lifecycle.value,
                    "scope": item.scope.value,
                    "failure_reason": item.failure_reason,
                }
            )
    swings.sort(
        key=lambda item: (
            timeframe_order[Timeframe(item["timeframe"])],
            item["pivot_index"],
            item["side"],
            (
                -1
                if item["confirmation_index"] is None
                else item["confirmation_index"]
            ),
            item["lifecycle"],
        )
    )
    bos_candidates.sort(
        key=lambda item: (
            timeframe_order[Timeframe(item["timeframe"])],
            item["pending_index"],
            item["direction"],
            item["target_pivot_index"],
        )
    )
    return {
        "swings": swings,
        "bos_candidates": bos_candidates,
        "context": {
            "unavailable_prior_same_side_count": (
                unavailable_prior_same_side_count
            ),
            "nonrecoverable_swing_count": nonrecoverable_swing_count,
            "nonrecoverable_bos_count": nonrecoverable_bos_count,
            "context_truncated": bool(
                unavailable_prior_same_side_count
                or nonrecoverable_swing_count
                or nonrecoverable_bos_count
            ),
        },
    }


def _forbidden_keys(value: Any, forbidden: frozenset[str]) -> list[str]:
    found: list[str] = []

    def walk(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                name = str(key)
                child_path = f"{path}.{name}" if path else name
                if name.casefold() in forbidden:
                    found.append(child_path)
                walk(child, child_path)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value, "")
    return found


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(to_primitive(payload))
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _assert_path_disjoint(
    target: Path,
    protected: Iterable[Path],
    *,
    label: str,
) -> None:
    target_absolute = target.resolve()
    for source in protected:
        source_absolute = source.resolve()
        if (
            target_absolute == source_absolute
            or target_absolute in source_absolute.parents
            or source_absolute in target_absolute.parents
        ):
            raise ValueError(f"{label} must be outside immutable trees")


def _protected_blind_transaction_root(blind_root: Path) -> Path:
    parent = blind_root.parent
    blind_set_root = parent.parent
    if (
        parent.name == "units"
        and (blind_set_root / "BLIND_SET_MANIFEST.json").is_file()
    ):
        return blind_set_root
    if (
        blind_root.name == "blind"
        and (parent / "COMPLETED.json").is_file()
        and (parent / "authority").is_dir()
    ):
        return parent
    return blind_root


def _protected_authority_packet_tree(authority_path: Path) -> Path:
    case_root = authority_path.parent.parent
    if (
        authority_path.name == "authority.json"
        and authority_path.parent.name == "authority"
        and case_root.parent.name == "cases"
        and (case_root / "COMPLETED.json").is_file()
    ):
        return case_root.parent.parent
    return case_root


def materialize_blind_unit(
    *,
    contract: Mapping[str, Any],
    selected_case: Mapping[str, Any],
    observation: MarketObservation,
    histories: Mapping[Timeframe, Sequence[Candle]],
    authority_root: str | Path,
    blind_root: str | Path,
    max_source_time_loaded: pd.Timestamp,
    max_engine_time_processed: pd.Timestamp,
    prefix_source_sha256: str,
    implementation_hashes: Mapping[str, str],
    tick_size: float,
    history_capacity: int,
    reset_epoch: int,
    last_reset_at: pd.Timestamp | None,
    last_reset_reason: str | None,
) -> tuple[Path, Path]:
    """Create separate immutable authority and reviewer-visible packet roots."""

    authority = Path(authority_root)
    blind = Path(blind_root)
    authority_absolute = authority.resolve()
    blind_absolute = blind.resolve()
    if (
        authority_absolute == blind_absolute
        or authority_absolute in blind_absolute.parents
        or blind_absolute in authority_absolute.parents
    ):
        raise ValueError(
            "authority and blind packet roots must not overlap"
        )
    if authority.exists() or blind.exists():
        raise FileExistsError("blind semantic packet roots are immutable")
    authority.mkdir(parents=True)
    blind.mkdir(parents=True)
    event_id = str(selected_case["semantic_event_id"])
    case_clock = _aware(selected_case["case_clock"], name="selected.case_clock")
    if observation.asof != case_clock:
        raise ValueError("pass-two observation did not reproduce the case clock")
    frame = observation.frame(
        Timeframe(str(selected_case["timeframe"]))
    )
    matched_by_id = [
        item
        for item in frame.structure_breaks
        if item.bos_id == str(selected_case["bos_id"])
    ]
    if len(matched_by_id) != 1:
        raise ValueError("pass-two replay did not reproduce the selected BOS")
    matched = matched_by_id[0]
    actual_class = classify_bos_case(
        matched,
        case_clock=case_clock,
    )
    actual_event_id = (
        None
        if actual_class is None
        else semantic_event_id(
            matched,
            case_class=actual_class,
            case_clock=case_clock,
        )
    )
    selected_identity = {
        "semantic_event_id": actual_event_id,
        "bos_id": matched.bos_id,
        "timeframe": matched.timeframe.value,
        "direction": matched.direction.value,
        "case_class": actual_class,
        "target_swing_id": matched.target_swing_id,
        "source_structure_id": matched.source_structure_id,
    }
    if any(
        selected_case.get(name) != value
        for name, value in selected_identity.items()
    ):
        raise ValueError(
            "pass-two selected metadata differs from the reproduced BOS"
        )
    if content_hash(matched) != selected_case["semantic_state_hash"]:
        raise ValueError("pass-two semantic state hash differs from selection")
    target_swings = [
        item
        for item in frame.swings
        if item.swing_id == matched.target_swing_id
    ]
    if len(target_swings) != 1:
        raise ValueError("pass-two target swing is absent or duplicated")
    target_swing = target_swings[0]
    prior_swing = next(
        (
            item
            for item in frame.swings
            if item.swing_id == target_swing.prior_same_side_id
        ),
        None,
    )
    source_structure = next(
        (
            item
            for item in frame.structures
            if item.structure_id == matched.source_structure_id
        ),
        None,
    )
    if not semantic_case_context_complete(
        observation,
        histories,
        matched,
    ):
        raise ValueError(
            "selected semantic case lacks complete reviewable context"
        )

    certificate = semantic_clock_certificate(
        observation,
        histories,
        case_clock=case_clock,
        max_source_time_loaded=max_source_time_loaded,
        max_engine_time_processed=max_engine_time_processed,
        prefix_source_sha256=prefix_source_sha256,
    )
    opaque_case_id = hashlib.sha256(
        f"{contract['audit_id']}|{event_id}".encode("utf-8")
    ).hexdigest()
    forbidden = frozenset(
        str(item).casefold()
        for item in contract["pre_reveal_forbidden_fields"]
    )
    image = SemanticCaseVisualizer().render_blind(
        observation,
        histories,
        blind / "case.png",
        opaque_case_id=opaque_case_id,
        tick_size=tick_size,
    )
    raw_evidence = _blind_raw_evidence(
        histories,
        opaque_case_id=opaque_case_id,
        case_clock=case_clock,
        tick_size=tick_size,
        prefix_source_sha256=prefix_source_sha256,
        history_capacity=history_capacity,
        reset_epoch=reset_epoch,
        last_reset_at=last_reset_at,
        last_reset_reason=last_reset_reason,
    )
    enumerated_truth = _enumerated_structure_truth(
        observation,
        raw_evidence,
    )
    if (
        not enumerated_truth["swings"]
        or not enumerated_truth["bos_candidates"]
    ):
        raise ValueError(
            "selected semantic case has no recoverable typed candidates"
        )
    raw_leaks = _forbidden_keys(raw_evidence, forbidden)
    if raw_leaks:
        raise ValueError(
            f"blind raw evidence contains forbidden fields: {raw_leaks}"
        )
    _write_new(blind / "blind_raw_evidence.json", raw_evidence)
    raw_evidence_sha = sha256_file(
        blind / "blind_raw_evidence.json"
    )
    authority_payload = {
        "format_version": 1,
        "artifact": "v3_structure_bos_case_authority",
        "audit_id": contract["audit_id"],
        "audit_contract_content_hash": content_hash(contract),
        "opaque_case_id": opaque_case_id,
        "selected_case": dict(selected_case),
        "observation_hash": content_hash(observation),
        "focus_bos": to_primitive(matched),
        "focus_target_swing": to_primitive(target_swing),
        "focus_prior_same_side_swing": to_primitive(prior_swing),
        "focus_source_structure": to_primitive(source_structure),
        "enumerated_truth": enumerated_truth,
        "semantic_clock_certificate": certificate,
        "blind_image_sha256": image.sha256,
        "blind_raw_evidence_sha256": raw_evidence_sha,
        "implementation_hashes": dict(
            sorted(implementation_hashes.items())
        ),
    }
    leaked = _forbidden_keys(authority_payload, forbidden)
    if leaked:
        raise ValueError(f"authority packet contains forbidden fields: {leaked}")
    _write_new(authority / "authority.json", authority_payload)

    blind_manifest = {
        "format_version": 1,
        "artifact": "v3_structure_bos_blind_case",
        "audit_id": contract["audit_id"],
        "audit_contract_content_hash": content_hash(contract),
        "review_unit_id": hashlib.sha256(
            (
                f"{opaque_case_id}|{case_clock.isoformat()}|"
                f"{image.sha256}|{raw_evidence_sha}"
            ).encode("utf-8")
        ).hexdigest(),
        "opaque_case_id": opaque_case_id,
        "case_clock": case_clock,
        "blind_image": "case.png",
        "blind_image_sha256": image.sha256,
        "blind_raw_evidence": "blind_raw_evidence.json",
        "blind_raw_evidence_sha256": raw_evidence_sha,
        "maximum_market_time": image.maximum_market_time,
    }
    _write_new(blind / "blind_manifest.json", blind_manifest)
    blind_manifest_sha = sha256_file(blind / "blind_manifest.json")
    template = {
        "review_unit_id": blind_manifest["review_unit_id"],
        "opaque_case_id": opaque_case_id,
        "case_clock": case_clock,
        "blind_image_sha256": image.sha256,
        "blind_manifest_sha256": blind_manifest_sha,
        "blind_raw_evidence_sha256": raw_evidence_sha,
        "reviewer_id": None,
        "reviewed_at": None,
        "no_future_attestation": None,
        "judgment": {
            "swings": [],
            "bos_candidates": [],
            "confidence": None,
            "issue_codes": [],
        },
    }
    _write_new(blind / "review_template.json", template)
    packet = {
        "format_version": 1,
        "artifact": "v3_structure_bos_blind_packet",
        "blind_manifest_sha256": blind_manifest_sha,
        "review_template_sha256": sha256_file(
            blind / "review_template.json"
        ),
        "blind_image_sha256": image.sha256,
        "blind_raw_evidence_sha256": raw_evidence_sha,
    }
    _write_new(blind / "BLIND_PACKET.json", packet)
    return authority / "authority.json", blind / "BLIND_PACKET.json"


def _blind_packet_hashes(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("blind packet root is not a regular directory")
    allowed_files = {
        "BLIND_PACKET.json",
        "blind_manifest.json",
        "blind_raw_evidence.json",
        "case.png",
        "review_template.json",
    }
    observed = {item.name for item in root.iterdir()}
    if observed != allowed_files:
        raise ValueError("blind packet file set changed")
    if any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
        raise ValueError("blind packet contains a non-regular file")
    packet = json.loads(
        (root / "BLIND_PACKET.json").read_text(encoding="utf-8")
    )
    if (
        set(packet)
        != {
            "format_version",
            "artifact",
            "blind_manifest_sha256",
            "review_template_sha256",
            "blind_image_sha256",
            "blind_raw_evidence_sha256",
        }
        or packet.get("format_version") != 1
        or packet.get("artifact")
        != "v3_structure_bos_blind_packet"
    ):
        raise ValueError("blind packet manifest schema changed")
    paths = {
        "blind_manifest_sha256": root / "blind_manifest.json",
        "review_template_sha256": root / "review_template.json",
        "blind_image_sha256": root / "case.png",
        "blind_raw_evidence_sha256": (
            root / "blind_raw_evidence.json"
        ),
    }
    for key, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError("blind packet contains a non-regular file")
        if packet.get(key) != sha256_file(path):
            raise ValueError(
                f"blind packet artifact changed: {path.name}"
            )
    manifest = json.loads(
        (root / "blind_manifest.json").read_text(encoding="utf-8")
    )
    if (
        set(manifest)
        != {
            "format_version",
            "artifact",
            "audit_id",
            "audit_contract_content_hash",
            "review_unit_id",
            "opaque_case_id",
            "case_clock",
            "blind_image",
            "blind_image_sha256",
            "blind_raw_evidence",
            "blind_raw_evidence_sha256",
            "maximum_market_time",
        }
        or manifest.get("format_version") != 1
        or manifest.get("artifact")
        != "v3_structure_bos_blind_case"
        or not str(manifest.get("audit_id", "")).strip()
        or HEX64.fullmatch(
            str(manifest.get("audit_contract_content_hash", ""))
        )
        is None
        or HEX64.fullmatch(str(manifest.get("review_unit_id", "")))
        is None
        or HEX64.fullmatch(str(manifest.get("opaque_case_id", "")))
        is None
        or manifest.get("blind_image") != "case.png"
        or manifest.get("blind_raw_evidence")
        != "blind_raw_evidence.json"
        or manifest.get("blind_image_sha256")
        != packet["blind_image_sha256"]
        or manifest.get("blind_raw_evidence_sha256")
        != packet["blind_raw_evidence_sha256"]
        or _aware(
            manifest.get("maximum_market_time"),
            name="blind.maximum_market_time",
        )
        > _aware(manifest.get("case_clock"), name="blind.case_clock")
    ):
        raise ValueError("blind case manifest identity changed")
    template = json.loads(
        (root / "review_template.json").read_text(encoding="utf-8")
    )
    if (
        set(template)
        != {
            "review_unit_id",
            "opaque_case_id",
            "case_clock",
            "blind_image_sha256",
            "blind_manifest_sha256",
            "blind_raw_evidence_sha256",
            "reviewer_id",
            "reviewed_at",
            "no_future_attestation",
            "judgment",
        }
        or template.get("review_unit_id")
        != manifest["review_unit_id"]
        or template.get("opaque_case_id")
        != manifest["opaque_case_id"]
        or template.get("case_clock") != manifest["case_clock"]
        or template.get("blind_image_sha256")
        != packet["blind_image_sha256"]
        or template.get("blind_manifest_sha256")
        != packet["blind_manifest_sha256"]
        or template.get("blind_raw_evidence_sha256")
        != packet["blind_raw_evidence_sha256"]
        or template.get("reviewer_id") is not None
        or template.get("reviewed_at") is not None
        or template.get("no_future_attestation") is not None
        or template.get("judgment")
        != {
            "swings": [],
            "bos_candidates": [],
            "confidence": None,
            "issue_codes": [],
        }
    ):
        raise ValueError("blind review template identity changed")
    raw = json.loads(
        (root / "blind_raw_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        raw.get("artifact")
        != "v3_structure_bos_blind_raw_evidence"
        or raw.get("opaque_case_id") != manifest["opaque_case_id"]
        or raw.get("case_clock") != manifest["case_clock"]
    ):
        raise ValueError("blind raw-evidence identity changed")
    return {
        "blind_packet_sha256": sha256_file(
            root / "BLIND_PACKET.json"
        ),
        **{
            key: sha256_file(path)
            for key, path in paths.items()
        },
    }


def _verify_locked_review_root(
    review_root: Path,
    blind_root: Path,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if review_root.is_symlink() or not review_root.is_dir():
        raise ValueError("review root is not a regular directory")
    observed = {
        item.name
        for item in review_root.iterdir()
    }
    if observed != {"review.json", "LOCKED_REVIEW.json"}:
        raise ValueError("review transaction tree is partial or changed")
    for item in review_root.iterdir():
        if item.is_symlink() or not item.is_file():
            raise ValueError(
                "review transaction contains a non-regular file"
            )
    review_path = review_root / "review.json"
    lock_path = review_root / "LOCKED_REVIEW.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    packet_hashes = _blind_packet_hashes(blind_root)
    if (
        set(lock)
        != {
            "format_version",
            "artifact",
            "audit_id",
            "audit_contract_content_hash",
            "review_unit_id",
            "blind_packet_sha256",
            "blind_manifest_sha256",
            "blind_image_sha256",
            "blind_raw_evidence_sha256",
            "review_template_sha256",
            "review_sha256",
        }
        or lock.get("format_version") != 1
        or lock.get("artifact") != "v3_locked_blind_review"
        or lock.get("audit_id") != contract.get("audit_id")
        or lock.get("audit_contract_content_hash")
        != content_hash(contract)
    ):
        raise ValueError("locked review contract identity changed")
    for key, value in packet_hashes.items():
        if lock.get(key) != value:
            raise ValueError(
                f"locked blind artifact changed: {key}"
            )
    if lock.get("review_sha256") != sha256_file(review_path):
        raise ValueError("locked review bytes changed")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (blind_root / "blind_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        lock.get("review_unit_id") != manifest.get("review_unit_id")
        or review.get("review_unit_id")
        != manifest.get("review_unit_id")
        or manifest.get("audit_id") != contract.get("audit_id")
        or manifest.get("audit_contract_content_hash")
        != content_hash(contract)
    ):
        raise ValueError("locked review-unit identity changed")
    _validate_review_document(
        review,
        blind_root=blind_root,
        contract=contract,
    )
    return lock


def _validate_enumerated_judgment(
    judgment: Mapping[str, Any],
    *,
    raw_evidence: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    if set(judgment) != {
        "swings",
        "bos_candidates",
        "confidence",
        "issue_codes",
    }:
        raise ValueError("blind review judgment schema changed")
    panels = raw_evidence.get("panels")
    if not isinstance(panels, Mapping):
        raise ValueError("blind raw-evidence panels are absent")
    limits = {
        timeframe.value: int(panels[timeframe.value]["history_count"])
        for timeframe in CORE_TIMEFRAMES
    }

    def index_value(
        value: Any,
        *,
        timeframe: str,
        name: str,
        optional: bool,
    ) -> int | None:
        if value is None and optional:
            return None
        if (
            type(value) is not int
            or value < 0
            or value >= limits[timeframe]
        ):
            raise ValueError(
                f"blind review {name} is outside raw evidence"
            )
        return value

    swings = judgment["swings"]
    if not isinstance(swings, list) or not swings:
        raise ValueError(
            "blind review must enumerate at least one swing"
        )
    swing_fields = {
        "timeframe",
        "side",
        "pivot_index",
        "prior_same_side_index",
        "relation",
        "observed_index",
        "confirmation_index",
        "broken_index",
        "lifecycle",
        "failure_reason",
    }
    seen_swings: set[str] = set()
    relations = {
        *(item.value for item in SwingRelation),
        "uncertain",
    }
    swing_lifecycles = {
        *(item.value for item in SwingLifecycle),
        "uncertain",
    }
    for raw in swings:
        if not isinstance(raw, Mapping) or set(raw) != swing_fields:
            raise ValueError("blind swing-candidate schema changed")
        timeframe = str(raw["timeframe"])
        if timeframe not in limits:
            raise ValueError("blind swing timeframe is invalid")
        if raw["side"] not in {
            *(item.value for item in SwingSide),
            "uncertain",
        }:
            raise ValueError("blind swing side is invalid")
        if raw["relation"] not in relations:
            raise ValueError("blind swing relation is invalid")
        if raw["lifecycle"] not in swing_lifecycles:
            raise ValueError("blind swing lifecycle is invalid")
        pivot = index_value(
            raw["pivot_index"],
            timeframe=timeframe,
            name="swing pivot index",
            optional=False,
        )
        prior = index_value(
            raw["prior_same_side_index"],
            timeframe=timeframe,
            name="prior swing index",
            optional=True,
        )
        observed = index_value(
            raw["observed_index"],
            timeframe=timeframe,
            name="swing observed index",
            optional=False,
        )
        confirmation = index_value(
            raw["confirmation_index"],
            timeframe=timeframe,
            name="swing confirmation index",
            optional=True,
        )
        broken = index_value(
            raw["broken_index"],
            timeframe=timeframe,
            name="swing broken index",
            optional=True,
        )
        if (
            (prior is not None and prior >= pivot)
            or observed < pivot
            or (
                confirmation is not None
                and confirmation <= pivot
            )
            or (
                broken is not None
                and (
                    confirmation is None
                    or broken <= confirmation
                )
            )
        ):
            raise ValueError("blind swing index order is invalid")
        lifecycle = raw["lifecycle"]
        failure_reason = raw["failure_reason"]
        if (
            lifecycle == SwingLifecycle.FORMING.value
            and (
                confirmation is not None
                or broken is not None
                or failure_reason is not None
                or raw["relation"] != SwingRelation.NONE.value
            )
        ):
            raise ValueError("blind forming-swing lifecycle is invalid")
        if (
            lifecycle == SwingLifecycle.FORMATION_FAILED.value
            and (
                confirmation is not None
                or broken is not None
                or failure_reason != "right_side_invalidated"
                or raw["relation"] != SwingRelation.NONE.value
            )
        ):
            raise ValueError("blind failed-swing lifecycle is invalid")
        if (
            lifecycle == SwingLifecycle.CONFIRMED.value
            and (
                confirmation is None
                or broken is not None
                or failure_reason is not None
            )
        ):
            raise ValueError("blind confirmed-swing lifecycle is invalid")
        if (
            lifecycle == SwingLifecycle.BROKEN.value
            and (
                confirmation is None
                or broken is None
                or failure_reason != "close_beyond_swing"
            )
        ):
            raise ValueError("blind broken-swing lifecycle is invalid")
        if (
            lifecycle == "uncertain"
            and failure_reason not in {None, "uncertain"}
        ):
            raise ValueError("blind uncertain swing reason is invalid")
        identity = json.dumps(
            to_primitive(dict(raw)),
            sort_keys=True,
            separators=(",", ":"),
        )
        if identity in seen_swings:
            raise ValueError("blind swing candidate is duplicated")
        seen_swings.add(identity)
    timeframe_order = {
        timeframe.value: index
        for index, timeframe in enumerate(CORE_TIMEFRAMES)
    }
    if swings != sorted(
        swings,
        key=lambda item: (
            timeframe_order[str(item["timeframe"])],
            int(item["pivot_index"]),
            str(item["side"]),
            (
                -1
                if item["confirmation_index"] is None
                else int(item["confirmation_index"])
            ),
            str(item["lifecycle"]),
        ),
    ):
        raise ValueError(
            "blind swing candidates are not in canonical order"
        )

    bos_candidates = judgment["bos_candidates"]
    if not isinstance(bos_candidates, list) or not bos_candidates:
        raise ValueError(
            "blind review must enumerate at least one BOS candidate"
        )
    bos_fields = {
        "timeframe",
        "direction",
        "target_pivot_index",
        "target_price_ticks",
        "pending_index",
        "wick_attempt_indices",
        "resolved_index",
        "lifecycle",
        "scope",
        "failure_reason",
    }
    seen_bos: set[str] = set()
    lifecycles = {
        *(item.value for item in BOSLifecycle),
        "uncertain",
    }
    scopes = {
        *(item.value for item in BOSScope),
        "uncertain",
    }
    for raw in bos_candidates:
        if not isinstance(raw, Mapping) or set(raw) != bos_fields:
            raise ValueError("blind BOS-candidate schema changed")
        timeframe = str(raw["timeframe"])
        if timeframe not in limits:
            raise ValueError("blind BOS timeframe is invalid")
        if raw["direction"] not in REVIEW_DIRECTIONS:
            raise ValueError("blind BOS direction is invalid")
        if raw["lifecycle"] not in lifecycles:
            raise ValueError("blind BOS lifecycle is invalid")
        if raw["scope"] not in scopes:
            raise ValueError("blind BOS scope is invalid")
        if (
            raw["failure_reason"] is not None
            and raw["failure_reason"] != "uncertain"
            and raw["failure_reason"] not in BOS_FAILURE_REASONS
        ):
            raise ValueError("blind BOS failure reason is invalid")
        target = index_value(
            raw["target_pivot_index"],
            timeframe=timeframe,
            name="BOS target index",
            optional=True,
        )
        pending = index_value(
            raw["pending_index"],
            timeframe=timeframe,
            name="BOS pending index",
            optional=True,
        )
        resolved = index_value(
            raw["resolved_index"],
            timeframe=timeframe,
            name="BOS resolved index",
            optional=True,
        )
        ticks = raw["target_price_ticks"]
        if ticks is not None and type(ticks) is not int:
            raise ValueError("blind BOS target ticks are invalid")
        attempts = raw["wick_attempt_indices"]
        if (
            not isinstance(attempts, list)
            or len(attempts) != len(set(attempts))
        ):
            raise ValueError("blind BOS attempt indices are invalid")
        attempt_values = [
            index_value(
                value,
                timeframe=timeframe,
                name="BOS attempt index",
                optional=False,
            )
            for value in attempts
        ]
        if attempt_values != sorted(attempt_values):
            raise ValueError(
                "blind BOS attempt indices are not chronological"
            )
        if (
            target is not None
            and pending is not None
            and pending <= target
        ):
            raise ValueError("blind BOS target/pending order is invalid")
        if attempt_values and pending is None:
            raise ValueError("blind BOS attempts require a pending index")
        if pending is not None and any(
            value <= pending for value in attempt_values
        ):
            raise ValueError("blind BOS attempt order is invalid")
        if resolved is not None and (
            pending is None or resolved <= pending
        ):
            raise ValueError("blind BOS resolution order is invalid")
        if resolved is not None and any(
            (
                value >= resolved
                if raw["lifecycle"] == BOSLifecycle.CONFIRMED.value
                else value > resolved
            )
            for value in attempt_values
        ):
            raise ValueError(
                "blind BOS attempt follows its resolution"
            )
        if (
            raw["lifecycle"] == BOSLifecycle.PENDING.value
            and (
                resolved is not None
                or raw["failure_reason"] is not None
            )
        ) or (
            raw["lifecycle"] == BOSLifecycle.CONFIRMED.value
            and (
                resolved is None
                or raw["failure_reason"] is not None
            )
        ) or (
            raw["lifecycle"] == BOSLifecycle.FAILED.value
            and (
                resolved is None
                or raw["failure_reason"] not in BOS_FAILURE_REASONS
            )
        ) or (
            raw["lifecycle"] == "uncertain"
            and raw["failure_reason"] not in {None, "uncertain"}
        ):
            raise ValueError(
                "blind BOS lifecycle/resolution disagree"
            )
        if (
            raw["lifecycle"] == BOSLifecycle.FAILED.value
            and attempt_values
            and attempt_values[-1] == resolved
            and raw["failure_reason"]
            not in BOS_SAME_CLOCK_FAILURE_REASONS
        ):
            raise ValueError(
                "blind same-clock failed BOS lacks supersession or "
                "structural invalidation"
            )
        identity = json.dumps(
            to_primitive(dict(raw)),
            sort_keys=True,
            separators=(",", ":"),
        )
        if identity in seen_bos:
            raise ValueError("blind BOS candidate is duplicated")
        seen_bos.add(identity)
    if bos_candidates != sorted(
        bos_candidates,
        key=lambda item: (
            timeframe_order[str(item["timeframe"])],
            (
                -1
                if item["pending_index"] is None
                else int(item["pending_index"])
            ),
            str(item["direction"]),
            (
                -1
                if item["target_pivot_index"] is None
                else int(item["target_pivot_index"])
            ),
        ),
    ):
        raise ValueError(
            "blind BOS candidates are not in canonical order"
        )
    enumerated_swing_pivots = {
        (str(item["timeframe"]), int(item["pivot_index"]))
        for item in swings
    }
    if any(
        item["target_pivot_index"] is not None
        and (
            str(item["timeframe"]),
            int(item["target_pivot_index"]),
        )
        not in enumerated_swing_pivots
        for item in bos_candidates
    ):
        raise ValueError(
            "blind BOS target is absent from enumerated swings"
        )

    raw_confidence = judgment["confidence"]
    if (
        isinstance(raw_confidence, bool)
        or not isinstance(raw_confidence, (int, float))
    ):
        raise ValueError("blind review confidence is not numeric")
    confidence = float(raw_confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("blind review confidence must be in [0, 1]")
    issue_codes = judgment["issue_codes"]
    registered = frozenset(contract["review_schema"]["issue_codes"])
    if (
        not isinstance(issue_codes, list)
        or len(issue_codes) != len(set(issue_codes))
        or any(item not in registered for item in issue_codes)
    ):
        raise ValueError("blind review contains unregistered issue codes")


def _validate_review_document(
    review: Mapping[str, Any],
    *,
    blind_root: Path,
    contract: Mapping[str, Any],
) -> None:
    packet_hashes = _blind_packet_hashes(blind_root)
    template = json.loads(
        (blind_root / "review_template.json").read_text(
            encoding="utf-8"
        )
    )
    if set(review) != set(template):
        raise ValueError("blind review top-level schema changed")
    for key in (
        "review_unit_id",
        "opaque_case_id",
        "case_clock",
        "blind_image_sha256",
        "blind_manifest_sha256",
        "blind_raw_evidence_sha256",
    ):
        if review.get(key) != template.get(key):
            raise ValueError(f"blind review identity mismatch: {key}")
    manifest = json.loads(
        (blind_root / "blind_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        manifest.get("audit_id") != contract.get("audit_id")
        or manifest.get("audit_contract_content_hash")
        != content_hash(contract)
        or packet_hashes["blind_manifest_sha256"]
        != review.get("blind_manifest_sha256")
    ):
        raise ValueError("blind review contract/packet identity changed")
    if review.get("no_future_attestation") is not True:
        raise ValueError("blind review requires a no-future attestation")
    if not str(review.get("reviewer_id", "")).strip():
        raise ValueError("blind reviewer identity is required")
    if _aware(
        review.get("reviewed_at"),
        name="reviewed_at",
    ) < _aware(
        manifest.get("case_clock"),
        name="blind case clock",
    ):
        raise ValueError("blind review predates its case clock")
    judgment = review.get("judgment")
    if not isinstance(judgment, Mapping):
        raise ValueError("blind review judgment must be a mapping")
    raw_evidence = json.loads(
        (blind_root / "blind_raw_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    _validate_enumerated_judgment(
        judgment,
        raw_evidence=raw_evidence,
        contract=contract,
    )
    serialized = json.dumps(
        to_primitive(dict(review)),
        sort_keys=True,
        ensure_ascii=False,
    )
    if ACTION_LANGUAGE.search(serialized):
        raise ValueError("blind review contains prohibited action language")


def _lock_blind_review_transaction(
    blind_root: str | Path,
    review_root: str | Path,
    review: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    protected_roots: Iterable[Path] = (),
) -> Path:
    root = Path(blind_root)
    target = Path(review_root)
    _assert_path_disjoint(
        target,
        (
            _protected_blind_transaction_root(root),
            *tuple(protected_roots),
        ),
        label="review transaction",
    )
    if target.exists() or target.is_symlink():
        raise FileExistsError("blind review is already locked")
    staging = target.parent / f".{target.name}.staging"
    if staging.exists() or staging.is_symlink():
        lock = _verify_locked_review_root(
            staging,
            root,
            contract=contract,
        )
        existing_review = json.loads(
            (staging / "review.json").read_text(encoding="utf-8")
        )
        if (
            existing_review != to_primitive(dict(review))
            or lock.get("audit_id") != contract.get("audit_id")
        ):
            raise ValueError(
                "orphan review transaction identity differs"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        return target / "LOCKED_REVIEW.json"
    packet_hashes = _blind_packet_hashes(root)
    _validate_review_document(
        review,
        blind_root=root,
        contract=contract,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    _write_new(staging / "review.json", dict(review))
    manifest = json.loads(
        (root / "blind_manifest.json").read_text(encoding="utf-8")
    )
    lock = {
        "format_version": 1,
        "artifact": "v3_locked_blind_review",
        "audit_id": contract["audit_id"],
        "audit_contract_content_hash": content_hash(contract),
        "review_unit_id": manifest["review_unit_id"],
        **packet_hashes,
        "review_sha256": sha256_file(staging / "review.json"),
    }
    _write_new(staging / "LOCKED_REVIEW.json", lock)
    _verify_locked_review_root(
        staging,
        root,
        contract=contract,
    )
    os.replace(staging, target)
    return target / "LOCKED_REVIEW.json"


def lock_blind_review(
    blind_root: str | Path,
    review_root: str | Path,
    review: Mapping[str, Any],
    *,
    blind_set_manifest: str | Path,
    selection_manifest: str | Path,
    pass2_manifest: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    """Lock one registered review without permitting engine-tree writes."""

    root = Path(blind_root)
    blind_set_path = Path(blind_set_manifest)
    selection_path = Path(selection_manifest)
    pass2_path = Path(pass2_manifest)
    _, units = _verify_blind_set_manifest(
        blind_set_path,
        pass2_path,
        selection_path,
        contract=contract,
    )
    manifest = json.loads(
        (root / "blind_manifest.json").read_text(encoding="utf-8")
    )
    review_id = str(manifest.get("review_unit_id", ""))
    unit = units.get(review_id)
    expected_root = (
        None
        if unit is None
        else blind_set_path.parent / str(unit["unit_path"])
    )
    if expected_root is None or root.resolve() != expected_root.resolve():
        raise ValueError("blind review unit is not in the registered export")
    engine_root = Path(
        str(
            json.loads(
                selection_path.read_text(encoding="utf-8")
            )["engine_output_root"]
        )
    ).resolve()
    return _lock_blind_review_transaction(
        root,
        review_root,
        review,
        contract=contract,
        protected_roots=(blind_set_path.parent, engine_root),
    )


def _registered_selection_dimensions(
    contract: Mapping[str, Any],
) -> tuple[tuple[str, ...], int, int]:
    selection = contract.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("semantic audit selection contract is absent")
    cases_per_bucket = int(selection.get("cases_per_bucket", 0))
    bucket_keys = tuple(
        "|".join(
            (
                str(timeframe),
                str(direction),
                str(case_class),
                str(year),
            )
        )
        for timeframe in selection.get("timeframes", ())
        for direction in selection.get("directions", ())
        for case_class in selection.get("case_classes", ())
        for year in selection.get("calendar_years", ())
    )
    expected_cases = len(bucket_keys) * cases_per_bucket
    if (
        cases_per_bucket <= 0
        or not bucket_keys
        or len(bucket_keys) != int(selection.get("bucket_count", -1))
        or expected_cases != int(selection.get("total_cases", -1))
    ):
        raise ValueError("semantic audit selection dimensions disagree")
    return bucket_keys, cases_per_bucket, expected_cases


def _verify_selection_authority_for_review(
    selection_path: Path,
    pass2_path: Path,
    *,
    contract: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    if (
        selection_path.is_symlink()
        or not selection_path.is_file()
        or pass2_path.is_symlink()
        or not pass2_path.is_file()
    ):
        raise ValueError("selection/Pass2 authority is not regular")
    selection = json.loads(
        selection_path.read_text(encoding="utf-8")
    )
    pass2 = json.loads(pass2_path.read_text(encoding="utf-8"))
    bucket_keys, cases_per_bucket, expected_cases = (
        _registered_selection_dimensions(contract)
    )
    raw_cases = selection.get("cases")
    case_hashes = pass2.get("case_commit_hashes")
    selection_fields = {
        "format_version",
        "artifact",
        "audit_id",
        "status",
        "audit_contract_sha256",
        "runner_contract_sha256",
        "engine_output_root",
        "source_sha256",
        "source_start",
        "source_end_exclusive",
        "source_rows",
        "produced_bars",
        "source_prefix_root",
        "produced_bar_prefix_root",
        "iterator_bindings",
        "implementation_hashes",
        "expected_bucket_count",
        "cases_per_bucket",
        "expected_case_count",
        "selected_case_count",
        "missing_buckets",
        "cases",
    }
    pass2_fields = {
        "format_version",
        "artifact",
        "audit_id",
        "engine_output_root",
        "selection_manifest_sha256",
        "source_sha256",
        "audit_contract_sha256",
        "runner_contract_sha256",
        "implementation_hashes",
        "iterator_bindings",
        "case_count",
        "final_source_rows_admitted",
        "final_produced_bars",
        "final_source_prefix_root",
        "final_produced_bar_prefix_root",
        "final_source_row_ordinal",
        "final_source_start",
        "final_causal_clock",
        "final_checkpoint_state_sha256",
        "case_commit_hashes",
    }
    if (
        set(selection) != selection_fields
        or set(pass2) != pass2_fields
        or selection.get("format_version") != 1
        or selection.get("artifact")
        != "v3_structure_bos_selection_authority"
        or selection.get("status") != "complete"
        or selection.get("audit_id") != contract.get("audit_id")
        or selection.get("expected_bucket_count") != len(bucket_keys)
        or selection.get("cases_per_bucket") != cases_per_bucket
        or selection.get("expected_case_count") != expected_cases
        or selection.get("selected_case_count") != expected_cases
        or selection.get("missing_buckets") != []
        or not isinstance(raw_cases, list)
        or len(raw_cases) != expected_cases
        or pass2.get("format_version") != 1
        or pass2.get("artifact") != "v3_semantic_pass2_manifest"
        or pass2.get("audit_id") != contract.get("audit_id")
        or pass2.get("case_count") != expected_cases
        or not Path(str(selection.get("engine_output_root", ""))).is_absolute()
        or selection_path.parent.resolve()
        != Path(str(selection.get("engine_output_root", ""))).resolve()
        or pass2_path.parent.resolve()
        != Path(str(selection.get("engine_output_root", ""))).resolve()
        or pass2.get("engine_output_root")
        != selection.get("engine_output_root")
        or any(
            HEX64.fullmatch(str(value)) is None
            for value in (
                selection.get("audit_contract_sha256"),
                selection.get("runner_contract_sha256"),
                selection.get("source_sha256"),
                selection.get("source_prefix_root"),
                selection.get("produced_bar_prefix_root"),
                pass2.get("final_source_prefix_root"),
                pass2.get("final_produced_bar_prefix_root"),
                pass2.get("final_checkpoint_state_sha256"),
            )
        )
        or _aware(
            selection.get("source_start"),
            name="selection source start",
        )
        >= _aware(
            selection.get("source_end_exclusive"),
            name="selection source end",
        )
        or type(selection.get("source_rows")) is not int
        or int(selection["source_rows"]) <= 0
        or type(selection.get("produced_bars")) is not int
        or int(selection["produced_bars"]) <= 0
        or type(pass2.get("final_source_rows_admitted")) is not int
        or int(pass2["final_source_rows_admitted"]) <= 0
        or type(pass2.get("final_produced_bars")) is not int
        or int(pass2["final_produced_bars"]) <= 0
        or type(pass2.get("final_source_row_ordinal")) is not int
        or int(pass2["final_source_row_ordinal"]) < 0
        or _aware(
            pass2.get("final_source_start"),
            name="Pass2 final source start",
        )
        >= _aware(
            pass2.get("final_causal_clock"),
            name="Pass2 final causal clock",
        )
        or pass2.get("selection_manifest_sha256")
        != sha256_file(selection_path)
        or pass2.get("audit_contract_sha256")
        != selection.get("audit_contract_sha256")
        or pass2.get("runner_contract_sha256")
        != selection.get("runner_contract_sha256")
        or pass2.get("source_sha256")
        != selection.get("source_sha256")
        or pass2.get("implementation_hashes")
        != selection.get("implementation_hashes")
        or not isinstance(pass2.get("implementation_hashes"), dict)
        or set(pass2.get("implementation_hashes", {}))
        != REQUIRED_DISCOVERY_IMPLEMENTATION_HASHES
        or any(
            HEX64.fullmatch(str(value)) is None
            for value in pass2.get(
                "implementation_hashes",
                {},
            ).values()
        )
        or pass2.get("iterator_bindings")
        != selection.get("iterator_bindings")
        or not isinstance(pass2.get("iterator_bindings"), dict)
        or set(pass2.get("iterator_bindings", {}))
        != REQUIRED_DISCOVERY_ITERATOR_BINDINGS
        or type(
            pass2.get("iterator_bindings", {}).get(
                "allow_data_gap_reset"
            )
        )
        is not bool
        or any(
            type(
                pass2.get("iterator_bindings", {}).get(name)
            )
            is not int
            or int(pass2["iterator_bindings"][name]) <= 0
            for name in (
                "maximum_history",
                "maximum_no_trade_gap_minutes",
                "source_batch_rows",
            )
        )
        or not isinstance(case_hashes, dict)
        or len(case_hashes) != expected_cases
    ):
        raise ValueError(
            "Pass2 is not bound to the registered complete selection"
        )
    by_event: dict[str, dict[str, Any]] = {}
    bucketed: dict[str, list[dict[str, Any]]] = {
        key: [] for key in bucket_keys
    }
    primitive_hash = str(
        contract["bindings"]["primitive_protocol_sha256"]
    )
    case_fields = {
        "semantic_event_id",
        "bos_id",
        "timeframe",
        "direction",
        "case_class",
        "calendar_year",
        "case_clock",
        "selection_score",
        "semantic_state_hash",
        "target_swing_id",
        "source_structure_id",
        "case_bar_synthetic",
        "case_source_row_start",
        "case_source_row_ordinal",
        "case_source_row_sha256",
        "case_bar_sha256",
        "source_prefix_root",
        "produced_bar_prefix_root",
        "source_rows_admitted",
        "produced_bars",
        "reset_epoch",
    }
    for raw in raw_cases:
        if not isinstance(raw, dict) or set(raw) != case_fields:
            raise ValueError("selection case is not a mapping")
        event_id = str(raw.get("semantic_event_id", ""))
        key = "|".join(
            (
                str(raw.get("timeframe", "")),
                str(raw.get("direction", "")),
                str(raw.get("case_class", "")),
                str(raw.get("calendar_year", "")),
            )
        )
        if (
            not event_id
            or event_id
            != "|".join(
                (
                    str(raw.get("bos_id", "")),
                    str(raw.get("case_class", "")),
                    _aware(
                        raw.get("case_clock"),
                        name="selection case clock",
                    ).isoformat(),
                )
            )
            or event_id in by_event
            or key not in bucketed
            or raw.get("selection_score")
            != selection_score(primitive_hash, event_id)
            or HEX64.fullmatch(str(raw.get("selection_score", "")))
            is None
            or HEX64.fullmatch(str(raw.get("semantic_state_hash", "")))
            is None
            or any(
                HEX64.fullmatch(str(raw.get(name, ""))) is None
                for name in (
                    "case_source_row_sha256",
                    "case_bar_sha256",
                    "source_prefix_root",
                    "produced_bar_prefix_root",
                )
            )
            or raw.get("case_bar_synthetic") is not False
            or type(raw.get("case_source_row_ordinal")) is not int
            or type(raw.get("source_rows_admitted")) is not int
            or type(raw.get("produced_bars")) is not int
            or type(raw.get("reset_epoch")) is not int
            or int(raw["case_source_row_ordinal"]) < 0
            or int(raw["source_rows_admitted"]) <= 0
            or int(raw["produced_bars"]) <= 0
            or int(raw["source_rows_admitted"])
            != int(raw["case_source_row_ordinal"]) + 1
            or int(raw["produced_bars"])
            < int(raw["source_rows_admitted"])
            or int(raw["reset_epoch"]) < 0
            or int(raw.get("calendar_year", -1))
            not in set(contract["selection"]["calendar_years"])
            or int(raw.get("calendar_year", -1))
            != int(
                _aware(
                    raw.get("case_clock"),
                    name="selection case clock",
                )
                .tz_convert("America/New_York")
                .year
            )
            or _aware(raw.get("case_source_row_start"), name="case source")
            + pd.Timedelta(minutes=1)
            != _aware(raw.get("case_clock"), name="case clock")
            or _aware(raw.get("case_clock"), name="case clock")
            > _aware(
                pass2.get("final_causal_clock"),
                name="Pass2 final causal clock",
            )
            or HEX64.fullmatch(
                str(case_hashes.get(event_id, ""))
            )
            is None
        ):
            raise ValueError(
                "selection case identity or commitment is invalid"
            )
        by_event[event_id] = dict(raw)
        bucketed[key].append(dict(raw))
    for key in bucket_keys:
        values = bucketed[key]
        values.sort(
            key=lambda item: (
                item["selection_score"],
                item["semantic_event_id"],
            )
        )
        if len(values) != cases_per_bucket:
            raise ValueError(
                f"registered review bucket count changed: {key}"
            )
    expected_order = [
        item["semantic_event_id"]
        for key in bucket_keys
        for item in bucketed[key]
    ]
    if (
        [str(item.get("semantic_event_id", "")) for item in raw_cases]
        != expected_order
        or set(case_hashes) != set(by_event)
    ):
        raise ValueError("selection case set or ordering changed")
    return selection, pass2, by_event


def _reviewer_export_creation_order(
    verified: Sequence[Mapping[str, Any]],
    authority_by_event: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return a physical export order independent of semantic-event order."""

    return sorted(
        verified,
        key=lambda value: str(
            authority_by_event[str(value["semantic_event_id"])][
                "review_unit_id"
            ]
        ),
    )


def materialize_blind_review_export(
    *,
    pass2_manifest_path: str | Path,
    selection_manifest_path: str | Path,
    packets_root: str | Path,
    destination: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    """Create a random-rekeyed, immutable reviewer export.

    Event identity and Pass2 completion commitments remain in an authority
    mapping under the original engine root.  Nothing event-enumerable is
    exposed to the reviewer.
    """

    pass2_path = Path(pass2_manifest_path)
    selection_path = Path(selection_manifest_path)
    packets = Path(packets_root)
    target = Path(destination)
    _assert_path_disjoint(
        target,
        (
            packets,
            selection_path.parent,
            pass2_path.parent,
        ),
        label="blind review export",
    )
    if target.exists() or target.is_symlink():
        raise FileExistsError("blind review export is immutable")
    if packets.is_symlink() or not packets.is_dir():
        raise ValueError("blind export packet root is not regular")
    if pass2_path.is_symlink() or not pass2_path.is_file():
        raise ValueError("blind export Pass2 manifest is not regular")
    selection, pass2, _ = (
        _verify_selection_authority_for_review(
            selection_path,
            pass2_path,
            contract=contract,
        )
    )
    case_hashes = pass2["case_commit_hashes"]
    engine_root = Path(str(selection["engine_output_root"])).resolve()
    if packets.resolve() != (engine_root / "packets").resolve():
        raise ValueError("blind export packet root left its bound engine root")
    verified: list[dict[str, Any]] = []
    seen_internal_review_ids: set[str] = set()
    for event_id, expected_completion in sorted(case_hashes.items()):
        case_id = hashlib.sha256(
            str(event_id).encode("utf-8")
        ).hexdigest()
        case_root = packets / "cases" / case_id
        completion = case_root / "COMPLETED.json"
        if (
            case_root.is_symlink()
            or not case_root.is_dir()
            or completion.is_symlink()
            or not completion.is_file()
            or sha256_file(completion) != expected_completion
        ):
            raise ValueError("Pass2 case completion changed")
        completion_payload = json.loads(
            completion.read_text(encoding="utf-8")
        )
        completion_hashes = completion_payload.get("file_hashes")
        completion_metadata = completion_payload.get(
            "completion_metadata"
        )
        if (
            completion_payload.get("artifact")
            != "v3_semantic_case_transaction"
            or completion_payload.get("semantic_event_id")
            != event_id
            or not isinstance(completion_metadata, dict)
            or completion_metadata.get(
                "selection_manifest_sha256"
            )
            != sha256_file(selection_path)
            or not isinstance(completion_hashes, dict)
            or set(completion_hashes)
            != {
                "authority/authority.json",
                "blind/BLIND_PACKET.json",
                "blind/blind_manifest.json",
                "blind/blind_raw_evidence.json",
                "blind/case.png",
                "blind/review_template.json",
            }
        ):
            raise ValueError("Pass2 case transaction identity changed")
        for relative, expected_digest in completion_hashes.items():
            source = case_root / relative
            if (
                HEX64.fullmatch(str(expected_digest)) is None
                or source.is_symlink()
                or not source.is_file()
                or sha256_file(source) != expected_digest
            ):
                raise ValueError("Pass2 case transaction bytes changed")
        blind_root = case_root / "blind"
        hashes = _blind_packet_hashes(blind_root)
        manifest = json.loads(
            (blind_root / "blind_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        review_id = str(manifest.get("review_unit_id", ""))
        if (
            not HEX64.fullmatch(review_id)
            or review_id in seen_internal_review_ids
            or manifest.get("audit_id") != pass2.get("audit_id")
            or manifest.get("audit_contract_content_hash")
            != content_hash(contract)
            or manifest.get("opaque_case_id") != hashlib.sha256(
                f"{contract['audit_id']}|{event_id}".encode("utf-8")
            ).hexdigest()
        ):
            raise ValueError(
                "blind export review-unit identity is invalid or duplicated"
            )
        seen_internal_review_ids.add(review_id)
        verified.append(
            {
                "semantic_event_id": event_id,
                "case_completion_sha256": expected_completion,
                "internal_review_unit_id": review_id,
                "internal_opaque_case_id": manifest["opaque_case_id"],
                "blind_root": blind_root,
                "internal_hashes": hashes,
            }
        )
    if len(verified) != int(pass2.get("case_count", -1)):
        raise ValueError("blind export case count differs from Pass2")

    authority_path = engine_root / "BLIND_EXPORT_AUTHORITY.json"
    if authority_path.is_symlink():
        raise ValueError("blind export authority mapping is not regular")
    if authority_path.exists():
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
    else:
        authority_units = [
            {
                "semantic_event_id": item["semantic_event_id"],
                "case_completion_sha256": item[
                    "case_completion_sha256"
                ],
                "internal_review_unit_id": item[
                    "internal_review_unit_id"
                ],
                "internal_opaque_case_id": item[
                    "internal_opaque_case_id"
                ],
                "review_unit_id": secrets.token_hex(32),
                "opaque_case_id": secrets.token_hex(32),
            }
            for item in verified
        ]
        authority = {
            "format_version": 1,
            "artifact": "v3_blind_export_authority",
            "audit_id": pass2["audit_id"],
            "audit_contract_content_hash": content_hash(contract),
            "engine_output_root": str(engine_root),
            "reviewer_export_root": str(target.resolve()),
            "selection_manifest_sha256": sha256_file(selection_path),
            "pass2_manifest_sha256": sha256_file(pass2_path),
            "case_count": len(authority_units),
            "units": authority_units,
        }
        _write_new(authority_path, authority)
    authority_fields = {
        "format_version",
        "artifact",
        "audit_id",
        "audit_contract_content_hash",
        "engine_output_root",
        "reviewer_export_root",
        "selection_manifest_sha256",
        "pass2_manifest_sha256",
        "case_count",
        "units",
    }
    authority_unit_fields = {
        "semantic_event_id",
        "case_completion_sha256",
        "internal_review_unit_id",
        "internal_opaque_case_id",
        "review_unit_id",
        "opaque_case_id",
    }
    authority_units = authority.get("units")
    if (
        set(authority) != authority_fields
        or authority.get("format_version") != 1
        or authority.get("artifact") != "v3_blind_export_authority"
        or authority.get("audit_id") != pass2.get("audit_id")
        or authority.get("audit_contract_content_hash")
        != content_hash(contract)
        or authority.get("engine_output_root") != str(engine_root)
        or authority.get("reviewer_export_root") != str(target.resolve())
        or authority.get("selection_manifest_sha256")
        != sha256_file(selection_path)
        or authority.get("pass2_manifest_sha256")
        != sha256_file(pass2_path)
        or authority.get("case_count") != len(verified)
        or not isinstance(authority_units, list)
        or any(
            not isinstance(item, dict)
            or set(item) != authority_unit_fields
            or any(
                HEX64.fullmatch(str(item.get(name, ""))) is None
                for name in (
                    "case_completion_sha256",
                    "internal_review_unit_id",
                    "internal_opaque_case_id",
                    "review_unit_id",
                    "opaque_case_id",
                )
            )
            for item in authority_units
        )
        or len(
            {str(item["semantic_event_id"]) for item in authority_units}
        )
        != len(verified)
        or len(
            {str(item["review_unit_id"]) for item in authority_units}
        )
        != len(verified)
        or len(
            {str(item["opaque_case_id"]) for item in authority_units}
        )
        != len(verified)
    ):
        raise ValueError("blind export authority mapping changed")
    authority_by_event = {
        str(item["semantic_event_id"]): item for item in authority_units
    }
    # Directory birthtime/ctime/inode order is reviewer-visible on common
    # filesystems.  Never create units in semantic-event order; use only the
    # already-generated random reviewer identity as the physical order key.
    export_items = _reviewer_export_creation_order(
        verified,
        authority_by_event,
    )
    for item in export_items:
        bound = authority_by_event.get(str(item["semantic_event_id"]))
        if (
            bound is None
            or bound["case_completion_sha256"]
            != item["case_completion_sha256"]
            or bound["internal_review_unit_id"]
            != item["internal_review_unit_id"]
            or bound["internal_opaque_case_id"]
            != item["internal_opaque_case_id"]
        ):
            raise ValueError("blind export authority unit changed")

    staging = target.parent / f".{target.name}.staging"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError("blind review export has a partial staging root")
    staging.mkdir(parents=True)
    units_root = staging / "units"
    units_root.mkdir()
    units: list[dict[str, Any]] = []
    created_review_ids: list[str] = []
    for item in export_items:
        bound = authority_by_event[str(item["semantic_event_id"])]
        review_id = str(bound["review_unit_id"])
        opaque_case_id = str(bound["opaque_case_id"])
        source_root = Path(item["blind_root"])
        unit_root = units_root / review_id
        unit_root.mkdir()
        created_review_ids.append(review_id)
        shutil.copyfile(source_root / "case.png", unit_root / "case.png")

        raw = json.loads(
            (source_root / "blind_raw_evidence.json").read_text(
                encoding="utf-8"
            )
        )
        raw["opaque_case_id"] = opaque_case_id
        _write_new(unit_root / "blind_raw_evidence.json", raw)
        raw_sha = sha256_file(unit_root / "blind_raw_evidence.json")

        manifest = json.loads(
            (source_root / "blind_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        manifest["review_unit_id"] = review_id
        manifest["opaque_case_id"] = opaque_case_id
        manifest["blind_raw_evidence_sha256"] = raw_sha
        _write_new(unit_root / "blind_manifest.json", manifest)
        manifest_sha = sha256_file(unit_root / "blind_manifest.json")

        template = json.loads(
            (source_root / "review_template.json").read_text(
                encoding="utf-8"
            )
        )
        template["review_unit_id"] = review_id
        template["opaque_case_id"] = opaque_case_id
        template["blind_manifest_sha256"] = manifest_sha
        template["blind_raw_evidence_sha256"] = raw_sha
        _write_new(unit_root / "review_template.json", template)
        packet = {
            "format_version": 1,
            "artifact": "v3_structure_bos_blind_packet",
            "blind_manifest_sha256": manifest_sha,
            "review_template_sha256": sha256_file(
                unit_root / "review_template.json"
            ),
            "blind_image_sha256": sha256_file(unit_root / "case.png"),
            "blind_raw_evidence_sha256": raw_sha,
        }
        _write_new(unit_root / "BLIND_PACKET.json", packet)
        hashes = _blind_packet_hashes(unit_root)
        units.append(
            {
                "review_unit_id": review_id,
                "opaque_case_id": opaque_case_id,
                "unit_path": f"units/{review_id}",
                **hashes,
            }
        )
    if created_review_ids != sorted(created_review_ids):
        raise AssertionError(
            "reviewer unit filesystem creation order is not random-ID order"
        )
    units.sort(key=lambda item: item["review_unit_id"])
    payload = {
        "format_version": 1,
        "artifact": "v3_structure_bos_blind_review_set",
        "audit_id": pass2["audit_id"],
        "audit_contract_content_hash": content_hash(contract),
        "pass2_manifest_sha256": sha256_file(pass2_path),
        "selection_manifest_sha256": (
            pass2["selection_manifest_sha256"]
        ),
        "source_sha256": pass2["source_sha256"],
        "audit_contract_sha256": pass2[
            "audit_contract_sha256"
        ],
        "runner_contract_sha256": pass2[
            "runner_contract_sha256"
        ],
        "implementation_hashes": pass2[
            "implementation_hashes"
        ],
        "case_count": len(units),
        "set_root": content_hash(units),
        "units": units,
    }
    _write_new(staging / "BLIND_SET_MANIFEST.json", payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, target)
    return target / "BLIND_SET_MANIFEST.json"


def _verify_blind_set_manifest(
    path: Path,
    pass2_path: Path,
    selection_path: Path,
    *,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if (
        path.is_symlink()
        or not path.is_file()
        or pass2_path.is_symlink()
        or not pass2_path.is_file()
    ):
        raise ValueError("blind review set authority is not regular")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _, pass2, selected_by_event = (
        _verify_selection_authority_for_review(
            selection_path,
            pass2_path,
            contract=contract,
        )
    )
    pass2_case_hashes = pass2["case_commit_hashes"]
    if (
        set(payload)
        != {
            "format_version",
            "artifact",
            "audit_id",
            "audit_contract_content_hash",
            "pass2_manifest_sha256",
            "selection_manifest_sha256",
            "source_sha256",
            "audit_contract_sha256",
            "runner_contract_sha256",
            "implementation_hashes",
            "case_count",
            "set_root",
            "units",
        }
        or payload.get("format_version") != 1
        or payload.get("artifact")
        != "v3_structure_bos_blind_review_set"
        or payload.get("audit_contract_content_hash")
        != content_hash(contract)
        or any(
            HEX64.fullmatch(str(pass2.get(name, ""))) is None
            for name in (
                "selection_manifest_sha256",
                "source_sha256",
                "audit_contract_sha256",
                "runner_contract_sha256",
            )
        )
        or not isinstance(
            pass2.get("implementation_hashes"),
            dict,
        )
        or any(
            HEX64.fullmatch(str(value)) is None
            for value in pass2.get(
                "implementation_hashes",
                {},
            ).values()
        )
        or not isinstance(pass2_case_hashes, dict)
        or len(pass2_case_hashes)
        != int(pass2.get("case_count", -1))
        or any(
            not str(event_id).strip()
            or HEX64.fullmatch(str(digest)) is None
            for event_id, digest in pass2_case_hashes.items()
        )
        or payload.get("pass2_manifest_sha256")
        != sha256_file(pass2_path)
        or payload.get("audit_id") != pass2.get("audit_id")
        or payload.get("selection_manifest_sha256")
        != pass2.get("selection_manifest_sha256")
        or payload.get("source_sha256") != pass2.get("source_sha256")
        or payload.get("audit_contract_sha256")
        != pass2.get("audit_contract_sha256")
        or payload.get("runner_contract_sha256")
        != pass2.get("runner_contract_sha256")
        or payload.get("implementation_hashes")
        != pass2.get("implementation_hashes")
    ):
        raise ValueError("blind review set differs from Pass2")
    raw_units = payload.get("units")
    if not isinstance(raw_units, list):
        raise ValueError("blind review set units are absent")
    units: dict[str, dict[str, Any]] = {}
    root = path.parent
    if root.is_symlink() or not root.is_dir():
        raise ValueError("blind review set root is not regular")
    if (
        path.name != "BLIND_SET_MANIFEST.json"
        or {item.name for item in root.iterdir()}
        != {"BLIND_SET_MANIFEST.json", "units"}
        or path.is_symlink()
        or not path.is_file()
        or (root / "units").is_symlink()
        or not (root / "units").is_dir()
    ):
        raise ValueError("blind review set top-level tree changed")
    engine_root = Path(str(pass2["engine_output_root"])).resolve()
    authority_path = engine_root / "BLIND_EXPORT_AUTHORITY.json"
    if authority_path.is_symlink() or not authority_path.is_file():
        raise ValueError("blind export authority mapping is absent")
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority_units = authority.get("units")
    if (
        set(authority)
        != {
            "format_version",
            "artifact",
            "audit_id",
            "audit_contract_content_hash",
            "engine_output_root",
            "reviewer_export_root",
            "selection_manifest_sha256",
            "pass2_manifest_sha256",
            "case_count",
            "units",
        }
        or authority.get("format_version") != 1
        or authority.get("artifact") != "v3_blind_export_authority"
        or authority.get("audit_id") != pass2.get("audit_id")
        or authority.get("audit_contract_content_hash")
        != content_hash(contract)
        or authority.get("engine_output_root") != str(engine_root)
        or authority.get("reviewer_export_root") != str(path.parent.resolve())
        or authority.get("selection_manifest_sha256")
        != sha256_file(selection_path)
        or authority.get("pass2_manifest_sha256")
        != sha256_file(pass2_path)
        or not isinstance(authority_units, list)
        or authority.get("case_count") != len(authority_units)
    ):
        raise ValueError("blind export authority mapping changed")
    authority_by_review: dict[str, dict[str, Any]] = {}
    for item in authority_units:
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "semantic_event_id",
                "case_completion_sha256",
                "internal_review_unit_id",
                "internal_opaque_case_id",
                "review_unit_id",
                "opaque_case_id",
            }
            or any(
                HEX64.fullmatch(str(item.get(name, ""))) is None
                for name in (
                    "case_completion_sha256",
                    "internal_review_unit_id",
                    "internal_opaque_case_id",
                    "review_unit_id",
                    "opaque_case_id",
                )
            )
            or str(item.get("semantic_event_id", ""))
            not in selected_by_event
            or str(item["review_unit_id"]) in authority_by_review
            or pass2_case_hashes.get(str(item["semantic_event_id"]))
            != item["case_completion_sha256"]
        ):
            raise ValueError("blind export authority unit changed")
        authority_by_review[str(item["review_unit_id"])] = dict(item)
    observed_event_commits: list[tuple[str, str]] = []
    for raw in raw_units:
        if (
            not isinstance(raw, dict)
            or set(raw)
            != {
                "review_unit_id",
                "opaque_case_id",
                "unit_path",
                "blind_packet_sha256",
                "blind_manifest_sha256",
                "review_template_sha256",
                "blind_image_sha256",
                "blind_raw_evidence_sha256",
            }
        ):
            raise ValueError("blind review set unit is invalid")
        review_id = str(raw.get("review_unit_id", ""))
        opaque_case_id = str(raw.get("opaque_case_id", ""))
        bound = authority_by_review.get(review_id)
        event_id = "" if bound is None else str(
            bound["semantic_event_id"]
        )
        if (
            not HEX64.fullmatch(review_id)
            or review_id in units
            or bound is None
            or event_id not in selected_by_event
            or HEX64.fullmatch(str(raw.get("opaque_case_id", "")))
            is None
            or opaque_case_id != bound["opaque_case_id"]
        ):
            raise ValueError(
                "blind review set contains duplicate or invalid units"
            )
        expected_path = f"units/{review_id}"
        if raw.get("unit_path") != expected_path:
            raise ValueError("blind review set unit path changed")
        unit_root = root / expected_path
        observed = {
            item.name
            for item in unit_root.iterdir()
        }
        if observed != {
            "BLIND_PACKET.json",
            "blind_manifest.json",
            "blind_raw_evidence.json",
            "case.png",
            "review_template.json",
        }:
            raise ValueError("blind review export unit tree changed")
        hashes = _blind_packet_hashes(unit_root)
        manifest = json.loads(
            (unit_root / "blind_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            manifest.get("review_unit_id") != review_id
            or manifest.get("opaque_case_id")
            != raw["opaque_case_id"]
            or manifest.get("audit_id") != payload.get("audit_id")
            or manifest.get("audit_contract_content_hash")
            != content_hash(contract)
        ):
            raise ValueError("blind review export identity changed")
        for key, value in hashes.items():
            if raw.get(key) != value:
                raise ValueError(
                    f"blind review export digest changed: {key}"
                )
        completion_sha = str(bound["case_completion_sha256"])
        if pass2_case_hashes.get(event_id) != completion_sha:
            raise ValueError("blind review case commitment changed")
        observed_event_commits.append((event_id, completion_sha))
        units[review_id] = {
            **dict(raw),
            "semantic_event_id": event_id,
            "case_completion_sha256": completion_sha,
            "internal_review_unit_id": bound[
                "internal_review_unit_id"
            ],
            "internal_opaque_case_id": bound[
                "internal_opaque_case_id"
            ],
        }
    expected_review_ids = {
        str(item["review_unit_id"])
        for item in raw_units
        if isinstance(item, Mapping)
    }
    if {
        item.name for item in (root / "units").iterdir()
    } != expected_review_ids or any(
        item.is_symlink() or not item.is_dir()
        for item in (root / "units").iterdir()
    ):
        raise ValueError("blind review set unit directory changed")
    if (
        len(units) != int(payload.get("case_count", -1))
        or len(units) != int(pass2.get("case_count", -1))
        or set(units) != set(authority_by_review)
        or len(
            {
                item["opaque_case_id"]
                for item in units.values()
            }
        )
        != len(units)
        or sorted(observed_event_commits)
        != sorted(
            (str(event_id), str(value))
            for event_id, value in pass2_case_hashes.items()
        )
        or payload.get("set_root")
        != content_hash(
            sorted(
                raw_units,
                key=lambda item: item["review_unit_id"],
            )
        )
    ):
        raise ValueError("blind review set count or root changed")
    return payload, units


def lock_review_set(
    review_roots: Iterable[str | Path],
    destination: str | Path,
    *,
    blind_set_manifest: str | Path,
    selection_manifest: str | Path,
    pass2_manifest: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    roots = tuple(Path(item) for item in review_roots)
    if len(roots) != len({path.resolve() for path in roots}):
        raise ValueError("blind review roots are duplicated")
    blind_set_path = Path(blind_set_manifest)
    selection_path = Path(selection_manifest)
    pass2_path = Path(pass2_manifest)
    blind_set, expected = _verify_blind_set_manifest(
        blind_set_path,
        pass2_path,
        selection_path,
        contract=contract,
    )
    _, _, registered_cases = _registered_selection_dimensions(contract)
    if len(expected) != registered_cases or len(roots) != registered_cases:
        raise ValueError("all registered blind reviews must be present")
    units: dict[str, dict[str, Any]] = {}
    for root in sorted(roots):
        _assert_path_disjoint(
            root,
            (
                blind_set_path.parent,
                selection_path.parent,
                pass2_path.parent,
            ),
            label="blind review transaction",
        )
        if root.is_symlink() or not root.is_dir():
            raise ValueError("blind review root is not regular")
        observed = {item.name for item in root.iterdir()}
        if observed != {"review.json", "LOCKED_REVIEW.json"}:
            raise ValueError("blind review transaction tree changed")
        review_path = root / "review.json"
        lock_path = root / "LOCKED_REVIEW.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review_id = str(lock.get("review_unit_id", ""))
        if (
            not HEX64.fullmatch(review_id)
            or review_id in units
            or review_id not in expected
            or review.get("review_unit_id") != review_id
            or lock.get("audit_id") != blind_set["audit_id"]
            or lock.get("review_sha256") != sha256_file(review_path)
        ):
            raise ValueError(
                "blind review identity is duplicate, mixed, or changed"
            )
        blind_root = (
            blind_set_path.parent / expected[review_id]["unit_path"]
        )
        verified_lock = _verify_locked_review_root(
            root,
            blind_root,
            contract=contract,
        )
        if verified_lock != lock:
            raise AssertionError("blind review verification changed lock")
        for key in (
            "blind_packet_sha256",
            "blind_manifest_sha256",
            "blind_image_sha256",
            "blind_raw_evidence_sha256",
            "review_template_sha256",
        ):
            if lock.get(key) != expected[review_id].get(key):
                raise ValueError(
                    f"blind review substituted packet evidence: {key}"
                )
        units[review_id] = {
            "review_unit_id": review_id,
            "review_root": str(root.resolve()),
            "locked_review_sha256": sha256_file(lock_path),
            "review_sha256": sha256_file(review_path),
        }
    if set(units) != set(expected):
        raise ValueError("blind review set is missing or has extra units")
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError("global review lock is immutable")
    _assert_path_disjoint(
        target,
        (
            *roots,
            blind_set_path.parent,
            selection_path.parent,
            pass2_path.parent,
        ),
        label="global review lock",
    )
    ordered_units = sorted(
        units.values(),
        key=lambda item: item["review_unit_id"],
    )
    payload = {
        "format_version": 1,
        "artifact": "v3_all_blind_reviews_locked",
        "audit_id": blind_set["audit_id"],
        "audit_contract_content_hash": content_hash(contract),
        "selection_manifest_sha256": sha256_file(selection_path),
        "pass2_manifest_sha256": sha256_file(pass2_path),
        "blind_set_manifest_sha256": sha256_file(blind_set_path),
        "case_count": len(units),
        "review_set_root": content_hash(ordered_units),
        "units": ordered_units,
    }
    _write_new(target, payload)
    return target


def _verify_global_review_lock(
    path: Path,
    *,
    blind_set_path: Path,
    selection_path: Path,
    pass2_path: Path,
    contract: Mapping[str, Any],
    blind_set: Mapping[str, Any],
    units: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, tuple[Path, dict[str, Any]]],
]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("semantic truth global review lock is not regular")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_units = payload.get("units")
    if (
        set(payload)
        != {
            "format_version",
            "artifact",
            "audit_id",
            "audit_contract_content_hash",
            "selection_manifest_sha256",
            "pass2_manifest_sha256",
            "blind_set_manifest_sha256",
            "case_count",
            "review_set_root",
            "units",
        }
        or payload.get("format_version") != 1
        or payload.get("artifact") != "v3_all_blind_reviews_locked"
        or payload.get("audit_id") != blind_set.get("audit_id")
        or payload.get("audit_contract_content_hash")
        != content_hash(contract)
        or payload.get("selection_manifest_sha256")
        != sha256_file(selection_path)
        or payload.get("pass2_manifest_sha256")
        != sha256_file(pass2_path)
        or payload.get("blind_set_manifest_sha256")
        != sha256_file(blind_set_path)
        or not isinstance(raw_units, list)
        or len(raw_units) != int(payload.get("case_count", -1))
        or len(raw_units) != len(units)
        or any(
            not isinstance(item, dict)
            or set(item)
            != {
                "review_unit_id",
                "review_root",
                "locked_review_sha256",
                "review_sha256",
            }
            for item in raw_units
        )
        or [item["review_unit_id"] for item in raw_units] != sorted(units)
        or payload.get("review_set_root") != content_hash(raw_units)
    ):
        raise ValueError(
            "semantic truth overlay lacks an exact global review lock"
        )
    verified_reviews: dict[str, tuple[Path, dict[str, Any]]] = {}
    for item in raw_units:
        review_id = str(item["review_unit_id"])
        review_root = Path(str(item["review_root"]))
        _assert_path_disjoint(
            review_root,
            (
                blind_set_path.parent,
                selection_path.parent,
                pass2_path.parent,
            ),
            label="locked review transaction",
        )
        blind_root = blind_set_path.parent / str(
            units[review_id]["unit_path"]
        )
        verified = _verify_locked_review_root(
            review_root,
            blind_root,
            contract=contract,
        )
        review_path = review_root / "review.json"
        lock_path = review_root / "LOCKED_REVIEW.json"
        if (
            HEX64.fullmatch(str(item["locked_review_sha256"])) is None
            or HEX64.fullmatch(str(item["review_sha256"])) is None
            or sha256_file(lock_path) != item["locked_review_sha256"]
            or sha256_file(review_path) != item["review_sha256"]
            or verified.get("review_unit_id") != review_id
        ):
            raise ValueError("semantic truth review transaction changed")
        verified_reviews[review_id] = (
            review_root,
            json.loads(review_path.read_text(encoding="utf-8")),
        )
    return payload, verified_reviews


def _semantic_visual_coverage(
    raw_evidence: Mapping[str, Any],
    enumerated_truth: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze an exact visual marker map for every typed candidate."""

    panels = raw_evidence.get("panels")
    swings = enumerated_truth.get("swings")
    bos_candidates = enumerated_truth.get("bos_candidates")
    if (
        not isinstance(panels, Mapping)
        or not isinstance(swings, list)
        or not isinstance(bos_candidates, list)
    ):
        raise ValueError("semantic visual coverage inputs are incomplete")
    price_panels: dict[str, dict[str, Any]] = {}
    index_maps: dict[str, dict[str, Any]] = {}
    for timeframe in CORE_TIMEFRAMES:
        name = timeframe.value
        panel = panels.get(name)
        rows = None if not isinstance(panel, Mapping) else panel.get("bars")
        if not isinstance(rows, list) or not rows:
            raise ValueError("semantic visual coverage panel is empty")
        maximum = int(SemanticCaseVisualizer.PANEL_BARS[timeframe])
        first = max(0, len(rows) - maximum)
        price_panels[name] = {
            "first_history_index": first,
            "last_history_index": len(rows) - 1,
            "bar_count": len(rows) - first,
            "maximum_bars": maximum,
            "axis_compressed": False,
        }

        def marker_index(value: Any, *, optional: bool) -> int | None:
            if value is None and optional:
                return None
            if type(value) is not int or value < 0 or value >= len(rows):
                raise ValueError(
                    "semantic visual marker is outside retained history"
                )
            return value

        swing_entries: list[dict[str, Any]] = []
        for candidate in swings:
            if candidate.get("timeframe") != name:
                continue
            swing_entries.append(
                {
                    "candidate_sha256": content_hash(candidate),
                    "pivot_index": marker_index(
                        candidate.get("pivot_index"),
                        optional=False,
                    ),
                    "observed_index": marker_index(
                        candidate.get("observed_index"),
                        optional=False,
                    ),
                    "confirmation_index": marker_index(
                        candidate.get("confirmation_index"),
                        optional=True,
                    ),
                    "broken_index": marker_index(
                        candidate.get("broken_index"),
                        optional=True,
                    ),
                }
            )
        bos_entries: list[dict[str, Any]] = []
        for candidate in bos_candidates:
            if candidate.get("timeframe") != name:
                continue
            attempts = candidate.get("wick_attempt_indices")
            if not isinstance(attempts, list):
                raise ValueError("semantic visual BOS attempts are invalid")
            bos_entries.append(
                {
                    "candidate_sha256": content_hash(candidate),
                    "target_pivot_index": marker_index(
                        candidate.get("target_pivot_index"),
                        optional=False,
                    ),
                    "pending_index": marker_index(
                        candidate.get("pending_index"),
                        optional=False,
                    ),
                    "wick_attempt_indices": [
                        marker_index(value, optional=False)
                        for value in attempts
                    ],
                    "resolved_index": marker_index(
                        candidate.get("resolved_index"),
                        optional=True,
                    ),
                }
            )
        index_maps[name] = {
            "history_rows": len(rows),
            "swing_candidates": swing_entries,
            "bos_candidates": bos_entries,
            "categorical_index_axis": True,
            "price_axis": False,
        }
    coverage = {
        "format_version": 1,
        "price_panels": price_panels,
        "candidate_index_maps": index_maps,
        "typed_swing_candidate_count": len(swings),
        "typed_bos_candidate_count": len(bos_candidates),
        "all_typed_candidates_in_index_maps": (
            sum(
                len(value["swing_candidates"])
                for value in index_maps.values()
            )
            == len(swings)
            and sum(
                len(value["bos_candidates"])
                for value in index_maps.values()
            )
            == len(bos_candidates)
        ),
        "price_axis_compression": False,
        "index_maps_are_categorical_not_price_axes": True,
    }
    if coverage["all_typed_candidates_in_index_maps"] is not True:
        raise AssertionError("semantic visual coverage omitted a typed candidate")
    return coverage


def _validate_semantic_visual_coverage(
    coverage: Mapping[str, Any],
    *,
    raw_evidence: Mapping[str, Any],
    enumerated_truth: Mapping[str, Any],
) -> None:
    expected = _semantic_visual_coverage(
        raw_evidence,
        enumerated_truth,
    )
    if to_primitive(dict(coverage)) != to_primitive(expected):
        raise ValueError("semantic truth visual coverage changed")


def materialize_semantic_truth_overlay(
    *,
    authority_path: str | Path,
    blind_set_manifest: str | Path,
    selection_manifest: str | Path,
    pass2_manifest: str | Path,
    global_review_lock: str | Path,
    review_unit_id: str,
    destination: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    """Reveal semantic truth only after the exact blind set is locked."""

    authority_source = Path(authority_path)
    blind_set_path = Path(blind_set_manifest)
    selection_path = Path(selection_manifest)
    pass2_path = Path(pass2_manifest)
    global_lock_path = Path(global_review_lock)
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError("semantic truth overlay is immutable")
    blind_set, units = _verify_blind_set_manifest(
        blind_set_path,
        pass2_path,
        selection_path,
        contract=contract,
    )
    _, verified_reviews = _verify_global_review_lock(
        global_lock_path,
        blind_set_path=blind_set_path,
        selection_path=selection_path,
        pass2_path=pass2_path,
        contract=contract,
        blind_set=blind_set,
        units=units,
    )
    if review_unit_id not in units:
        raise ValueError("semantic truth review unit is not registered")
    unit = units[review_unit_id]
    blind_root = blind_set_path.parent / unit["unit_path"]
    review_root, review = verified_reviews[review_unit_id]
    case_root = authority_source.parent.parent
    _assert_path_disjoint(
        target,
        (
            _protected_authority_packet_tree(authority_source),
            blind_set_path.parent,
            selection_path.parent,
            pass2_path.parent,
            *(root for root, _ in verified_reviews.values()),
        ),
        label="semantic truth overlay",
    )
    if authority_source.is_symlink() or not authority_source.is_file():
        raise ValueError("semantic truth authority is not regular")
    authority = json.loads(
        authority_source.read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (blind_root / "blind_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    raw = json.loads(
        (blind_root / "blind_raw_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    internal_raw_path = case_root / "blind" / "blind_raw_evidence.json"
    if internal_raw_path.is_symlink() or not internal_raw_path.is_file():
        raise ValueError("semantic truth internal raw evidence is absent")
    internal_raw = json.loads(
        internal_raw_path.read_text(encoding="utf-8")
    )
    sanitized_internal_raw = dict(internal_raw)
    sanitized_internal_raw["opaque_case_id"] = unit["opaque_case_id"]
    if (
        set(authority)
        != {
            "format_version",
            "artifact",
            "audit_id",
            "audit_contract_content_hash",
            "opaque_case_id",
            "selected_case",
            "observation_hash",
            "focus_bos",
            "focus_target_swing",
            "focus_prior_same_side_swing",
            "focus_source_structure",
            "enumerated_truth",
            "semantic_clock_certificate",
            "blind_image_sha256",
            "blind_raw_evidence_sha256",
            "implementation_hashes",
        }
        or authority.get("format_version") != 1
        or authority.get("artifact")
        != "v3_structure_bos_case_authority"
        or authority.get("audit_id") != contract.get("audit_id")
        or authority.get("audit_contract_content_hash")
        != content_hash(contract)
        or manifest.get("review_unit_id") != review_unit_id
        or manifest.get("opaque_case_id") != unit["opaque_case_id"]
        or authority.get("opaque_case_id")
        != unit["internal_opaque_case_id"]
        or authority.get("blind_image_sha256")
        != sha256_file(blind_root / "case.png")
        or authority.get("blind_raw_evidence_sha256")
        != sha256_file(internal_raw_path)
        or raw != sanitized_internal_raw
    ):
        raise ValueError("semantic truth authority/blind identity differs")
    certificate = authority.get("semantic_clock_certificate")
    if not isinstance(certificate, dict):
        raise ValueError("semantic truth clock certificate is absent")
    certificate_body = dict(certificate)
    certificate_digest = certificate_body.pop(
        "certificate_sha256",
        None,
    )
    case_clock = _aware(
        authority.get("selected_case", {}).get("case_clock"),
        name="authority case clock",
    )
    clock_records = certificate.get("clock_records")
    if (
        certificate_digest != content_hash(certificate_body)
        or certificate.get("artifact")
        != "v3_semantic_clock_certificate"
        or certificate.get("case_clock")
        != authority.get("selected_case", {}).get("case_clock")
        or certificate.get("observation_hash")
        != authority.get("observation_hash")
        or certificate.get("prefix_source_sha256")
        != raw.get("prefix_source_sha256")
        or _aware(
            certificate.get("max_source_time_loaded"),
            name="authority source cutoff",
        )
        >= case_clock
        or _aware(
            certificate.get("max_engine_time_processed"),
            name="authority engine cutoff",
        )
        != case_clock
        or _aware(
            certificate.get("maximum_market_time"),
            name="authority market cutoff",
        )
        > case_clock
        or _aware(
            certificate.get("maximum_semantic_known_at"),
            name="authority semantic cutoff",
        )
        > case_clock
        or not isinstance(clock_records, list)
        or any(
            not isinstance(item, dict)
            or set(item) != {"path", "clock"}
            or _aware(
                item.get("clock"),
                name="authority clock record",
            )
            > case_clock
            for item in clock_records
        )
        or raw.get("opaque_case_id") != unit.get("opaque_case_id")
        or internal_raw.get("opaque_case_id")
        != authority.get("opaque_case_id")
        or raw.get("case_clock")
        != authority.get("selected_case", {}).get("case_clock")
    ):
        raise ValueError("semantic truth clock certificate changed")

    selected = authority["selected_case"]
    event_id = str(selected.get("semantic_event_id", ""))
    completion_path = case_root / "COMPLETED.json"
    _, pass2, selected_by_event = (
        _verify_selection_authority_for_review(
            selection_path,
            pass2_path,
            contract=contract,
        )
    )
    completion = (
        json.loads(completion_path.read_text(encoding="utf-8"))
        if completion_path.is_file() and not completion_path.is_symlink()
        else {}
    )
    if (
        not event_id
        or event_id != unit.get("semantic_event_id")
        or selected_by_event.get(event_id) != selected
        or authority.get("implementation_hashes")
        != pass2.get("implementation_hashes")
        or pass2.get("case_commit_hashes", {}).get(event_id)
        != unit["case_completion_sha256"]
        or completion_path.is_symlink()
        or not completion_path.is_file()
        or sha256_file(completion_path)
        != unit["case_completion_sha256"]
        or completion.get("semantic_event_id") != event_id
        or completion.get("file_hashes", {}).get(
            "authority/authority.json"
        )
        != sha256_file(authority_source)
    ):
        raise ValueError(
            "semantic truth authority lacks its exact Pass2 commitment"
        )
    timeframe = str(selected["timeframe"])
    rows = raw["panels"][timeframe]["bars"]

    def row_index(
        clock: str | None,
        *,
        field: str,
    ) -> int | None:
        if clock is None:
            return None
        matches = [
            int(row["bar_index"])
            for row in rows
            if row[field] == clock
        ]
        if len(matches) > 1:
            raise ValueError("semantic truth clock mapping is ambiguous")
        return None if not matches else matches[0]

    target_swing = authority["focus_target_swing"]
    prior_swing = authority.get("focus_prior_same_side_swing")
    bos = authority["focus_bos"]
    mapping = {
        "target_pivot_index": row_index(
            target_swing["pivot_start"],
            field="start",
        ),
        "target_confirmation_index": row_index(
            target_swing["confirmed_at"],
            field="end",
        ),
        "prior_same_side_pivot_index": (
            None
            if prior_swing is None
            else row_index(
                prior_swing["pivot_start"],
                field="start",
            )
        ),
        "pending_index": row_index(
            bos["pending_at"],
            field="end",
        ),
        "last_attempt_index": row_index(
            bos.get("last_attempt_at"),
            field="end",
        ),
        "attempt_indices": [
            row_index(value, field="end")
            for value in bos.get("attempt_clocks", ())
        ],
        "resolved_index": row_index(
            bos.get("resolved_at"),
            field="end",
        ),
    }
    required_mapping = (
        mapping["target_pivot_index"],
        mapping["target_confirmation_index"],
        mapping["pending_index"],
        *mapping["attempt_indices"],
        *(
            ()
            if bos.get("resolved_at") is None
            else (mapping["resolved_index"],)
        ),
    )
    if any(value is None for value in required_mapping):
        raise ValueError(
            "selected semantic truth is not fully materializable"
        )
    enumerated_truth = authority.get("enumerated_truth")
    if not isinstance(enumerated_truth, Mapping):
        raise ValueError("semantic authority enumeration is absent")
    _validate_enumerated_judgment(
        {
            "swings": enumerated_truth.get("swings"),
            "bos_candidates": enumerated_truth.get("bos_candidates"),
            "confidence": 1.0,
            "issue_codes": [],
        },
        raw_evidence=raw,
        contract=contract,
    )
    focus_candidate = {
        "timeframe": timeframe,
        "direction": bos["direction"],
        "target_pivot_index": mapping["target_pivot_index"],
        "target_price_ticks": bos["target_ticks"],
        "pending_index": mapping["pending_index"],
        "wick_attempt_indices": mapping["attempt_indices"],
        "resolved_index": mapping["resolved_index"],
        "lifecycle": bos["lifecycle"],
        "scope": bos["scope"],
        "failure_reason": bos["failure_reason"],
    }
    if focus_candidate not in enumerated_truth["bos_candidates"]:
        raise ValueError(
            "semantic authority enumeration omits the selected BOS"
        )
    reviewed = review["judgment"]

    def omissions(
        expected_values: Sequence[Mapping[str, Any]],
        observed_values: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        expected_by_key = {
            canonical_json(to_primitive(dict(item))).decode("utf-8"): dict(item)
            for item in expected_values
        }
        observed_by_key = {
            canonical_json(to_primitive(dict(item))).decode("utf-8"): dict(item)
            for item in observed_values
        }
        return (
            [
                expected_by_key[key]
                for key in sorted(expected_by_key.keys() - observed_by_key.keys())
            ],
            [
                observed_by_key[key]
                for key in sorted(observed_by_key.keys() - expected_by_key.keys())
            ],
        )

    omitted_swings, extra_swings = omissions(
        enumerated_truth["swings"],
        reviewed["swings"],
    )
    omitted_bos, extra_bos = omissions(
        enumerated_truth["bos_candidates"],
        reviewed["bos_candidates"],
    )
    swing_exact = reviewed["swings"] == enumerated_truth["swings"]
    bos_exact = (
        reviewed["bos_candidates"]
        == enumerated_truth["bos_candidates"]
    )
    computed_issues = (
        ["MULTIPLE_CANDIDATES_OMITTED"]
        if omitted_swings or omitted_bos
        else []
    )
    visual_coverage = _semantic_visual_coverage(
        raw,
        enumerated_truth,
    )
    truth = {
        "format_version": 1,
        "artifact": "v3_structure_bos_semantic_truth",
        "audit_id": blind_set["audit_id"],
        "review_unit_id": review_unit_id,
        "case_clock": selected["case_clock"],
        "timeframe": timeframe,
        "selected_case": selected,
        "target_swing": target_swing,
        "prior_same_side_swing": prior_swing,
        "source_structure": authority.get("focus_source_structure"),
        "bos": bos,
        "raw_index_mapping": mapping,
        "enumerated_truth": enumerated_truth,
        "locked_review_judgment": reviewed,
        "swing_exact_agreement": swing_exact,
        "bos_exact_agreement": bos_exact,
        "strict_case_agreement": swing_exact and bos_exact,
        "omitted_swings": omitted_swings,
        "extra_swings": extra_swings,
        "omitted_bos_candidates": omitted_bos,
        "extra_bos_candidates": extra_bos,
        "computed_issue_codes": computed_issues,
        "visualization_coverage": visual_coverage,
        "context_truncated": bool(
            enumerated_truth.get("context", {}).get(
                "context_truncated",
                True,
            )
        ),
        "future_present": False,
        "action_present": False,
    }

    staging = target.parent / f".{target.name}.staging"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(
            "semantic truth overlay has a partial staging root"
        )
    staging.mkdir(parents=True)
    _write_new(staging / "semantic_truth.json", truth)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tick_size = float(raw["tick_size"])
    figure, axes = plt.subplots(
        4,
        2,
        figsize=(18, 14),
        dpi=110,
        constrained_layout=True,
        squeeze=False,
    )
    for row_number, panel_timeframe in enumerate(CORE_TIMEFRAMES):
        price_axis = axes[row_number][0]
        index_axis = axes[row_number][1]
        panel_rows = raw["panels"][panel_timeframe.value]["bars"]
        panel_coverage = visual_coverage["price_panels"][
            panel_timeframe.value
        ]
        first = int(panel_coverage["first_history_index"])
        visible_rows = panel_rows[first:]
        candles = tuple(
            Candle(
                timeframe=panel_timeframe,
                start=row["start"],
                end=row["end"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                symbol=row["symbol"],
                instrument_id=row["instrument_id"],
                observed_minutes=row["observed_minutes"],
                expected_minutes=row["expected_minutes"],
                complete=row["complete"],
                real_minutes=row["real_minutes"],
                synthetic_minutes=row["synthetic_minutes"],
            )
            for row in visible_rows
        )
        SemanticCaseVisualizer._candles(
            price_axis,
            candles,
            first_history_index=first,
            tick_size=tick_size,
        )
        panel_swings = [
            value
            for value in enumerated_truth["swings"]
            if value["timeframe"] == panel_timeframe.value
        ]
        panel_bos = [
            value
            for value in enumerated_truth["bos_candidates"]
            if value["timeframe"] == panel_timeframe.value
        ]
        for index, candidate in enumerate(panel_swings):
            history_index = candidate["pivot_index"]
            if history_index >= first:
                price_axis.axvline(
                    history_index - first,
                    color=(
                        "#2563eb"
                        if candidate["side"] == SwingSide.HIGH.value
                        else "#0891b2"
                    ),
                    linewidth=0.7,
                    linestyle=":",
                    alpha=0.45,
                    label=(
                        "all typed swings"
                        if index == 0
                        else None
                    ),
                )
        for index, candidate in enumerate(panel_bos):
            for label, history_index, color, linestyle in (
                (
                    "all BOS pending",
                    candidate["pending_index"],
                    "#d97706",
                    "--",
                ),
                *tuple(
                    (
                        "all BOS attempts",
                        attempt,
                        "#7c3aed",
                        ":",
                    )
                    for attempt in candidate[
                        "wick_attempt_indices"
                    ]
                ),
                (
                    "all BOS resolved",
                    candidate["resolved_index"],
                    "#111827",
                    "-.",
                ),
            ):
                if history_index is not None and history_index >= first:
                    price_axis.axvline(
                        history_index - first,
                        color=color,
                        linewidth=0.65,
                        linestyle=linestyle,
                        alpha=0.40,
                        label=label if index == 0 else None,
                    )
        if panel_timeframe.value == timeframe:
            overlays = (
                ("target", mapping["target_pivot_index"], "#2563eb"),
                ("pending", mapping["pending_index"], "#d97706"),
                *tuple(
                    (
                        f"wick attempt {index + 1}",
                        attempt,
                        "#7c3aed",
                    )
                    for index, attempt in enumerate(
                        mapping["attempt_indices"]
                    )
                ),
                ("resolved", mapping["resolved_index"], "#111827"),
            )
            for label, history_index, color in overlays:
                if (
                    history_index is not None
                    and history_index >= first
                ):
                    price_axis.axvline(
                        history_index - first,
                        color=color,
                        linewidth=1.1,
                        label=label,
                    )
        if price_axis.get_legend_handles_labels()[0]:
            price_axis.legend(loc="upper left", fontsize=7)
        price_axis.set_title(
            (
                f"{panel_timeframe.value} fixed latest-price window "
                f"({len(visible_rows)}/"
                f"{panel_coverage['maximum_bars']} bars; no compression)"
            ),
            loc="left",
            fontsize=9,
        )
        index_map = visual_coverage["candidate_index_maps"][
            panel_timeframe.value
        ]
        swing_points = {
            "swing pivot": ([], [], "#2563eb", "o"),
            "swing observed": ([], [], "#0891b2", "^"),
            "swing confirmed": ([], [], "#0f766e", "s"),
            "swing broken": ([], [], "#b91c1c", "x"),
        }
        for candidate in index_map["swing_candidates"]:
            for label, field, lane in (
                ("swing pivot", "pivot_index", 0),
                ("swing observed", "observed_index", 1),
                ("swing confirmed", "confirmation_index", 2),
                ("swing broken", "broken_index", 3),
            ):
                history_index = candidate[field]
                if history_index is not None:
                    swing_points[label][0].append(history_index)
                    swing_points[label][1].append(lane)
        bos_points = {
            "BOS target": ([], [], "#2563eb", "o"),
            "BOS pending": ([], [], "#d97706", ">"),
            "BOS attempt": ([], [], "#7c3aed", "D"),
            "BOS resolved": ([], [], "#111827", "x"),
        }
        for candidate in index_map["bos_candidates"]:
            bos_points["BOS target"][0].append(
                candidate["target_pivot_index"]
            )
            bos_points["BOS target"][1].append(5)
            bos_points["BOS pending"][0].append(
                candidate["pending_index"]
            )
            bos_points["BOS pending"][1].append(6)
            for history_index in candidate["wick_attempt_indices"]:
                bos_points["BOS attempt"][0].append(history_index)
                bos_points["BOS attempt"][1].append(7)
            if candidate["resolved_index"] is not None:
                bos_points["BOS resolved"][0].append(
                    candidate["resolved_index"]
                )
                bos_points["BOS resolved"][1].append(8)
        for label, (x_values, y_values, color, marker) in (
            *swing_points.items(),
            *bos_points.items(),
        ):
            if x_values:
                index_axis.scatter(
                    x_values,
                    y_values,
                    color=color,
                    marker=marker,
                    s=18,
                    linewidths=0.7,
                    alpha=0.75,
                    label=f"{label} ({len(x_values)})",
                )
        if panel_timeframe.value == timeframe:
            focus_values = [
                mapping["target_pivot_index"],
                mapping["pending_index"],
                *mapping["attempt_indices"],
                mapping["resolved_index"],
            ]
            focus_values = [
                value for value in focus_values if value is not None
            ]
            if focus_values:
                index_axis.scatter(
                    focus_values,
                    [9] * len(focus_values),
                    facecolors="none",
                    edgecolors="#dc2626",
                    marker="o",
                    s=52,
                    linewidths=1.2,
                    label=f"selected focal lifecycle ({len(focus_values)})",
                )
        index_axis.set_xlim(-0.5, max(0.5, len(panel_rows) - 0.5))
        index_axis.set_ylim(-0.75, 9.75)
        index_axis.set_yticks(range(10))
        index_axis.set_yticklabels(
            [
                "swing pivot",
                "swing observed",
                "swing confirmed",
                "swing broken",
                "",
                "BOS target",
                "BOS pending",
                "BOS attempt",
                "BOS resolved",
                "selected focus",
            ],
            fontsize=7,
        )
        index_axis.grid(
            True,
            axis="x",
            color="#dbe4ee",
            linewidth=0.4,
            alpha=0.7,
        )
        if index_axis.get_legend_handles_labels()[0]:
            index_axis.legend(
                loc="upper left",
                fontsize=6,
                ncol=2,
            )
        index_axis.set_title(
            (
                f"{panel_timeframe.value} complete typed-candidate index map "
                f"({len(panel_swings)} swings, {len(panel_bos)} BOS; "
                "categorical index, not a price axis)"
            ),
            loc="left",
            fontsize=9,
        )
    figure.suptitle(
        "LOCKED BLIND REVIEW → STRUCTURE/BOS TRUTH OVERLAY",
        fontsize=13,
        weight="bold",
    )
    figure.savefig(
        staging / "semantic_truth_overlay.png",
        bbox_inches="tight",
    )
    plt.close(figure)
    lock = {
        "format_version": 1,
        "artifact": "v3_semantic_truth_overlay_lock",
        "audit_id": blind_set["audit_id"],
        "audit_contract_content_hash": content_hash(contract),
        "review_unit_id": review_unit_id,
        "authority_path": str(authority_source.resolve()),
        "authority_sha256": sha256_file(authority_source),
        "selection_manifest_sha256": sha256_file(selection_path),
        "pass2_manifest_sha256": sha256_file(pass2_path),
        "blind_set_manifest_sha256": sha256_file(blind_set_path),
        "global_review_lock_sha256": sha256_file(global_lock_path),
        "locked_review_sha256": sha256_file(
            review_root / "review.json"
        ),
        "semantic_truth_sha256": sha256_file(
            staging / "semantic_truth.json"
        ),
        "semantic_truth_overlay_sha256": sha256_file(
            staging / "semantic_truth_overlay.png"
        ),
        "strict_case_agreement": truth["strict_case_agreement"],
        "future_present": False,
    }
    _write_new(staging / "TRUTH_OVERLAY.json", lock)
    os.replace(staging, target)
    return target / "TRUTH_OVERLAY.json"


def compute_semantic_review_gate(
    contract: Mapping[str, Any],
    *,
    case_count: int,
    strict_case_agreement_count: int,
) -> dict[str, Any]:
    """Compute the frozen gate without a caller-supplied threshold."""

    if (
        type(case_count) is not int
        or type(strict_case_agreement_count) is not int
        or case_count < 1
        or not 0 <= strict_case_agreement_count <= case_count
    ):
        raise ValueError("semantic agreement counts are invalid")
    gate_text = str(
        contract.get("release_gates", {}).get(
            "blind_strict_case_agreement",
            "",
        )
    )
    gate_match = re.fullmatch(
        r"at least ([0-9]+)/([0-9]+); "
        r"missing and uncertain count as disagreement",
        gate_text,
    )
    if gate_match is None or int(gate_match.group(2)) != case_count:
        raise ValueError("semantic blind agreement gate is not computable")
    required = int(gate_match.group(1))
    formula_gate = str(
        contract.get("release_gates", {}).get(
            "formula_and_causal_agreement",
            "",
        )
    )
    formula_match = re.fullmatch(r"([0-9]+)/([0-9]+)", formula_gate)
    if (
        formula_match is None
        or int(formula_match.group(1)) != case_count
        or int(formula_match.group(2)) != case_count
    ):
        raise ValueError("semantic causal agreement gate is not computable")
    return {
        "case_count": case_count,
        "strict_case_agreement_count": strict_case_agreement_count,
        "strict_case_agreement_required": required,
        "strict_case_agreement_passed": (
            strict_case_agreement_count >= required
        ),
    }


def materialize_semantic_review_scorecard(
    truth_overlay_roots: Iterable[str | Path],
    destination: str | Path,
    *,
    blind_set_manifest: str | Path,
    selection_manifest: str | Path,
    pass2_manifest: str | Path,
    global_review_lock: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    """Aggregate the preregistered blind gate from exact truth overlays."""

    roots = tuple(Path(value) for value in truth_overlay_roots)
    if len(roots) != len({value.resolve() for value in roots}):
        raise ValueError("semantic truth overlay roots are duplicated")
    blind_set_path = Path(blind_set_manifest)
    selection_path = Path(selection_manifest)
    pass2_path = Path(pass2_manifest)
    global_lock_path = Path(global_review_lock)
    blind_set, units = _verify_blind_set_manifest(
        blind_set_path,
        pass2_path,
        selection_path,
        contract=contract,
    )
    _, verified_reviews = _verify_global_review_lock(
        global_lock_path,
        blind_set_path=blind_set_path,
        selection_path=selection_path,
        pass2_path=pass2_path,
        contract=contract,
        blind_set=blind_set,
        units=units,
    )
    if len(roots) != len(units):
        raise ValueError("every registered case requires one truth overlay")
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError("semantic review scorecard is immutable")
    _assert_path_disjoint(
        target,
        (
            blind_set_path.parent,
            selection_path.parent,
            pass2_path.parent,
            *(root for root, _ in verified_reviews.values()),
            *roots,
        ),
        label="semantic review scorecard",
    )
    expected_lock_fields = {
        "format_version",
        "artifact",
        "audit_id",
        "audit_contract_content_hash",
        "review_unit_id",
        "authority_path",
        "authority_sha256",
        "selection_manifest_sha256",
        "pass2_manifest_sha256",
        "blind_set_manifest_sha256",
        "global_review_lock_sha256",
        "locked_review_sha256",
        "semantic_truth_sha256",
        "semantic_truth_overlay_sha256",
        "strict_case_agreement",
        "future_present",
    }
    results: dict[str, dict[str, Any]] = {}
    authority_packet_trees: set[Path] = set()
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("semantic truth overlay root is not regular")
        observed_files = {value.name for value in root.iterdir()}
        if observed_files != {
            "semantic_truth.json",
            "semantic_truth_overlay.png",
            "TRUTH_OVERLAY.json",
        } or any(
            value.is_symlink() or not value.is_file()
            for value in root.iterdir()
        ):
            raise ValueError("semantic truth overlay tree changed")
        lock_path = root / "TRUTH_OVERLAY.json"
        truth_path = root / "semantic_truth.json"
        image_path = root / "semantic_truth_overlay.png"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        review_id = str(lock.get("review_unit_id", ""))
        if (
            set(lock) != expected_lock_fields
            or lock.get("format_version") != 1
            or lock.get("artifact")
            != "v3_semantic_truth_overlay_lock"
            or lock.get("audit_id") != contract.get("audit_id")
            or lock.get("audit_contract_content_hash")
            != content_hash(contract)
            or review_id not in units
            or review_id in results
            or lock.get("selection_manifest_sha256")
            != sha256_file(selection_path)
            or lock.get("pass2_manifest_sha256")
            != sha256_file(pass2_path)
            or lock.get("blind_set_manifest_sha256")
            != sha256_file(blind_set_path)
            or lock.get("global_review_lock_sha256")
            != sha256_file(global_lock_path)
            or lock.get("semantic_truth_sha256")
            != sha256_file(truth_path)
            or lock.get("semantic_truth_overlay_sha256")
            != sha256_file(image_path)
            or lock.get("future_present") is not False
        ):
            raise ValueError("semantic truth overlay lock changed")
        review_root, review = verified_reviews[review_id]
        authority_path = Path(str(lock["authority_path"]))
        authority_packet_trees.add(
            _protected_authority_packet_tree(authority_path).resolve()
        )
        if (
            authority_path.is_symlink()
            or not authority_path.is_file()
            or lock.get("authority_sha256")
            != sha256_file(authority_path)
            or lock.get("locked_review_sha256")
            != sha256_file(review_root / "review.json")
        ):
            raise ValueError(
                "semantic scorecard authority/review commitment changed"
            )
        authority = json.loads(
            authority_path.read_text(encoding="utf-8")
        )
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        raw_evidence_path = (
            authority_path.parent.parent
            / "blind"
            / "blind_raw_evidence.json"
        )
        if (
            raw_evidence_path.is_symlink()
            or not raw_evidence_path.is_file()
        ):
            raise ValueError(
                "semantic scorecard raw visual evidence is absent"
            )
        raw_evidence = json.loads(
            raw_evidence_path.read_text(encoding="utf-8")
        )
        _validate_semantic_visual_coverage(
            truth.get("visualization_coverage", {}),
            raw_evidence=raw_evidence,
            enumerated_truth=truth.get("enumerated_truth", {}),
        )
        swing_exact = (
            truth.get("locked_review_judgment", {}).get("swings")
            == truth.get("enumerated_truth", {}).get("swings")
        )
        bos_exact = (
            truth.get("locked_review_judgment", {}).get("bos_candidates")
            == truth.get("enumerated_truth", {}).get("bos_candidates")
        )
        strict = swing_exact and bos_exact
        event_id = str(
            authority.get("selected_case", {}).get(
                "semantic_event_id",
                "",
            )
        )
        completion_path = authority_path.parent.parent / "COMPLETED.json"
        completion = (
            json.loads(completion_path.read_text(encoding="utf-8"))
            if completion_path.is_file()
            and not completion_path.is_symlink()
            else {}
        )
        if (
            authority.get("audit_id") != contract.get("audit_id")
            or authority.get("audit_contract_content_hash")
            != content_hash(contract)
            or authority.get("opaque_case_id")
            != units[review_id]["internal_opaque_case_id"]
            or authority.get("enumerated_truth")
            != truth.get("enumerated_truth")
            or review.get("judgment")
            != truth.get("locked_review_judgment")
            or truth.get("review_unit_id") != review_id
            or truth.get("swing_exact_agreement") is not swing_exact
            or truth.get("bos_exact_agreement") is not bos_exact
            or truth.get("strict_case_agreement") is not strict
            or lock.get("strict_case_agreement") is not strict
            or truth.get("context_truncated") is not False
            or truth.get("future_present") is not False
            or truth.get("action_present") is not False
            or truth.get("visualization_coverage", {}).get(
                "all_typed_candidates_in_index_maps"
            )
            is not True
            or truth.get("visualization_coverage", {}).get(
                "price_axis_compression"
            )
            is not False
            or event_id != units[review_id]["semantic_event_id"]
            or completion_path.is_symlink()
            or not completion_path.is_file()
            or sha256_file(completion_path)
            != units[review_id]["case_completion_sha256"]
            or completion.get("semantic_event_id") != event_id
            or completion.get("file_hashes", {}).get(
                "authority/authority.json"
            )
            != lock.get("authority_sha256")
        ):
            raise ValueError("semantic scorecard case truth changed")
        omitted = bool(
            truth.get("omitted_swings")
            or truth.get("omitted_bos_candidates")
        )
        expected_computed = (
            ["MULTIPLE_CANDIDATES_OMITTED"] if omitted else []
        )
        if truth.get("computed_issue_codes") != expected_computed:
            raise ValueError("semantic scorecard omission issue changed")
        results[review_id] = {
            "review_unit_id": review_id,
            "semantic_event_id": event_id,
            "truth_overlay_lock_sha256": sha256_file(lock_path),
            "strict_case_agreement": strict,
            "swing_exact_agreement": swing_exact,
            "bos_exact_agreement": bos_exact,
            "omitted_swing_count": len(
                truth.get("omitted_swings", ())
            ),
            "extra_swing_count": len(truth.get("extra_swings", ())),
            "omitted_bos_count": len(
                truth.get("omitted_bos_candidates", ())
            ),
            "extra_bos_count": len(
                truth.get("extra_bos_candidates", ())
            ),
        }
    if set(results) != set(units):
        raise ValueError("semantic scorecard case set is incomplete")
    _assert_path_disjoint(
        target,
        authority_packet_trees,
        label="semantic review scorecard",
    )
    ordered = sorted(
        results.values(),
        key=lambda item: item["review_unit_id"],
    )
    strict_count = sum(
        bool(item["strict_case_agreement"]) for item in ordered
    )
    gate = compute_semantic_review_gate(
        contract,
        case_count=len(ordered),
        strict_case_agreement_count=strict_count,
    )
    payload = {
        "format_version": 1,
        "artifact": "v3_structure_bos_semantic_review_scorecard",
        "audit_id": contract["audit_id"],
        "audit_contract_content_hash": content_hash(contract),
        "selection_manifest_sha256": sha256_file(selection_path),
        "pass2_manifest_sha256": sha256_file(pass2_path),
        "blind_set_manifest_sha256": sha256_file(blind_set_path),
        "global_review_lock_sha256": sha256_file(global_lock_path),
        "case_count": gate["case_count"],
        "pass2_reproduction_count": len(ordered),
        "strict_case_agreement_count": gate[
            "strict_case_agreement_count"
        ],
        "strict_case_agreement_required": gate[
            "strict_case_agreement_required"
        ],
        "strict_case_agreement_passed": gate[
            "strict_case_agreement_passed"
        ],
        "result_root": content_hash(ordered),
        "results": ordered,
        "future_present": False,
        "economic_authority": False,
    }
    _write_new(target, payload)
    return target


__all__ = [
    "CASE_CLASSES",
    "DEFAULT_AUDIT_CONTRACT",
    "FixedSemanticCaseSelector",
    "SemanticCase",
    "SemanticCaseVisualizer",
    "SemanticImageArtifact",
    "classify_bos_case",
    "compute_semantic_review_gate",
    "load_audit_contract",
    "lock_blind_review",
    "lock_review_set",
    "materialize_blind_review_export",
    "materialize_blind_unit",
    "materialize_semantic_review_scorecard",
    "materialize_semantic_truth_overlay",
    "selection_score",
    "semantic_case_context_complete",
    "semantic_clock_certificate",
    "semantic_event_id",
]
