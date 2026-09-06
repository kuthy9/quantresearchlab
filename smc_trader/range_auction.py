"""Causal H1 dealing-range and completed-1m manipulation primitives."""
from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import pandas as pd

from .model import (
    BALANCE_PRICE_TEST_KINDS,
    BALANCE_CLAIM_ABANDONED,
    BALANCE_CLAIM_CONFIRMED,
    Candle,
    DealingRangeLifecycle,
    DealingRangeState,
    RANGE_AUCTION_HARD_BOUNDARY_REASONS,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolState,
    ManipulationLifecycle,
    ManipulationSourceDisposition,
    ManipulationSourceDispositionKind,
    ManipulationState,
    RANGE_MATURITY_GATE_NAMES,
    RANGE_PAIR_FUNNEL_COUNTS,
    RangeFormationFunnelSnapshot,
    SupportResistanceLifecycle,
    SupportResistanceState,
    Timeframe,
    aware_timestamp,
    clamp,
)


@dataclass(frozen=True)
class RangeAuctionProtocol:
    """Executable mirror of the frozen Group 4 contract."""

    protocol_hash: str
    source_group12_protocol_hash: str
    tick_size: float
    h1_atr_period: int
    m1_atr_period: int
    reacceptance_hold_bars: int
    outside_acceptance_closes: int
    resolution_deadline_real_1m_bars: int
    minimum_candidate_real_h1_bars: int
    maximum_forming_real_h1_bars: int
    minimum_boundary_touches_each: int
    minimum_midpoint_crossings: int
    minimum_inside_close_fraction: float
    maximum_width_atr_at_formation: float
    compression_early_real_h1_bars: int
    compression_late_real_h1_bars: int
    maximum_compression_ratio: float
    maximum_ranges: int
    maximum_manipulations: int
    balance_price_test_band_atr_fraction: float
    balance_price_test_band_minimum_ticks: int
    balance_deep_penetration_atr_fraction: float
    balance_minimum_price_test_generations_each: int
    protocol_version: str = "3.2.0-group4.1"
    balance_sub_protocol_version: str = "balance_range_v1.2"

    def __post_init__(self) -> None:
        hashes = (
            self.protocol_hash,
            self.source_group12_protocol_hash,
        )
        if (
            any(
                len(value) != 64
                or any(character not in "0123456789abcdef"
                       for character in value)
                for value in hashes
            )
            or not self.protocol_version
            or not math.isfinite(float(self.tick_size))
            or self.tick_size <= 0.0
            or self.h1_atr_period < 1
            or self.m1_atr_period < 1
            or self.reacceptance_hold_bars != 1
            or self.outside_acceptance_closes != 2
            or self.resolution_deadline_real_1m_bars != 5
            or self.minimum_candidate_real_h1_bars < 2
            or self.maximum_forming_real_h1_bars
            < self.minimum_candidate_real_h1_bars
            or self.minimum_boundary_touches_each < 2
            or self.minimum_midpoint_crossings < 1
            or not 0.0 < self.minimum_inside_close_fraction <= 1.0
            or self.maximum_width_atr_at_formation <= 0.0
            or self.compression_early_real_h1_bars < 1
            or self.compression_late_real_h1_bars < 1
            or self.maximum_compression_ratio <= 0.0
            or self.maximum_ranges < 1
            or self.maximum_manipulations < 1
            or not self.balance_sub_protocol_version
            or not 0.0 < self.balance_price_test_band_atr_fraction <= 1.0
            or self.balance_price_test_band_minimum_ticks < 1
            or not 0.0 < self.balance_deep_penetration_atr_fraction <= 1.0
            or self.balance_minimum_price_test_generations_each < 2
        ):
            raise ValueError("Group 4 protocol differs from its frozen contract")

    @classmethod
    def from_file(cls, path: str | Path) -> "RangeAuctionProtocol":
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = Path(__file__).resolve().parents[1] / source
        raw = source.read_bytes()
        payload = json.loads(raw)
        parameters = payload["engineering_parameters"]
        return cls(
            protocol_hash=hashlib.sha256(raw).hexdigest(),
            source_group12_protocol_hash=payload["upstream"][
                "group12_protocol_sha256"
            ],
            tick_size=float(payload["tick_size"]),
            h1_atr_period=int(parameters["h1_atr_period"]),
            m1_atr_period=int(parameters["m1_atr_period"]),
            reacceptance_hold_bars=int(
                parameters["reacceptance_hold_bars"]
            ),
            outside_acceptance_closes=int(
                parameters["outside_acceptance_closes"]
            ),
            resolution_deadline_real_1m_bars=int(
                parameters["resolution_deadline_real_1m_bars"]
            ),
            minimum_candidate_real_h1_bars=int(
                parameters["minimum_candidate_real_h1_bars"]
            ),
            maximum_forming_real_h1_bars=int(
                parameters["maximum_forming_real_h1_bars"]
            ),
            minimum_boundary_touches_each=int(
                parameters["minimum_boundary_touches_each"]
            ),
            minimum_midpoint_crossings=int(
                parameters["minimum_midpoint_crossings"]
            ),
            minimum_inside_close_fraction=float(
                parameters["minimum_inside_close_fraction"]
            ),
            maximum_width_atr_at_formation=float(
                parameters["maximum_width_atr_at_formation"]
            ),
            compression_early_real_h1_bars=int(
                parameters["compression_early_real_h1_bars"]
            ),
            compression_late_real_h1_bars=int(
                parameters["compression_late_real_h1_bars"]
            ),
            maximum_compression_ratio=float(
                parameters["maximum_compression_ratio"]
            ),
            maximum_ranges=int(
                parameters["retained_dealing_range_states"]
            ),
            maximum_manipulations=int(
                parameters["retained_manipulation_states"]
            ),
            balance_price_test_band_atr_fraction=float(
                parameters["balance_price_test_band_atr_fraction"]
            ),
            balance_price_test_band_minimum_ticks=int(
                parameters["balance_price_test_band_minimum_ticks"]
            ),
            balance_deep_penetration_atr_fraction=float(
                parameters["balance_deep_penetration_atr_fraction"]
            ),
            balance_minimum_price_test_generations_each=int(
                parameters["balance_minimum_price_test_generations_each"]
            ),
            protocol_version=payload["protocol_version"],
            balance_sub_protocol_version=payload[
                "balance_sub_protocol_version"
            ],
        )



@dataclass(frozen=True)
class RangeAuctionUpdate:
    dealing_ranges: tuple[DealingRangeState, ...]
    manipulations: tuple[ManipulationState, ...]
    range_boundary_inventory: tuple[LiquidityInventoryItem, ...]
    range_transitions: tuple[DealingRangeState, ...] = ()
    manipulation_transitions: tuple[ManipulationState, ...] = ()
    ambiguous_sweep_item_ids: tuple[str, ...] = ()
    atr_unready_sweep_item_ids: tuple[str, ...] = ()
    source_dispositions: tuple[ManipulationSourceDisposition, ...] = ()
    range_funnel: tuple[RangeFormationFunnelSnapshot, ...] = ()
    boundary_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "dealing_ranges",
            "manipulations",
            "range_boundary_inventory",
            "range_transitions",
            "manipulation_transitions",
            "ambiguous_sweep_item_ids",
            "atr_unready_sweep_item_ids",
            "source_dispositions",
            "range_funnel",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            self.boundary_reason is not None
            and self.boundary_reason not in RANGE_AUCTION_HARD_BOUNDARY_REASONS
        ):
            raise ValueError("Group 4 update has an unregistered boundary")
        if (
            len(self.ambiguous_sweep_item_ids)
            != len(set(self.ambiguous_sweep_item_ids))
            or len(self.atr_unready_sweep_item_ids)
            != len(set(self.atr_unready_sweep_item_ids))
            or set(self.ambiguous_sweep_item_ids)
            & set(self.atr_unready_sweep_item_ids)
        ):
            raise ValueError("Group 4 unclassified sweep identities repeat")
        if any(
            not isinstance(item, ManipulationSourceDisposition)
            for item in self.source_dispositions
        ):
            raise TypeError(
                "Group 4 source disposition is not typed"
            )
        disposition_ids = tuple(
            item.source_inventory_item_id
            for item in self.source_dispositions
        )
        if len(disposition_ids) != len(set(disposition_ids)):
            raise ValueError(
                "Group 4 source dispositions are not mutually exclusive"
            )
        ambiguous_ids = {
            item.source_inventory_item_id
            for item in self.source_dispositions
            if item.disposition
            is ManipulationSourceDispositionKind.AMBIGUOUS_DUAL_SIDE
        }
        atr_unready_ids = {
            item.source_inventory_item_id
            for item in self.source_dispositions
            if item.disposition
            is ManipulationSourceDispositionKind.ATR_UNREADY
        }
        if self.source_dispositions and (
            ambiguous_ids != set(self.ambiguous_sweep_item_ids)
            or atr_unready_ids != set(self.atr_unready_sweep_item_ids)
        ):
            raise ValueError(
                "Group 4 compatibility sweep identities disagree with "
                "source dispositions"
            )
        if any(
            not isinstance(item, RangeFormationFunnelSnapshot)
            for item in self.range_funnel
        ):
            raise TypeError("Group 4 range funnel record is not typed")
        funnel_clocks = tuple(item.observed_at for item in self.range_funnel)
        if (
            funnel_clocks != tuple(sorted(funnel_clocks))
            or len(funnel_clocks) != len(set(funnel_clocks))
            or (self.boundary_reason is not None and funnel_clocks)
        ):
            raise ValueError("Group 4 range funnel clocks are invalid")


@dataclass
class _RangeWork:
    bars: deque[Candle]
    true_ranges: deque[float]
    # Whether the previous completed bar was inside each boundary's tolerance
    # band, and the verdict of every test generation so far.  Occupancy is what
    # makes a run of bars hugging one level a single test.
    lower_in_band: bool = False
    upper_in_band: bool = False
    lower_test_kinds: list[str] = field(default_factory=list)
    upper_test_kinds: list[str] = field(default_factory=list)

    def clone(self) -> "_RangeWork":
        return _RangeWork(
            deque(self.bars, maxlen=self.bars.maxlen),
            deque(
                self.true_ranges,
                maxlen=self.true_ranges.maxlen,
            ),
            self.lower_in_band,
            self.upper_in_band,
            list(self.lower_test_kinds),
            list(self.upper_test_kinds),
        )


@dataclass(frozen=True)
class _RangePairPartition:
    live_structural_pairs: tuple[
        tuple[SupportResistanceState, SupportResistanceState],
        ...,
    ]
    invalid_geometry_pairs: tuple[
        tuple[SupportResistanceState, SupportResistanceState],
        ...,
    ]
    geometry_valid_pairs: tuple[
        tuple[SupportResistanceState, SupportResistanceState],
        ...,
    ]
    close_outside_pairs: tuple[
        tuple[SupportResistanceState, SupportResistanceState],
        ...,
    ]
    admitted_pairs: tuple[
        tuple[SupportResistanceState, SupportResistanceState],
        ...,
    ]
    cold_blocked_pairs: tuple[
        tuple[SupportResistanceState, SupportResistanceState],
        ...,
    ]
    available_pairs: tuple[
        tuple[SupportResistanceState, SupportResistanceState],
        ...,
    ]


@dataclass(frozen=True)
class _RangeGateEvaluation:
    range_id: str
    gates: tuple[tuple[str, float, float, float], ...]
    unmet: tuple[str, ...]


@dataclass(frozen=True)
class _ManipulationSource:
    side: str
    source_kind: str
    source_id: str
    source_protocol_hash: str
    source_timeframe: Timeframe
    inventory: LiquidityInventoryItem
    formed_at: pd.Timestamp
    eligible_at: pd.Timestamp
    lower_bound: float
    upper_bound: float
    boundary_price: float


def _identity(*parts: object) -> str:
    payload = "|".join(
        value.isoformat()
        if isinstance(value, pd.Timestamp)
        else str(value)
        for value in parts
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _true_range(candle: Candle, prior_close: float | None) -> float:
    if prior_close is None:
        return float(candle.high - candle.low)
    return float(
        max(
            candle.high - candle.low,
            abs(candle.high - prior_close),
            abs(candle.low - prior_close),
        )
    )


class CausalRangeAuctionTracker:
    """Incrementally describes range formation and sourced excursions."""

    def __init__(self, protocol: RangeAuctionProtocol) -> None:
        self.protocol = protocol
        self._ranges: dict[str, DealingRangeState] = {}
        self._range_order: deque[str] = deque()
        self._range_work: dict[str, _RangeWork] = {}
        self._range_inventory: dict[str, LiquidityInventoryItem] = {}
        self._manipulations: dict[str, ManipulationState] = {}
        self._manipulation_order: deque[str] = deque()
        self._h1_true_ranges: deque[float] = deque(
            maxlen=protocol.h1_atr_period
        )
        self._m1_true_ranges: deque[float] = deque(
            maxlen=protocol.m1_atr_period
        )
        self._prior_h1_close: float | None = None
        self._prior_m1_close: float | None = None
        self._identity: tuple[str, int] | None = None
        self._last_h1_end: pd.Timestamp | None = None
        self._last_m1_end: pd.Timestamp | None = None
        self._last_h1_raw_end: pd.Timestamp | None = None
        self._last_m1_raw_end: pd.Timestamp | None = None
        self._blocked_cold_pairs: set[tuple[str, str]] = set()
        self._last_h1_input: tuple[object, ...] | None = None
        self._last_m1_input: tuple[object, ...] | None = None
        self._last_h1_output: RangeAuctionUpdate | None = None
        self._last_m1_output: RangeAuctionUpdate | None = None
        self._last_boundary_input: tuple[object, ...] | None = None
        self._last_boundary_output: RangeAuctionUpdate | None = None

    @property
    def last_h1_end(self) -> pd.Timestamp | None:
        return self._last_h1_end

    @property
    def last_m1_end(self) -> pd.Timestamp | None:
        return self._last_m1_end

    def _transaction_clone(self) -> "CausalRangeAuctionTracker":
        candidate = copy(self)
        candidate._ranges = dict(self._ranges)
        candidate._range_order = deque(self._range_order)
        candidate._range_work = {
            key: value.clone()
            for key, value in self._range_work.items()
        }
        candidate._range_inventory = dict(self._range_inventory)
        candidate._manipulations = dict(self._manipulations)
        candidate._manipulation_order = deque(
            self._manipulation_order
        )
        candidate._h1_true_ranges = deque(
            self._h1_true_ranges,
            maxlen=self._h1_true_ranges.maxlen,
        )
        candidate._m1_true_ranges = deque(
            self._m1_true_ranges,
            maxlen=self._m1_true_ranges.maxlen,
        )
        candidate._blocked_cold_pairs = set(self._blocked_cold_pairs)
        return candidate

    def _commit(self, candidate: "CausalRangeAuctionTracker") -> None:
        self.__dict__.update(candidate.__dict__)

    def snapshot(self) -> RangeAuctionUpdate:
        return RangeAuctionUpdate(
            dealing_ranges=tuple(
                self._ranges[key] for key in self._range_order
            ),
            manipulations=tuple(
                self._manipulations[key]
                for key in self._manipulation_order
            ),
            range_boundary_inventory=tuple(
                sorted(
                    self._range_inventory.values(),
                    key=lambda item: (
                        item.confirmed_at,
                        item.item_id,
                    ),
                )
            ),
        )

    def mark_existing_source_pairs_ineligible(
        self,
        support_resistance: Iterable[SupportResistanceState],
    ) -> None:
        """Fail closed when the retained prefix cannot prove pair history."""

        zones = tuple(support_resistance)
        if any(
            not isinstance(zone, SupportResistanceState)
            or zone.timeframe is not Timeframe.H1
            for zone in zones
        ):
            raise ValueError("Group 4 cold pair source is not typed H1")
        supports = tuple(
            zone
            for zone in zones
            if (
                zone.source_kind == "structural_swing"
                and zone.side == "support"
                and self._source_is_live(zone)
            )
        )
        resistances = tuple(
            zone
            for zone in zones
            if (
                zone.source_kind == "structural_swing"
                and zone.side == "resistance"
                and self._source_is_live(zone)
            )
        )
        self._blocked_cold_pairs.update(
            (lower.zone_id, upper.zone_id)
            for lower in supports
            for upper in resistances
        )
        self._last_h1_input = None
        self._last_h1_output = None
        self._last_boundary_input = None
        self._last_boundary_output = None

    def _output(
        self,
        *,
        range_transitions: Iterable[DealingRangeState] = (),
        manipulation_transitions: Iterable[ManipulationState] = (),
        ambiguous: Iterable[str] = (),
        atr_unready: Iterable[str] = (),
        source_dispositions: Iterable[
            ManipulationSourceDisposition
        ] = (),
        range_funnel: Iterable[RangeFormationFunnelSnapshot] = (),
        boundary_reason: str | None = None,
    ) -> RangeAuctionUpdate:
        snapshot = self.snapshot()
        return RangeAuctionUpdate(
            dealing_ranges=snapshot.dealing_ranges,
            manipulations=snapshot.manipulations,
            range_boundary_inventory=(
                snapshot.range_boundary_inventory
            ),
            range_transitions=tuple(range_transitions),
            manipulation_transitions=tuple(
                manipulation_transitions
            ),
            ambiguous_sweep_item_ids=tuple(sorted(set(ambiguous))),
            atr_unready_sweep_item_ids=tuple(
                sorted(set(atr_unready))
            ),
            source_dispositions=tuple(source_dispositions),
            range_funnel=tuple(range_funnel),
            boundary_reason=boundary_reason,
        )

    def _validate_contract(self, candle: Candle) -> None:
        identity = (candle.symbol, int(candle.instrument_id))
        if self._identity is not None and identity != self._identity:
            raise ValueError(
                "Group 4 contract changed without a hard boundary"
            )
        self._identity = identity

    def _validate_h1(
        self,
        candle: Candle,
        zones: Sequence[SupportResistanceState],
    ) -> None:
        if (
            not isinstance(candle, Candle)
            or candle.timeframe is not Timeframe.H1
            or not candle.complete
        ):
            raise ValueError("Group 4 requires a completed H1 candle")
        if any(
            not isinstance(zone, SupportResistanceState)
            or zone.timeframe is not Timeframe.H1
            for zone in zones
        ):
            raise ValueError("Group 4 received a non-H1 range source")
        zone_ids = tuple(zone.zone_id for zone in zones)
        if (
            len(zone_ids) != len(set(zone_ids))
            or any(
                value is not None and value > candle.end
                for zone in zones
                for value in (
                    zone.confirmed_at,
                    zone.tested_at,
                    zone.broken_at,
                    zone.reaccepted_at,
                    zone.retired_at,
                    *zone.touch_times,
                )
            )
        ):
            raise ValueError(
                "Group 4 H1 source identity or knowledge clock is invalid"
            )
        self._validate_contract(candle)

    @staticmethod
    def _live_range(
        ranges: Iterable[DealingRangeState],
    ) -> DealingRangeState | None:
        live = tuple(
            state
            for state in ranges
            if state.lifecycle
            in {
                DealingRangeLifecycle.ACTIVE,
            }
        )
        if len(live) > 1:
            raise RuntimeError("Group 4 retained more than one live range")
        return live[0] if live else None

    def _range_statistics(
        self,
        state: DealingRangeState,
        work: _RangeWork,
        lower: SupportResistanceState | None,
        upper: SupportResistanceState | None,
    ) -> dict[str, object]:
        bars = tuple(work.bars)
        closes = tuple(float(candle.close) for candle in bars)
        sides = tuple(
            -1 if value < state.midpoint else 1
            for value in closes
            if value != state.midpoint
        )
        crossings = sum(
            left != right for left, right in zip(sides, sides[1:])
        )
        inside_fraction = (
            sum(
                state.lower_bound <= value <= state.upper_bound
                for value in closes
            )
            / len(closes)
        )
        ranges = tuple(work.true_ranges)
        early_count = self.protocol.compression_early_real_h1_bars
        late_count = self.protocol.compression_late_real_h1_bars
        compression_ratio = (
            median(ranges[-late_count:])
            / max(median(ranges[:early_count]), self.protocol.tick_size)
            if len(ranges) >= max(early_count, late_count)
            else 1.0
        )
        lower_count = (
            state.lower_touch_count
            if lower is None
            else int(lower.total_touch_count)
        )
        upper_count = (
            state.upper_touch_count
            if upper is None
            else int(upper.total_touch_count)
        )
        narrowness = clamp(
            1.0
            - state.width_atr_at_formation
            / self.protocol.maximum_width_atr_at_formation
        )
        compression = clamp(1.0 - compression_ratio)
        boundary = clamp(min(lower_count, upper_count) / 3.0)
        crossing = clamp(crossings / 4.0)
        strength = (
            narrowness
            + compression
            + boundary
            + crossing
            + inside_fraction
        ) / 5.0
        return {
            "candidate_real_h1_bars": len(bars),
            "lower_touch_count": lower_count,
            "upper_touch_count": upper_count,
            "midpoint_crossings": crossings,
            "inside_close_fraction": inside_fraction,
            "compression_ratio": compression_ratio,
            "narrowness_strength": narrowness,
            "compression_strength": compression,
            "boundary_test_strength": boundary,
            "crossing_strength": crossing,
            "strength": strength,
            "age_h1_bars": len(bars) - 1,
        }

    def _balance_price_test(
        self,
        state: DealingRangeState,
        work: _RangeWork,
        candle: Candle,
        prior_atr: float,
    ) -> dict[str, object]:
        """Record this completed bar's interaction with each frozen boundary.

        Balance asks whether price repeatedly traded into and was rejected by
        both sides.  The evidence is therefore the bar's own extremes against
        the frozen boundary, never the source zone's structural touch count --
        that only moves when another confirmed swing forms inside the zone,
        which is a different claim about a different thing.

        One generation is one continuous visit: price has to leave the
        tolerance band before the next test can open, so a run of bars hugging
        one level is a single test.  A generation keeps the deepest
        interaction it reached, which is why a shallow print followed by a deep
        one reads as one deep test rather than two.
        """

        band = max(
            self.protocol.tick_size
            * self.protocol.balance_price_test_band_minimum_ticks,
            self.protocol.balance_price_test_band_atr_fraction * prior_atr,
        )
        deep = self.protocol.balance_deep_penetration_atr_fraction * prior_atr
        close = float(candle.close)
        for side, beyond, closed_outside in (
            (
                "upper",
                float(candle.high) - state.upper_bound,
                close > state.upper_bound,
            ),
            (
                "lower",
                state.lower_bound - float(candle.low),
                close < state.lower_bound,
            ),
        ):
            in_band = beyond >= -band
            kinds = getattr(work, f"{side}_test_kinds")
            was_in_band = getattr(work, f"{side}_in_band")
            if in_band:
                if closed_outside:
                    kind = "close_outside"
                elif beyond > deep:
                    kind = "deep_penetration"
                elif beyond > 0.0:
                    kind = "shallow_penetration"
                else:
                    kind = "touch_only"
                if not was_in_band:
                    kinds.append(kind)
                elif BALANCE_PRICE_TEST_KINDS.index(
                    kind
                ) > BALANCE_PRICE_TEST_KINDS.index(kinds[-1]):
                    kinds[-1] = kind
            setattr(work, f"{side}_in_band", in_band)
        return {
            "balance_lower_test_generations": len(work.lower_test_kinds),
            "balance_upper_test_generations": len(work.upper_test_kinds),
            "balance_lower_test_kinds": tuple(work.lower_test_kinds),
            "balance_upper_test_kinds": tuple(work.upper_test_kinds),
        }

    def _range_gate_evaluation(
        self,
        state: DealingRangeState,
        statistics: dict[str, object],
    ) -> _RangeGateEvaluation:
        actuals = {
            "duration": float(statistics["candidate_real_h1_bars"]),
            "bilateral_price_tests": float(
                min(
                    int(statistics["balance_lower_test_generations"]),
                    int(statistics["balance_upper_test_generations"]),
                )
            ),
            "midpoint_crossing": float(
                statistics["midpoint_crossings"]
            ),
            "inside_close_fraction": float(
                statistics["inside_close_fraction"]
            ),
            "width": float(state.width_atr_at_formation),
            "compression": float(statistics["compression_ratio"]),
        }
        thresholds = {
            "duration": float(
                self.protocol.minimum_candidate_real_h1_bars
            ),
            "bilateral_price_tests": float(
                self.protocol.balance_minimum_price_test_generations_each
            ),
            "midpoint_crossing": float(
                self.protocol.minimum_midpoint_crossings
            ),
            "inside_close_fraction": float(
                self.protocol.minimum_inside_close_fraction
            ),
            "width": float(
                self.protocol.maximum_width_atr_at_formation
            ),
            "compression": float(
                self.protocol.maximum_compression_ratio
            ),
        }
        maximum_gates = {"width", "compression"}
        gates = tuple(
            (
                name,
                actuals[name],
                thresholds[name],
                (
                    thresholds[name] - actuals[name]
                    if name in maximum_gates
                    else actuals[name] - thresholds[name]
                )
                / thresholds[name],
            )
            for name in RANGE_MATURITY_GATE_NAMES
        )
        return _RangeGateEvaluation(
            range_id=state.range_id,
            gates=gates,
            unmet=tuple(
                name for name, _, _, margin in gates if margin < 0.0
            ),
        )

    @staticmethod
    def _source_is_live(
        state: SupportResistanceState | None,
    ) -> bool:
        return (
            state is not None
            and state.lifecycle
            in {
                SupportResistanceLifecycle.ACTIVE,
                SupportResistanceLifecycle.TESTED,
            }
        )

    def _terminal_range(
        self,
        state: DealingRangeState,
        observed_at: pd.Timestamp,
        reason: str,
        **changes: object,
    ) -> DealingRangeState:
        updates = dict(changes)
        updates.update(
            lifecycle=DealingRangeLifecycle.BROKEN,
            broken_at=observed_at,
            state_started_at=observed_at,
            last_updated_at=observed_at,
            transition_reason=reason,
        )
        terminal = replace(
            state,
            **updates,
        )
        self._ranges[state.range_id] = terminal
        self._range_work.pop(state.range_id, None)
        self._sync_range_inventory(terminal)
        return terminal

    def _balance_claim_open(self, state: DealingRangeState) -> bool:
        """Whether this candidate is still being tested for balance."""

        return (
            state.lifecycle is DealingRangeLifecycle.ACTIVE
            and state.transition_reason != BALANCE_CLAIM_ABANDONED
            and state.range_id in self._range_work
        )

    def _advance_live_range(
        self,
        candle: Candle,
        zones_by_id: dict[str, SupportResistanceState],
        true_range: float,
        prior_atr: float,
    ) -> tuple[DealingRangeState | None, _RangeGateEvaluation | None]:
        state = self._live_range(self._ranges.values())
        if state is None:
            return None, None
        lower = zones_by_id.get(state.lower_source_zone_id)
        upper = zones_by_id.get(state.upper_source_zone_id)
        if not self._balance_claim_open(state):
            updated = replace(
                state,
                last_updated_at=candle.end,
                age_h1_bars=state.age_h1_bars + 1,
            )
            if (
                candle.close < state.lower_bound
                or candle.close > state.upper_bound
            ):
                return (
                    self._terminal_range(
                        updated,
                        candle.end,
                        "close_beyond_frozen_range",
                    ),
                    None,
                )
            self._ranges[state.range_id] = updated
            self._sync_range_inventory(updated)
            return None, None
        work = self._range_work[state.range_id]
        if work.bars[-1].end < candle.end:
            work.bars.append(candle)
            work.true_ranges.append(float(true_range))
        statistics = {
            **self._range_statistics(state, work, lower, upper),
            **self._balance_price_test(state, work, candle, prior_atr),
        }
        update_fields = {
            **statistics,
            "lower_source_tested_at": (
                state.lower_source_tested_at
                if lower is None
                else lower.tested_at
            ),
            "upper_source_tested_at": (
                state.upper_source_tested_at
                if upper is None
                else upper.tested_at
            ),
            "lower_source_member_swing_ids": (
                state.lower_source_member_swing_ids
                if lower is None
                else lower.member_swing_ids
            ),
            "upper_source_member_swing_ids": (
                state.upper_source_member_swing_ids
                if upper is None
                else upper.member_swing_ids
            ),
            "last_updated_at": candle.end,
        }
        if (
            candle.close < state.lower_bound
            or candle.close > state.upper_bound
        ):
            return (
                self._terminal_range(
                    state,
                    candle.end,
                    "close_beyond_frozen_range",
                    **update_fields,
                ),
                None,
            )
        if not self._source_is_live(lower) or not self._source_is_live(upper):
            return (
                self._terminal_range(
                    state,
                    candle.end,
                    "forming_source_invalidated",
                    **update_fields,
                ),
                None,
            )
        gate_evaluation = self._range_gate_evaluation(state, statistics)
        mature = (
            statistics["candidate_real_h1_bars"]
            >= self.protocol.minimum_candidate_real_h1_bars
            and statistics["balance_lower_test_generations"]
            >= self.protocol.balance_minimum_price_test_generations_each
            and statistics["balance_upper_test_generations"]
            >= self.protocol.balance_minimum_price_test_generations_each
            and statistics["midpoint_crossings"]
            >= self.protocol.minimum_midpoint_crossings
            and statistics["inside_close_fraction"]
            >= self.protocol.minimum_inside_close_fraction
            and state.width_atr_at_formation
            <= self.protocol.maximum_width_atr_at_formation
            and statistics["compression_ratio"]
            <= self.protocol.maximum_compression_ratio
        )
        if mature != (not gate_evaluation.unmet):
            raise RuntimeError(
                "Group 4 maturity gate diagnostic disagrees with reducer"
            )
        if mature:
            # The registered standard is met.  That settles the balance claim
            # and promotes the boundaries; the range itself is still exactly
            # the location it was on the bar before, so its lifecycle does not
            # move and this is not a range transition.
            confirmed = replace(
                state,
                **update_fields,
                balance_confirmed_at=candle.end,
                state_started_at=candle.end,
                transition_reason=BALANCE_CLAIM_CONFIRMED,
            )
            self._ranges[state.range_id] = confirmed
            self._range_work.pop(state.range_id, None)
            self._create_range_inventory(confirmed)
            return None, gate_evaluation
        if (
            statistics["candidate_real_h1_bars"]
            >= self.protocol.maximum_forming_real_h1_bars
        ):
            # The candidate ran out of room to prove balance.  That verdict
            # belongs to the balance claim alone: the structural interval is
            # still an interval and still locates price, so it stays FORMING
            # and is only ended by price closing outside it.
            abandoned = replace(
                state,
                **update_fields,
                state_started_at=candle.end,
                transition_reason=BALANCE_CLAIM_ABANDONED,
            )
            self._ranges[state.range_id] = abandoned
            self._range_work.pop(state.range_id, None)
            # Not a transition of the range entity: its lifecycle is still
            # FORMING and was already recorded at creation.  Only the balance
            # claim ended, and the funnel diagnostic carries that verdict.
            return None, gate_evaluation
        updated = replace(
            state,
            **update_fields,
            transition_reason=None,
        )
        self._ranges[state.range_id] = updated
        return None, gate_evaluation

    @staticmethod
    def _pair_sort_key(
        pair: tuple[SupportResistanceState, SupportResistanceState],
    ) -> tuple[float, int, str, str]:
        lower, upper = pair
        return (
            upper.upper_bound - lower.lower_bound,
            -max(lower.confirmed_at.value, upper.confirmed_at.value),
            lower.zone_id,
            upper.zone_id,
        )

    def _range_pair_partition(
        self,
        candle: Candle,
        zones: Sequence[SupportResistanceState],
    ) -> _RangePairPartition:
        supports = tuple(
            zone
            for zone in zones
            if (
                zone.source_kind == "structural_swing"
                and zone.side == "support"
                and self._source_is_live(zone)
            )
        )
        resistances = tuple(
            zone
            for zone in zones
            if (
                zone.source_kind == "structural_swing"
                and zone.side == "resistance"
                and self._source_is_live(zone)
            )
        )
        live_structural = tuple(
            (lower, upper)
            for lower in supports
            for upper in resistances
        )
        geometry_valid = tuple(
            pair
            for pair in live_structural
            if pair[0].upper_bound < pair[1].lower_bound
        )
        invalid_geometry = tuple(
            pair
            for pair in live_structural
            if pair[0].upper_bound >= pair[1].lower_bound
        )
        at_price = tuple(
            pair
            for pair in geometry_valid
            if pair[0].lower_bound <= candle.close <= pair[1].upper_bound
        )
        close_outside = tuple(
            pair for pair in geometry_valid if pair not in at_price
        )
        admitted = {
            (
                state.lower_source_zone_id,
                state.upper_source_zone_id,
            )
            for state in self._ranges.values()
        }
        admitted_pairs = tuple(
            pair
            for pair in at_price
            if (pair[0].zone_id, pair[1].zone_id) in admitted
        )
        not_admitted = tuple(
            pair
            for pair in at_price
            if (pair[0].zone_id, pair[1].zone_id) not in admitted
        )
        cold_blocked = tuple(
            pair
            for pair in not_admitted
            if (pair[0].zone_id, pair[1].zone_id)
            in self._blocked_cold_pairs
        )
        available = tuple(
            pair
            for pair in not_admitted
            if (pair[0].zone_id, pair[1].zone_id)
            not in self._blocked_cold_pairs
        )
        ordered = lambda values: tuple(
            sorted(values, key=self._pair_sort_key)
        )
        return _RangePairPartition(
            live_structural_pairs=ordered(live_structural),
            invalid_geometry_pairs=ordered(invalid_geometry),
            geometry_valid_pairs=ordered(geometry_valid),
            close_outside_pairs=ordered(close_outside),
            admitted_pairs=ordered(admitted_pairs),
            cold_blocked_pairs=ordered(cold_blocked),
            available_pairs=ordered(available),
        )

    def _eligible_pairs(
        self,
        candle: Candle,
        zones: Sequence[SupportResistanceState],
    ) -> tuple[
        tuple[SupportResistanceState, SupportResistanceState],
        ...,
    ]:
        return self._range_pair_partition(candle, zones).available_pairs

    def _create_range(
        self,
        candle: Candle,
        zones: Sequence[SupportResistanceState],
        true_range: float,
    ) -> DealingRangeState | None:
        if self._live_range(self._ranges.values()) is not None:
            return None
        if len(self._h1_true_ranges) < self.protocol.h1_atr_period:
            return None
        pairs = self._eligible_pairs(candle, zones)
        if not pairs:
            return None
        lower, upper = pairs[0]
        formation_atr = max(
            sum(self._h1_true_ranges) / len(self._h1_true_ranges),
            self.protocol.tick_size,
        )
        lower_bound = float(lower.lower_bound)
        upper_bound = float(upper.upper_bound)
        width = upper_bound - lower_bound
        midpoint = (lower_bound + upper_bound) / 2.0
        range_id = _identity(
            self.protocol.protocol_hash,
            candle.symbol,
            candle.instrument_id,
            lower.zone_id,
            upper.zone_id,
            candle.end,
        )
        narrowness_strength = clamp(
            1.0
            - (width / formation_atr)
            / self.protocol.maximum_width_atr_at_formation
        )
        boundary_test_strength = clamp(
            min(
                int(lower.total_touch_count),
                int(upper.total_touch_count),
            )
            / 3.0
        )
        inside_close_fraction = 1.0
        strength = (
            narrowness_strength
            + boundary_test_strength
            + inside_close_fraction
        ) / 5.0
        state = DealingRangeState(
            range_id=range_id,
            protocol_hash=self.protocol.protocol_hash,
            source_group12_protocol_hash=(
                self.protocol.source_group12_protocol_hash
            ),
            symbol=candle.symbol,
            instrument_id=int(candle.instrument_id),
            timeframe=Timeframe.H1,
            lifecycle=DealingRangeLifecycle.ACTIVE,
            lower_source_zone_id=lower.zone_id,
            upper_source_zone_id=upper.zone_id,
            lower_source_confirmed_at=lower.confirmed_at,
            upper_source_confirmed_at=upper.confirmed_at,
            lower_source_tested_at=lower.tested_at,
            upper_source_tested_at=upper.tested_at,
            lower_source_member_swing_ids=lower.member_swing_ids,
            upper_source_member_swing_ids=upper.member_swing_ids,
            lower_source_lower_bound=lower.lower_bound,
            lower_source_upper_bound=lower.upper_bound,
            upper_source_lower_bound=upper.lower_bound,
            upper_source_upper_bound=upper.upper_bound,
            formed_at=candle.end,
            balance_confirmed_at=None,
            broken_at=None,
            state_started_at=candle.end,
            last_updated_at=candle.end,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            midpoint=midpoint,
            value_price=midpoint,
            formation_atr=formation_atr,
            width_points=width,
            width_atr_at_formation=width / formation_atr,
            candidate_real_h1_bars=1,
            lower_touch_count=int(lower.total_touch_count),
            upper_touch_count=int(upper.total_touch_count),
            midpoint_crossings=0,
            inside_close_fraction=inside_close_fraction,
            compression_ratio=1.0,
            narrowness_strength=narrowness_strength,
            compression_strength=0.0,
            boundary_test_strength=boundary_test_strength,
            crossing_strength=0.0,
            strength=strength,
            age_h1_bars=0,
            transition_reason="source_pair_selected",
        )
        self._ranges[range_id] = state
        self._range_order.append(range_id)
        self._range_work[range_id] = _RangeWork(
            deque(
                (candle,),
                maxlen=self.protocol.maximum_forming_real_h1_bars,
            ),
            deque(
                (float(true_range),),
                maxlen=self.protocol.maximum_forming_real_h1_bars,
            ),
        )
        self._ensure_range_capacity(
            {zone.zone_id for zone in zones}
        )
        return state

    def _range_item_id(
        self,
        state: DealingRangeState,
        side: str,
    ) -> str:
        digest = _identity(
            self.protocol.protocol_hash,
            state.range_id,
            side,
            state.balance_confirmed_at,
        )
        return f"range_boundary:{digest}"

    def _create_range_inventory(
        self,
        state: DealingRangeState,
    ) -> None:
        if state.balance_confirmed_at is None:
            raise RuntimeError(
                "a range whose balance claim has not settled has no inventory"
            )
        for side, price, zone_id in (
            (
                "below",
                state.lower_bound,
                state.lower_source_zone_id,
            ),
            (
                "above",
                state.upper_bound,
                state.upper_source_zone_id,
            ),
        ):
            item = LiquidityInventoryItem(
                item_id=self._range_item_id(state, side),
                timeframe=Timeframe.H1,
                side=side,
                kind="range_boundary",
                price=price,
                lower_bound=price,
                upper_bound=price,
                formed_at=state.formed_at,
                confirmed_at=state.balance_confirmed_at,
                lifecycle=LiquidityInventoryLifecycle.VISIBLE,
                source_ids=(state.range_id, zone_id),
                age_bars=state.age_h1_bars,
                strength=state.strength,
            )
            self._range_inventory[item.item_id] = item

    def _sync_range_inventory(
        self,
        state: DealingRangeState,
    ) -> None:
        for item_id, item in tuple(self._range_inventory.items()):
            if state.range_id not in item.source_ids:
                continue
            self._range_inventory[item_id] = replace(
                item,
                age_bars=state.age_h1_bars,
            )

    def _ensure_range_capacity(
        self,
        current_zone_ids: set[str],
    ) -> None:
        while len(self._ranges) > self.protocol.maximum_ranges:
            removable = next(
                (
                    range_id
                    for range_id in self._range_order
                    if (
                        self._ranges[range_id].lifecycle
                        is DealingRangeLifecycle.BROKEN
                        and not {
                            self._ranges[range_id].lower_source_zone_id,
                            self._ranges[range_id].upper_source_zone_id,
                        }.issubset(current_zone_ids)
                        and all(
                            item.lifecycle
                            is LiquidityInventoryLifecycle.CONSUMED
                            for item in self._range_inventory.values()
                            if range_id in item.source_ids
                        )
                    )
                ),
                None,
            )
            if removable is None:
                raise RuntimeError(
                    "Group 4 range capacity has no safe terminal eviction"
                )
            self._ranges.pop(removable)
            self._range_work.pop(removable, None)
            self._range_order.remove(removable)
            for item_id, item in tuple(
                self._range_inventory.items()
            ):
                if removable in item.source_ids:
                    self._range_inventory.pop(item_id)

    def _apply_h1(
        self,
        candle: Candle,
        zones: tuple[SupportResistanceState, ...],
    ) -> RangeAuctionUpdate:
        if (
            self._last_h1_raw_end is not None
            and candle.end <= self._last_h1_raw_end
        ):
            raise ValueError("duplicate or out-of-order Group 4 H1 candle")
        self._last_h1_raw_end = candle.end
        if not candle.real_completed:
            return self._output()
        true_range = _true_range(candle, self._prior_h1_close)
        # The tolerance band is sized from the volatility known *before* this
        # bar, so it is read off the window before this bar joins it.
        prior_atr = (
            sum(self._h1_true_ranges) / len(self._h1_true_ranges)
            if self._h1_true_ranges
            else float(self.protocol.tick_size)
        )
        if self._prior_h1_close is not None and true_range > 0.0:
            self._h1_true_ranges.append(
                max(true_range, self.protocol.tick_size)
            )
        self._prior_h1_close = float(candle.close)
        zones_by_id = {zone.zone_id: zone for zone in zones}
        transition, gate_evaluation = self._advance_live_range(
            candle,
            zones_by_id,
            true_range,
            prior_atr,
        )
        pair_partition = self._range_pair_partition(candle, zones)
        available_count = len(pair_partition.available_pairs)
        same_bar_terminal_blocked = 0
        live_range_blocked = 0
        atr_unready = 0
        eligible_count = 0
        if (
            transition is not None
            and transition.lifecycle is DealingRangeLifecycle.BROKEN
        ):
            same_bar_terminal_blocked = available_count
        elif self._live_range(self._ranges.values()) is not None:
            live_range_blocked = available_count
        elif len(self._h1_true_ranges) < self.protocol.h1_atr_period:
            atr_unready = available_count
        else:
            eligible_count = available_count
        created = None
        if transition is None or (
            transition.lifecycle is not DealingRangeLifecycle.BROKEN
        ):
            created = self._create_range(
                candle,
                zones,
                true_range,
            )
        if bool(created is not None) != bool(eligible_count):
            raise RuntimeError(
                "Group 4 range funnel disagrees with range creation"
            )
        if created is not None:
            selected_pair = (
                created.lower_source_zone_id,
                created.upper_source_zone_id,
            )
            expected_pair = pair_partition.available_pairs[0]
            if selected_pair != (
                expected_pair[0].zone_id,
                expected_pair[1].zone_id,
            ):
                raise RuntimeError(
                    "Group 4 range selection disagrees with its funnel"
                )
        else:
            selected_pair = None
        range_funnel = RangeFormationFunnelSnapshot(
            observed_at=candle.end,
            pair_counts=(
                (
                    "live_structural_pairs",
                    len(pair_partition.live_structural_pairs),
                ),
                (
                    "invalid_geometry_pairs",
                    len(pair_partition.invalid_geometry_pairs),
                ),
                (
                    "geometry_valid_pairs",
                    len(pair_partition.geometry_valid_pairs),
                ),
                (
                    "close_outside_pair_pairs",
                    len(pair_partition.close_outside_pairs),
                ),
                (
                    "already_admitted_pairs",
                    len(pair_partition.admitted_pairs),
                ),
                (
                    "cold_start_blocked_pairs",
                    len(pair_partition.cold_blocked_pairs),
                ),
                (
                    "same_bar_terminal_blocked_pairs",
                    same_bar_terminal_blocked,
                ),
                ("live_range_blocked_pairs", live_range_blocked),
                ("atr_unready_pairs", atr_unready),
                ("eligible_pairs", eligible_count),
                ("forming_selected", int(created is not None)),
            ),
            selected_source_pair_ids=selected_pair,
            selected_range_id=(
                None if created is None else created.range_id
            ),
            maturity_range_id=(
                None
                if gate_evaluation is None
                else gate_evaluation.range_id
            ),
            maturity_gates=(
                ()
                if gate_evaluation is None
                else gate_evaluation.gates
            ),
            unmet_maturity_gates=(
                ()
                if gate_evaluation is None
                else gate_evaluation.unmet
            ),
        )
        self._ensure_range_capacity(set(zones_by_id))
        self._last_h1_end = candle.end
        return self._output(
            range_transitions=tuple(
                value
                for value in (transition, created)
                if value is not None
            ),
            range_funnel=(range_funnel,),
        )

    def on_completed_h1(
        self,
        candle: Candle,
        support_resistance: Iterable[SupportResistanceState],
    ) -> RangeAuctionUpdate:
        zones = tuple(support_resistance)
        input_value = (candle, zones)
        if (
            input_value == self._last_h1_input
            and self._last_h1_output is not None
        ):
            return self._last_h1_output
        candidate = self._transaction_clone()
        try:
            candidate._validate_h1(candle, zones)
            output = candidate._apply_h1(candle, zones)
            candidate._last_h1_input = input_value
            candidate._last_h1_output = output
            candidate._last_boundary_input = None
            candidate._last_boundary_output = None
        except Exception:
            raise
        self._commit(candidate)
        return output

    def _pool_by_inventory(
        self,
        item: LiquidityInventoryItem,
        pools: Sequence[LiquidityPoolState],
    ) -> LiquidityPoolState | None:
        pool_id = (
            item.item_id[len("pool:") :]
            if item.item_id.startswith("pool:")
            else None
        )
        matches = tuple(
            pool
            for pool in pools
            if (
                pool_id is not None
                and pool.pool_id == pool_id
                and pool.timeframe is item.timeframe
            )
        )
        if len(matches) > 1:
            raise ValueError("duplicate Group 4 pool source identity")
        if not matches:
            return None
        pool = matches[0]
        expected_kind = (
            "equal_highs" if pool.side == "above" else "equal_lows"
        )
        expected_price = (
            pool.upper_bound
            if pool.side == "above"
            else pool.lower_bound
        )
        if (
            pool.lifecycle.value != "formed"
            or item.side != pool.side
            or item.kind != expected_kind
            or item.price != expected_price
            or item.lower_bound != pool.lower_bound
            or item.upper_bound != pool.upper_bound
            or item.formed_at != pool.formed_at
            or item.confirmed_at != pool.confirmed_at
            or item.source_ids != pool.member_swing_ids
        ):
            raise ValueError(
                "Group 4 pool inventory and source geometry disagree"
            )
        return pool

    def _range_by_inventory(
        self,
        item: LiquidityInventoryItem,
    ) -> DealingRangeState | None:
        retained = self._range_inventory.get(item.item_id)
        if retained is None:
            return None
        if retained != item:
            raise ValueError(
                "Group 4 range inventory differs from retained source"
            )
        matches = tuple(
            self._ranges[source_id]
            for source_id in item.source_ids
            if source_id in self._ranges
        )
        if len(matches) > 1:
            raise ValueError("range inventory maps to multiple ranges")
        return matches[0] if matches else None

    def _source_from_item(
        self,
        item: LiquidityInventoryItem,
        pools: Sequence[LiquidityPoolState],
        candle: Candle,
        prior_close: float,
    ) -> tuple[
        _ManipulationSource | None,
        ManipulationSourceDispositionKind | None,
    ]:
        if (
            item.lifecycle is not LiquidityInventoryLifecycle.VISIBLE
            or item.confirmed_at > candle.start
        ):
            return (
                None,
                ManipulationSourceDispositionKind.REJECTED_SOURCE_MISSING_OR_STALE,
            )
        if item.kind == "range_boundary":
            state = self._range_by_inventory(item)
            expected_zone_id = (
                state.upper_source_zone_id
                if state is not None and item.side == "above"
                else state.lower_source_zone_id
                if state is not None
                else None
            )
            expected_price = (
                state.upper_bound
                if state is not None and item.side == "above"
                else state.lower_bound
                if state is not None
                else None
            )
            if (
                state is None
                or state.balance_confirmed_at is None
                or item.timeframe is not Timeframe.H1
                or item.price != expected_price
                or item.lower_bound != expected_price
                or item.upper_bound != expected_price
                or item.formed_at != state.formed_at
                or item.confirmed_at != state.balance_confirmed_at
                or len(item.source_ids) != 2
                or set(item.source_ids)
                != {state.range_id, expected_zone_id}
                or state.balance_confirmed_at > candle.start
                or (
                    state.broken_at is not None
                    and state.broken_at <= candle.end
                )
            ):
                return (
                    None,
                    ManipulationSourceDispositionKind.REJECTED_SOURCE_MISSING_OR_STALE,
                )
            if not state.lower_bound <= prior_close <= state.upper_bound:
                return (
                    None,
                    ManipulationSourceDispositionKind.REJECTED_PRIOR_CLOSE,
                )
            return (
                _ManipulationSource(
                    side=item.side,
                    source_kind="mature_range_boundary",
                    source_id=state.range_id,
                    source_protocol_hash=state.protocol_hash,
                    source_timeframe=Timeframe.H1,
                    inventory=item,
                    formed_at=state.formed_at,
                    eligible_at=state.balance_confirmed_at,
                    lower_bound=state.lower_bound,
                    upper_bound=state.upper_bound,
                    boundary_price=item.price,
                ),
                None,
            )
        if item.kind not in {"equal_highs", "equal_lows"}:
            return (
                None,
                ManipulationSourceDispositionKind.REJECTED_SOURCE_MISSING_OR_STALE,
            )
        pool = self._pool_by_inventory(item, pools)
        if pool is None or pool.confirmed_at > candle.start:
            return (
                None,
                ManipulationSourceDispositionKind.REJECTED_SOURCE_MISSING_OR_STALE,
            )
        if (
            (item.side == "above" and prior_close > pool.upper_bound)
            or (item.side == "below" and prior_close < pool.lower_bound)
        ):
            return (
                None,
                ManipulationSourceDispositionKind.REJECTED_PRIOR_CLOSE,
            )
        return (
            _ManipulationSource(
                side=item.side,
                source_kind="formed_liquidity_pool",
                source_id=pool.pool_id,
                source_protocol_hash=(
                    self.protocol.source_group12_protocol_hash
                ),
                source_timeframe=pool.timeframe,
                inventory=item,
                formed_at=pool.formed_at,
                eligible_at=pool.confirmed_at,
                lower_bound=pool.lower_bound,
                upper_bound=pool.upper_bound,
                boundary_price=item.price,
            ),
            None,
        )

    @staticmethod
    def _crossed(
        source: _ManipulationSource,
        candle: Candle,
    ) -> bool:
        return (
            candle.high > source.boundary_price
            if source.side == "above"
            else candle.low < source.boundary_price
        )

    @staticmethod
    def _close_outside(
        state_or_source: ManipulationState | _ManipulationSource,
        close: float,
    ) -> tuple[bool, str | None]:
        lower_bound = (
            state_or_source.source_lower_bound
            if isinstance(state_or_source, ManipulationState)
            else state_or_source.lower_bound
        )
        upper_bound = (
            state_or_source.source_upper_bound
            if isinstance(state_or_source, ManipulationState)
            else state_or_source.upper_bound
        )
        # Acceptance is always measured against the actually swept side.
        # The opposite range boundary is a separate liquidity source, not a
        # second way to accept this auction outside.
        if state_or_source.side == "above":
            return close > upper_bound, (
                "above"
                if close > upper_bound
                else None
            )
        return close < lower_bound, (
            "below"
            if close < lower_bound
            else None
        )

    def _resolve_live_manipulation(
        self,
        candle: Candle,
    ) -> ManipulationState | None:
        live = tuple(
            state
            for state in self._manipulations.values()
            if (
                state.lifecycle is ManipulationLifecycle.SWEPT
                and state.censored_at is None
            )
        )
        if len(live) > 1:
            raise RuntimeError(
                "Group 4 retained more than one live manipulation"
            )
        if not live or candle.end <= live[0].swept_at:
            return None
        state = live[0]
        outside, resolved_side = self._close_outside(
            state,
            float(candle.close),
        )
        age = state.age_1m_bars + 1
        outside_bars = state.outside_completed_bars + int(outside)
        if outside:
            outside_run = (
                state.outside_run + 1
                if state.outside_run_side == resolved_side
                else 1
            )
            advanced = replace(
                state,
                last_updated_at=candle.end,
                outside_completed_bars=outside_bars,
                age_1m_bars=age,
                reentry_candidate_at=None,
                reentry_candidate_price=None,
                inside_hold_bars=0,
                reentry_failed_at=(
                    candle.end
                    if state.reentry_candidate_at is not None
                    else state.reentry_failed_at
                ),
                outside_run=outside_run,
                outside_run_side=resolved_side,
            )
            if outside_run >= self.protocol.outside_acceptance_closes:
                resolved = replace(
                    advanced,
                    lifecycle=ManipulationLifecycle.ACCEPTED_OUTSIDE,
                    accepted_outside_at=candle.end,
                    resolved_at=candle.end,
                    state_started_at=candle.end,
                    resolved_side=resolved_side,
                    transition_reason="consecutive_closes_held_outside",
                )
            else:
                resolved = advanced
        else:
            if state.reentry_candidate_at is None:
                resolved = replace(
                    state,
                    last_updated_at=candle.end,
                    outside_completed_bars=outside_bars,
                    age_1m_bars=age,
                    reentry_candidate_at=candle.end,
                    reentry_candidate_price=float(candle.close),
                    inside_hold_bars=0,
                    outside_run=0,
                    outside_run_side=None,
                )
            else:
                hold = state.inside_hold_bars + 1
                advanced = replace(
                    state,
                    last_updated_at=candle.end,
                    outside_completed_bars=outside_bars,
                    age_1m_bars=age,
                    inside_hold_bars=hold,
                    outside_run=0,
                    outside_run_side=None,
                )
                if hold >= self.protocol.reacceptance_hold_bars:
                    resolved = replace(
                        advanced,
                        lifecycle=ManipulationLifecycle.REACCEPTED,
                        reaccepted_at=candle.end,
                        resolved_at=candle.end,
                        state_started_at=candle.end,
                        reentry_price=(
                            state.reentry_candidate_price
                        ),
                        transition_reason=(
                            "reentry_held_inside_swept_boundary"
                        ),
                    )
                else:
                    resolved = advanced
        if (
            resolved.lifecycle is ManipulationLifecycle.SWEPT
            and age >= self.protocol.resolution_deadline_real_1m_bars
        ):
            resolved = replace(
                resolved,
                last_updated_at=candle.end,
                censored_at=candle.end,
                deadline_at=candle.end,
                deadline_elapsed=True,
                transition_reason="deadline_elapsed",
            )
        self._manipulations[state.manipulation_id] = resolved
        if (
            resolved.lifecycle is ManipulationLifecycle.SWEPT
            and resolved.censored_at is None
        ):
            return None
        return resolved

    def _consume_range_crossings(
        self,
        items: Sequence[LiquidityInventoryItem],
        candle: Candle,
    ) -> None:
        for item in items:
            if item.kind != "range_boundary":
                continue
            retained = self._range_inventory.get(item.item_id)
            if (
                retained is None
                or retained.lifecycle
                is not LiquidityInventoryLifecycle.VISIBLE
            ):
                continue
            self._range_inventory[item.item_id] = replace(
                retained,
                lifecycle=LiquidityInventoryLifecycle.CONSUMED,
                consumed_at=candle.end,
                lifecycle_reason="range_boundary_consumed",
            )

    def _create_manipulation(
        self,
        source: _ManipulationSource,
        coincident_source_ids: tuple[str, ...],
        crossed_source_ids: tuple[str, ...],
        candle: Candle,
        prior_atr: float,
    ) -> ManipulationState:
        extreme = (
            float(candle.high)
            if source.side == "above"
            else float(candle.low)
        )
        penetration = (
            extreme - source.boundary_price
            if source.side == "above"
            else source.boundary_price - extreme
        )
        penetration_atr = penetration / max(
            prior_atr,
            self.protocol.tick_size,
        )
        outside, _ = self._close_outside(
            source,
            float(candle.close),
        )
        manipulation_id = _identity(
            self.protocol.protocol_hash,
            candle.symbol,
            candle.instrument_id,
            source.source_kind,
            source.source_id,
            source.side,
            candle.end,
        )
        state = ManipulationState(
            manipulation_id=manipulation_id,
            protocol_hash=self.protocol.protocol_hash,
            source_group12_protocol_hash=(
                self.protocol.source_group12_protocol_hash
            ),
            symbol=candle.symbol,
            instrument_id=int(candle.instrument_id),
            timeframe=Timeframe.M1,
            lifecycle=ManipulationLifecycle.SWEPT,
            side=source.side,
            source_kind=source.source_kind,
            source_id=source.source_id,
            source_protocol_hash=source.source_protocol_hash,
            source_timeframe=source.source_timeframe,
            source_inventory_item_id=source.inventory.item_id,
            source_inventory_lifecycle=(
                LiquidityInventoryLifecycle.VISIBLE
            ),
            coincident_source_ids=coincident_source_ids,
            source_formed_at=source.formed_at,
            source_eligible_at=source.eligible_at,
            source_lower_bound=source.lower_bound,
            source_upper_bound=source.upper_bound,
            formed_at=candle.end,
            confirmed_at=candle.end,
            swept_at=candle.end,
            reaccepted_at=None,
            accepted_outside_at=None,
            resolved_at=None,
            state_started_at=candle.end,
            last_updated_at=candle.end,
            sweep_extreme=extreme,
            close_outside_on_sweep=outside,
            reentry_price=None,
            resolved_side=None,
            outside_completed_bars=int(outside),
            penetration_atr=penetration_atr,
            strength=clamp(penetration_atr),
            age_1m_bars=0,
            transition_reason="source_swept",
            censored_at=None,
            reentry_candidate_at=None,
            reentry_candidate_price=None,
            inside_hold_bars=0,
            reentry_failed_at=None,
            outside_run=int(outside),
            outside_run_side=(source.side if outside else None),
            # The fifth future real-completed bar is not knowable at sweep
            # time across gaps/closures.  Record its actual clock only when
            # the real-bar deadline is reached.
            deadline_at=None,
            deadline_elapsed=False,
            crossed_source_ids=crossed_source_ids,
        )
        self._manipulations[manipulation_id] = state
        self._manipulation_order.append(manipulation_id)
        self._ensure_manipulation_capacity()
        return state

    def _ensure_manipulation_capacity(self) -> None:
        while (
            len(self._manipulations)
            > self.protocol.maximum_manipulations
        ):
            removable = next(
                (
                    identity
                    for identity in self._manipulation_order
                    if not (
                        self._manipulations[identity].lifecycle
                        is ManipulationLifecycle.SWEPT
                        and self._manipulations[identity].censored_at is None
                    )
                ),
                None,
            )
            if removable is None:
                raise RuntimeError(
                    "Group 4 manipulation capacity has no terminal eviction"
                )
            self._manipulations.pop(removable)
            self._manipulation_order.remove(removable)

    def _compact_terminal_manipulations_without_sources(
        self,
        pools: Sequence[LiquidityPoolState],
        *,
        current_end: pd.Timestamp,
    ) -> None:
        retained_source_ids = set(self._range_inventory)
        retained_source_ids.update(
            f"pool:{pool.pool_id}" for pool in pools
        )
        for manipulation_id in tuple(self._manipulation_order):
            state = self._manipulations[manipulation_id]
            if state.source_inventory_item_id in retained_source_ids:
                continue
            if (
                (
                    state.lifecycle is ManipulationLifecycle.SWEPT
                    and state.censored_at is None
                )
                or state.last_updated_at >= current_end
            ):
                continue
            self._manipulations.pop(manipulation_id)
            self._manipulation_order.remove(manipulation_id)

    def _apply_m1(
        self,
        candle: Candle,
        prior_inventory: tuple[LiquidityInventoryItem, ...],
        pools: tuple[LiquidityPoolState, ...],
        completed_h1: Candle | None,
        h1_zones: tuple[SupportResistanceState, ...],
    ) -> RangeAuctionUpdate:
        if (
            self._last_m1_raw_end is not None
            and candle.end <= self._last_m1_raw_end
        ):
            raise ValueError("duplicate or out-of-order Group 4 1m candle")
        self._last_m1_raw_end = candle.end
        if not candle.real_completed:
            range_transitions: tuple[DealingRangeState, ...] = ()
            range_funnel: tuple[RangeFormationFunnelSnapshot, ...] = ()
            if completed_h1 is not None:
                h1_output = self._apply_h1(completed_h1, h1_zones)
                range_transitions = h1_output.range_transitions
                range_funnel = h1_output.range_funnel
            return self._output(
                range_transitions=range_transitions,
                range_funnel=range_funnel,
            )
        self._compact_terminal_manipulations_without_sources(
            pools,
            current_end=candle.end,
        )
        prior_close = self._prior_m1_close
        prior_atr = (
            sum(self._m1_true_ranges) / len(self._m1_true_ranges)
            if (
                len(self._m1_true_ranges)
                == self.protocol.m1_atr_period
            )
            else None
        )
        had_live_before = any(
            state.lifecycle is ManipulationLifecycle.SWEPT
            and state.censored_at is None
            for state in self._manipulations.values()
        )
        resolved = self._resolve_live_manipulation(candle)
        candidate_sources: list[_ManipulationSource] = []
        disposition_by_item_id: dict[
            str,
            ManipulationSourceDispositionKind,
        ] = {}
        crossed_items = tuple(
            item
            for item in prior_inventory
            if (
                item.lifecycle is LiquidityInventoryLifecycle.VISIBLE
                and item.kind
                in {"range_boundary", "equal_highs", "equal_lows"}
                and item.confirmed_at <= candle.start
                and (
                    candle.high > item.upper_bound
                    if item.side == "above"
                    else candle.low < item.lower_bound
                )
            )
        )
        crossed_item_ids = tuple(item.item_id for item in crossed_items)
        if len(crossed_item_ids) != len(set(crossed_item_ids)):
            raise ValueError(
                "Group 4 raw crossed source identity repeats"
            )

        def assign(
            item_id: str,
            disposition: ManipulationSourceDispositionKind,
        ) -> None:
            if item_id in disposition_by_item_id:
                raise RuntimeError(
                    "Group 4 source received multiple dispositions"
                )
            disposition_by_item_id[item_id] = disposition

        if prior_close is None:
            for item in crossed_items:
                assign(
                    item.item_id,
                    ManipulationSourceDispositionKind.REJECTED_PRIOR_CLOSE,
                )
        else:
            for item in crossed_items:
                source, rejection = self._source_from_item(
                    item,
                    pools,
                    candle,
                    prior_close,
                )
                if rejection is not None:
                    assign(item.item_id, rejection)
                    continue
                if source is None or not self._crossed(source, candle):
                    raise RuntimeError(
                        "Group 4 admitted source disagrees with its raw "
                        "crossing"
                    )
                candidate_sources.append(source)
        self._consume_range_crossings(crossed_items, candle)
        range_transitions: tuple[DealingRangeState, ...] = ()
        range_funnel: tuple[RangeFormationFunnelSnapshot, ...] = ()
        if completed_h1 is not None:
            h1_output = self._apply_h1(completed_h1, h1_zones)
            range_transitions = h1_output.range_transitions
            range_funnel = h1_output.range_funnel
        same_clock_invalidated_range_ids = {
            state.range_id
            for state in range_transitions
            if (
                state.lifecycle is DealingRangeLifecycle.BROKEN
                and state.broken_at == candle.end
            )
        }
        retained_candidates: list[_ManipulationSource] = []
        for source in candidate_sources:
            if (
                source.source_kind == "mature_range_boundary"
                and source.source_id in same_clock_invalidated_range_ids
            ):
                assign(
                    source.inventory.item_id,
                    ManipulationSourceDispositionKind.REJECTED_RANGE_INVALIDATED_SAME_CLOCK,
                )
                continue
            if source.source_kind == "mature_range_boundary":
                state = self._ranges.get(source.source_id)
                if (
                    state is None
                    or state.balance_confirmed_at is None
                ):
                    raise RuntimeError(
                        "Group 4 range source changed without a same-clock "
                        "terminal transition"
                    )
            retained_candidates.append(source)
        candidate_sources = retained_candidates
        crossed_sides = {source.side for source in candidate_sources}
        ambiguous = (
            tuple(
                source.inventory.item_id
                for source in candidate_sources
            )
            if len(crossed_sides) > 1
            else ()
        )
        if ambiguous:
            for source in candidate_sources:
                assign(
                    source.inventory.item_id,
                    ManipulationSourceDispositionKind.AMBIGUOUS_DUAL_SIDE,
                )
        # A cold causal prefix may encounter an already-visible liquidity
        # source before fourteen real predecessor minutes exist.  The sweep
        # cannot be normalized without its *prior* ATR, but that is a normal
        # not-yet-ready condition rather than corrupt input.  Consume the
        # observed crossing, expose its identity as unclassified, and keep
        # warming the ATR; never backfill this event with a future ATR.
        atr_unready = bool(
            candidate_sources and prior_atr is None and not ambiguous
        )
        atr_unready_ids = (
            tuple(
                sorted(
                    source.inventory.item_id
                    for source in candidate_sources
                )
            )
            if atr_unready
            else ()
        )
        if atr_unready:
            for source in candidate_sources:
                assign(
                    source.inventory.item_id,
                    ManipulationSourceDispositionKind.ATR_UNREADY,
                )
        created = None
        actionable_sources = bool(
            candidate_sources
            and not ambiguous
            and prior_atr is not None
        )
        if actionable_sources and had_live_before:
            blocked_disposition = (
                ManipulationSourceDispositionKind.BLOCKED_EXISTING_LIVE
                if resolved is None
                else ManipulationSourceDispositionKind.BLOCKED_LIVE_RESOLVED_SAME_BAR
            )
            for source in candidate_sources:
                assign(source.inventory.item_id, blocked_disposition)
        elif actionable_sources:
            ordered = sorted(
                candidate_sources,
                key=lambda source: (
                    abs(source.boundary_price - prior_close),
                    (
                        0
                        if source.source_kind
                        == "mature_range_boundary"
                        else 1
                    ),
                    source.eligible_at,
                    source.source_id,
                ),
            )
            primary = ordered[0]
            assign(
                primary.inventory.item_id,
                ManipulationSourceDispositionKind.SELECTED_PRIMARY,
            )
            coincident = tuple(
                source.source_id
                for source in ordered[1:]
                if math.isclose(
                    source.boundary_price,
                    primary.boundary_price,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            )
            for source in ordered[1:]:
                assign(
                    source.inventory.item_id,
                    (
                        ManipulationSourceDispositionKind.ATTACHED_COINCIDENT_SECONDARY
                        if math.isclose(
                            source.boundary_price,
                            primary.boundary_price,
                            rel_tol=0.0,
                            abs_tol=0.0,
                        )
                        else ManipulationSourceDispositionKind.ATTACHED_SAME_SIDE_SECONDARY
                    ),
                )
            created = self._create_manipulation(
                primary,
                coincident,
                tuple(
                    dict.fromkeys(
                        (
                            primary.source_id,
                            *(
                                source.source_id
                                for source in ordered
                                if source is not primary
                            ),
                        )
                    )
                ),
                candle,
                prior_atr,
            )
        if set(disposition_by_item_id) != set(crossed_item_ids):
            raise RuntimeError(
                "Group 4 raw crossed source disposition is not conserved"
            )
        source_dispositions = tuple(
            ManipulationSourceDisposition(
                source_inventory_item_id=item_id,
                observed_at=candle.end,
                disposition=disposition_by_item_id[item_id],
            )
            for item_id in sorted(disposition_by_item_id)
        )
        true_range = _true_range(candle, self._prior_m1_close)
        if self._prior_m1_close is not None and true_range > 0.0:
            self._m1_true_ranges.append(
                max(true_range, self.protocol.tick_size)
            )
        self._prior_m1_close = float(candle.close)
        self._last_m1_end = candle.end
        self._compact_terminal_manipulations_without_sources(
            pools,
            current_end=candle.end,
        )
        return self._output(
            range_transitions=range_transitions,
            manipulation_transitions=tuple(
                value
                for value in (resolved, created)
                if value is not None
            ),
            ambiguous=ambiguous,
            atr_unready=atr_unready_ids,
            source_dispositions=source_dispositions,
            range_funnel=range_funnel,
        )

    def on_completed_update(
        self,
        candle: Candle,
        *,
        prior_inventory: Iterable[LiquidityInventoryItem],
        liquidity_pools: Iterable[LiquidityPoolState],
        completed_h1: Candle | None = None,
        h1_support_resistance: Iterable[
            SupportResistanceState
        ] = (),
    ) -> RangeAuctionUpdate:
        if (
            not isinstance(candle, Candle)
            or candle.timeframe is not Timeframe.M1
            or not candle.complete
            or candle.expected_minutes != 1
        ):
            raise ValueError("Group 4 requires a completed 1m candle")
        inventory = tuple(prior_inventory)
        pools = tuple(liquidity_pools)
        zones = tuple(h1_support_resistance)
        input_value = (
            candle,
            inventory,
            pools,
            completed_h1,
            zones,
        )
        if (
            input_value == self._last_m1_input
            and self._last_m1_output is not None
        ):
            return self._last_m1_output
        candidate = self._transaction_clone()
        try:
            candidate._validate_contract(candle)
            if any(
                not isinstance(item, LiquidityInventoryItem)
                for item in inventory
            ):
                raise TypeError("Group 4 inventory source is not typed")
            inventory_item_ids = tuple(
                item.item_id for item in inventory
            )
            if len(inventory_item_ids) != len(set(inventory_item_ids)):
                raise ValueError(
                    "Group 4 inventory source identity repeats"
                )
            if any(
                not isinstance(pool, LiquidityPoolState)
                for pool in pools
            ):
                raise TypeError("Group 4 pool source is not typed")
            if completed_h1 is not None:
                candidate._validate_h1(completed_h1, zones)
                if completed_h1.end != candle.end:
                    raise ValueError(
                        "Group 4 H1 and 1m completion clocks disagree"
                    )
                if (
                    not candle.real_completed
                    and completed_h1.real_completed
                ):
                    raise ValueError(
                        "synthetic Group 4 1m cannot carry a real H1 bar"
                    )
            output = candidate._apply_m1(
                candle,
                inventory,
                pools,
                completed_h1,
                zones,
            )
            candidate._last_m1_input = input_value
            candidate._last_m1_output = output
            candidate._last_boundary_input = None
            candidate._last_boundary_output = None
        except Exception:
            raise
        self._commit(candidate)
        return output

    def bootstrap_completed_1m_prefix(
        self,
        candles: Iterable[Candle],
        *,
        pool_inventory: Iterable[LiquidityInventoryItem],
        liquidity_pools: Iterable[LiquidityPoolState],
    ) -> RangeAuctionUpdate:
        """Cold-reconstruct first crossings from one retained causal prefix."""

        prefix = tuple(candles)
        pool_items = tuple(
            replace(
                item,
                lifecycle=LiquidityInventoryLifecycle.VISIBLE,
                targeted_at=None,
                consumed_at=None,
                lifecycle_reason=None,
            )
            for item in pool_inventory
            if item.kind in {"equal_highs", "equal_lows"}
        )
        pools = tuple(liquidity_pools)
        sources = (
            *pool_items,
            *(
                item
                for item in self._range_inventory.values()
                if item.lifecycle is LiquidityInventoryLifecycle.VISIBLE
            ),
        )
        input_value = (
            "cold_prefix",
            prefix,
            pool_items,
            pools,
        )
        if (
            input_value == self._last_m1_input
            and self._last_m1_output is not None
        ):
            return self._last_m1_output
        if not prefix:
            if sources:
                raise ValueError(
                    "Group 4 cold attachment lacks a completed 1m prefix"
                )
            return self.snapshot()
        real = tuple(
            candle for candle in prefix if candle.real_completed
        )
        if sources and not real:
            raise ValueError(
                "Group 4 cold attachment has no real completed minute"
            )
        if any(
            not isinstance(candle, Candle)
            or candle.timeframe is not Timeframe.M1
            or not candle.complete
            or candle.expected_minutes != 1
            for candle in prefix
        ):
            raise ValueError("Group 4 cold prefix is not completed 1m data")
        identities = {
            (candle.symbol, int(candle.instrument_id))
            for candle in prefix
        }
        if (
            len(identities) != 1
            or (
                self._identity is not None
                and next(iter(identities)) != self._identity
            )
            or any(
                later.end <= earlier.end
                for earlier, later in zip(prefix, prefix[1:])
            )
        ):
            raise ValueError(
                "Group 4 cold prefix contract or clock is inconsistent"
            )
        candidate = self._transaction_clone()
        candidate._identity = next(iter(identities))
        for item in pool_items:
            if candidate._pool_by_inventory(item, pools) is None:
                raise ValueError(
                    "Group 4 cold source lacks its exact pool state"
                )
        if any(
            not any(
                candle.real_completed
                and candle.end <= item.confirmed_at
                for candle in prefix
            )
            for item in sources
        ):
            raise ValueError(
                "Group 4 cold prefix lacks a real source predecessor"
            )
        try:
            consumed_pool_ids: set[str] = set()
            output = candidate.snapshot()
            transitions: list[ManipulationState] = []
            ambiguous: list[str] = []
            atr_unready: list[str] = []
            source_dispositions: list[
                ManipulationSourceDisposition
            ] = []
            for candle in prefix:
                inventory = tuple(
                    item
                    for item in (
                        *pool_items,
                        *candidate._range_inventory.values(),
                    )
                    if (
                        item.confirmed_at <= candle.start
                        and item.item_id not in consumed_pool_ids
                        and item.lifecycle
                        is LiquidityInventoryLifecycle.VISIBLE
                    )
                )
                output = candidate._apply_m1(
                    candle,
                    inventory,
                    pools,
                    None,
                    (),
                )
                transitions.extend(output.manipulation_transitions)
                ambiguous.extend(output.ambiguous_sweep_item_ids)
                atr_unready.extend(output.atr_unready_sweep_item_ids)
                source_dispositions.extend(output.source_dispositions)
                for item in inventory:
                    if (
                        candle.real_completed
                        and item.kind in {"equal_highs", "equal_lows"}
                        and (
                            candle.high > item.upper_bound
                            if item.side == "above"
                            else candle.low < item.lower_bound
                        )
                    ):
                        consumed_pool_ids.add(item.item_id)
            output = RangeAuctionUpdate(
                dealing_ranges=output.dealing_ranges,
                manipulations=output.manipulations,
                range_boundary_inventory=(
                    output.range_boundary_inventory
                ),
                manipulation_transitions=tuple(transitions),
                ambiguous_sweep_item_ids=tuple(
                    sorted(set(ambiguous))
                ),
                atr_unready_sweep_item_ids=tuple(
                    sorted(set(atr_unready))
                ),
                source_dispositions=tuple(source_dispositions),
            )
            candidate._last_m1_input = input_value
            candidate._last_m1_output = output
            candidate._last_boundary_input = None
            candidate._last_boundary_output = None
        except Exception:
            raise
        self._commit(candidate)
        return output

    def on_boundary(
        self,
        reason: str,
        observed_at: pd.Timestamp,
    ) -> RangeAuctionUpdate:
        observed_at = aware_timestamp(
            observed_at,
            name="group4.boundary.observed_at",
        )
        input_value = (reason, observed_at)
        if (
            input_value == self._last_boundary_input
            and self._last_boundary_output is not None
        ):
            return self._last_boundary_output
        if reason not in RANGE_AUCTION_HARD_BOUNDARY_REASONS:
            raise ValueError("Group 4 boundary reason is not hard")
        raw_ends = tuple(
            value
            for value in (
                self._last_h1_raw_end,
                self._last_m1_raw_end,
            )
            if value is not None
        )
        if raw_ends and observed_at <= max(raw_ends):
            raise ValueError(
                "Group 4 boundary is duplicate or out of order"
            )
        candidate = self._transaction_clone()
        try:
            range_transitions = tuple(
                candidate._terminal_range(
                    state,
                    observed_at,
                    reason,
                )
                for state in tuple(candidate._ranges.values())
                if state.lifecycle
                in {
                    DealingRangeLifecycle.ACTIVE,
                }
            )
            manipulation_transitions = tuple(
                replace(
                    state,
                    last_updated_at=observed_at,
                    transition_reason=reason,
                    censored_at=observed_at,
                )
                for state in candidate._manipulations.values()
                if (
                    state.lifecycle is ManipulationLifecycle.SWEPT
                    and state.censored_at is None
                )
            )
            candidate._ranges.clear()
            candidate._range_order.clear()
            candidate._range_work.clear()
            candidate._range_inventory.clear()
            candidate._manipulations.clear()
            candidate._manipulation_order.clear()
            candidate._h1_true_ranges.clear()
            candidate._m1_true_ranges.clear()
            candidate._prior_h1_close = None
            candidate._prior_m1_close = None
            candidate._identity = None
            candidate._last_h1_end = None
            candidate._last_m1_end = None
            candidate._last_h1_raw_end = None
            candidate._last_m1_raw_end = observed_at
            candidate._blocked_cold_pairs.clear()
            output = candidate._output(
                range_transitions=range_transitions,
                manipulation_transitions=manipulation_transitions,
                boundary_reason=reason,
            )
            candidate._last_h1_input = None
            candidate._last_h1_output = None
            candidate._last_m1_input = None
            candidate._last_m1_output = None
            candidate._last_boundary_input = input_value
            candidate._last_boundary_output = output
        except Exception:
            raise
        self._commit(candidate)
        return output


__all__ = [
    "CausalRangeAuctionTracker",
    "RangeAuctionProtocol",
    "RangeAuctionUpdate",
]
