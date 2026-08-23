"""Deterministic Phase 8 execution-method research on frozen intents.

The module is deliberately separate from Decision, RiskManager, broker adapters,
and the future order FSM.  It compares entry methods as paired counterfactuals
for one immutable research intent.  Minute BBO supplies only displayed best-level
capacity; Phase 6 ``F`` volume supplies an explicitly labelled passive-fill
proxy.  Neither input reveals queue position, so this module never reports
queue-level fills as fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import pandas as pd

from .execution import TopOfBook
from .mbo_mechanism import validate_mbo_mechanism_frame
from .model import Bar, Direction, aware_timestamp

if TYPE_CHECKING:
    from .trade_intent import TradeIntent


EXECUTION_RESEARCH_SCHEMA_VERSION = "phase8_execution_research_v1.1"
EXECUTION_RESEARCH_MODEL_VERSION = "minute_bbo_mbo_proxy_ohlc_adverse_first_v2"
# Canonical JSON payload fingerprint.  Any protocol edit requires a version
# bump and an intentional update of this identity.
EXECUTION_RESEARCH_PROTOCOL_SHA256 = (
    "d04f5020aec6d48d66e9c22df574ffde34b2f0e4363753e0041af0a0b1e80701"
)


class ExecutionResearchError(ValueError):
    """Raised when an execution study is not identity-safe or causal."""


class ExecutionMethod(str, Enum):
    MARKET = "market"
    FVG_50_LIMIT = "fvg_50_limit"
    OB_50_LIMIT = "ob_50_limit"
    RECLAIM_LIMIT = "reclaim_limit"
    BREAKOUT_LIMIT = "breakout_limit"
    AGGRESSIVE_LIMIT = "aggressive_limit"
    PASSIVE_LIMIT = "passive_limit"


ORDERED_EXECUTION_METHODS = tuple(ExecutionMethod)
LIMIT_EXECUTION_METHODS = ORDERED_EXECUTION_METHODS[1:]


class OrderResearchStatus(str, Enum):
    FILLED = "filled"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    CENSORED = "censored"


_TERMINAL_REASONS_BY_STATUS = {
    OrderResearchStatus.FILLED: frozenset({"full_fill"}),
    OrderResearchStatus.EXPIRED: frozenset({"good_til_time_expired"}),
    OrderResearchStatus.CANCELLED: frozenset(
        {
            "invalidation_cancelled_remainder",
            "target_cancelled_remainder",
            "invalidation_before_proxy_fill",
            "target_before_proxy_fill",
        }
    ),
    OrderResearchStatus.REJECTED: frozenset(
        {"quantity_limit", "off_tick_price", "invalid_price_geometry"}
    ),
    OrderResearchStatus.CENSORED: frozenset(
        {
            "missing_clock",
            "missing_top_of_book",
            "invalid_top_of_book",
            "crossed_top_of_book",
            "stale_top_of_book",
            "reset_source",
            "synthetic_source",
            "missing_mbo_mechanism",
            "fractional_contract_capacity",
            "insufficient_displayed_bbo_depth",
        }
    ),
}


def _timestamp(value: Any, *, name: str) -> pd.Timestamp:
    try:
        result = aware_timestamp(pd.Timestamp(value), name=name)
    except (TypeError, ValueError) as exc:
        raise ExecutionResearchError(f"{name} must be timezone aware") from exc
    return result.tz_convert("UTC")


def _finite(value: Any, *, name: str, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ExecutionResearchError(f"{name} is invalid")
    return result


def _normal(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Bar):
        return {
            "start": value.start.isoformat(),
            "open": value.open,
            "high": value.high,
            "low": value.low,
            "close": value.close,
            "volume": value.volume,
            "symbol": value.symbol,
            "instrument_id": value.instrument_id,
            "synthetic_no_trade": value.synthetic_no_trade,
            "data_gap_before_minutes": value.data_gap_before_minutes,
        }
    if isinstance(value, TopOfBook):
        return {
            "observed_at": value.observed_at.isoformat(),
            "bid": value.bid,
            "ask": value.ask,
            "bid_size": value.bid_size,
            "ask_size": value.ask_size,
        }
    if isinstance(value, Mapping):
        return {str(key): _normal(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normal(item) for item in value]
    if is_dataclass(value):
        return {
            item.name: _normal(getattr(value, item.name))
            for item in fields(value)
        }
    if hasattr(value, "item"):
        return _normal(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _normal(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _exact_ids(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if (
        not result
        or len(result) != len(set(result))
        or any(not isinstance(value, str) or not value for value in result)
    ):
        raise ExecutionResearchError(f"{name} must be unique non-empty identities")
    return tuple(sorted(result))


def _sha256_text(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExecutionResearchError(f"{name} must be a lowercase SHA-256")
    return value


def _contract_identity(value: str | int, *, name: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ExecutionResearchError(f"{name} must be an exact string or integer")
    if isinstance(value, int) and value < 0:
        raise ExecutionResearchError(f"{name} integer cannot be negative")
    if isinstance(value, str) and (not value or value != value.strip()):
        raise ExecutionResearchError(f"{name} string is empty or not canonical")
    return value


@dataclass(frozen=True)
class ExecutionResearchConfig:
    """Versioned, canonical preregistration for the bounded v1 model."""

    schema_version: str
    model_version: str
    status: str
    authority: str
    ordered_methods: tuple[ExecutionMethod, ...]
    market_comparator: ExecutionMethod
    tick_size: float
    point_value: float
    commission_per_contract_per_side: float
    maximum_book_age_seconds: float
    clock_step_seconds: int
    maximum_quantity: int
    realized_spread_horizon_seconds: int
    minimum_complete_pairs: int
    canonical_payload: str
    config_id: str

    def __post_init__(self) -> None:
        try:
            payload = json.loads(self.canonical_payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ExecutionResearchError("canonical execution config is invalid") from exc
        fingerprint = hashlib.sha256(self.canonical_payload.encode("utf-8")).hexdigest()
        if (
            fingerprint != EXECUTION_RESEARCH_PROTOCOL_SHA256
            or self.config_id != f"execution-research-config:{fingerprint}"
            or self.schema_version != payload.get("schema_version")
            or self.model_version != payload.get("model_version")
            or self.status != payload.get("status")
            or self.authority != payload.get("authority")
            or tuple(self.ordered_methods)
            != tuple(ExecutionMethod(value) for value in payload.get("ordered_methods", ()))
            or self.market_comparator
            is not ExecutionMethod(payload.get("market_comparator"))
            or self.tick_size != float(payload.get("tick_size"))
            or self.point_value != float(payload.get("point_value"))
            or self.commission_per_contract_per_side
            != float(payload.get("commission_per_contract_per_side"))
            or self.maximum_book_age_seconds
            != float(payload.get("maximum_book_age_seconds"))
            or self.clock_step_seconds != int(payload.get("clock_step_seconds"))
            or self.maximum_quantity != int(payload.get("maximum_quantity"))
            or self.realized_spread_horizon_seconds
            != int(payload.get("realized_spread_horizon_seconds"))
            or self.minimum_complete_pairs
            != int(payload.get("inference", {}).get("minimum_complete_pairs"))
        ):
            raise ExecutionResearchError("execution research config identity drift")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExecutionResearchConfig":
        required = {
            "schema_version",
            "model_version",
            "status",
            "authority",
            "ordered_methods",
            "market_comparator",
            "tick_size",
            "point_value",
            "commission_per_contract_per_side",
            "maximum_book_age_seconds",
            "clock_step_seconds",
            "maximum_quantity",
            "realized_spread_horizon_seconds",
            "clock_model",
            "fill_model",
            "censor_policy",
            "metric_definitions",
            "inference",
        }
        if set(payload) != required:
            raise ExecutionResearchError("execution research config fields differ")
        methods = tuple(ExecutionMethod(item) for item in payload["ordered_methods"])
        inference = payload["inference"]
        fill_model = payload["fill_model"]
        clock_model = payload["clock_model"]
        censor_policy = payload["censor_policy"]
        metrics = payload["metric_definitions"]
        if not all(
            isinstance(value, Mapping)
            for value in (inference, clock_model, fill_model, censor_policy, metrics)
        ):
            raise ExecutionResearchError("execution research protocol sections are invalid")
        if (
            payload["schema_version"] != EXECUTION_RESEARCH_SCHEMA_VERSION
            or payload["model_version"] != EXECUTION_RESEARCH_MODEL_VERSION
            or payload["status"] != "development_unvalidated"
            or payload["authority"] != "research_only_never_submit"
            or methods != ORDERED_EXECUTION_METHODS
            or ExecutionMethod(payload["market_comparator"])
            is not ExecutionMethod.MARKET
            or inference.get("unit") != "research_intent_id"
            or inference.get("pairing")
            != "same_intent_all_methods_no_cross_intent_matching"
            or inference.get("primary_metric")
            != "implementation_shortfall_points"
            or inference.get("primary_pair_availability")
            != "both_full_fill_and_primary_metric_non_null_ignore_secondary_metric_censors"
            or inference.get("contrasts")
            != "each_non_market_method_minus_market"
            or inference.get("test") != "two_sided_exact_sign_test"
            or inference.get("multiplicity")
            != "holm_across_six_fixed_contrasts"
            or inference.get("cross_intent_independence_claimed") is not False
            or inference.get("confirmatory_claim") is not False
            or inference.get("tuning") is not False
            or clock_model
            != {
                "entry_gtt_field": "expires_at_compatibility_alias_for_entry_expires_at",
                "analysis_end_field": "analysis_ends_at",
                "default_analysis_end": "entry_expires_at_for_legacy_callers",
                "post_gtt_policy": (
                    "existing_position_stop_target_and_observation_continue;"
                    "new_entry_fills_forbidden"
                ),
            }
            or fill_model.get("queue_truth_available") is not False
            or censor_policy.get("invalid_source_is_unfilled") is not False
            or censor_policy.get("silent_fallback_allowed") is not False
            or set(metrics)
            != {
                "implementation_shortfall_points",
                "slippage_points",
                "realized_spread_points",
                "adverse_selection_points",
                "missed_opportunity_points",
                "mae_mfe",
                "target_before_invalidation",
            }
        ):
            raise ExecutionResearchError("execution research preregistration changed")
        tick_size = _finite(payload["tick_size"], name="tick_size", positive=True)
        point_value = _finite(payload["point_value"], name="point_value", positive=True)
        commission = _finite(
            payload["commission_per_contract_per_side"],
            name="commission_per_contract_per_side",
        )
        maximum_age = _finite(
            payload["maximum_book_age_seconds"],
            name="maximum_book_age_seconds",
        )
        integer_values = (
            payload["clock_step_seconds"],
            payload["maximum_quantity"],
            payload["realized_spread_horizon_seconds"],
            inference.get("minimum_complete_pairs"),
        )
        if (
            commission < 0.0
            or maximum_age < 0.0
            or any(type(value) is not int or value <= 0 for value in integer_values)
        ):
            raise ExecutionResearchError("execution research numeric contract is invalid")
        canonical_payload = json.dumps(
            _normal(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        identity = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
        if identity != EXECUTION_RESEARCH_PROTOCOL_SHA256:
            raise ExecutionResearchError(
                "execution research config fingerprint changed; version bump required"
            )
        return cls(
            schema_version=payload["schema_version"],
            model_version=payload["model_version"],
            status=payload["status"],
            authority=payload["authority"],
            ordered_methods=methods,
            market_comparator=ExecutionMethod.MARKET,
            tick_size=tick_size,
            point_value=point_value,
            commission_per_contract_per_side=commission,
            maximum_book_age_seconds=maximum_age,
            clock_step_seconds=int(payload["clock_step_seconds"]),
            maximum_quantity=int(payload["maximum_quantity"]),
            realized_spread_horizon_seconds=int(
                payload["realized_spread_horizon_seconds"]
            ),
            minimum_complete_pairs=int(inference["minimum_complete_pairs"]),
            canonical_payload=canonical_payload,
            config_id=f"execution-research-config:{identity}",
        )


def load_execution_research_config(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ExecutionResearchConfig:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ExecutionResearchError("execution research config is not a regular file")
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual != _sha256_text(
        expected_sha256, name="expected config SHA-256"
    ):
        raise ExecutionResearchError("execution research config SHA-256 mismatch")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExecutionResearchError("execution research config is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ExecutionResearchError("execution research config root is invalid")
    return ExecutionResearchConfig.from_payload(payload)


@dataclass(frozen=True)
class ExecutionResearchIntent:
    """Exact frozen research fact shared by every counterfactual method."""

    source_trade_intent_id: str
    created_at: pd.Timestamp
    expires_at: pd.Timestamp
    symbol: str
    instrument_id: str | int
    vendor_instrument_id: int
    instrument_mapping_id: str
    instrument_mapping_sha256: str
    side: Direction
    quantity: int
    tick_size: float
    point_value: float
    arrival_mid: float
    invalidation_price: float
    target_price: float
    method_prices: tuple[tuple[ExecutionMethod, float], ...]
    source_identity_ids: tuple[str, ...]
    paired_stratum: tuple[str, ...]
    analysis_ends_at: pd.Timestamp | None = None
    schema_version: str = EXECUTION_RESEARCH_SCHEMA_VERSION
    submission_allowed: bool = False
    research_intent_id: str = field(init=False)

    def __post_init__(self) -> None:
        created = _timestamp(self.created_at, name="research intent created_at")
        expires = _timestamp(self.expires_at, name="research intent expires_at")
        analysis_ends = (
            expires
            if self.analysis_ends_at is None
            else _timestamp(
                self.analysis_ends_at,
                name="research intent analysis_ends_at",
            )
        )
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "analysis_ends_at", analysis_ends)
        object.__setattr__(self, "side", Direction(self.side))
        object.__setattr__(
            self,
            "instrument_id",
            _contract_identity(self.instrument_id, name="research intent instrument_id"),
        )
        if type(self.vendor_instrument_id) is not int or self.vendor_instrument_id < 0:
            raise ExecutionResearchError("vendor instrument_id must be a nonnegative integer")
        if not self.instrument_mapping_id:
            raise ExecutionResearchError("instrument mapping identity is required")
        _sha256_text(
            self.instrument_mapping_sha256, name="instrument mapping SHA-256"
        )
        sources = _exact_ids(self.source_identity_ids, name="source_identity_ids")
        strata = tuple(self.paired_stratum)
        object.__setattr__(self, "source_identity_ids", sources)
        object.__setattr__(self, "paired_stratum", strata)
        prices = tuple(
            (ExecutionMethod(method), _finite(price, name=f"{method} price", positive=True))
            for method, price in self.method_prices
        )
        object.__setattr__(self, "method_prices", prices)
        arrival = _finite(self.arrival_mid, name="arrival_mid", positive=True)
        invalidation = _finite(
            self.invalidation_price, name="invalidation_price", positive=True
        )
        target = _finite(self.target_price, name="target_price", positive=True)
        tick_size = _finite(self.tick_size, name="intent tick_size", positive=True)
        point_value = _finite(self.point_value, name="intent point_value", positive=True)
        object.__setattr__(self, "arrival_mid", arrival)
        object.__setattr__(self, "invalidation_price", invalidation)
        object.__setattr__(self, "target_price", target)
        object.__setattr__(self, "tick_size", tick_size)
        object.__setattr__(self, "point_value", point_value)
        if (
            self.schema_version != EXECUTION_RESEARCH_SCHEMA_VERSION
            or self.submission_allowed is not False
            or not isinstance(self.source_trade_intent_id, str)
            or not self.source_trade_intent_id
            or expires <= created
            or analysis_ends < expires
            or not _tick_aligned(invalidation, tick_size)
            or not _tick_aligned(target, tick_size)
            or not self.symbol
            or type(self.quantity) is not int
            or self.quantity <= 0
            or tuple(method for method, _ in prices) != LIMIT_EXECUTION_METHODS
            or len(set(method for method, _ in prices)) != len(prices)
            or not strata
            or any(not isinstance(value, str) or not value for value in strata)
            or (
                self.side is Direction.LONG
                and not invalidation < arrival < target
            )
            or (
                self.side is Direction.SHORT
                and not target < arrival < invalidation
            )
        ):
            raise ExecutionResearchError("research intent contract is invalid")
        payload = {
            name: value
            for name, value in self.__dict__.items()
            if name != "research_intent_id"
        }
        object.__setattr__(
            self,
            "research_intent_id",
            f"execution-research-intent:{canonical_sha256(payload)}",
        )

    @property
    def entry_expires_at(self) -> pd.Timestamp:
        """Versioned entry GTT; ``expires_at`` remains its compatibility name."""

        return self.expires_at

    def price_for(self, method: ExecutionMethod) -> float | None:
        method = ExecutionMethod(method)
        if method is ExecutionMethod.MARKET:
            return None
        return dict(self.method_prices)[method]


def research_intent_from_trade_intent(
    intent: "TradeIntent",
    *,
    arrival_book: TopOfBook,
    tick_size: float,
    vendor_instrument_id: int,
    instrument_mapping_id: str,
    instrument_mapping_sha256: str,
    method_prices: Mapping[ExecutionMethod, float],
    paired_stratum: Sequence[str],
    analysis_ends_at: pd.Timestamp | None = None,
) -> ExecutionResearchIntent:
    """Freeze one Phase 7 TradeIntent without inventing missing method prices."""

    from .trade_intent import TradeIntent

    if not isinstance(intent, TradeIntent):
        raise TypeError("execution research requires a typed TradeIntent")
    if not isinstance(arrival_book, TopOfBook):
        raise TypeError("execution research requires a typed arrival TopOfBook")
    if arrival_book.observed_at > intent.created_at:
        raise ExecutionResearchError("arrival BBO is later than the TradeIntent")
    if set(method_prices) != set(LIMIT_EXECUTION_METHODS):
        raise ExecutionResearchError("all and only fixed method prices are required")
    provided = tuple(
        (method, method_prices[method]) for method in LIMIT_EXECUTION_METHODS
    )
    return ExecutionResearchIntent(
        source_trade_intent_id=intent.intent_id,
        created_at=intent.created_at,
        expires_at=intent.expires_at,
        symbol=intent.symbol,
        instrument_id=intent.instrument_id,
        vendor_instrument_id=vendor_instrument_id,
        instrument_mapping_id=instrument_mapping_id,
        instrument_mapping_sha256=instrument_mapping_sha256,
        side=intent.side,
        quantity=intent.quantity,
        tick_size=tick_size,
        point_value=intent.point_value,
        arrival_mid=(arrival_book.bid + arrival_book.ask) / 2.0,
        invalidation_price=intent.invalidation.price,
        target_price=intent.targets[0].price,
        method_prices=provided,
        source_identity_ids=(intent.intent_id, *intent.source_identity_ids),
        paired_stratum=tuple(paired_stratum),
        analysis_ends_at=analysis_ends_at,
    )


@dataclass(frozen=True)
class MinuteExecutionInput:
    """One completed-minute BBO/MBO fact with exact source bindings."""

    decision_time: pd.Timestamp
    symbol: str
    instrument_id: str | int
    vendor_instrument_id: int
    instrument_mapping_id: str
    instrument_mapping_sha256: str
    bar: Bar
    book: TopOfBook | None
    book_valid: bool
    invalid_reason: str | None
    passive_bid_fill_volume: float | None
    passive_ask_fill_volume: float | None
    displayed_bid_add_volume: float | None
    displayed_ask_add_volume: float | None
    displayed_bid_cancel_volume: float | None
    displayed_ask_cancel_volume: float | None
    aggressor_buy_volume: float | None
    aggressor_sell_volume: float | None
    source_reset: bool
    synthetic_source: bool
    source_artifact_id: str
    source_artifact_sha256: str
    source_row_sha256: str
    input_id: str = field(init=False)

    def __post_init__(self) -> None:
        clock = _timestamp(self.decision_time, name="execution input decision_time")
        object.__setattr__(self, "decision_time", clock)
        object.__setattr__(
            self,
            "instrument_id",
            _contract_identity(self.instrument_id, name="execution input instrument_id"),
        )
        if (
            type(self.vendor_instrument_id) is not int
            or self.vendor_instrument_id < 0
            or not self.instrument_mapping_id
        ):
            raise ExecutionResearchError("execution input instrument mapping is invalid")
        _sha256_text(
            self.instrument_mapping_sha256, name="instrument mapping SHA-256"
        )
        if not isinstance(self.bar, Bar) or self.bar.end != clock:
            raise ExecutionResearchError("bar must end at the execution decision clock")
        if self.bar.instrument_id != self.vendor_instrument_id:
            raise ExecutionResearchError("vendor instrument differs from its OHLCV bar")
        if not self.symbol or self.symbol != self.bar.symbol:
            raise ExecutionResearchError("execution input symbol differs from its OHLCV bar")
        if self.bar.synthetic_no_trade and not self.synthetic_source:
            raise ExecutionResearchError("synthetic bar must be labelled synthetic source")
        if self.book is not None and self.book.observed_at > clock:
            raise ExecutionResearchError("future BBO cannot enter execution research")
        if not isinstance(self.book_valid, bool):
            raise ExecutionResearchError("book_valid must be bool")
        if self.book_valid and (self.book is None or self.invalid_reason is not None):
            raise ExecutionResearchError("valid BBO identity contradicts input fields")
        if not self.book_valid and not self.invalid_reason:
            raise ExecutionResearchError("invalid BBO requires an exact reason")
        if not self.source_artifact_id:
            raise ExecutionResearchError("source artifact identity is required")
        _sha256_text(self.source_artifact_sha256, name="source artifact SHA-256")
        _sha256_text(self.source_row_sha256, name="source row SHA-256")
        payload = {
            name: value for name, value in self.__dict__.items() if name != "input_id"
        }
        object.__setattr__(
            self,
            "input_id",
            f"execution-minute-input:{canonical_sha256(payload)}",
        )

    def source_censor_reason(
        self, config: ExecutionResearchConfig
    ) -> str | None:
        if self.synthetic_source:
            return "synthetic_source"
        if self.source_reset:
            return "reset_source"
        if not self.book_valid:
            reason = str(self.invalid_reason)
            if "missing" in reason:
                return "missing_top_of_book"
            if "cross" in reason:
                return "crossed_top_of_book"
            return "invalid_top_of_book"
        if self.book is None:
            return "missing_top_of_book"
        age = (self.decision_time - self.book.observed_at).total_seconds()
        if age < 0.0:
            raise ExecutionResearchError("future BBO cannot enter execution research")
        if age > config.maximum_book_age_seconds:
            return "stale_top_of_book"
        if any(
            abs(float(value) - round(float(value))) > 1e-9
            for value in (self.book.bid_size, self.book.ask_size)
        ):
            return "fractional_contract_capacity"
        mechanism = (
            self.passive_bid_fill_volume,
            self.passive_ask_fill_volume,
            self.displayed_bid_add_volume,
            self.displayed_ask_add_volume,
            self.displayed_bid_cancel_volume,
            self.displayed_ask_cancel_volume,
            self.aggressor_buy_volume,
            self.aggressor_sell_volume,
        )
        if any(
            value is None
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in mechanism
        ):
            return "missing_mbo_mechanism"
        if any(
            abs(float(value) - round(float(value))) > 1e-9
            for value in (
                self.passive_bid_fill_volume,
                self.passive_ask_fill_volume,
            )
        ):
            return "fractional_contract_capacity"
        return None


def minute_execution_inputs_from_phase6_frame(
    frame: pd.DataFrame,
    bars: Sequence[Bar],
    *,
    source_artifact_id: str,
    source_artifact_sha256: str,
    logical_instrument_id: str | int,
    instrument_mapping_id: str,
    instrument_mapping_sha256: str,
    reset_clocks: Sequence[pd.Timestamp] = (),
) -> tuple[MinuteExecutionInput, ...]:
    """Adapt an already hash-bound Phase 6 frame without changing its grain.

    The caller supplies the verified artifact identity/SHA.  This adapter then
    validates the public Phase 6 schema and binds every derived input to a hash
    of the complete minute row.  It does not load a looser subset or substitute
    values when a field is absent.
    """

    if not source_artifact_id:
        raise ExecutionResearchError("source artifact identity is required")
    artifact_sha = _sha256_text(
        source_artifact_sha256, name="source artifact SHA-256"
    )
    values = validate_mbo_mechanism_frame(frame)
    by_clock: dict[pd.Timestamp, Bar] = {}
    for bar in bars:
        if not isinstance(bar, Bar) or bar.end in by_clock:
            raise ExecutionResearchError("Phase 8 bars are invalid or duplicated")
        by_clock[bar.end] = bar
    frame_clocks = tuple(values["decision_time"])
    if set(frame_clocks) != set(by_clock) or len(frame_clocks) != len(by_clock):
        raise ExecutionResearchError("Phase 6 frame and OHLCV clocks differ")
    reset = {_timestamp(clock, name="reset clock") for clock in reset_clocks}
    if not reset.issubset(set(frame_clocks)):
        raise ExecutionResearchError("reset clock is outside the Phase 6 frame")

    results: list[MinuteExecutionInput] = []
    for row in values.to_dict(orient="records"):
        clock = _timestamp(row["decision_time"], name="Phase 6 decision_time")
        bar = by_clock[clock]
        if (
            str(row["symbol"]) != bar.symbol
            or int(row["instrument_id"]) != bar.instrument_id
        ):
            raise ExecutionResearchError("Phase 6 and OHLCV contract identities differ")
        book_valid = bool(row["book_valid"])
        book = (
            TopOfBook(
                observed_at=row["book_observed_at"],
                bid=float(row["bid"]),
                ask=float(row["ask"]),
                bid_size=float(row["bid_size"]),
                ask_size=float(row["ask_size"]),
            )
            if book_valid
            else None
        )
        invalid_reason = None if book_valid else str(row["invalid_reason"])
        results.append(
            MinuteExecutionInput(
                decision_time=clock,
                symbol=str(row["symbol"]),
                instrument_id=logical_instrument_id,
                vendor_instrument_id=int(row["instrument_id"]),
                instrument_mapping_id=instrument_mapping_id,
                instrument_mapping_sha256=instrument_mapping_sha256,
                bar=bar,
                book=book,
                book_valid=book_valid,
                invalid_reason=invalid_reason,
                passive_bid_fill_volume=float(row["passive_bid_fill_volume"]),
                passive_ask_fill_volume=float(row["passive_ask_fill_volume"]),
                displayed_bid_add_volume=float(row["displayed_bid_add_volume"]),
                displayed_ask_add_volume=float(row["displayed_ask_add_volume"]),
                displayed_bid_cancel_volume=float(row["displayed_bid_cancel_volume"]),
                displayed_ask_cancel_volume=float(row["displayed_ask_cancel_volume"]),
                aggressor_buy_volume=float(row["aggressor_buy_volume"]),
                aggressor_sell_volume=float(row["aggressor_sell_volume"]),
                source_reset=clock in reset,
                synthetic_source=bool(bar.synthetic_no_trade),
                source_artifact_id=source_artifact_id,
                source_artifact_sha256=artifact_sha,
                source_row_sha256=canonical_sha256(row),
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class ExecutionMethodOutcome:
    source_trade_intent_id: str
    research_intent_id: str
    config_id: str
    method: ExecutionMethod
    side: Direction
    intended_quantity: int
    arrival_mid: float
    point_value: float
    commission_per_contract_per_side: float
    source_input_ids: tuple[str, ...]
    order_price: float | None
    terminal_status: OrderResearchStatus
    terminal_reason: str
    last_evaluated_at: pd.Timestamp
    filled_quantity: float
    fill_fraction: float
    partial_fill: bool
    first_fill_at: pd.Timestamp | None
    full_fill_at: pd.Timestamp | None
    time_to_first_fill_seconds: float | None
    time_to_full_fill_seconds: float | None
    average_fill_price: float | None
    implementation_shortfall_points: float | None
    slippage_points: float | None
    realized_spread_points: float | None
    adverse_selection_points: float | None
    missed_opportunity_points: float | None
    mae_points: float | None
    mfe_points: float | None
    target_before_invalidation: bool | None
    position_outcome: str | None
    displayed_defense_proxy_contracts: float
    queue_truth_available: bool
    analysis_censor_reasons: tuple[str, ...]
    outcome_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", ExecutionMethod(self.method))
        object.__setattr__(self, "side", Direction(self.side))
        object.__setattr__(self, "terminal_status", OrderResearchStatus(self.terminal_status))
        object.__setattr__(
            self,
            "last_evaluated_at",
            _timestamp(self.last_evaluated_at, name="outcome last_evaluated_at"),
        )
        if self.first_fill_at is not None:
            object.__setattr__(
                self,
                "first_fill_at",
                _timestamp(self.first_fill_at, name="outcome first_fill_at"),
            )
        if self.full_fill_at is not None:
            object.__setattr__(
                self,
                "full_fill_at",
                _timestamp(self.full_fill_at, name="outcome full_fill_at"),
            )
        source_ids = tuple(self.source_input_ids)
        censor_reasons = tuple(self.analysis_censor_reasons)
        numeric_optional = (
            self.order_price,
            self.time_to_first_fill_seconds,
            self.time_to_full_fill_seconds,
            self.average_fill_price,
            self.implementation_shortfall_points,
            self.slippage_points,
            self.realized_spread_points,
            self.adverse_selection_points,
            self.missed_opportunity_points,
            self.mae_points,
            self.mfe_points,
        )
        has_fill = self.filled_quantity > 0.0
        is_full = abs(self.fill_fraction - 1.0) <= 1e-12
        is_partial = 0.0 < self.fill_fraction < 1.0
        if (
            not self.source_trade_intent_id
            or not self.research_intent_id
            or not self.config_id
            or not source_ids
            or len(source_ids) != len(set(source_ids))
            or any(not isinstance(value, str) or not value for value in source_ids)
            or not self.terminal_reason
            or self.terminal_reason
            not in _TERMINAL_REASONS_BY_STATUS[self.terminal_status]
            or type(self.intended_quantity) is not int
            or self.intended_quantity <= 0
            or not math.isfinite(float(self.arrival_mid))
            or self.arrival_mid <= 0.0
            or not math.isfinite(float(self.point_value))
            or self.point_value <= 0.0
            or not math.isfinite(float(self.commission_per_contract_per_side))
            or self.commission_per_contract_per_side < 0.0
            or self.queue_truth_available is not False
            or not math.isfinite(float(self.filled_quantity))
            or self.filled_quantity < 0.0
            or abs(self.filled_quantity - round(self.filled_quantity)) > 1e-9
            or not math.isfinite(float(self.fill_fraction))
            or not 0.0 <= self.fill_fraction <= 1.0
            or (has_fill != (self.fill_fraction > 0.0))
            or abs(
                self.fill_fraction
                - self.filled_quantity / float(self.intended_quantity)
            )
            > 1e-12
            or self.partial_fill != is_partial
            or (
                has_fill
                and abs(
                    self.filled_quantity / self.fill_fraction
                    - round(self.filled_quantity / self.fill_fraction)
                )
                > 1e-9
            )
            or (self.terminal_status is OrderResearchStatus.FILLED) != is_full
            or (self.first_fill_at is not None) != has_fill
            or (self.full_fill_at is not None) != is_full
            or (self.time_to_first_fill_seconds is not None) != has_fill
            or (self.time_to_full_fill_seconds is not None) != is_full
            or (self.average_fill_price is not None) != has_fill
            or (self.implementation_shortfall_points is not None) != has_fill
            or (self.slippage_points is not None) != has_fill
            or (not has_fill and self.mae_points is not None)
            or (not has_fill and self.mfe_points is not None)
            or any(
                value is not None and not math.isfinite(float(value))
                for value in numeric_optional
            )
            or (
                self.method is ExecutionMethod.MARKET
                and self.order_price is not None
            )
            or (
                self.method is not ExecutionMethod.MARKET
                and (self.order_price is None or self.order_price <= 0.0)
            )
            or (
                has_fill
                and (
                    self.average_fill_price is None
                    or self.implementation_shortfall_points is None
                    or self.slippage_points is None
                    or not math.isclose(
                        self.implementation_shortfall_points,
                        self.side.sign
                        * (self.average_fill_price - self.arrival_mid)
                        + self.commission_per_contract_per_side / self.point_value,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        self.slippage_points,
                        self.side.sign
                        * (
                            self.average_fill_price
                            - (
                                self.arrival_mid
                                if self.order_price is None
                                else self.order_price
                            )
                        ),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )
            )
            or (
                self.first_fill_at is not None
                and self.first_fill_at > self.last_evaluated_at
            )
            or (
                self.full_fill_at is not None
                and (
                    self.first_fill_at is None
                    or self.full_fill_at < self.first_fill_at
                    or self.full_fill_at > self.last_evaluated_at
                )
            )
            or any(
                value is not None and value < 0.0
                for value in (
                    self.time_to_first_fill_seconds,
                    self.time_to_full_fill_seconds,
                    self.adverse_selection_points,
                    self.missed_opportunity_points,
                    self.mae_points,
                    self.mfe_points,
                )
            )
            or not math.isfinite(float(self.displayed_defense_proxy_contracts))
            or self.displayed_defense_proxy_contracts < 0.0
            or censor_reasons != tuple(sorted(set(censor_reasons)))
            or any(not isinstance(value, str) or not value for value in censor_reasons)
            or (
                (self.realized_spread_points is None)
                != (self.adverse_selection_points is None)
            )
            or (
                self.target_before_invalidation is True
                and self.position_outcome != "target"
            )
            or (
                self.target_before_invalidation is False
                and self.position_outcome
                not in {"invalidation", "same_bar_ambiguous_invalidation_first"}
            )
            or (
                self.terminal_status is OrderResearchStatus.REJECTED
                and has_fill
            )
        ):
            raise ExecutionResearchError("execution method outcome invariants failed")
        object.__setattr__(self, "source_input_ids", source_ids)
        object.__setattr__(self, "analysis_censor_reasons", censor_reasons)
        payload = {
            name: value for name, value in self.__dict__.items() if name != "outcome_id"
        }
        object.__setattr__(
            self, "outcome_id", f"execution-method-outcome:{canonical_sha256(payload)}"
        )


@dataclass(frozen=True)
class PairedExecutionStudy:
    intent_contract: ExecutionResearchIntent
    protocol_contract: ExecutionResearchConfig
    source_trade_intent_id: str
    research_intent_id: str
    config_id: str
    paired_stratum: tuple[str, ...]
    source_input_ids: tuple[str, ...]
    outcomes: tuple[ExecutionMethodOutcome, ...]
    study_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.intent_contract, ExecutionResearchIntent)
            or not isinstance(self.protocol_contract, ExecutionResearchConfig)
            or self.intent_contract.source_trade_intent_id
            != self.source_trade_intent_id
            or self.intent_contract.research_intent_id != self.research_intent_id
            or self.intent_contract.paired_stratum != self.paired_stratum
            or self.protocol_contract.config_id != self.config_id
            or not self.source_trade_intent_id
            or not self.research_intent_id
            or not self.config_id
            or not self.paired_stratum
            or any(not isinstance(value, str) or not value for value in self.paired_stratum)
            or not self.source_input_ids
            or len(self.source_input_ids) != len(set(self.source_input_ids))
            or len({outcome.outcome_id for outcome in self.outcomes})
            != len(self.outcomes)
            or tuple(outcome.method for outcome in self.outcomes)
            != ORDERED_EXECUTION_METHODS
            or any(
                outcome.research_intent_id != self.research_intent_id
                or outcome.source_trade_intent_id != self.source_trade_intent_id
                or outcome.config_id != self.config_id
                or outcome.source_input_ids != self.source_input_ids
                or outcome.side is not self.intent_contract.side
                or outcome.intended_quantity != self.intent_contract.quantity
                or outcome.arrival_mid != self.intent_contract.arrival_mid
                or outcome.point_value != self.intent_contract.point_value
                or outcome.commission_per_contract_per_side
                != self.protocol_contract.commission_per_contract_per_side
                or outcome.order_price
                != self.intent_contract.price_for(outcome.method)
                for outcome in self.outcomes
            )
        ):
            raise ExecutionResearchError("paired outcome intent/input parity failed")
        payload = {
            "source_trade_intent_id": self.source_trade_intent_id,
            "research_intent_id": self.research_intent_id,
            "config_id": self.config_id,
            "paired_stratum": self.paired_stratum,
            "source_input_ids": self.source_input_ids,
            "outcome_ids": tuple(outcome.outcome_id for outcome in self.outcomes),
        }
        object.__setattr__(self, "study_id", f"paired-execution-study:{canonical_sha256(payload)}")


def _tick_aligned(price: float, tick_size: float) -> bool:
    ticks = float(price) / float(tick_size)
    return abs(ticks - round(ticks)) <= 1e-8


def _limit_touched(direction: Direction, price: float, bar: Bar) -> bool:
    return bar.low <= price if direction is Direction.LONG else bar.high >= price


def _stop_touched(intent: ExecutionResearchIntent, bar: Bar) -> bool:
    return (
        bar.low <= intent.invalidation_price
        if intent.side is Direction.LONG
        else bar.high >= intent.invalidation_price
    )


def _target_touched(intent: ExecutionResearchIntent, bar: Bar) -> bool:
    return (
        bar.high >= intent.target_price
        if intent.side is Direction.LONG
        else bar.low <= intent.target_price
    )


def _marketable(direction: Direction, limit_price: float, book: TopOfBook) -> bool:
    return (
        limit_price >= book.ask
        if direction is Direction.LONG
        else limit_price <= book.bid
    )


def _best_price_and_size(direction: Direction, book: TopOfBook) -> tuple[float, int]:
    price, raw_size = (
        (float(book.ask), float(book.ask_size))
        if direction is Direction.LONG
        else (float(book.bid), float(book.bid_size))
    )
    return price, math.floor(max(0.0, raw_size))


def _passive_capacity(direction: Direction, value: MinuteExecutionInput) -> int:
    raw = (
        value.passive_bid_fill_volume
        if direction is Direction.LONG
        else value.passive_ask_fill_volume
    )
    return math.floor(max(0.0, float(raw or 0.0)))


def _defense_proxy(direction: Direction, value: MinuteExecutionInput) -> float:
    if direction is Direction.LONG:
        added, cancelled = (
            value.displayed_bid_add_volume,
            value.displayed_bid_cancel_volume,
        )
    else:
        added, cancelled = (
            value.displayed_ask_add_volume,
            value.displayed_ask_cancel_volume,
        )
    return max(0.0, float(added or 0.0) - float(cancelled or 0.0))


def _rejected_method(
    intent: ExecutionResearchIntent,
    config: ExecutionResearchConfig,
    method: ExecutionMethod,
    inputs: tuple[MinuteExecutionInput, ...],
    reason: str,
) -> ExecutionMethodOutcome:
    return ExecutionMethodOutcome(
        source_trade_intent_id=intent.source_trade_intent_id,
        research_intent_id=intent.research_intent_id,
        config_id=config.config_id,
        method=method,
        side=intent.side,
        intended_quantity=intent.quantity,
        arrival_mid=intent.arrival_mid,
        point_value=intent.point_value,
        commission_per_contract_per_side=config.commission_per_contract_per_side,
        source_input_ids=tuple(item.input_id for item in inputs),
        order_price=intent.price_for(method),
        terminal_status=OrderResearchStatus.REJECTED,
        terminal_reason=reason,
        last_evaluated_at=intent.created_at,
        filled_quantity=0.0,
        fill_fraction=0.0,
        partial_fill=False,
        first_fill_at=None,
        full_fill_at=None,
        time_to_first_fill_seconds=None,
        time_to_full_fill_seconds=None,
        average_fill_price=None,
        implementation_shortfall_points=None,
        slippage_points=None,
        realized_spread_points=None,
        adverse_selection_points=None,
        missed_opportunity_points=None,
        mae_points=None,
        mfe_points=None,
        target_before_invalidation=None,
        position_outcome=None,
        displayed_defense_proxy_contracts=0.0,
        queue_truth_available=False,
        analysis_censor_reasons=(),
    )


def _evaluate_method(
    intent: ExecutionResearchIntent,
    inputs: tuple[MinuteExecutionInput, ...],
    config: ExecutionResearchConfig,
    method: ExecutionMethod,
) -> ExecutionMethodOutcome:
    order_price = intent.price_for(method)
    if intent.quantity > config.maximum_quantity:
        return _rejected_method(intent, config, method, inputs, "quantity_limit")
    if order_price is not None:
        if not _tick_aligned(order_price, config.tick_size):
            return _rejected_method(intent, config, method, inputs, "off_tick_price")
        if (
            intent.side is Direction.LONG
            and not intent.invalidation_price < order_price < intent.target_price
        ) or (
            intent.side is Direction.SHORT
            and not intent.target_price < order_price < intent.invalidation_price
        ):
            return _rejected_method(intent, config, method, inputs, "invalid_price_geometry")

    fills: list[tuple[pd.Timestamp, float, float]] = []
    filled = 0.0
    defense_proxy = 0.0
    status: OrderResearchStatus | None = None
    terminal_reason = ""
    position_outcome: str | None = None
    target_before: bool | None = None
    mae = 0.0
    mfe = 0.0
    last_clock = intent.created_at
    analysis_censors: list[str] = []
    previous_clock: pd.Timestamp | None = None
    terminal_mid: float | None = None
    path_metrics_available = True

    for item in inputs:
        if (
            item.decision_time < intent.created_at
            or item.decision_time > intent.analysis_ends_at
        ):
            continue
        last_clock = item.decision_time
        expected = (
            intent.created_at
            if previous_clock is None
            else previous_clock
            + pd.Timedelta(config.clock_step_seconds, unit="s")
        )
        if item.decision_time != expected:
            # A missing pre-GTT clock can hide an entry fill and therefore
            # censors the primary fill outcome.  Once GTT has elapsed, the
            # entry quantity is frozen and the same gap affects only the
            # still-open position/path metrics.
            if (
                filled < intent.quantity - 1e-9
                and expected < intent.entry_expires_at
            ):
                status = OrderResearchStatus.CENSORED
                terminal_reason = "missing_clock"
            else:
                if status is None:
                    status = (
                        OrderResearchStatus.FILLED
                        if filled >= intent.quantity - 1e-9
                        else OrderResearchStatus.EXPIRED
                    )
                    terminal_reason = (
                        "full_fill"
                        if status is OrderResearchStatus.FILLED
                        else "good_til_time_expired"
                    )
                analysis_censors.append("missing_clock")
                if filled > 0.0:
                    position_outcome = position_outcome or "censored"
                    path_metrics_available = False
            break
        previous_clock = item.decision_time
        if (
            item.symbol != intent.symbol
            or item.instrument_id != intent.instrument_id
            or item.vendor_instrument_id != intent.vendor_instrument_id
            or item.instrument_mapping_id != intent.instrument_mapping_id
            or item.instrument_mapping_sha256 != intent.instrument_mapping_sha256
        ):
            raise ExecutionResearchError("intent and minute input contract differ")

        source_reason = item.source_censor_reason(config)
        if source_reason is not None:
            if (
                filled < intent.quantity - 1e-9
                and item.decision_time < intent.entry_expires_at
            ):
                status = OrderResearchStatus.CENSORED
                terminal_reason = source_reason
            else:
                if status is None:
                    status = (
                        OrderResearchStatus.FILLED
                        if filled >= intent.quantity - 1e-9
                        else OrderResearchStatus.EXPIRED
                    )
                    terminal_reason = (
                        "full_fill"
                        if status is OrderResearchStatus.FILLED
                        else "good_til_time_expired"
                    )
                analysis_censors.append(source_reason)
                if filled > 0.0:
                    position_outcome = position_outcome or "censored"
                    path_metrics_available = False
            break
        assert item.book is not None
        terminal_mid = (item.book.bid + item.book.ask) / 2.0
        if (
            item.decision_time < intent.entry_expires_at
            and filled < intent.quantity - 1e-9
        ):
            defense_proxy += _defense_proxy(intent.side, item)

        previous_filled = filled
        candidate_kind: str | None = None
        candidate_price: float | None = None
        candidate_quantity = 0.0
        candidate_depth_censor = False
        if (
            item.decision_time < intent.entry_expires_at
            and filled < intent.quantity - 1e-9
        ):
            remaining = float(intent.quantity) - filled
            if method is ExecutionMethod.MARKET:
                if item.decision_time != intent.created_at:
                    status = OrderResearchStatus.CENSORED
                    terminal_reason = "missing_clock"
                    break
                price, capacity = _best_price_and_size(intent.side, item.book)
                candidate_kind = "quote"
                candidate_price = price
                candidate_quantity = min(remaining, capacity)
                candidate_depth_censor = candidate_quantity < remaining - 1e-9
            else:
                assert order_price is not None
                if _marketable(intent.side, order_price, item.book):
                    best, capacity = _best_price_and_size(intent.side, item.book)
                    candidate_kind = "quote"
                    candidate_price = best
                    candidate_quantity = min(remaining, capacity)
                    candidate_depth_censor = candidate_quantity < remaining - 1e-9
                elif (
                    item.decision_time > intent.created_at
                    and _limit_touched(intent.side, order_price, item.bar)
                ):
                    candidate_quantity = min(
                        remaining, _passive_capacity(intent.side, item)
                    )
                    if candidate_quantity > 0.0:
                        candidate_kind = "resting"
                        candidate_price = order_price

        # A quote observed at this completed clock is available only after the
        # bar.  Existing partial exposure therefore resolves this bar before a
        # new quote fill can change its average price or quantity.
        if (
            previous_filled > 0.0
            and item.decision_time > intent.created_at
        ):
            prior_average = (
                sum(quantity * price for _, quantity, price in fills)
                / previous_filled
            )
            adverse = item.bar.low if intent.side is Direction.LONG else item.bar.high
            favorable = item.bar.high if intent.side is Direction.LONG else item.bar.low
            mae = max(
                mae,
                max(0.0, -intent.side.sign * (adverse - prior_average)),
            )
            mfe = max(
                mfe,
                max(0.0, intent.side.sign * (favorable - prior_average)),
            )
            stop = _stop_touched(intent, item.bar)
            target = _target_touched(intent, item.bar)
            if stop or target:
                position_outcome = (
                    "same_bar_ambiguous_invalidation_first"
                    if stop and target
                    else "invalidation"
                    if stop
                    else "target"
                )
                target_before = not stop
                if previous_filled < intent.quantity - 1e-9:
                    if item.decision_time <= intent.entry_expires_at:
                        status = OrderResearchStatus.CANCELLED
                        terminal_reason = (
                            "invalidation_cancelled_remainder"
                            if stop
                            else "target_cancelled_remainder"
                        )
                break

        # A completed bar can prove that the structural target was reached
        # while the entry was still pending.  An end-clock quote is known only
        # after that bar, so it cannot resurrect the entry.  A resting proxy
        # fill that depends on the same bar retains the separately frozen
        # adverse-first ambiguity rule below.
        if (
            previous_filled == 0.0
            and item.decision_time > intent.created_at
            and candidate_kind != "resting"
        ):
            stop = _stop_touched(intent, item.bar)
            target = _target_touched(intent, item.bar)
            if stop or target:
                status = OrderResearchStatus.CANCELLED
                terminal_reason = (
                    "invalidation_before_proxy_fill"
                    if stop
                    else "target_before_proxy_fill"
                )
                break

        if candidate_quantity > 0.0:
            assert candidate_price is not None
            fills.append((item.decision_time, candidate_quantity, candidate_price))
            filled += candidate_quantity
        if candidate_depth_censor:
            status = OrderResearchStatus.CENSORED
            terminal_reason = "insufficient_displayed_bbo_depth"
            break

        # A resting fill is inferred from this completed bar.  Its intrabar
        # location is unknown, so adverse excursion/stop is charged and a
        # favorable target on that same bar is not credited.
        if (
            candidate_kind == "resting"
            and candidate_quantity > 0.0
            and item.decision_time > intent.created_at
        ):
            average = sum(quantity * price for _, quantity, price in fills) / filled
            adverse = item.bar.low if intent.side is Direction.LONG else item.bar.high
            mae = max(mae, max(0.0, -intent.side.sign * (adverse - average)))
            stop = _stop_touched(intent, item.bar)
            target = _target_touched(intent, item.bar)
            if stop:
                position_outcome = (
                    "same_bar_ambiguous_invalidation_first" if target else "invalidation"
                )
                target_before = False
                if filled < intent.quantity - 1e-9:
                    status = OrderResearchStatus.CANCELLED
                    terminal_reason = "invalidation_cancelled_remainder"
                break
        if filled >= intent.quantity - 1e-9 and status is None:
            status = OrderResearchStatus.FILLED
            terminal_reason = "full_fill"
        if item.decision_time >= intent.entry_expires_at:
            if filled < intent.quantity - 1e-9:
                status = OrderResearchStatus.EXPIRED
                terminal_reason = "good_til_time_expired"
            if filled == 0.0:
                break
        if item.decision_time >= intent.analysis_ends_at:
            break

    if status is None:
        if filled >= intent.quantity - 1e-9:
            status = OrderResearchStatus.FILLED
            terminal_reason = "full_fill"
        elif previous_clock is None or previous_clock < intent.entry_expires_at:
            status = OrderResearchStatus.CENSORED
            terminal_reason = "missing_clock"
        else:
            status = OrderResearchStatus.EXPIRED
            terminal_reason = "good_til_time_expired"

    if (
        filled > 0.0
        and position_outcome is None
        and (previous_clock is None or previous_clock < intent.analysis_ends_at)
    ):
        analysis_censors.append("missing_clock")
        position_outcome = "censored"
        path_metrics_available = False

    average_fill = (
        sum(quantity * price for _, quantity, price in fills) / filled
        if filled > 0.0
        else None
    )
    first_fill = fills[0][0] if fills else None
    full_fill = (
        next(
            clock
            for index, (clock, _, _) in enumerate(fills)
            if sum(quantity for _, quantity, _ in fills[: index + 1])
            >= intent.quantity - 1e-9
        )
        if filled >= intent.quantity - 1e-9
        else None
    )
    implementation_shortfall = None
    slippage = None
    realized_spread = None
    adverse_selection = None
    if average_fill is not None:
        implementation_shortfall = (
            intent.side.sign * (average_fill - intent.arrival_mid)
            + config.commission_per_contract_per_side / config.point_value
        )
        reference = intent.arrival_mid if order_price is None else order_price
        slippage = intent.side.sign * (average_fill - reference)
        horizon = first_fill + pd.Timedelta(
            config.realized_spread_horizon_seconds, unit="s"
        )
        horizon_input = next(
            (item for item in inputs if item.decision_time >= horizon), None
        )
        if horizon_input is None:
            analysis_censors.append("realized_spread_horizon_unavailable")
        else:
            horizon_reason = horizon_input.source_censor_reason(config)
            if horizon_reason is not None or horizon_input.book is None:
                analysis_censors.append(
                    f"realized_spread_horizon_{horizon_reason or 'missing_top_of_book'}"
                )
            else:
                horizon_mid = (horizon_input.book.bid + horizon_input.book.ask) / 2.0
                realized_spread = 2.0 * intent.side.sign * (
                    average_fill - horizon_mid
                )
                adverse_selection = max(
                    0.0, intent.side.sign * (average_fill - horizon_mid)
                )
    fill_fraction = min(1.0, filled / float(intent.quantity))
    missed = (
        None
        if terminal_mid is None or not path_metrics_available
        else (1.0 - fill_fraction)
        * max(0.0, intent.side.sign * (terminal_mid - intent.arrival_mid))
    )
    return ExecutionMethodOutcome(
        source_trade_intent_id=intent.source_trade_intent_id,
        research_intent_id=intent.research_intent_id,
        config_id=config.config_id,
        method=method,
        side=intent.side,
        intended_quantity=intent.quantity,
        arrival_mid=intent.arrival_mid,
        point_value=intent.point_value,
        commission_per_contract_per_side=config.commission_per_contract_per_side,
        source_input_ids=tuple(item.input_id for item in inputs),
        order_price=order_price,
        terminal_status=status,
        terminal_reason=terminal_reason,
        last_evaluated_at=last_clock,
        filled_quantity=float(filled),
        fill_fraction=float(fill_fraction),
        partial_fill=bool(0.0 < filled < intent.quantity - 1e-9),
        first_fill_at=first_fill,
        full_fill_at=full_fill,
        time_to_first_fill_seconds=(
            None
            if first_fill is None
            else (first_fill - intent.created_at).total_seconds()
        ),
        time_to_full_fill_seconds=(
            None
            if full_fill is None
            else (full_fill - intent.created_at).total_seconds()
        ),
        average_fill_price=average_fill,
        implementation_shortfall_points=implementation_shortfall,
        slippage_points=slippage,
        realized_spread_points=realized_spread,
        adverse_selection_points=adverse_selection,
        missed_opportunity_points=missed,
        mae_points=(mae if filled > 0.0 and path_metrics_available else None),
        mfe_points=(mfe if filled > 0.0 and path_metrics_available else None),
        target_before_invalidation=target_before,
        position_outcome=position_outcome,
        displayed_defense_proxy_contracts=float(defense_proxy),
        queue_truth_available=False,
        analysis_censor_reasons=tuple(sorted(set(analysis_censors))),
    )


def evaluate_execution_research_intent(
    intent: ExecutionResearchIntent,
    inputs: Sequence[MinuteExecutionInput],
    config: ExecutionResearchConfig,
) -> PairedExecutionStudy:
    """Evaluate the fixed method family on one identical intent/input ledger."""

    if not isinstance(intent, ExecutionResearchIntent):
        raise TypeError("execution research requires ExecutionResearchIntent")
    if not isinstance(config, ExecutionResearchConfig):
        raise TypeError("execution research requires ExecutionResearchConfig")
    if intent.tick_size != config.tick_size or intent.point_value != config.point_value:
        raise ExecutionResearchError(
            "intent tick_size/point_value differ from the frozen execution protocol"
        )
    ordered = tuple(sorted(inputs, key=lambda item: item.decision_time))
    if (
        not ordered
        or any(not isinstance(item, MinuteExecutionInput) for item in ordered)
        or len({item.decision_time for item in ordered}) != len(ordered)
        or len({item.input_id for item in ordered}) != len(ordered)
        or ordered[0].decision_time != intent.created_at
        or any(
            item.decision_time < intent.created_at
            or item.decision_time > intent.analysis_ends_at
            or item.symbol != intent.symbol
            or item.instrument_id != intent.instrument_id
            or item.vendor_instrument_id != intent.vendor_instrument_id
            or item.instrument_mapping_id != intent.instrument_mapping_id
            or item.instrument_mapping_sha256 != intent.instrument_mapping_sha256
            for item in ordered
        )
    ):
        raise ExecutionResearchError("execution input ledger is invalid or not intent-parallel")
    arrival_source_reason = ordered[0].source_censor_reason(config)
    if arrival_source_reason is None:
        assert ordered[0].book is not None
        causal_arrival_mid = (ordered[0].book.bid + ordered[0].book.ask) / 2.0
        if not math.isclose(
            intent.arrival_mid,
            causal_arrival_mid,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ExecutionResearchError(
                "intent arrival_mid differs from the first causal BBO"
            )
    outcomes = tuple(
        _evaluate_method(intent, ordered, config, method)
        for method in config.ordered_methods
    )
    return PairedExecutionStudy(
        intent_contract=intent,
        protocol_contract=config,
        source_trade_intent_id=intent.source_trade_intent_id,
        research_intent_id=intent.research_intent_id,
        config_id=config.config_id,
        paired_stratum=intent.paired_stratum,
        source_input_ids=tuple(item.input_id for item in ordered),
        outcomes=outcomes,
    )


@dataclass(frozen=True)
class ExecutionMethodSummary:
    method: ExecutionMethod
    intents: int
    eligible_non_censored: int
    any_fill_count: int
    full_fill_count: int
    partial_fill_count: int
    censored_count: int
    cancelled_count: int
    expired_count: int
    rejected_count: int
    any_fill_probability: float | None
    full_fill_probability: float | None
    median_time_to_first_fill_seconds: float | None
    median_time_to_full_fill_seconds: float | None
    mean_implementation_shortfall_points: float | None
    mean_realized_spread_points: float | None
    mean_adverse_selection_points: float | None
    mean_missed_opportunity_points: float | None
    mean_mae_points: float | None
    mean_mfe_points: float | None
    target_before_invalidation_rate: float | None


@dataclass(frozen=True)
class PairedInferenceContrast:
    method: ExecutionMethod
    comparator: ExecutionMethod
    complete_pairs: int
    mean_difference_points: float | None
    median_difference_points: float | None
    method_better: int
    comparator_better: int
    ties: int
    sign_test_pvalue: float
    holm_adjusted_pvalue: float
    status: str
    cross_intent_independence_claimed: bool
    confirmatory_claim: bool


@dataclass(frozen=True)
class ExecutionResearchSummary:
    config_id: str
    study_ids: tuple[str, ...]
    method_summaries: tuple[ExecutionMethodSummary, ...]
    paired_contrasts: tuple[PairedInferenceContrast, ...]
    summary_id: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "config_id": self.config_id,
            "study_ids": self.study_ids,
            "method_summaries": self.method_summaries,
            "paired_contrasts": self.paired_contrasts,
        }
        object.__setattr__(self, "summary_id", f"execution-research-summary:{canonical_sha256(payload)}")


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _exact_sign_pvalue(wins: int, losses: int) -> float:
    count = wins + losses
    if count == 0:
        return 1.0
    smaller = min(wins, losses)
    tail = sum(math.comb(count, index) for index in range(smaller + 1)) / (2**count)
    return min(1.0, 2.0 * tail)


def summarize_execution_research(
    studies: Sequence[PairedExecutionStudy],
    config: ExecutionResearchConfig,
) -> ExecutionResearchSummary:
    """Summarize fixed same-intent pairs without tuning or cross-intent matching."""

    ordered_studies = tuple(sorted(studies, key=lambda item: item.research_intent_id))
    if (
        not ordered_studies
        or len({item.research_intent_id for item in ordered_studies})
        != len(ordered_studies)
        or len({item.source_trade_intent_id for item in ordered_studies})
        != len(ordered_studies)
        or any(
            item.config_id != config.config_id
            or item.protocol_contract != config
            for item in ordered_studies
        )
    ):
        raise ExecutionResearchError(
            "execution studies are empty, duplicate-source, or mixed-config"
        )
    by_method = {
        method: tuple(
            next(outcome for outcome in study.outcomes if outcome.method is method)
            for study in ordered_studies
        )
        for method in config.ordered_methods
    }
    summaries: list[ExecutionMethodSummary] = []
    for method, outcomes in by_method.items():
        eligible = tuple(
            outcome
            for outcome in outcomes
            if outcome.terminal_status is not OrderResearchStatus.CENSORED
        )
        first_times = [
            outcome.time_to_first_fill_seconds
            for outcome in outcomes
            if outcome.time_to_first_fill_seconds is not None
        ]
        full_times = [
            outcome.time_to_full_fill_seconds
            for outcome in outcomes
            if outcome.time_to_full_fill_seconds is not None
        ]
        resolved_targets = [
            outcome.target_before_invalidation
            for outcome in outcomes
            if outcome.target_before_invalidation is not None
        ]
        def values(name: str) -> list[float]:
            return [
                float(value)
                for outcome in outcomes
                if (value := getattr(outcome, name)) is not None
            ]
        summaries.append(
            ExecutionMethodSummary(
                method=method,
                intents=len(outcomes),
                eligible_non_censored=len(eligible),
                any_fill_count=sum(outcome.filled_quantity > 0 for outcome in eligible),
                full_fill_count=sum(
                    outcome.terminal_status is OrderResearchStatus.FILLED
                    for outcome in eligible
                ),
                partial_fill_count=sum(outcome.partial_fill for outcome in outcomes),
                censored_count=sum(
                    outcome.terminal_status is OrderResearchStatus.CENSORED
                    for outcome in outcomes
                ),
                cancelled_count=sum(
                    outcome.terminal_status is OrderResearchStatus.CANCELLED
                    for outcome in outcomes
                ),
                expired_count=sum(
                    outcome.terminal_status is OrderResearchStatus.EXPIRED
                    for outcome in outcomes
                ),
                rejected_count=sum(
                    outcome.terminal_status is OrderResearchStatus.REJECTED
                    for outcome in outcomes
                ),
                any_fill_probability=(
                    None
                    if not eligible
                    else sum(outcome.filled_quantity > 0 for outcome in eligible)
                    / len(eligible)
                ),
                full_fill_probability=(
                    None
                    if not eligible
                    else sum(
                        outcome.terminal_status is OrderResearchStatus.FILLED
                        for outcome in eligible
                    )
                    / len(eligible)
                ),
                median_time_to_first_fill_seconds=(
                    None if not first_times else float(median(first_times))
                ),
                median_time_to_full_fill_seconds=(
                    None if not full_times else float(median(full_times))
                ),
                mean_implementation_shortfall_points=_mean(
                    values("implementation_shortfall_points")
                ),
                mean_realized_spread_points=_mean(values("realized_spread_points")),
                mean_adverse_selection_points=_mean(
                    values("adverse_selection_points")
                ),
                mean_missed_opportunity_points=_mean(
                    values("missed_opportunity_points")
                ),
                mean_mae_points=_mean(values("mae_points")),
                mean_mfe_points=_mean(values("mfe_points")),
                target_before_invalidation_rate=(
                    None
                    if not resolved_targets
                    else sum(bool(value) for value in resolved_targets)
                    / len(resolved_targets)
                ),
            )
        )

    raw: list[dict[str, Any]] = []
    market = by_method[ExecutionMethod.MARKET]
    for method in LIMIT_EXECUTION_METHODS:
        differences: list[float] = []
        for treatment, comparator in zip(by_method[method], market, strict=True):
            if (
                treatment.terminal_status is OrderResearchStatus.FILLED
                and comparator.terminal_status is OrderResearchStatus.FILLED
                and treatment.implementation_shortfall_points is not None
                and comparator.implementation_shortfall_points is not None
            ):
                differences.append(
                    treatment.implementation_shortfall_points
                    - comparator.implementation_shortfall_points
                )
        wins = sum(value < -1e-12 for value in differences)
        losses = sum(value > 1e-12 for value in differences)
        ties = len(differences) - wins - losses
        raw.append(
            {
                "method": method,
                "complete_pairs": len(differences),
                "mean": _mean(differences),
                "median": None if not differences else float(median(differences)),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "p": _exact_sign_pvalue(wins, losses),
            }
        )
    adjusted = [1.0] * len(raw)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(raw)), key=lambda i: raw[i]["p"])):
        candidate = min(1.0, (len(raw) - rank) * raw[index]["p"])
        running = max(running, candidate)
        adjusted[index] = running
    contrasts = tuple(
        PairedInferenceContrast(
            method=value["method"],
            comparator=ExecutionMethod.MARKET,
            complete_pairs=value["complete_pairs"],
            mean_difference_points=value["mean"],
            median_difference_points=value["median"],
            method_better=value["wins"],
            comparator_better=value["losses"],
            ties=value["ties"],
            sign_test_pvalue=float(value["p"]),
            holm_adjusted_pvalue=float(adjusted[index]),
            status=(
                "descriptive_research_only"
                if value["complete_pairs"] >= config.minimum_complete_pairs
                else "underpowered_descriptive_only"
            ),
            cross_intent_independence_claimed=False,
            confirmatory_claim=False,
        )
        for index, value in enumerate(raw)
    )
    return ExecutionResearchSummary(
        config_id=config.config_id,
        study_ids=tuple(study.study_id for study in ordered_studies),
        method_summaries=tuple(summaries),
        paired_contrasts=contrasts,
    )


__all__ = [
    "EXECUTION_RESEARCH_MODEL_VERSION",
    "EXECUTION_RESEARCH_PROTOCOL_SHA256",
    "EXECUTION_RESEARCH_SCHEMA_VERSION",
    "ExecutionMethod",
    "ExecutionMethodOutcome",
    "ExecutionMethodSummary",
    "ExecutionResearchConfig",
    "ExecutionResearchError",
    "ExecutionResearchIntent",
    "ExecutionResearchSummary",
    "LIMIT_EXECUTION_METHODS",
    "MinuteExecutionInput",
    "ORDERED_EXECUTION_METHODS",
    "OrderResearchStatus",
    "PairedExecutionStudy",
    "PairedInferenceContrast",
    "canonical_sha256",
    "evaluate_execution_research_intent",
    "load_execution_research_config",
    "minute_execution_inputs_from_phase6_frame",
    "research_intent_from_trade_intent",
    "summarize_execution_research",
]
