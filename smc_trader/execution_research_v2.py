"""Preregistered Phase 8 v2 boundary contracts.

This module adds the missing *input* and *admission* contracts around the
existing :mod:`smc_trader.execution_research` v1.1 evaluator.  It deliberately
does not read empirical inputs, evaluate outcomes, select a winning method, or
write research artifacts.  Its responsibilities are limited to:

* integer-tick method-price provenance and explicit pre-outcome availability;
* one primary execution policy plus fixed one-factor-at-a-time sensitivities;
* one selected method/price per never-submit executable instruction;
* a config-bound, auditable shadow risk-admission protocol;
* content-addressed append-only intent/research-case JSONL records; and
* validation of an inert runner manifest without opening dataset bindings.

The existing Phase 8 v1.1 evaluator remains the outcome engine.  A future
runner may adapt the validated records into that evaluator only after its own
manifest is frozen and admitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .execution_fsm import EXECUTION_AUTHORITY, account_state_fingerprint
from .execution_research import (
    ORDERED_EXECUTION_METHODS,
    ExecutionMethod,
    ExecutionResearchError,
)
from .model import AccountState, Direction, aware_timestamp
from .trade_intent import EntryMethod, TradeIntent


EXECUTION_RESEARCH_V2_SCHEMA_VERSION = "phase8_execution_research_v2.0"
METHOD_PRICE_PROVENANCE_SCHEMA_VERSION = "phase8_method_price_provenance_v1"
METHOD_PRICE_SET_SCHEMA_VERSION = "phase8_method_price_set_v1"
EXECUTABLE_INSTRUCTION_SCHEMA_VERSION = "phase8_executable_instruction_v1"
RISK_ADMISSION_SCHEMA_VERSION = "phase8_risk_admission_v1"
PHASE8_LEDGER_SCHEMA_VERSION = "phase8_intent_research_case_ledger_v1"
PHASE8_RUNNER_MANIFEST_SCHEMA_VERSION = "phase8_runner_manifest_v2"

# Exact-byte identities of the preregistered files.  They are intentionally
# updated only together with their configs and focused contract tests.
EXECUTION_RESEARCH_V2_CONFIG_SHA256 = (
    "9c3b86f5ed4cbbcfad56c1af537be2878a078679197f12eb30e66b5eea4214e5"
)
RISK_ADMISSION_PROTOCOL_SHA256 = (
    "9eae13b583d1918c9b1f12e787b5ecf158d5ed4808cdfe89c97e1b5c504947b2"
)


class Phase8ContractError(ExecutionResearchError):
    """Raised when a Phase 8 v2 boundary is non-causal or identity-unsafe."""


class MethodAvailability(str, Enum):
    AVAILABLE = "available_preoutcome"
    NOT_APPLICABLE = "not_applicable_preoutcome"
    CENSORED = "unavailable_censored_preoutcome"


class VariantDimension(str, Enum):
    PRIMARY = "primary"
    WAIT = "wait"
    CANCEL = "cancel"
    STOP = "stop"
    TARGET = "target"


class LedgerRecordKind(str, Enum):
    INTENT = "trade_intent"
    RESEARCH_CASE = "execution_research_case"


_EXECUTION_LOADER_SEAL = object()
_INSTRUCTION_BUILDER_SEAL = object()
_RISK_EVALUATION_SEAL = object()


def _timestamp(value: Any, *, name: str) -> pd.Timestamp:
    try:
        result = aware_timestamp(pd.Timestamp(value), name=name)
    except (TypeError, ValueError) as exc:
        raise Phase8ContractError(f"{name} must be timezone aware") from exc
    return result.tz_convert("UTC")


def _identity(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in ("\n", "\r", "\x00"))
    ):
        raise Phase8ContractError(f"{name} must be canonical non-empty text")
    return value


def _sha256(value: Any, *, name: str) -> str:
    result = _identity(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise Phase8ContractError(f"{name} must be a lowercase SHA-256")
    return result


def _finite(value: Any, *, name: str, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise Phase8ContractError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _ids(
    values: Iterable[str],
    *,
    name: str,
    allow_empty: bool = False,
    sort: bool = False,
) -> tuple[str, ...]:
    result = tuple(values)
    if (
        (not allow_empty and not result)
        or len(result) != len(set(result))
        or any(not isinstance(value, str) or not value for value in result)
    ):
        raise Phase8ContractError(f"{name} must contain unique non-empty identities")
    return tuple(sorted(result)) if sort else result


def _normal(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if is_dataclass(value):
        return {
            item.name: _normal(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {
            str(key): _normal(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normal(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise Phase8ContractError("canonical payload cannot contain non-finite floats")
    if hasattr(value, "item"):
        return _normal(value.item())
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _normal(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise Phase8ContractError("value is not canonical-JSON serializable") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Phase8ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_loads_exact(raw: bytes | str, *, name: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except Phase8ContractError:
        raise
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase8ContractError(f"{name} is not valid duplicate-free JSON") from exc


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise Phase8ContractError(f"{name} keys are not exact; missing={missing}, extra={extra}")


def _price_to_ticks(price: Any, tick_size: Any, *, name: str) -> int:
    numeric_price = _finite(price, name=name, positive=True)
    numeric_tick = _finite(tick_size, name=f"{name} tick_size", positive=True)
    ratio = numeric_price / numeric_tick
    rounded = int(round(ratio))
    if not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise Phase8ContractError(f"{name} is off tick")
    return rounded


@dataclass(frozen=True)
class MethodPriceProvenance:
    """One method's causal price fact or explicit pre-outcome unavailability."""

    method: ExecutionMethod
    availability: MethodAvailability
    price_ticks: int | None
    tick_size: float
    source_semantic_type: str
    source_object_id: str | None
    source_generation_id: str | None
    source_event_ids: tuple[str, ...]
    source_known_at: pd.Timestamp | None
    snapshot_asof: pd.Timestamp
    derivation_id: str
    derivation_input_ids: tuple[str, ...]
    rounding_rule: str
    source_protocol_sha256: str
    availability_reason: str
    schema_version: str = METHOD_PRICE_PROVENANCE_SCHEMA_VERSION
    provenance_id: str = field(init=False)

    def __post_init__(self) -> None:
        method = ExecutionMethod(self.method)
        availability = MethodAvailability(self.availability)
        tick_size = _finite(self.tick_size, name="method price tick_size", positive=True)
        snapshot = _timestamp(self.snapshot_asof, name="method price snapshot_asof")
        events = _ids(
            self.source_event_ids,
            name="method price source_event_ids",
            allow_empty=availability is MethodAvailability.NOT_APPLICABLE,
        )
        inputs = _ids(
            self.derivation_input_ids,
            name="method price derivation_input_ids",
            allow_empty=availability is MethodAvailability.NOT_APPLICABLE,
        )
        _identity(self.source_semantic_type, name="method price source_semantic_type")
        _identity(self.derivation_id, name="method price derivation_id")
        _identity(self.rounding_rule, name="method price rounding_rule")
        _identity(self.availability_reason, name="method price availability_reason")
        _sha256(self.source_protocol_sha256, name="method price source_protocol_sha256")
        if self.schema_version != METHOD_PRICE_PROVENANCE_SCHEMA_VERSION:
            raise Phase8ContractError("method price schema version is invalid")

        known: pd.Timestamp | None = None
        if availability in {
            MethodAvailability.AVAILABLE,
            MethodAvailability.CENSORED,
        }:
            if availability is MethodAvailability.AVAILABLE and (
                type(self.price_ticks) is not int or self.price_ticks <= 0
            ):
                raise Phase8ContractError(
                    "available method price must be positive integer ticks"
                )
            if availability is MethodAvailability.CENSORED and self.price_ticks is not None:
                raise Phase8ContractError("censored method price cannot carry a latent price")
            if self.source_object_id is None or self.source_generation_id is None:
                raise Phase8ContractError(
                    "available or censored method price requires object and generation lineage"
                )
            _identity(self.source_object_id, name="method price source_object_id")
            _identity(self.source_generation_id, name="method price source_generation_id")
            if self.source_known_at is None:
                raise Phase8ContractError(
                    "available or censored method price requires source_known_at"
                )
            known = _timestamp(self.source_known_at, name="method price source_known_at")
            if known > snapshot:
                raise Phase8ContractError("method price cannot use future-known lineage")
        else:
            if (
                self.price_ticks is not None
                or self.source_object_id is not None
                or self.source_generation_id is not None
                or events
                or self.source_known_at is not None
                or inputs
            ):
                raise Phase8ContractError(
                    "unavailable method price cannot carry a latent price or source fact"
                )

        object.__setattr__(self, "method", method)
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "tick_size", tick_size)
        object.__setattr__(self, "snapshot_asof", snapshot)
        object.__setattr__(self, "source_known_at", known)
        object.__setattr__(self, "source_event_ids", events)
        object.__setattr__(self, "derivation_input_ids", inputs)
        payload = self.to_payload(include_identity=False)
        object.__setattr__(self, "provenance_id", f"method-price:{_digest(payload)[:32]}")

    @property
    def price(self) -> float | None:
        return None if self.price_ticks is None else self.price_ticks * self.tick_size

    def to_payload(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "method": self.method.value,
            "availability": self.availability.value,
            "price_ticks": self.price_ticks,
            "tick_size": self.tick_size,
            "source_semantic_type": self.source_semantic_type,
            "source_object_id": self.source_object_id,
            "source_generation_id": self.source_generation_id,
            "source_event_ids": list(self.source_event_ids),
            "source_known_at": (
                None if self.source_known_at is None else self.source_known_at.isoformat()
            ),
            "snapshot_asof": self.snapshot_asof.isoformat(),
            "derivation_id": self.derivation_id,
            "derivation_input_ids": list(self.derivation_input_ids),
            "rounding_rule": self.rounding_rule,
            "source_protocol_sha256": self.source_protocol_sha256,
            "availability_reason": self.availability_reason,
        }
        if include_identity:
            payload["provenance_id"] = self.provenance_id
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MethodPriceProvenance":
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "method",
                "availability",
                "price_ticks",
                "tick_size",
                "source_semantic_type",
                "source_object_id",
                "source_generation_id",
                "source_event_ids",
                "source_known_at",
                "snapshot_asof",
                "derivation_id",
                "derivation_input_ids",
                "rounding_rule",
                "source_protocol_sha256",
                "availability_reason",
                "provenance_id",
            },
            name="method price payload",
        )
        result = cls(
            schema_version=payload["schema_version"],
            method=ExecutionMethod(payload["method"]),
            availability=MethodAvailability(payload["availability"]),
            price_ticks=payload["price_ticks"],
            tick_size=payload["tick_size"],
            source_semantic_type=payload["source_semantic_type"],
            source_object_id=payload["source_object_id"],
            source_generation_id=payload["source_generation_id"],
            source_event_ids=tuple(payload["source_event_ids"]),
            source_known_at=payload["source_known_at"],
            snapshot_asof=payload["snapshot_asof"],
            derivation_id=payload["derivation_id"],
            derivation_input_ids=tuple(payload["derivation_input_ids"]),
            rounding_rule=payload["rounding_rule"],
            source_protocol_sha256=payload["source_protocol_sha256"],
            availability_reason=payload["availability_reason"],
        )
        if result.provenance_id != payload["provenance_id"]:
            raise Phase8ContractError("method price provenance identity conflicts with content")
        return result


@dataclass(frozen=True)
class MethodPriceSet:
    """All methods at one clock, including explicit unavailable methods."""

    source_trade_intent_id: str
    snapshot_asof: pd.Timestamp
    tick_size: float
    methods: tuple[MethodPriceProvenance, ...]
    schema_version: str = METHOD_PRICE_SET_SCHEMA_VERSION
    method_price_set_id: str = field(init=False)

    def __post_init__(self) -> None:
        _identity(self.source_trade_intent_id, name="method price set source intent")
        snapshot = _timestamp(self.snapshot_asof, name="method price set snapshot_asof")
        tick_size = _finite(self.tick_size, name="method price set tick_size", positive=True)
        methods = tuple(self.methods)
        if (
            self.schema_version != METHOD_PRICE_SET_SCHEMA_VERSION
            or tuple(item.method for item in methods) != ORDERED_EXECUTION_METHODS
            or any(not isinstance(item, MethodPriceProvenance) for item in methods)
            or any(item.snapshot_asof != snapshot for item in methods)
            or any(not math.isclose(item.tick_size, tick_size, rel_tol=0.0, abs_tol=1e-12) for item in methods)
            or not any(item.availability is MethodAvailability.AVAILABLE for item in methods)
        ):
            raise Phase8ContractError(
                "method price set must contain every method once, in preregistered order, at one clock"
            )
        object.__setattr__(self, "snapshot_asof", snapshot)
        object.__setattr__(self, "tick_size", tick_size)
        object.__setattr__(self, "methods", methods)
        object.__setattr__(
            self,
            "method_price_set_id",
            f"method-price-set:{_digest(self.to_payload(include_identity=False))[:32]}",
        )

    def for_method(self, method: ExecutionMethod) -> MethodPriceProvenance:
        typed = ExecutionMethod(method)
        return self.methods[ORDERED_EXECUTION_METHODS.index(typed)]

    def to_payload(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "source_trade_intent_id": self.source_trade_intent_id,
            "snapshot_asof": self.snapshot_asof.isoformat(),
            "tick_size": self.tick_size,
            "methods": [item.to_payload() for item in self.methods],
        }
        if include_identity:
            payload["method_price_set_id"] = self.method_price_set_id
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MethodPriceSet":
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "source_trade_intent_id",
                "snapshot_asof",
                "tick_size",
                "methods",
                "method_price_set_id",
            },
            name="method price set payload",
        )
        result = cls(
            schema_version=payload["schema_version"],
            source_trade_intent_id=payload["source_trade_intent_id"],
            snapshot_asof=payload["snapshot_asof"],
            tick_size=payload["tick_size"],
            methods=tuple(MethodPriceProvenance.from_payload(item) for item in payload["methods"]),
        )
        if result.method_price_set_id != payload["method_price_set_id"]:
            raise Phase8ContractError("method price set identity conflicts with content")
        return result


@dataclass(frozen=True)
class ExecutionVariant:
    """One fixed primary or OFAT policy; it contains no outcome fields."""

    variant_id: str
    changed_dimension: VariantDimension
    wait_policy_id: str
    wait_limit_completed_m1_bars: int | None
    cancel_policy_id: str
    stop_policy_id: str
    target_policy_id: str
    executable_instruction_eligible: bool
    inferential_role: str
    outcome_tuning_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_dimension", VariantDimension(self.changed_dimension))
        for name in (
            "variant_id",
            "wait_policy_id",
            "cancel_policy_id",
            "stop_policy_id",
            "target_policy_id",
            "inferential_role",
        ):
            _identity(getattr(self, name), name=f"variant {name}")
        if (
            self.wait_limit_completed_m1_bars is not None
            and (type(self.wait_limit_completed_m1_bars) is not int or self.wait_limit_completed_m1_bars <= 0)
        ):
            raise Phase8ContractError("variant wait limit must be positive completed M1 bars")
        if type(self.executable_instruction_eligible) is not bool:
            raise Phase8ContractError("variant executable eligibility must be boolean")
        if type(self.outcome_tuning_allowed) is not bool or self.outcome_tuning_allowed:
            raise Phase8ContractError("outcome-driven variant tuning is forbidden")

    def to_payload(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "changed_dimension": self.changed_dimension.value,
            "wait_policy_id": self.wait_policy_id,
            "wait_limit_completed_m1_bars": self.wait_limit_completed_m1_bars,
            "cancel_policy_id": self.cancel_policy_id,
            "stop_policy_id": self.stop_policy_id,
            "target_policy_id": self.target_policy_id,
            "executable_instruction_eligible": self.executable_instruction_eligible,
            "inferential_role": self.inferential_role,
            "outcome_tuning_allowed": self.outcome_tuning_allowed,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExecutionVariant":
        _require_exact_keys(
            payload,
            {
                "variant_id",
                "changed_dimension",
                "wait_policy_id",
                "wait_limit_completed_m1_bars",
                "cancel_policy_id",
                "stop_policy_id",
                "target_policy_id",
                "executable_instruction_eligible",
                "inferential_role",
                "outcome_tuning_allowed",
            },
            name="execution variant",
        )
        return cls(**payload)


_EXPECTED_VARIANT_PAYLOADS: tuple[dict[str, Any], ...] = (
    {
        "variant_id": "primary_v1",
        "changed_dimension": "primary",
        "wait_policy_id": "intent_native_expiry",
        "wait_limit_completed_m1_bars": None,
        "cancel_policy_id": "all_frozen_intent_conditions",
        "stop_policy_id": "intent_structural_invalidation",
        "target_policy_id": "primary_dol",
        "executable_instruction_eligible": True,
        "inferential_role": "primary",
        "outcome_tuning_allowed": False,
    },
    {
        "variant_id": "wait_1m_v1",
        "changed_dimension": "wait",
        "wait_policy_id": "one_completed_m1_bar",
        "wait_limit_completed_m1_bars": 1,
        "cancel_policy_id": "all_frozen_intent_conditions",
        "stop_policy_id": "intent_structural_invalidation",
        "target_policy_id": "primary_dol",
        "executable_instruction_eligible": False,
        "inferential_role": "ofat_sensitivity",
        "outcome_tuning_allowed": False,
    },
    {
        "variant_id": "wait_5m_v1",
        "changed_dimension": "wait",
        "wait_policy_id": "five_completed_m1_bars",
        "wait_limit_completed_m1_bars": 5,
        "cancel_policy_id": "all_frozen_intent_conditions",
        "stop_policy_id": "intent_structural_invalidation",
        "target_policy_id": "primary_dol",
        "executable_instruction_eligible": False,
        "inferential_role": "ofat_sensitivity",
        "outcome_tuning_allowed": False,
    },
    {
        "variant_id": "cancel_gtt_only_v1",
        "changed_dimension": "cancel",
        "wait_policy_id": "intent_native_expiry",
        "wait_limit_completed_m1_bars": None,
        "cancel_policy_id": "gtt_only",
        "stop_policy_id": "intent_structural_invalidation",
        "target_policy_id": "primary_dol",
        "executable_instruction_eligible": False,
        "inferential_role": "ofat_sensitivity",
        "outcome_tuning_allowed": False,
    },
    {
        "variant_id": "cancel_price_terminal_only_v1",
        "changed_dimension": "cancel",
        "wait_policy_id": "intent_native_expiry",
        "wait_limit_completed_m1_bars": None,
        "cancel_policy_id": "price_terminal_only",
        "stop_policy_id": "intent_structural_invalidation",
        "target_policy_id": "primary_dol",
        "executable_instruction_eligible": False,
        "inferential_role": "ofat_sensitivity",
        "outcome_tuning_allowed": False,
    },
    {
        "variant_id": "stop_entry_zone_failure_v1",
        "changed_dimension": "stop",
        "wait_policy_id": "intent_native_expiry",
        "wait_limit_completed_m1_bars": None,
        "cancel_policy_id": "all_frozen_intent_conditions",
        "stop_policy_id": "entry_zone_failure_boundary",
        "target_policy_id": "primary_dol",
        "executable_instruction_eligible": False,
        "inferential_role": "ofat_sensitivity",
        "outcome_tuning_allowed": False,
    },
    {
        "variant_id": "target_1r_capped_dol_v1",
        "changed_dimension": "target",
        "wait_policy_id": "intent_native_expiry",
        "wait_limit_completed_m1_bars": None,
        "cancel_policy_id": "all_frozen_intent_conditions",
        "stop_policy_id": "intent_structural_invalidation",
        "target_policy_id": "one_r_capped_by_primary_dol",
        "executable_instruction_eligible": False,
        "inferential_role": "ofat_sensitivity",
        "outcome_tuning_allowed": False,
    },
)


def _validate_ofat_variants(variants: Sequence[ExecutionVariant]) -> None:
    if [item.to_payload() for item in variants] != list(_EXPECTED_VARIANT_PAYLOADS):
        raise Phase8ContractError("execution variants differ from the preregistered primary/OFAT registry")
    primary = variants[0]
    policy_fields = (
        "wait_policy_id",
        "wait_limit_completed_m1_bars",
        "cancel_policy_id",
        "stop_policy_id",
        "target_policy_id",
    )
    dimensions = {
        VariantDimension.WAIT: {"wait_policy_id", "wait_limit_completed_m1_bars"},
        VariantDimension.CANCEL: {"cancel_policy_id"},
        VariantDimension.STOP: {"stop_policy_id"},
        VariantDimension.TARGET: {"target_policy_id"},
    }
    for variant in variants[1:]:
        changed = {
            name for name in policy_fields if getattr(variant, name) != getattr(primary, name)
        }
        if changed != dimensions[variant.changed_dimension]:
            raise Phase8ContractError("sensitivity variant is not one-factor-at-a-time")


_METHOD_RULES = {
    "market": {
        "source_semantic_type": "causal_arrival_bbo",
        "derivation_id": "first_causal_valid_best_quote_ticks_v1",
        "rounding_rule": "reject_off_tick_no_rounding",
    },
    "fvg_50_limit": {
        "source_semantic_type": "fvg_generation",
        "derivation_id": "fvg_midpoint_ticks_v1",
        "rounding_rule": "nearest_tick_half_even_v1",
    },
    "ob_50_limit": {
        "source_semantic_type": "qualified_order_block_generation",
        "derivation_id": "qualified_ob_midpoint_ticks_v1",
        "rounding_rule": "nearest_tick_half_even_v1",
    },
    "reclaim_limit": {
        "source_semantic_type": "liquidity_interaction_generation",
        "derivation_id": "first_outside_close_reclaim_ticks_v1",
        "rounding_rule": "reject_off_tick_no_rounding",
    },
    "breakout_limit": {
        "source_semantic_type": "structure_transition_generation",
        "derivation_id": "first_confirming_outside_close_ticks_v1",
        "rounding_rule": "reject_off_tick_no_rounding",
    },
    "aggressive_limit": {
        "source_semantic_type": "causal_arrival_bbo",
        "derivation_id": "near_quote_one_tick_improvement_v1",
        "rounding_rule": "exact_integer_tick_arithmetic",
    },
    "passive_limit": {
        "source_semantic_type": "causal_arrival_bbo",
        "derivation_id": "far_quote_one_tick_improvement_v1",
        "rounding_rule": "exact_integer_tick_arithmetic",
    },
}


@dataclass(frozen=True)
class Phase8ExecutionResearchProtocol:
    """Exact preregistration loaded from the frozen Phase 8 v2 config."""

    schema_version: str
    protocol_version: str
    status: str
    authority: str
    ordered_methods: tuple[ExecutionMethod, ...]
    variants: tuple[ExecutionVariant, ...]
    canonical_config_sha256: str
    source_file_sha256: str
    protocol_id: str
    _loader_seal: object = field(init=False, default=None, repr=False, compare=False)

    def variant(self, variant_id: str) -> ExecutionVariant:
        matches = tuple(item for item in self.variants if item.variant_id == variant_id)
        if len(matches) != 1:
            raise Phase8ContractError("unknown execution variant")
        return matches[0]

    @property
    def variant_registry_id(self) -> str:
        return f"phase8-variant-registry:{_digest([item.to_payload() for item in self.variants])[:32]}"

    def validate_method_price_set(self, method_prices: MethodPriceSet) -> None:
        if not isinstance(method_prices, MethodPriceSet):
            raise TypeError("method price validation requires MethodPriceSet")
        for item in method_prices.methods:
            rule = _METHOD_RULES[item.method.value]
            if (
                item.source_semantic_type != rule["source_semantic_type"]
                or item.derivation_id != rule["derivation_id"]
                or (
                    item.rounding_rule
                    != (
                        "not_applicable"
                        if item.availability is MethodAvailability.NOT_APPLICABLE
                        else rule["rounding_rule"]
                    )
                )
            ):
                raise Phase8ContractError("method price derivation differs from preregistration")
        if method_prices.for_method(ExecutionMethod.MARKET).availability is not MethodAvailability.AVAILABLE:
            raise Phase8ContractError("paired Phase 8 research requires an available market comparator")


def _expected_execution_config_payload() -> dict[str, Any]:
    return {
        "schema_version": EXECUTION_RESEARCH_V2_SCHEMA_VERSION,
        "protocol_version": "method_provenance_ofat_runner_contract_v1",
        "status": "preregistered_infrastructure_only_not_authorized_to_run",
        "authority": "research_only_never_submit",
        "ordered_methods": [method.value for method in ORDERED_EXECUTION_METHODS],
        "method_price_contract": {
            "unit": "positive_integer_ticks",
            "availability": "explicit_per_method_preoutcome",
            "required_lineage": [
                "source_object_id",
                "source_generation_id",
                "source_event_ids",
                "source_known_at",
                "derivation_id",
                "derivation_input_ids",
                "source_protocol_sha256",
            ],
            "known_at_rule": "source_known_at_lte_snapshot_asof",
            "unavailable_price_imputation_allowed": False,
            "bare_float_method_prices_allowed": False,
            "method_rules": _METHOD_RULES,
        },
        "instruction_selection": {
            "rule": "first_available_method_in_frozen_trade_intent_preference_order",
            "known_at": "source_trade_intent.created_at",
            "derivation_id": "source_trade_intent_first_available_preference_v1",
            "post_outcome_field_access": False,
            "missing_disposition": "instruction_not_formed",
            "entry_method_mapping": {
                "market_entry": "market",
                "fvg_50_limit": "fvg_50_limit",
                "ob_50_limit": "ob_50_limit",
                "reclaim_entry": "reclaim_limit",
            },
        },
        "policy_definitions": {
            "wait": {
                "intent_native_expiry": {
                    "deadline": "source_trade_intent.expires_at",
                    "real_completed_m1_bars": None,
                    "synthetic_or_gap_bars_count": False,
                },
                "one_completed_m1_bar": {
                    "deadline": "min(source_trade_intent.expires_at,known_at_of_first_subsequent_real_completed_m1_bar)",
                    "real_completed_m1_bars": 1,
                    "synthetic_or_gap_bars_count": False,
                },
                "five_completed_m1_bars": {
                    "deadline": "min(source_trade_intent.expires_at,known_at_of_fifth_subsequent_real_completed_m1_bar)",
                    "real_completed_m1_bars": 5,
                    "synthetic_or_gap_bars_count": False,
                },
            },
            "cancel": {
                "all_frozen_intent_conditions": {
                    "sources": "exact_source_trade_intent.cancel_conditions_plus_gtt",
                    "evaluation_clock": "causal_known_at",
                    "same_clock_precedence": "stop_then_target_then_cancel_then_fill",
                },
                "gtt_only": {
                    "sources": "source_trade_intent.expires_at_only",
                    "evaluation_clock": "causal_known_at",
                    "same_clock_precedence": "gtt_before_new_fill_at_or_after_deadline",
                },
                "price_terminal_only": {
                    "sources": "primary_stop_or_primary_target_touch_on_real_completed_m1",
                    "evaluation_clock": "completed_bar_known_at",
                    "same_clock_precedence": "adverse_stop_before_target_before_new_fill",
                },
            },
            "stop": {
                "intent_structural_invalidation": {
                    "price": "source_trade_intent.invalidation.price_ticks",
                    "known_at_rule": "invalidation.observed_at_lte_intent.created_at",
                    "missing_disposition": "case_rejected",
                },
                "entry_zone_failure_boundary": {
                    "price": "long_lower_or_short_upper_boundary_of_selected_method_source_generation",
                    "known_at_rule": "boundary_known_at_lte_intent.created_at",
                    "missing_disposition": "variant_censored_not_imputed",
                },
            },
            "target": {
                "primary_dol": {
                    "price": "source_trade_intent.targets[0].price_ticks",
                    "known_at_rule": "target.confirmed_at_lte_intent.created_at",
                    "missing_disposition": "case_rejected",
                },
                "one_r_capped_by_primary_dol": {
                    "risk_unit": "abs(selected_entry_ticks-primary_stop_ticks)",
                    "long_price": "min(selected_entry_ticks+risk_unit,primary_dol_ticks)",
                    "short_price": "max(selected_entry_ticks-risk_unit,primary_dol_ticks)",
                    "missing_or_nonpositive_disposition": "variant_censored_not_imputed",
                },
            },
        },
        "variants": list(_EXPECTED_VARIANT_PAYLOADS),
        "comparison": {
            "primary_variant_id": "primary_v1",
            "sensitivity_design": "fixed_one_factor_at_a_time",
            "same_intent_pairing_only": True,
            "result_driven_variant_selection_allowed": False,
            "post_outcome_method_availability_allowed": False,
            "primary_metric": "implementation_shortfall_points",
            "market_comparator": "market",
        },
        "runner_contract": {
            "validate_only_until_manifest_frozen": True,
            "sealed_input_open_during_validation": False,
            "artifact_write_during_validation": False,
            "v1_1_outcome_engine_compatibility": True,
        },
    }


def load_execution_research_v2_config(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> Phase8ExecutionResearchProtocol:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise Phase8ContractError("Phase 8 v2 config is not a regular file")
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXECUTION_RESEARCH_V2_CONFIG_SHA256:
        raise Phase8ContractError("Phase 8 v2 config bytes differ from preregistration")
    if expected_sha256 is not None and actual != _sha256(expected_sha256, name="expected Phase 8 config SHA-256"):
        raise Phase8ContractError("Phase 8 v2 config SHA-256 mismatch")
    payload = _json_loads_exact(raw, name="Phase 8 v2 config")
    if not isinstance(payload, Mapping) or payload != _expected_execution_config_payload():
        raise Phase8ContractError("Phase 8 v2 config semantics differ from preregistration")
    variants = tuple(ExecutionVariant.from_payload(item) for item in payload["variants"])
    _validate_ofat_variants(variants)
    canonical = _digest(payload)
    result = Phase8ExecutionResearchProtocol(
        schema_version=payload["schema_version"],
        protocol_version=payload["protocol_version"],
        status=payload["status"],
        authority=payload["authority"],
        ordered_methods=tuple(ExecutionMethod(value) for value in payload["ordered_methods"]),
        variants=variants,
        canonical_config_sha256=canonical,
        source_file_sha256=actual,
        protocol_id=f"phase8-execution-v2:{canonical}",
    )
    object.__setattr__(result, "_loader_seal", _EXECUTION_LOADER_SEAL)
    return result


_ENTRY_METHOD_BINDING = {
    ExecutionMethod.MARKET: EntryMethod.MARKET_ENTRY,
    ExecutionMethod.FVG_50_LIMIT: EntryMethod.FVG_50_LIMIT,
    ExecutionMethod.OB_50_LIMIT: EntryMethod.OB_50_LIMIT,
    ExecutionMethod.RECLAIM_LIMIT: EntryMethod.RECLAIM_ENTRY,
}
_ENTRY_TO_EXECUTION_METHOD = {
    entry_method: execution_method
    for execution_method, entry_method in _ENTRY_METHOD_BINDING.items()
}


@dataclass(frozen=True)
class ExecutableTradeInstruction:
    """One pre-outcome selected method and price for one non-zero TradeIntent."""

    intent: TradeIntent
    method_price: MethodPriceProvenance
    method_price_set_id: str
    execution_protocol_id: str
    execution_protocol_sha256: str
    variant_id: str
    selection_event_id: str
    selection_known_at: pd.Timestamp
    selection_derivation_id: str
    stop_price_ticks: int
    target_price_ticks: int
    schema_version: str = EXECUTABLE_INSTRUCTION_SCHEMA_VERSION
    authority: str = EXECUTION_AUTHORITY
    submission_allowed: bool = False
    instruction_id: str = field(init=False)
    _builder_seal: object = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.intent, TradeIntent):
            raise TypeError("executable instruction requires TradeIntent")
        if not isinstance(self.method_price, MethodPriceProvenance):
            raise TypeError("executable instruction requires MethodPriceProvenance")
        known = _timestamp(self.selection_known_at, name="method selection known_at")
        for name in (
            "method_price_set_id",
            "execution_protocol_id",
            "variant_id",
            "selection_event_id",
            "selection_derivation_id",
        ):
            _identity(getattr(self, name), name=f"instruction {name}")
        _sha256(self.execution_protocol_sha256, name="instruction execution protocol SHA-256")
        expected_protocol_id = (
            f"phase8-execution-v2:{_digest(_expected_execution_config_payload())}"
        )
        expected_stop_ticks = _price_to_ticks(
            self.intent.invalidation.price,
            self.method_price.tick_size,
            name="intent invalidation",
        )
        expected_target_ticks = _price_to_ticks(
            self.intent.targets[0].price,
            self.method_price.tick_size,
            name="intent primary target",
        )
        if (
            self.schema_version != EXECUTABLE_INSTRUCTION_SCHEMA_VERSION
            or self.authority != EXECUTION_AUTHORITY
            or type(self.submission_allowed) is not bool
            or self.submission_allowed
            or self.intent.quantity <= 0
            or self.method_price.availability is not MethodAvailability.AVAILABLE
            or self.method_price.snapshot_asof != self.intent.created_at.tz_convert("UTC")
            or known != self.method_price.snapshot_asof
            or self.method_price.source_known_at is None
            or self.method_price.source_known_at > known
            or self.execution_protocol_id != expected_protocol_id
            or self.execution_protocol_sha256 != EXECUTION_RESEARCH_V2_CONFIG_SHA256
            or self.variant_id != "primary_v1"
            or self.selection_derivation_id
            != "source_trade_intent_first_available_preference_v1"
            or self.stop_price_ticks != expected_stop_ticks
            or self.target_price_ticks != expected_target_ticks
            or type(self.stop_price_ticks) is not int
            or type(self.target_price_ticks) is not int
            or min(self.stop_price_ticks, self.target_price_ticks) <= 0
        ):
            raise Phase8ContractError("executable instruction contract is invalid")
        bound_entry_method = _ENTRY_METHOD_BINDING.get(self.method_price.method)
        if bound_entry_method is None or bound_entry_method not in self.intent.entry_method_preferences:
            raise Phase8ContractError("selected method is not authorized by the source TradeIntent")
        entry_ticks = self.method_price.price_ticks
        assert entry_ticks is not None
        if self.intent.side is Direction.LONG:
            geometry_ok = self.stop_price_ticks < entry_ticks < self.target_price_ticks
        else:
            geometry_ok = self.target_price_ticks < entry_ticks < self.stop_price_ticks
        if not geometry_ok:
            raise Phase8ContractError("instruction stop/entry/target geometry is invalid")
        object.__setattr__(self, "selection_known_at", known)
        payload = {
            "schema_version": self.schema_version,
            "source_trade_intent_id": self.intent.intent_id,
            "method_price_provenance_id": self.method_price.provenance_id,
            "method_price_set_id": self.method_price_set_id,
            "execution_protocol_id": self.execution_protocol_id,
            "execution_protocol_sha256": self.execution_protocol_sha256,
            "variant_id": self.variant_id,
            "selection_event_id": self.selection_event_id,
            "selection_known_at": known.isoformat(),
            "selection_derivation_id": self.selection_derivation_id,
            "stop_price_ticks": self.stop_price_ticks,
            "target_price_ticks": self.target_price_ticks,
            "authority": self.authority,
            "submission_allowed": self.submission_allowed,
        }
        object.__setattr__(self, "instruction_id", f"executable-instruction:{_digest(payload)[:32]}")

    @property
    def entry_price(self) -> float:
        price = self.method_price.price
        assert price is not None
        return price

    @property
    def stop_price(self) -> float:
        return self.stop_price_ticks * self.method_price.tick_size

    @property
    def target_price(self) -> float:
        return self.target_price_ticks * self.method_price.tick_size


def build_executable_trade_instruction(
    intent: TradeIntent,
    method_prices: MethodPriceSet,
    *,
    protocol: Phase8ExecutionResearchProtocol,
    selection_event_id: str,
) -> ExecutableTradeInstruction:
    """Select the first causally available frozen preference; never use outcomes."""

    if not isinstance(intent, TradeIntent):
        raise TypeError("instruction builder requires TradeIntent")
    if not isinstance(protocol, Phase8ExecutionResearchProtocol):
        raise TypeError("instruction builder requires loaded Phase 8 v2 protocol")
    if protocol._loader_seal is not _EXECUTION_LOADER_SEAL:
        raise TypeError("instruction builder requires loader-authenticated Phase 8 protocol")
    if method_prices.source_trade_intent_id != intent.intent_id:
        raise Phase8ContractError("method price set does not bind the source TradeIntent")
    if method_prices.snapshot_asof != intent.created_at.tz_convert("UTC"):
        raise Phase8ContractError("method prices must be frozen at TradeIntent creation")
    if (
        intent.invalidation.observed_at.tz_convert("UTC") > method_prices.snapshot_asof
        or intent.targets[0].confirmed_at.tz_convert("UTC") > method_prices.snapshot_asof
    ):
        raise Phase8ContractError("instruction stop/target lineage was not known at intent creation")
    if not math.isclose(method_prices.tick_size, 0.25, rel_tol=0.0, abs_tol=1e-12):
        raise Phase8ContractError("method price tick size differs from the preregistered contract")
    protocol.validate_method_price_set(method_prices)
    primary = protocol.variant("primary_v1")
    if not primary.executable_instruction_eligible:
        raise Phase8ContractError("primary variant is not executable-instruction eligible")
    provenance: MethodPriceProvenance | None = None
    for preference in intent.entry_method_preferences:
        execution_method = _ENTRY_TO_EXECUTION_METHOD.get(preference)
        if execution_method is None:
            continue
        candidate = method_prices.for_method(execution_method)
        if candidate.availability is MethodAvailability.AVAILABLE:
            provenance = candidate
            break
    if provenance is None:
        raise Phase8ContractError(
            "no frozen TradeIntent entry preference is causally available"
        )
    tick_size = provenance.tick_size
    if not math.isclose(intent.point_value, 20.0, rel_tol=0.0, abs_tol=1e-12):
        # The frozen v1.1 protocol currently supports the NQ point value only.
        raise Phase8ContractError("source TradeIntent point value differs from the preregistered contract")
    stop_ticks = _price_to_ticks(intent.invalidation.price, tick_size, name="intent invalidation")
    target_ticks = _price_to_ticks(intent.targets[0].price, tick_size, name="intent primary target")
    result = ExecutableTradeInstruction(
        intent=intent,
        method_price=provenance,
        method_price_set_id=method_prices.method_price_set_id,
        execution_protocol_id=protocol.protocol_id,
        execution_protocol_sha256=protocol.source_file_sha256,
        variant_id=primary.variant_id,
        selection_event_id=selection_event_id,
        selection_known_at=method_prices.snapshot_asof,
        selection_derivation_id="source_trade_intent_first_available_preference_v1",
        stop_price_ticks=stop_ticks,
        target_price_ticks=target_ticks,
    )
    object.__setattr__(result, "_builder_seal", _INSTRUCTION_BUILDER_SEAL)
    return result


_RISK_LOADER_SEAL = object()


@dataclass(frozen=True, init=False)
class RiskAdmissionProtocol:
    """Loader-only exact risk protocol; no caller-supplied fingerprint field."""

    schema_version: str
    protocol_version: str
    status: str
    authority: str
    requires_flat_account: bool
    tick_size: float
    point_value: float
    maximum_quantity: int
    maximum_single_trade_risk_fraction: float
    maximum_total_open_risk_fraction: float
    source_file_sha256: str
    canonical_config_sha256: str
    protocol_id: str
    _loader_seal: object = field(repr=False, compare=False)

    @classmethod
    def _loaded(cls, payload: Mapping[str, Any], *, source_sha256: str) -> "RiskAdmissionProtocol":
        result = object.__new__(cls)
        limits = payload["limits"]
        values = {
            "schema_version": payload["schema_version"],
            "protocol_version": payload["protocol_version"],
            "status": payload["status"],
            "authority": payload["authority"],
            "requires_flat_account": payload["requires_flat_account"],
            "tick_size": float(limits["tick_size"]),
            "point_value": float(limits["point_value"]),
            "maximum_quantity": int(limits["maximum_quantity"]),
            "maximum_single_trade_risk_fraction": float(limits["maximum_single_trade_risk_fraction"]),
            "maximum_total_open_risk_fraction": float(limits["maximum_total_open_risk_fraction"]),
            "source_file_sha256": source_sha256,
            "canonical_config_sha256": _digest(payload),
            "protocol_id": f"phase8-risk-admission:{_digest(payload)}",
            "_loader_seal": _RISK_LOADER_SEAL,
        }
        for name, value in values.items():
            object.__setattr__(result, name, value)
        return result


def _expected_risk_config_payload() -> dict[str, Any]:
    return {
        "schema_version": RISK_ADMISSION_SCHEMA_VERSION,
        "protocol_version": "shadow_exact_instruction_risk_v1",
        "status": "preregistered_shadow_admission_enabled",
        "authority": "shadow_research_only",
        "requires_flat_account": True,
        "caller_supplied_fingerprint_allowed": False,
        "limits": {
            "tick_size": 0.25,
            "point_value": 20.0,
            "maximum_quantity": 100,
            "maximum_single_trade_risk_fraction": 0.01,
            "maximum_total_open_risk_fraction": 0.02,
        },
        "checks": [
            "loaded_exact_protocol_bytes",
            "one_available_preoutcome_method_price",
            "source_trade_intent_identity_conserved",
            "exact_account_quantity_and_point_value",
            "exact_intent_risk_budget_snapshot",
            "method_price_stop_risk_lte_frozen_intent_budget",
            "flat_account",
            "directional_stop_entry_target_geometry",
            "admission_clock_before_intent_expiry",
            "never_submit_authority",
        ],
    }


def load_risk_admission_protocol(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> RiskAdmissionProtocol:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise Phase8ContractError("risk admission config is not a regular file")
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != RISK_ADMISSION_PROTOCOL_SHA256:
        raise Phase8ContractError("risk admission config bytes differ from preregistration")
    if expected_sha256 is not None and actual != _sha256(expected_sha256, name="expected risk config SHA-256"):
        raise Phase8ContractError("risk admission config SHA-256 mismatch")
    payload = _json_loads_exact(raw, name="risk admission config")
    if not isinstance(payload, Mapping) or payload != _expected_risk_config_payload():
        raise Phase8ContractError("risk admission config semantics differ from preregistration")
    return RiskAdmissionProtocol._loaded(payload, source_sha256=actual)


@dataclass(frozen=True)
class RiskAdmissionDecision:
    assessed_at: pd.Timestamp
    instruction_id: str
    source_trade_intent_id: str
    account_snapshot_id: str
    account_snapshot_fingerprint: str
    risk_protocol_id: str
    risk_protocol_sha256: str
    position_risk_amount: float
    position_risk_fraction: float
    passed: bool
    reasons: tuple[str, ...]
    schema_version: str = RISK_ADMISSION_SCHEMA_VERSION
    authority: str = EXECUTION_AUTHORITY
    submission_allowed: bool = False
    decision_id: str = field(init=False)
    _evaluation_seal: object = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        assessed = _timestamp(self.assessed_at, name="risk admission assessed_at")
        for name in (
            "instruction_id",
            "source_trade_intent_id",
            "account_snapshot_id",
            "risk_protocol_id",
        ):
            _identity(getattr(self, name), name=f"risk admission {name}")
        _sha256(self.account_snapshot_fingerprint, name="account snapshot fingerprint")
        _sha256(self.risk_protocol_sha256, name="risk protocol SHA-256")
        reasons = _ids(self.reasons, name="risk admission reasons", allow_empty=self.passed)
        risk_amount = _finite(self.position_risk_amount, name="position risk amount")
        risk_fraction = _finite(self.position_risk_fraction, name="position risk fraction")
        expected_risk_protocol_id = (
            f"phase8-risk-admission:{_digest(_expected_risk_config_payload())}"
        )
        if (
            self.schema_version != RISK_ADMISSION_SCHEMA_VERSION
            or self.authority != EXECUTION_AUTHORITY
            or type(self.submission_allowed) is not bool
            or self.submission_allowed
            or type(self.passed) is not bool
            or risk_amount < 0.0
            or risk_fraction < 0.0
            or (self.passed and reasons)
            or self.risk_protocol_id != expected_risk_protocol_id
            or self.risk_protocol_sha256 != RISK_ADMISSION_PROTOCOL_SHA256
        ):
            raise Phase8ContractError("risk admission decision contract is invalid")
        object.__setattr__(self, "assessed_at", assessed)
        object.__setattr__(self, "reasons", reasons)
        payload = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "decision_id" and not item.name.startswith("_")
        }
        object.__setattr__(self, "decision_id", f"risk-admission:{_digest(payload)[:32]}")


@dataclass(frozen=True)
class RiskAdmittedExecutableTradeInstruction:
    instruction: ExecutableTradeInstruction
    admission: RiskAdmissionDecision
    authority: str = EXECUTION_AUTHORITY
    submission_allowed: bool = False
    admitted_instruction_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.instruction, ExecutableTradeInstruction)
            or not isinstance(self.admission, RiskAdmissionDecision)
            or not self.admission.passed
            or self.admission._evaluation_seal is not _RISK_EVALUATION_SEAL
            or self.admission.instruction_id != self.instruction.instruction_id
            or self.admission.source_trade_intent_id != self.instruction.intent.intent_id
            or self.authority != EXECUTION_AUTHORITY
            or type(self.submission_allowed) is not bool
            or self.submission_allowed
        ):
            raise Phase8ContractError("risk-admitted instruction does not conserve identity")
        identity = _digest(
            {
                "instruction_id": self.instruction.instruction_id,
                "admission_id": self.admission.decision_id,
            }
        )
        object.__setattr__(self, "admitted_instruction_id", f"risk-admitted-instruction:{identity[:32]}")


def evaluate_risk_admission(
    instruction: ExecutableTradeInstruction,
    account: AccountState,
    protocol: RiskAdmissionProtocol,
    *,
    assessed_at: pd.Timestamp,
) -> RiskAdmissionDecision:
    """Evaluate exact shadow risk without accepting any fingerprint argument."""

    if not isinstance(instruction, ExecutableTradeInstruction):
        raise TypeError("risk admission requires ExecutableTradeInstruction")
    if instruction._builder_seal is not _INSTRUCTION_BUILDER_SEAL:
        raise TypeError("risk admission requires a formally built executable instruction")
    if not isinstance(account, AccountState):
        raise TypeError("risk admission requires AccountState")
    if (
        not isinstance(protocol, RiskAdmissionProtocol)
        or getattr(protocol, "_loader_seal", None) is not _RISK_LOADER_SEAL
    ):
        raise TypeError("risk admission requires a loader-authenticated protocol")
    clock = _timestamp(assessed_at, name="risk admission clock")
    intent = instruction.intent
    price_risk = abs(instruction.entry_price - instruction.stop_price)
    amount = price_risk * float(intent.point_value) * int(intent.quantity)
    fraction = amount / float(account.equity)
    expected_budget = float(account.equity) * float(account.requested_risk_fraction)
    reasons: list[str] = []
    if protocol.requires_flat_account and (
        account.position is not None or account.open_risk_fraction > 1e-12
    ):
        reasons.append("account_not_flat")
    if account.quantity != intent.quantity:
        reasons.append("quantity_mismatch")
    if account.quantity > protocol.maximum_quantity:
        reasons.append("quantity_limit")
    if not math.isclose(account.point_value, intent.point_value, rel_tol=0.0, abs_tol=1e-12):
        reasons.append("account_point_value_mismatch")
    if not math.isclose(intent.point_value, protocol.point_value, rel_tol=0.0, abs_tol=1e-12):
        reasons.append("protocol_point_value_mismatch")
    if not math.isclose(instruction.method_price.tick_size, protocol.tick_size, rel_tol=0.0, abs_tol=1e-12):
        reasons.append("protocol_tick_size_mismatch")
    if not math.isclose(account.requested_risk_fraction, intent.risk_budget_fraction, rel_tol=0.0, abs_tol=1e-12):
        reasons.append("intent_risk_fraction_mismatch")
    if not math.isclose(expected_budget, intent.risk_budget_amount, rel_tol=1e-12, abs_tol=1e-9):
        reasons.append("intent_risk_budget_amount_mismatch")
    if amount > intent.risk_budget_amount + 1e-9:
        reasons.append("method_price_risk_exceeds_intent_budget")
    if fraction > protocol.maximum_single_trade_risk_fraction + 1e-12:
        reasons.append("single_trade_risk_limit")
    if account.open_risk_fraction + fraction > protocol.maximum_total_open_risk_fraction + 1e-12:
        reasons.append("total_open_risk_limit")
    if clock < intent.created_at.tz_convert("UTC") or clock >= intent.expires_at.tz_convert("UTC"):
        reasons.append("admission_clock_outside_intent")
    unique_reasons = tuple(dict.fromkeys(reasons))
    result = RiskAdmissionDecision(
        assessed_at=clock,
        instruction_id=instruction.instruction_id,
        source_trade_intent_id=intent.intent_id,
        account_snapshot_id=intent.account_snapshot_id,
        account_snapshot_fingerprint=account_state_fingerprint(account),
        risk_protocol_id=protocol.protocol_id,
        risk_protocol_sha256=protocol.source_file_sha256,
        position_risk_amount=amount,
        position_risk_fraction=fraction,
        passed=not unique_reasons,
        reasons=unique_reasons,
    )
    object.__setattr__(result, "_evaluation_seal", _RISK_EVALUATION_SEAL)
    return result


def risk_admit_executable_trade_instruction(
    instruction: ExecutableTradeInstruction,
    account: AccountState,
    protocol: RiskAdmissionProtocol,
    *,
    assessed_at: pd.Timestamp,
) -> RiskAdmittedExecutableTradeInstruction:
    decision = evaluate_risk_admission(
        instruction,
        account,
        protocol,
        assessed_at=assessed_at,
    )
    if not decision.passed:
        raise Phase8ContractError(f"risk admission rejected: {','.join(decision.reasons)}")
    return RiskAdmittedExecutableTradeInstruction(instruction=instruction, admission=decision)


@dataclass(frozen=True)
class IntentLedgerRecord:
    """Full canonical primitive snapshot of one non-zero TradeIntent."""

    source_trade_intent_id: str
    created_at: pd.Timestamp
    canonical_intent_json: str
    trade_intent_sha256: str
    schema_version: str = PHASE8_LEDGER_SCHEMA_VERSION
    record_kind: LedgerRecordKind = LedgerRecordKind.INTENT
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        _identity(self.source_trade_intent_id, name="intent ledger source_trade_intent_id")
        created = _timestamp(self.created_at, name="intent ledger created_at")
        _sha256(self.trade_intent_sha256, name="intent ledger trade_intent_sha256")
        if self.schema_version != PHASE8_LEDGER_SCHEMA_VERSION or LedgerRecordKind(self.record_kind) is not LedgerRecordKind.INTENT:
            raise Phase8ContractError("intent ledger schema/kind is invalid")
        decoded = _json_loads_exact(self.canonical_intent_json, name="canonical TradeIntent snapshot")
        if not isinstance(decoded, Mapping):
            raise Phase8ContractError("canonical TradeIntent snapshot must be an object")
        canonical = _canonical_json(decoded)
        if canonical != self.canonical_intent_json:
            raise Phase8ContractError("TradeIntent snapshot is not canonical JSON")
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != self.trade_intent_sha256:
            raise Phase8ContractError("TradeIntent snapshot digest mismatch")
        if decoded.get("intent_id") != self.source_trade_intent_id:
            raise Phase8ContractError("TradeIntent snapshot identity mismatch")
        if _timestamp(decoded.get("created_at"), name="snapshotted TradeIntent created_at") != created:
            raise Phase8ContractError("TradeIntent snapshot clock mismatch")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "record_kind", LedgerRecordKind.INTENT)
        key = {
            "record_kind": LedgerRecordKind.INTENT.value,
            "source_trade_intent_id": self.source_trade_intent_id,
            "trade_intent_sha256": self.trade_intent_sha256,
        }
        object.__setattr__(self, "record_id", f"phase8-intent-record:{_digest(key)[:32]}")

    @classmethod
    def from_trade_intent(cls, intent: TradeIntent) -> "IntentLedgerRecord":
        if not isinstance(intent, TradeIntent):
            raise TypeError("intent ledger requires TradeIntent")
        if intent.quantity <= 0:
            raise Phase8ContractError("zero-quantity TradeIntent cannot enter the Phase 8 ledger")
        canonical = _canonical_json(intent)
        return cls(
            source_trade_intent_id=intent.intent_id,
            created_at=intent.created_at,
            canonical_intent_json=canonical,
            trade_intent_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_kind": self.record_kind.value,
            "record_id": self.record_id,
            "source_trade_intent_id": self.source_trade_intent_id,
            "created_at": self.created_at.isoformat(),
            "canonical_intent_json": self.canonical_intent_json,
            "trade_intent_sha256": self.trade_intent_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "IntentLedgerRecord":
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "record_kind",
                "record_id",
                "source_trade_intent_id",
                "created_at",
                "canonical_intent_json",
                "trade_intent_sha256",
            },
            name="intent ledger record",
        )
        result = cls(
            schema_version=payload["schema_version"],
            record_kind=LedgerRecordKind(payload["record_kind"]),
            source_trade_intent_id=payload["source_trade_intent_id"],
            created_at=payload["created_at"],
            canonical_intent_json=payload["canonical_intent_json"],
            trade_intent_sha256=payload["trade_intent_sha256"],
        )
        if result.record_id != payload["record_id"]:
            raise Phase8ContractError("intent ledger record identity conflicts with content")
        return result


@dataclass(frozen=True)
class ResearchCaseLedgerRecord:
    """Outcome-free pairing case bound to one intent and full method registry."""

    source_intent_record_id: str
    source_trade_intent_id: str
    created_at: pd.Timestamp
    method_price_set: MethodPriceSet
    execution_protocol_id: str
    execution_protocol_sha256: str
    risk_protocol_id: str
    risk_protocol_sha256: str
    variant_registry_id: str
    source_artifact_ids: tuple[str, ...]
    schema_version: str = PHASE8_LEDGER_SCHEMA_VERSION
    record_kind: LedgerRecordKind = LedgerRecordKind.RESEARCH_CASE
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_intent_record_id",
            "source_trade_intent_id",
            "execution_protocol_id",
            "risk_protocol_id",
            "variant_registry_id",
        ):
            _identity(getattr(self, name), name=f"research case {name}")
        created = _timestamp(self.created_at, name="research case created_at")
        artifacts = _ids(self.source_artifact_ids, name="research case source_artifact_ids", sort=True)
        _sha256(self.execution_protocol_sha256, name="research case execution protocol SHA-256")
        _sha256(self.risk_protocol_sha256, name="research case risk protocol SHA-256")
        expected_execution_id = (
            f"phase8-execution-v2:{_digest(_expected_execution_config_payload())}"
        )
        expected_risk_id = (
            f"phase8-risk-admission:{_digest(_expected_risk_config_payload())}"
        )
        expected_variant_registry_id = (
            "phase8-variant-registry:"
            + _digest(_EXPECTED_VARIANT_PAYLOADS)[:32]
        )
        if (
            self.schema_version != PHASE8_LEDGER_SCHEMA_VERSION
            or LedgerRecordKind(self.record_kind) is not LedgerRecordKind.RESEARCH_CASE
            or not isinstance(self.method_price_set, MethodPriceSet)
            or self.method_price_set.source_trade_intent_id != self.source_trade_intent_id
            or self.method_price_set.snapshot_asof != created
            or self.execution_protocol_id != expected_execution_id
            or self.execution_protocol_sha256 != EXECUTION_RESEARCH_V2_CONFIG_SHA256
            or self.risk_protocol_id != expected_risk_id
            or self.risk_protocol_sha256 != RISK_ADMISSION_PROTOCOL_SHA256
            or self.variant_registry_id != expected_variant_registry_id
        ):
            raise Phase8ContractError("research case ledger contract is invalid")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "source_artifact_ids", artifacts)
        object.__setattr__(self, "record_kind", LedgerRecordKind.RESEARCH_CASE)
        key = self.to_payload(include_identity=False)
        object.__setattr__(self, "record_id", f"phase8-research-case:{_digest(key)[:32]}")

    def to_payload(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "record_kind": self.record_kind.value,
            "source_intent_record_id": self.source_intent_record_id,
            "source_trade_intent_id": self.source_trade_intent_id,
            "created_at": self.created_at.isoformat(),
            "method_price_set": self.method_price_set.to_payload(),
            "execution_protocol_id": self.execution_protocol_id,
            "execution_protocol_sha256": self.execution_protocol_sha256,
            "risk_protocol_id": self.risk_protocol_id,
            "risk_protocol_sha256": self.risk_protocol_sha256,
            "variant_registry_id": self.variant_registry_id,
            "source_artifact_ids": list(self.source_artifact_ids),
        }
        if include_identity:
            payload["record_id"] = self.record_id
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ResearchCaseLedgerRecord":
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "record_kind",
                "record_id",
                "source_intent_record_id",
                "source_trade_intent_id",
                "created_at",
                "method_price_set",
                "execution_protocol_id",
                "execution_protocol_sha256",
                "risk_protocol_id",
                "risk_protocol_sha256",
                "variant_registry_id",
                "source_artifact_ids",
            },
            name="research case ledger record",
        )
        result = cls(
            schema_version=payload["schema_version"],
            record_kind=LedgerRecordKind(payload["record_kind"]),
            source_intent_record_id=payload["source_intent_record_id"],
            source_trade_intent_id=payload["source_trade_intent_id"],
            created_at=payload["created_at"],
            method_price_set=MethodPriceSet.from_payload(payload["method_price_set"]),
            execution_protocol_id=payload["execution_protocol_id"],
            execution_protocol_sha256=payload["execution_protocol_sha256"],
            risk_protocol_id=payload["risk_protocol_id"],
            risk_protocol_sha256=payload["risk_protocol_sha256"],
            variant_registry_id=payload["variant_registry_id"],
            source_artifact_ids=tuple(payload["source_artifact_ids"]),
        )
        if result.record_id != payload["record_id"]:
            raise Phase8ContractError("research case identity conflicts with content")
        return result


def build_research_case_record(
    intent_record: IntentLedgerRecord,
    method_prices: MethodPriceSet,
    execution_protocol: Phase8ExecutionResearchProtocol,
    risk_protocol: RiskAdmissionProtocol,
    *,
    source_artifact_ids: Sequence[str],
) -> ResearchCaseLedgerRecord:
    if not isinstance(intent_record, IntentLedgerRecord):
        raise TypeError("research case requires IntentLedgerRecord")
    if (
        not isinstance(execution_protocol, Phase8ExecutionResearchProtocol)
        or execution_protocol._loader_seal is not _EXECUTION_LOADER_SEAL
    ):
        raise TypeError("research case requires loader-authenticated execution protocol")
    if (
        not isinstance(risk_protocol, RiskAdmissionProtocol)
        or getattr(risk_protocol, "_loader_seal", None) is not _RISK_LOADER_SEAL
    ):
        raise TypeError("research case requires loader-authenticated risk protocol")
    if method_prices.source_trade_intent_id != intent_record.source_trade_intent_id:
        raise Phase8ContractError("research case does not bind the intent ledger record")
    execution_protocol.validate_method_price_set(method_prices)
    return ResearchCaseLedgerRecord(
        source_intent_record_id=intent_record.record_id,
        source_trade_intent_id=intent_record.source_trade_intent_id,
        created_at=intent_record.created_at,
        method_price_set=method_prices,
        execution_protocol_id=execution_protocol.protocol_id,
        execution_protocol_sha256=execution_protocol.source_file_sha256,
        risk_protocol_id=risk_protocol.protocol_id,
        risk_protocol_sha256=risk_protocol.source_file_sha256,
        variant_registry_id=execution_protocol.variant_registry_id,
        source_artifact_ids=tuple(source_artifact_ids),
    )


Phase8LedgerRecord = IntentLedgerRecord | ResearchCaseLedgerRecord


class Phase8AppendOnlyLedger:
    """In-memory append-only contract with deterministic JSONL interchange."""

    def __init__(self, records: Iterable[Phase8LedgerRecord] = ()) -> None:
        self._records: list[Phase8LedgerRecord] = []
        self._by_record_id: dict[str, Phase8LedgerRecord] = {}
        self._logical_keys: dict[tuple[str, str], str] = {}
        for record in records:
            self.append(record)

    @property
    def records(self) -> tuple[Phase8LedgerRecord, ...]:
        return tuple(self._records)

    @staticmethod
    def _logical_key(record: Phase8LedgerRecord) -> tuple[str, str]:
        return (record.record_kind.value, record.source_trade_intent_id)

    def append(self, record: Phase8LedgerRecord) -> bool:
        if not isinstance(record, (IntentLedgerRecord, ResearchCaseLedgerRecord)):
            raise TypeError("Phase 8 ledger accepts only formal intent/research-case records")
        existing = self._by_record_id.get(record.record_id)
        if existing is not None:
            if existing != record:
                raise Phase8ContractError("ledger record identity collision")
            return False
        logical_key = self._logical_key(record)
        prior_id = self._logical_keys.get(logical_key)
        if prior_id is not None:
            raise Phase8ContractError("append would revise an existing logical ledger fact")
        if isinstance(record, ResearchCaseLedgerRecord):
            intent_record = self._by_record_id.get(record.source_intent_record_id)
            if not isinstance(intent_record, IntentLedgerRecord):
                raise Phase8ContractError("research case must follow its exact intent record")
        self._records.append(record)
        self._by_record_id[record.record_id] = record
        self._logical_keys[logical_key] = record.record_id
        return True

    def to_jsonl(self) -> str:
        if not self._records:
            return ""
        return "".join(_canonical_json(record.to_payload()) + "\n" for record in self._records)

    @classmethod
    def from_jsonl(cls, text: str) -> "Phase8AppendOnlyLedger":
        if not isinstance(text, str):
            raise TypeError("Phase 8 ledger JSONL must be text")
        records: list[Phase8LedgerRecord] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line:
                raise Phase8ContractError(f"blank JSONL record at line {line_number}")
            payload = _json_loads_exact(line, name=f"Phase 8 ledger line {line_number}")
            if not isinstance(payload, Mapping):
                raise Phase8ContractError("Phase 8 ledger record must be an object")
            kind = payload.get("record_kind")
            if kind == LedgerRecordKind.INTENT.value:
                records.append(IntentLedgerRecord.from_payload(payload))
            elif kind == LedgerRecordKind.RESEARCH_CASE.value:
                records.append(ResearchCaseLedgerRecord.from_payload(payload))
            else:
                raise Phase8ContractError("unknown Phase 8 ledger record kind")
        return cls(records)


@dataclass(frozen=True)
class Phase8RunnerValidation:
    manifest_sha256: str
    execution_protocol_sha256: str
    risk_protocol_sha256: str
    ready: bool
    validate_only: bool
    opened_dataset_bindings: tuple[str, ...]
    written_artifacts: tuple[str, ...]
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha256(self.manifest_sha256, name="runner manifest SHA-256")
        _sha256(self.execution_protocol_sha256, name="runner execution protocol SHA-256")
        _sha256(self.risk_protocol_sha256, name="runner risk protocol SHA-256")
        if (
            self.ready
            or not self.validate_only
            or self.opened_dataset_bindings
            or self.written_artifacts
            or not self.blockers
        ):
            raise Phase8ContractError("inert runner validation result is invalid")


_EXPECTED_RUNNER_BLOCKERS = (
    "manifest_not_frozen",
    "experiment_identity_missing",
    "fit_validation_cohorts_not_bound",
    "intent_research_case_ledger_not_bound",
    "minute_execution_input_ledger_not_bound",
    "ohlcv_artifact_not_bound",
    "phase6_mbo_artifact_not_bound",
    "instrument_mapping_registry_not_bound",
    "formal_runner_not_implemented",
    "outputs_not_registered",
)


def validate_phase8_runner_manifest(path: str | Path) -> Phase8RunnerValidation:
    """Validate only the inert template and configs; never open data bindings."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise Phase8ContractError("Phase 8 runner manifest is not a regular file")
    raw = source.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    payload = _json_loads_exact(raw, name="Phase 8 runner manifest")
    if not isinstance(payload, Mapping):
        raise Phase8ContractError("Phase 8 runner manifest root must be an object")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "status",
            "authority",
            "frozen_before_run",
            "experiment_id",
            "sealed_oos",
            "protocol_bindings",
            "dataset_bindings",
            "runner_contract",
            "readiness_blockers",
            "outputs",
        },
        name="Phase 8 runner manifest",
    )
    bindings = payload["protocol_bindings"]
    dataset_bindings = payload["dataset_bindings"]
    outputs = payload["outputs"]
    if not all(
        isinstance(value, Mapping)
        for value in (bindings, dataset_bindings, outputs)
    ):
        raise Phase8ContractError("runner manifest binding objects are invalid")
    if (
        payload["schema_version"] != PHASE8_RUNNER_MANIFEST_SCHEMA_VERSION
        or payload["status"] != "preregistered_infrastructure_only_not_authorized_to_run"
        or payload["frozen_before_run"] is not False
        or payload["experiment_id"] is not None
        or payload["authority"]
        != {
            "research_only": True,
            "order_submission": False,
            "execution_authorized": False,
            "sealed_oos_reveal_authorized": False,
        }
        or payload["sealed_oos"] != {"opened": False, "path": None, "sha256": None}
        or payload["runner_contract"]
        != {
            "mode": "validate_contract_only",
            "read_dataset_bindings": False,
            "write_artifacts": False,
            "v1_1_outcome_engine_compatibility": True,
        }
        or tuple(payload["readiness_blockers"]) != _EXPECTED_RUNNER_BLOCKERS
        or any(value is not None for value in dataset_bindings.values())
        or any(value is not None for value in outputs.values())
    ):
        raise Phase8ContractError("runner manifest is not the inert preregistration template")

    _require_exact_keys(
        bindings,
        {"execution_research_v2", "risk_admission_v1"},
        name="runner protocol bindings",
    )
    _require_exact_keys(
        dataset_bindings,
        {
            "fit_cohort",
            "validation_cohort",
            "intent_research_case_ledger",
            "minute_execution_input_ledger",
            "ohlcv_artifact",
            "phase6_mbo_artifact",
            "instrument_mapping_registry",
        },
        name="runner dataset bindings",
    )
    _require_exact_keys(
        outputs,
        {
            "intent_research_case_ledger",
            "per_method_outcomes",
            "paired_variant_results",
            "summary",
        },
        name="runner outputs",
    )
    repository_root = source.resolve().parents[2]
    execution_binding = bindings["execution_research_v2"]
    risk_binding = bindings["risk_admission_v1"]
    if not isinstance(execution_binding, Mapping) or not isinstance(
        risk_binding, Mapping
    ):
        raise Phase8ContractError("runner protocol binding entries are invalid")
    _require_exact_keys(
        execution_binding,
        {"path", "sha256"},
        name="runner execution protocol binding",
    )
    _require_exact_keys(
        risk_binding,
        {"path", "sha256"},
        name="runner risk protocol binding",
    )
    if execution_binding.get("path") != "configs/execution_research_v2.json":
        raise Phase8ContractError("runner execution protocol path is not preregistered")
    if risk_binding.get("path") != "configs/risk_admission_v1.json":
        raise Phase8ContractError("runner risk protocol path is not preregistered")
    execution_protocol = load_execution_research_v2_config(
        repository_root / execution_binding["path"],
        expected_sha256=execution_binding.get("sha256"),
    )
    risk_protocol = load_risk_admission_protocol(
        repository_root / risk_binding["path"],
        expected_sha256=risk_binding.get("sha256"),
    )
    return Phase8RunnerValidation(
        manifest_sha256=manifest_sha,
        execution_protocol_sha256=execution_protocol.source_file_sha256,
        risk_protocol_sha256=risk_protocol.source_file_sha256,
        ready=False,
        validate_only=True,
        opened_dataset_bindings=(),
        written_artifacts=(),
        blockers=_EXPECTED_RUNNER_BLOCKERS,
    )


__all__ = [
    "EXECUTABLE_INSTRUCTION_SCHEMA_VERSION",
    "EXECUTION_RESEARCH_V2_CONFIG_SHA256",
    "EXECUTION_RESEARCH_V2_SCHEMA_VERSION",
    "METHOD_PRICE_PROVENANCE_SCHEMA_VERSION",
    "METHOD_PRICE_SET_SCHEMA_VERSION",
    "PHASE8_LEDGER_SCHEMA_VERSION",
    "PHASE8_RUNNER_MANIFEST_SCHEMA_VERSION",
    "RISK_ADMISSION_PROTOCOL_SHA256",
    "RISK_ADMISSION_SCHEMA_VERSION",
    "ExecutableTradeInstruction",
    "ExecutionVariant",
    "IntentLedgerRecord",
    "LedgerRecordKind",
    "MethodAvailability",
    "MethodPriceProvenance",
    "MethodPriceSet",
    "Phase8AppendOnlyLedger",
    "Phase8ContractError",
    "Phase8ExecutionResearchProtocol",
    "Phase8RunnerValidation",
    "ResearchCaseLedgerRecord",
    "RiskAdmissionDecision",
    "RiskAdmissionProtocol",
    "RiskAdmittedExecutableTradeInstruction",
    "VariantDimension",
    "build_executable_trade_instruction",
    "build_research_case_record",
    "evaluate_risk_admission",
    "load_execution_research_v2_config",
    "load_risk_admission_protocol",
    "risk_admit_executable_trade_instruction",
    "validate_phase8_runner_manifest",
]
