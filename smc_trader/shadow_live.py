"""No-order Phase 9 shadow stream and exact replay parity primitives.

The runner intentionally reuses :class:`ContinuousSMCEngine` and the Phase 8
execution reducer.  It owns no broker adapter and accepts only a
``NullExecutionGateway``.  Every completed-clock input is retained before the
engine is advanced, then a compact record binds the causal input, Eye store,
MarketSnapshot, Brain projections, legacy Decision/Risk outputs, and all
registered execution FSMs.

This module is an engineering/parity surface.  It is not a live-trading mode,
does not authorize orders, and does not turn shadow probabilities into action.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from .engine import ContinuousSMCEngine
from .execution_fsm import (
    EXECUTION_PROTOCOL_FINGERPRINT,
    ExecutionEventEnvelope,
    ExecutionFSM,
    RiskApprovedTradeIntent,
    account_state_fingerprint,
)
from .model import (
    AccountState,
    Bar,
    EngineSnapshot,
    EventKind,
    EventOrigin,
    SMC_SEMANTIC_VERSION,
    to_primitive,
)
from .observation import ExecutionRealityInput
from .semantics import load_semantic_selection


SHADOW_LIVE_SCHEMA_VERSION = "phase9_shadow_live_v1.3"
SHADOW_LIVE_STATUS = "engineering_validation_only"
SHADOW_LIVE_AUTHORITY = "null_gateway_no_external_submission"
SHADOW_COMPONENT_DIGEST_VERSION = "phase9_shadow_component_digest_v3"
SHADOW_LEGACY_COMPONENT_DIGEST_VERSION = (
    "phase9_shadow_component_digest_legacy_v1_2"
)
SHADOW_RECORD_FIELDS = (
    "input_digest",
    "journal_prefix_fingerprint",
    "observation_fingerprint",
    "audit_event_store_fingerprint",
    "market_snapshot_fingerprint",
    "market_state_fingerprint",
    "relation_fingerprint",
    "session_fingerprint",
    "neutral_market_state_fingerprint",
    "belief_fingerprint",
    "path_state_fingerprint",
    "path_update_fingerprint",
    "dol_probability_fingerprint",
    "signal_assessment_fingerprint",
    "trade_intent_fingerprint",
    "decision_fingerprint",
    "risk_fingerprint",
    "engine_snapshot_fingerprint",
    "runtime_action_policy_fingerprint",
    "runtime_bindings_fingerprint",
    "execution_approval_fingerprint",
    "execution_event_fingerprint",
    "execution_state_fingerprint",
    "protocol_id",
    "external_submission_attempts",
)

SHADOW_RUNTIME_BINDING_KEYS = (
    "dol_probability_protocol_fingerprint",
    "dol_ranking_protocol_fingerprint",
    "execution_protocol_fingerprint",
    "model_config_sha256",
    "path_protocol_fingerprint",
    "semantic_version",
    "shadow_component_digest_version",
    "signal_policy_protocol_fingerprint",
)
_LEGACY_SHADOW_RUNTIME_BINDING_KEYS = tuple(
    key
    for key in SHADOW_RUNTIME_BINDING_KEYS
    if key != "shadow_component_digest_version"
)
SHADOW_COMPACT_RUNTIME_CHECKPOINT_SCHEMA = "shadow_compact_runtime_v8"

_SHADOW_COMPONENT_FINGERPRINT_FIELDS = tuple(
    name
    for name in SHADOW_RECORD_FIELDS
    if name
    not in {
        "input_digest",
        "journal_prefix_fingerprint",
        "protocol_id",
        "external_submission_attempts",
    }
)


class ShadowLiveError(ValueError):
    """Raised when a shadow input, identity, or parity contract fails closed."""


def _aware(value: Any, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ShadowLiveError(f"{name} must be timezone aware")
    return result.tz_convert("UTC")


def _identity(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ShadowLiveError(f"{name} must be a non-empty identity")
    return value


def _require_shadow_boundary_attack_real_bar(
    bar: Any,
    boundary: Any,
) -> None:
    """Fail closed on a non-real BAR during boundary checkpoint rebuild."""

    if (
        bar is None
        or bar.kind is not EventKind.BAR_COMPLETED
        or bar.origin is not EventOrigin.NORMALIZED_DATA
        or bar.evidence.get("real_completed") is not True
        or bar.evidence.get("clock_only") is not False
        or bar.timeframe is not boundary.timeframe
        or bar.known_at != boundary.known_at
    ):
        raise ValueError(
            "boundary attack real BAR is absent from audit history"
        )


def _sha256_text(value: Any, *, name: str) -> str:
    value = _identity(value, name=name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ShadowLiveError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _prefixed_digest(value: Any, *, prefix: str, name: str) -> str:
    value = _identity(value, name=name)
    if not value.startswith(prefix):
        raise ShadowLiveError(f"{name} has the wrong identity namespace")
    _sha256_text(value[len(prefix) :], name=name)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        to_primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _digest_primitive(value: Any) -> str:
    """Hash an already-normalized JSON value without walking it again."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_contract_default(value: Any) -> Any:
    """Expose canonical DTO fields directly to the C JSON encoder.

    Current component parity previously materialized a second complete
    primitive object graph before handing it to ``json.dumps``.  Canonical
    component DTOs use string (including string-enum) mapping keys, so the
    encoder can traverse their immutable graph directly while producing the
    exact same sorted JSON bytes.  The generic ``to_primitive`` path remains
    the authority outside this bounded parity hot path.
    """

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key.value if isinstance(key, Enum) else key): item
            for key, item in value.items()
        }
    if is_dataclass(value):
        return {
            item.name: getattr(value, item.name)
            for item in fields(value)
        }
    raise TypeError(
        f"object of type {type(value).__name__} is not canonical JSON"
    )


def _digest_canonical_contract(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_canonical_contract_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _component_digest(
    role: str,
    payload: Mapping[str, Any],
    *,
    version: str,
) -> str:
    return _digest_primitive(
        {
            "component_digest_version": version,
            "role": role,
            "payload": payload,
        }
    )


def _canonical_contract_component_digest(
    role: str,
    payload: Mapping[str, Any],
    *,
    version: str,
) -> str:
    return _digest_canonical_contract(
        {
            "component_digest_version": version,
            "role": role,
            "payload": payload,
        }
    )


@dataclass(frozen=True)
class ShadowComponentDigestBundle:
    """One version-bound set of exact parity component fingerprints."""

    version: str
    fingerprints: tuple[tuple[str, str], ...]
    bundle_id: str = field(init=False)

    def __post_init__(self) -> None:
        values = tuple(self.fingerprints)
        object.__setattr__(self, "fingerprints", values)
        if (
            self.version
            not in {
                SHADOW_COMPONENT_DIGEST_VERSION,
                SHADOW_LEGACY_COMPONENT_DIGEST_VERSION,
            }
            or tuple(key for key, _ in values)
            != _SHADOW_COMPONENT_FINGERPRINT_FIELDS
        ):
            raise ShadowLiveError("shadow component digest bundle changed")
        for key, value in values:
            _sha256_text(value, name=f"component digest {key}")
        object.__setattr__(
            self,
            "bundle_id",
            "shadow-component-bundle:"
            + _component_digest(
                "bundle",
                dict(values),
                version=self.version,
            ),
        )

    def as_record_fields(self) -> dict[str, str]:
        return dict(self.fingerprints)


def _execution_component_maps(
    runner: "ShadowLiveRunner",
) -> tuple[dict[str, str], dict[str, str]]:
    return (
        {
            identity: fsm.store.event_fingerprint
            for identity, fsm in sorted(runner.execution_fsms.items())
        },
        {
            identity: fsm.store.state_fingerprint
            for identity, fsm in sorted(runner.execution_fsms.items())
        },
    )


def _legacy_shadow_component_digest_bundle(
    runner: "ShadowLiveRunner",
    snapshot: EngineSnapshot,
) -> ShadowComponentDigestBundle:
    """Retain exact v1.2 fingerprint semantics for old checkpoints."""

    market = snapshot.market_snapshot
    belief = snapshot.belief
    execution_event_map, execution_state_map = _execution_component_maps(
        runner
    )
    values = {
        "observation_fingerprint": _digest(snapshot.observation),
        "audit_event_store_fingerprint": (
            runner.engine.observer.audit_store.fingerprint()
        ),
        "market_snapshot_fingerprint": (
            _digest(None) if market is None else market.fingerprint
        ),
        "market_state_fingerprint": (
            _digest(None)
            if market is None
            else _digest(market.replay_payload())
        ),
        "relation_fingerprint": (
            _digest(None) if market is None else _digest(market.relations)
        ),
        "session_fingerprint": (
            _digest(None) if market is None else _digest(market.session)
        ),
        "neutral_market_state_fingerprint": _digest(
            snapshot.neutral_market_state
        ),
        "belief_fingerprint": _digest(belief),
        "path_state_fingerprint": _digest(belief.path_competition_state),
        "path_update_fingerprint": _digest(
            belief.path_update_records_this_clock
        ),
        "dol_probability_fingerprint": _digest(belief.dol_probabilities),
        "signal_assessment_fingerprint": _digest(belief.signal_assessments),
        "trade_intent_fingerprint": _digest(belief.trade_intents),
        "decision_fingerprint": _digest(snapshot.decision),
        "risk_fingerprint": _digest(snapshot.risk),
        "engine_snapshot_fingerprint": _digest(snapshot),
        "runtime_action_policy_fingerprint": (
            runner._runtime_action_policy_fingerprint
        ),
        "runtime_bindings_fingerprint": (
            runner._runtime_bindings_fingerprint
        ),
        "execution_approval_fingerprint": _digest(
            runner._approval_digests
        ),
        "execution_event_fingerprint": _digest(execution_event_map),
        "execution_state_fingerprint": _digest(execution_state_map),
    }
    return ShadowComponentDigestBundle(
        version=SHADOW_LEGACY_COMPONENT_DIGEST_VERSION,
        fingerprints=tuple(
            (name, values[name])
            for name in _SHADOW_COMPONENT_FINGERPRINT_FIELDS
        ),
    )


def _current_shadow_component_digest_bundle(
    runner: "ShadowLiveRunner",
    snapshot: EngineSnapshot,
) -> ShadowComponentDigestBundle:
    """Traverse each large immutable component at most once per clock."""

    market = snapshot.market_snapshot
    observation = snapshot.observation
    belief = snapshot.belief
    if observation.market_snapshot is not market:
        raise ShadowLiveError(
            "shadow observation and Engine snapshot market identities differ"
        )

    if market is None:
        market_snapshot_fingerprint = _component_digest(
            "market_snapshot",
            {"present": False},
            version=SHADOW_COMPONENT_DIGEST_VERSION,
        )
        market_state_fingerprint = _component_digest(
            "market_replay_state",
            {"present": False},
            version=SHADOW_COMPONENT_DIGEST_VERSION,
        )
        relation_fingerprint = _digest_primitive(None)
        session_fingerprint = _digest_primitive(None)
    else:
        timeframe_fingerprint = _digest_canonical_contract(
            market.timeframe_states
        )
        relation_fingerprint = _digest_canonical_contract(market.relations)
        session_fingerprint = _digest_canonical_contract(market.session)
        events_fingerprint = _digest_canonical_contract(
            market.events_this_update
        )
        market_snapshot_fingerprint = _canonical_contract_component_digest(
            "market_snapshot",
            {
                "asof": market.asof,
                "symbol": market.symbol,
                "instrument_id": market.instrument_id,
                "price": market.price,
                "semantic_version": market.semantic_version,
                "semantic_registry_identity": (
                    market.semantic_registry_identity
                ),
                "timeframe_states_fingerprint": timeframe_fingerprint,
                "relations_fingerprint": relation_fingerprint,
                "session_fingerprint": session_fingerprint,
                "events_this_update_fingerprint": events_fingerprint,
                "labels": market.labels,
                "authority": market.authority,
            },
            version=SHADOW_COMPONENT_DIGEST_VERSION,
        )
        market_state_fingerprint = _canonical_contract_component_digest(
            "market_replay_state",
            {
                "timeframes_fingerprint": timeframe_fingerprint,
                "relations_fingerprint": relation_fingerprint,
                "session_fingerprint": session_fingerprint,
                "authority": market.authority,
            },
            version=SHADOW_COMPONENT_DIGEST_VERSION,
        )

    observation_payload = {
        item.name: (
            {
                "component_digest_version": SHADOW_COMPONENT_DIGEST_VERSION,
                "fingerprint": market_snapshot_fingerprint,
            }
            if item.name == "market_snapshot"
            else getattr(observation, item.name)
        )
        for item in fields(observation)
    }
    observation_fingerprint = _canonical_contract_component_digest(
        "market_observation",
        observation_payload,
        version=SHADOW_COMPONENT_DIGEST_VERSION,
    )

    belief_fingerprint = _digest_canonical_contract(belief)
    path_state_fingerprint = _digest_canonical_contract(
        belief.path_competition_state
    )
    path_update_fingerprint = _digest_canonical_contract(
        belief.path_update_records_this_clock
    )
    dol_probability_fingerprint = _digest_canonical_contract(
        belief.dol_probabilities
    )
    signal_assessment_fingerprint = _digest_canonical_contract(
        belief.signal_assessments
    )
    trade_intent_fingerprint = _digest_canonical_contract(
        belief.trade_intents
    )
    neutral_market_state_fingerprint = _digest_canonical_contract(
        snapshot.neutral_market_state
    )
    decision_fingerprint = _digest_canonical_contract(snapshot.decision)
    risk_fingerprint = _digest_canonical_contract(snapshot.risk)
    engine_snapshot_fingerprint = _component_digest(
        "engine_snapshot",
        {
            "observation_fingerprint": observation_fingerprint,
            "belief_fingerprint": belief_fingerprint,
            "decision_fingerprint": decision_fingerprint,
            "risk_fingerprint": risk_fingerprint,
            "neutral_market_state_fingerprint": (
                neutral_market_state_fingerprint
            ),
            "market_snapshot_fingerprint": market_snapshot_fingerprint,
        },
        version=SHADOW_COMPONENT_DIGEST_VERSION,
    )
    execution_event_map, execution_state_map = _execution_component_maps(
        runner
    )
    values = {
        "observation_fingerprint": observation_fingerprint,
        "audit_event_store_fingerprint": (
            runner.engine.observer.audit_store.fingerprint()
        ),
        "market_snapshot_fingerprint": market_snapshot_fingerprint,
        "market_state_fingerprint": market_state_fingerprint,
        "relation_fingerprint": relation_fingerprint,
        "session_fingerprint": session_fingerprint,
        "neutral_market_state_fingerprint": (
            neutral_market_state_fingerprint
        ),
        "belief_fingerprint": belief_fingerprint,
        "path_state_fingerprint": path_state_fingerprint,
        "path_update_fingerprint": path_update_fingerprint,
        "dol_probability_fingerprint": dol_probability_fingerprint,
        "signal_assessment_fingerprint": signal_assessment_fingerprint,
        "trade_intent_fingerprint": trade_intent_fingerprint,
        "decision_fingerprint": decision_fingerprint,
        "risk_fingerprint": risk_fingerprint,
        "engine_snapshot_fingerprint": engine_snapshot_fingerprint,
        "runtime_action_policy_fingerprint": (
            runner._runtime_action_policy_fingerprint
        ),
        "runtime_bindings_fingerprint": (
            runner._runtime_bindings_fingerprint
        ),
        "execution_approval_fingerprint": _digest_canonical_contract(
            runner._approval_digests
        ),
        "execution_event_fingerprint": _digest_canonical_contract(
            execution_event_map
        ),
        "execution_state_fingerprint": _digest_canonical_contract(
            execution_state_map
        ),
    }
    return ShadowComponentDigestBundle(
        version=SHADOW_COMPONENT_DIGEST_VERSION,
        fingerprints=tuple(
            (name, values[name])
            for name in _SHADOW_COMPONENT_FINGERPRINT_FIELDS
        ),
    )


class _CanonicalStringSequenceDigest:
    """Incrementally hash the exact canonical JSON representation of strings."""

    def __init__(self, values: Iterable[str] = ()) -> None:
        self._hasher = hashlib.sha256()
        self._hasher.update(b"[")
        self._count = 0
        for value in values:
            self.append(value)

    @property
    def count(self) -> int:
        return self._count

    def append(self, value: str) -> None:
        _identity(value, name="incremental digest value")
        if self._count:
            self._hasher.update(b",")
        self._hasher.update(_canonical_json(value).encode("utf-8"))
        self._count += 1

    @property
    def fingerprint(self) -> str:
        current = self._hasher.copy()
        current.update(b"]")
        return current.hexdigest()


@dataclass(frozen=True)
class ShadowLiveProtocol:
    schema_version: str
    status: str
    authority: str
    external_submission_allowed: bool
    input_grain: str
    duplicate_policy: str
    ordering_policy: str
    evidence_clock_policy: str
    contract_quantity_policy: str
    execution_event_delivery_policy: str
    failure_policy: str
    parity_policy: str
    checkpoint_policy: str
    instrument_mapping: tuple[tuple[str, Any], ...]
    record_fields: tuple[str, ...]
    canonical_payload: str
    protocol_id: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            payload = json.loads(self.canonical_payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ShadowLiveError("shadow protocol canonical payload is invalid") from exc
        required = {
            "schema_version",
            "status",
            "authority",
            "external_submission_allowed",
            "input_grain",
            "duplicate_policy",
            "ordering_policy",
            "evidence_clock_policy",
            "contract_quantity_policy",
            "execution_event_delivery_policy",
            "failure_policy",
            "parity_policy",
            "checkpoint_policy",
            "instrument_mapping",
            "record_fields",
        }
        if (
            "schema_version" not in vars(self)
            or not isinstance(payload, Mapping)
            or set(payload) != required
            or self.schema_version != payload.get("schema_version")
            or self.status != payload.get("status")
            or self.authority != payload.get("authority")
            or self.external_submission_allowed
            is not payload.get("external_submission_allowed")
            or self.input_grain != payload.get("input_grain")
            or self.duplicate_policy != payload.get("duplicate_policy")
            or self.ordering_policy != payload.get("ordering_policy")
            or self.evidence_clock_policy != payload.get("evidence_clock_policy")
            or self.contract_quantity_policy
            != payload.get("contract_quantity_policy")
            or self.execution_event_delivery_policy
            != payload.get("execution_event_delivery_policy")
            or self.failure_policy != payload.get("failure_policy")
            or self.parity_policy != payload.get("parity_policy")
            or self.checkpoint_policy != payload.get("checkpoint_policy")
            or dict(self.instrument_mapping) != payload.get("instrument_mapping")
            or self.record_fields != tuple(payload.get("record_fields", ()))
            or self.schema_version != SHADOW_LIVE_SCHEMA_VERSION
            or self.status != SHADOW_LIVE_STATUS
            or self.authority != SHADOW_LIVE_AUTHORITY
            or self.external_submission_allowed is not False
            or self.record_fields != SHADOW_RECORD_FIELDS
            or self.input_grain
            != "one_exact_completed_bar_plus_execution_reality_and_execution_facts"
            or self.duplicate_policy
            != "identical_feed_event_id_is_idempotent_conflicting_payload_fail_stop"
            or self.ordering_policy
            != "strictly_increasing_completed_bar_clock_except_identical_duplicate"
            or self.evidence_clock_policy
            != "exact_execution_and_account_identity_with_observed_at_le_known_at_le_completed_clock_recomputed_age_and_consistent_bbo"
            or self.contract_quantity_policy
            != "positive_whole_contract_quantity_and_displayed_capacity"
            or self.execution_event_delivery_policy
            != "first_clock_start_le_known_at_le_end_then_last_committed_asof_lt_known_at_le_current_asof"
            or self.failure_policy
            != "first_failure_is_terminal_attempt_journal_replays_rejected_inputs_and_parity_includes_records_journal_failure_and_gateway"
            or self.parity_policy
            != "all_registered_fields_and_terminal_state_exact_every_clock_with_canonical_scene_delta_identity_order"
            or self.checkpoint_policy
            != "pickle_resume_and_cold_attempt_replay_must_match_exactly"
        ):
            raise ShadowLiveError("shadow protocol preregistration changed")
        mapping = dict(self.instrument_mapping)
        mapping_required = {
            "mapping_id",
            "logical_instrument_id",
            "vendor_instrument_id",
            "vendor_symbol",
            "tick_size",
            "point_value",
            "mapping_sha256",
        }
        mapping_body = {
            key: mapping[key]
            for key in (
                "mapping_id",
                "logical_instrument_id",
                "vendor_instrument_id",
                "vendor_symbol",
                "tick_size",
                "point_value",
            )
            if key in mapping
        }
        if (
            set(mapping) != mapping_required
            or not isinstance(mapping.get("mapping_id"), str)
            or not mapping.get("mapping_id")
            or not isinstance(mapping.get("logical_instrument_id"), (str, int))
            or isinstance(mapping.get("logical_instrument_id"), bool)
            or type(mapping.get("vendor_instrument_id")) is not int
            or mapping["vendor_instrument_id"] < 0
            or not isinstance(mapping.get("vendor_symbol"), str)
            or not mapping.get("vendor_symbol")
            or not isinstance(mapping.get("tick_size"), (int, float))
            or isinstance(mapping.get("tick_size"), bool)
            or not math.isfinite(float(mapping["tick_size"]))
            or float(mapping["tick_size"]) <= 0.0
            or not isinstance(mapping.get("point_value"), (int, float))
            or isinstance(mapping.get("point_value"), bool)
            or not math.isfinite(float(mapping["point_value"]))
            or float(mapping["point_value"]) <= 0.0
            or mapping.get("mapping_sha256") != _digest(mapping_body)
        ):
            raise ShadowLiveError("shadow instrument mapping is invalid")
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        if canonical != self.canonical_payload:
            raise ShadowLiveError("shadow protocol payload is not canonical")
        object.__setattr__(self, "protocol_id", f"shadow-live-protocol:{_digest(payload)}")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ShadowLiveProtocol":
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return cls(
            schema_version=str(payload.get("schema_version")),
            status=str(payload.get("status")),
            authority=str(payload.get("authority")),
            external_submission_allowed=payload.get("external_submission_allowed"),
            input_grain=str(payload.get("input_grain")),
            duplicate_policy=str(payload.get("duplicate_policy")),
            ordering_policy=str(payload.get("ordering_policy")),
            evidence_clock_policy=str(payload.get("evidence_clock_policy")),
            contract_quantity_policy=str(payload.get("contract_quantity_policy")),
            execution_event_delivery_policy=str(
                payload.get("execution_event_delivery_policy")
            ),
            failure_policy=str(payload.get("failure_policy")),
            parity_policy=str(payload.get("parity_policy")),
            checkpoint_policy=str(payload.get("checkpoint_policy")),
            instrument_mapping=tuple(
                sorted(dict(payload.get("instrument_mapping", {})).items())
            ),
            record_fields=tuple(payload.get("record_fields", ())),
            canonical_payload=canonical,
        )


def load_shadow_live_protocol(
    path: str | Path = "configs/shadow_live_v1.json",
    *,
    expected_sha256: str | None = None,
) -> ShadowLiveProtocol:
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = Path(__file__).resolve().parents[1] / source
    raw = source.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ShadowLiveError("shadow protocol file SHA-256 mismatch")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ShadowLiveError("shadow protocol file is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ShadowLiveError("shadow protocol root must be a mapping")
    return ShadowLiveProtocol.from_payload(payload)


def shadow_runtime_bindings_from_model_config(
    path: str | Path = "configs/model.json",
) -> tuple[tuple[str, str], ...]:
    """Bind model bytes, the canonical foundation, and Phase 7/8 protocols."""

    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = Path(__file__).resolve().parents[1] / source
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
        selection = load_semantic_selection(
            payload["semantic_selection"],
            root=Path(__file__).resolve().parents[1],
        )
        path_values = payload["path_hypotheses"]
        dol_values = payload["dol_probability"]
        signal_values = payload["signal_policy"]
        bindings = {
            "model_config_sha256": hashlib.sha256(raw).hexdigest(),
            "semantic_version": selection.atomic_semantics_version,
            "path_protocol_fingerprint": path_values[
                "path_protocol_fingerprint"
            ],
            "dol_ranking_protocol_fingerprint": path_values[
                "dol_protocol_fingerprint"
            ],
            "dol_probability_protocol_fingerprint": dol_values[
                "protocol_fingerprint"
            ],
            "signal_policy_protocol_fingerprint": signal_values[
                "protocol_fingerprint"
            ],
            "execution_protocol_fingerprint": EXECUTION_PROTOCOL_FINGERPRINT,
            "shadow_component_digest_version": (
                SHADOW_COMPONENT_DIGEST_VERSION
            ),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ShadowLiveError("model config lacks exact shadow runtime bindings") from exc
    if set(bindings) != set(SHADOW_RUNTIME_BINDING_KEYS):
        raise ShadowLiveError("shadow runtime binding keys changed")
    for key, value in bindings.items():
        if not isinstance(value, str) or not value:
            raise ShadowLiveError(f"shadow runtime binding {key} is invalid")
        if key == "shadow_component_digest_version":
            if value != SHADOW_COMPONENT_DIGEST_VERSION:
                raise ShadowLiveError(
                    "shadow component digest version binding changed"
                )
        elif key != "semantic_version" and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ShadowLiveError(f"shadow runtime binding {key} is not SHA-256")
    return tuple(sorted(bindings.items()))


class NullExecutionGateway:
    """A capability-negative gateway: every external submission is an error."""

    __slots__ = ("_submission_attempts",)

    def __init__(self) -> None:
        self._submission_attempts = 0

    @property
    def submission_attempts(self) -> int:
        return self._submission_attempts

    def submit(self, _: Any) -> None:
        self._submission_attempts += 1
        raise ShadowLiveError("NullExecutionGateway forbids external submission")


_NULL_EXECUTION_GATEWAY_SUBMIT = NullExecutionGateway.submit


@dataclass(frozen=True)
class ShadowClockInput:
    feed_event_id: str
    received_at: pd.Timestamp
    bar: Bar
    execution: ExecutionRealityInput
    execution_observed_at: pd.Timestamp
    execution_known_at: pd.Timestamp
    execution_source_event_id: str
    account: AccountState
    account_observed_at: pd.Timestamp
    account_known_at: pd.Timestamp
    account_snapshot_id: str
    source_event_ids: tuple[str, ...]
    approved_intents: tuple[RiskApprovedTradeIntent, ...] = ()
    execution_events: tuple[ExecutionEventEnvelope, ...] = ()
    input_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _identity(self.feed_event_id, name="feed_event_id")
        received = _aware(self.received_at, name="shadow received_at")
        object.__setattr__(self, "received_at", received)
        if not isinstance(self.bar, Bar):
            raise TypeError("shadow input requires Bar")
        if received < self.bar.end:
            raise ShadowLiveError("shadow input was received before bar completion")
        if not isinstance(self.execution, ExecutionRealityInput):
            raise TypeError("shadow input requires ExecutionRealityInput")
        if not isinstance(self.account, AccountState):
            raise TypeError("shadow input requires AccountState")
        execution_observed_at = _aware(
            self.execution_observed_at,
            name="execution_observed_at",
        )
        execution_known_at = _aware(
            self.execution_known_at,
            name="execution_known_at",
        )
        account_observed_at = _aware(
            self.account_observed_at,
            name="account_observed_at",
        )
        account_known_at = _aware(
            self.account_known_at,
            name="account_known_at",
        )
        execution_source_event_id = _identity(
            self.execution_source_event_id,
            name="execution_source_event_id",
        )
        account_snapshot_id = _identity(
            self.account_snapshot_id,
            name="account_snapshot_id",
        )
        if not (
            execution_observed_at <= execution_known_at <= self.bar.end
            and account_observed_at <= account_known_at <= self.bar.end
        ):
            raise ShadowLiveError(
                "shadow execution/account evidence is future-known or clock-inverted"
            )
        expected_age = (self.bar.end - execution_observed_at).total_seconds()
        if (
            not math.isfinite(float(self.execution.data_age_seconds))
            or not math.isclose(
                float(self.execution.data_age_seconds),
                expected_age,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ShadowLiveError(
                "execution data age does not match its exact observed clock"
            )
        if type(self.execution.quantity) is not int or self.execution.quantity <= 0:
            raise ShadowLiveError("shadow execution quantity must be whole contracts")
        if type(self.account.quantity) is not int or self.account.quantity <= 0:
            raise ShadowLiveError("shadow account quantity must be whole contracts")
        displayed_sizes = (
            self.execution.size_available,
            self.execution.bid_size,
            self.execution.ask_size,
        )
        if any(
            value is not None
            and (
                not math.isfinite(float(value))
                or float(value) < 0.0
                or not float(value).is_integer()
            )
            for value in displayed_sizes
        ):
            raise ShadowLiveError(
                "shadow displayed execution capacity must be whole contracts"
            )
        bbo_values = (
            self.execution.bid,
            self.execution.ask,
            self.execution.bid_size,
            self.execution.ask_size,
        )
        if any(value is not None for value in bbo_values):
            if any(value is None for value in bbo_values):
                raise ShadowLiveError("shadow execution BBO is incomplete")
            bid = float(self.execution.bid)
            ask = float(self.execution.ask)
            bid_size = float(self.execution.bid_size)
            ask_size = float(self.execution.ask_size)
            expected_spread = ask - bid
            expected_size = min(bid_size, ask_size)
            expected_imbalance = (bid_size - ask_size) / max(
                1.0, bid_size + ask_size
            )
            if (
                not all(
                    math.isfinite(value)
                    for value in (bid, ask, bid_size, ask_size)
                )
                or bid <= 0.0
                or ask <= bid
                or self.execution.spread_points is None
                or self.execution.size_available is None
                or self.execution.depth_imbalance is None
                or not math.isclose(
                    float(self.execution.spread_points),
                    expected_spread,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(self.execution.size_available),
                    expected_size,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(self.execution.depth_imbalance),
                    expected_imbalance,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ShadowLiveError(
                    "shadow execution BBO redundant fields are inconsistent"
                )
        account_values = (
            self.account.equity,
            self.account.open_risk_fraction,
            self.account.requested_risk_fraction,
            self.account.point_value,
        )
        if any(not math.isfinite(float(value)) for value in account_values):
            raise ShadowLiveError("shadow account state must be finite")
        object.__setattr__(self, "execution_observed_at", execution_observed_at)
        object.__setattr__(self, "execution_known_at", execution_known_at)
        object.__setattr__(
            self,
            "execution_source_event_id",
            execution_source_event_id,
        )
        object.__setattr__(self, "account_observed_at", account_observed_at)
        object.__setattr__(self, "account_known_at", account_known_at)
        object.__setattr__(self, "account_snapshot_id", account_snapshot_id)
        sources = tuple(sorted(self.source_event_ids))
        if (
            not sources
            or len(sources) != len(set(sources))
            or any(not isinstance(value, str) or not value for value in sources)
            or execution_source_event_id == account_snapshot_id
            or execution_source_event_id not in sources
            or account_snapshot_id not in sources
        ):
            raise ShadowLiveError(
                "shadow source_event_ids omit exact execution/account evidence"
            )
        object.__setattr__(self, "source_event_ids", sources)
        approvals = tuple(
            sorted(self.approved_intents, key=lambda item: item.approved_intent_id)
        )
        events = tuple(
            sorted(
                self.execution_events,
                key=lambda item: (
                    item.fact.approved_intent_id,
                    item.vendor_sequence,
                    item.known_at,
                    item.event_id,
                ),
            )
        )
        if (
            any(not isinstance(item, RiskApprovedTradeIntent) for item in approvals)
            or len({item.approved_intent_id for item in approvals}) != len(approvals)
            or any(not isinstance(item, ExecutionEventEnvelope) for item in events)
            or len({item.event_id for item in events}) != len(events)
            or any(item.known_at > self.bar.end for item in events)
        ):
            raise ShadowLiveError("shadow execution facts are invalid or future-known")
        object.__setattr__(self, "approved_intents", approvals)
        object.__setattr__(self, "execution_events", events)
        payload = {
            name: getattr(self, name)
            for name in (
                "feed_event_id",
                "received_at",
                "bar",
                "execution",
                "execution_observed_at",
                "execution_known_at",
                "execution_source_event_id",
                "account",
                "account_observed_at",
                "account_known_at",
                "account_snapshot_id",
                "source_event_ids",
                "approved_intents",
                "execution_events",
            )
        }
        object.__setattr__(self, "input_digest", _digest(payload))


class ShadowInputJournal:
    """Append-only completed-clock journal with reconnect-safe deduplication."""

    def __init__(self) -> None:
        self._attempts: list[ShadowClockInput] = []
        self._events: list[ShadowClockInput] = []
        self._attempt_digests: list[str] = []
        self._event_digests: list[str] = []
        self._attempt_sequence_digest = _CanonicalStringSequenceDigest()
        self._event_sequence_digest = _CanonicalStringSequenceDigest()
        self._by_feed_id: dict[str, ShadowClockInput] = {}
        self._by_clock: dict[pd.Timestamp, str] = {}
        self._execution_evidence_digests: dict[str, str] = {}
        self._account_evidence_digests: dict[str, str] = {}

    @property
    def events(self) -> tuple[ShadowClockInput, ...]:
        return tuple(self._events)

    @property
    def attempts(self) -> tuple[ShadowClockInput, ...]:
        return tuple(self._attempts)

    @property
    def fingerprint(self) -> str:
        return self._event_sequence_digest.fingerprint

    @property
    def attempt_fingerprint(self) -> str:
        return self._attempt_sequence_digest.fingerprint

    def __len__(self) -> int:
        return len(self._events)

    def get(self, feed_event_id: str) -> ShadowClockInput | None:
        return self._by_feed_id.get(feed_event_id)

    def record_attempt(self, value: ShadowClockInput) -> None:
        if not isinstance(value, ShadowClockInput):
            raise TypeError("shadow journal attempts require ShadowClockInput")
        self._attempts.append(value)
        self._attempt_digests.append(value.input_digest)
        self._attempt_sequence_digest.append(value.input_digest)

    def require_incremental_consistent(self) -> None:
        """Check constant-time invariants used on the per-clock hot path."""

        if (
            len(self._attempt_digests) != len(self._attempts)
            or len(self._event_digests) != len(self._events)
            or self._attempt_sequence_digest.count != len(self._attempts)
            or self._event_sequence_digest.count != len(self._events)
            or len(self._by_feed_id) != len(self._events)
            or len(self._by_clock) != len(self._events)
            or (
                self._events
                and (
                    self._event_digests[-1] != self._events[-1].input_digest
                    or self._by_feed_id.get(self._events[-1].feed_event_id)
                    is not self._events[-1]
                    or self._by_clock.get(self._events[-1].bar.end)
                    != self._events[-1].feed_event_id
                )
            )
            or (
                self._attempts
                and self._attempt_digests[-1] != self._attempts[-1].input_digest
            )
        ):
            raise ShadowLiveError("shadow journal incremental indexes drifted")

    def require_consistent(self) -> None:
        """Rebuild every mutable index from immutable accepted events."""

        for value in (*self._attempts, *self._events):
            if not isinstance(value, ShadowClockInput):
                raise ShadowLiveError(
                    "shadow journal history changed input type"
                )
            rebuilt = ShadowClockInput(
                **{
                    item.name: getattr(value, item.name)
                    for item in fields(ShadowClockInput)
                    if item.init
                }
            )
            if rebuilt != value or rebuilt.input_digest != value.input_digest:
                raise ShadowLiveError(
                    "shadow input content does not bind input_digest"
                )
        clone = ShadowInputJournal()
        for value in self._events:
            clone.append(value)
        if (
            clone._by_feed_id != self._by_feed_id
            or clone._by_clock != self._by_clock
            or clone._execution_evidence_digests
            != self._execution_evidence_digests
            or clone._account_evidence_digests != self._account_evidence_digests
            or any(not isinstance(value, ShadowClockInput) for value in self._attempts)
            or self._attempt_digests
            != [value.input_digest for value in self._attempts]
            or self._event_digests != [value.input_digest for value in self._events]
            or self._attempt_sequence_digest.count != len(self._attempts)
            or self._event_sequence_digest.count != len(self._events)
            or self.attempt_fingerprint != _digest(tuple(self._attempt_digests))
            or self.fingerprint != _digest(tuple(self._event_digests))
        ):
            raise ShadowLiveError("shadow journal indexes or evidence registry drifted")
        accepted_index = 0
        for attempt in self._attempts:
            if (
                accepted_index < len(self._events)
                and attempt.input_digest == self._events[accepted_index].input_digest
            ):
                accepted_index += 1
        if accepted_index != len(self._events):
            raise ShadowLiveError("shadow accepted journal is not ordered within attempts")

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_attempt_sequence_digest", None)
        state.pop("_event_sequence_digest", None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(state)
        if "_attempt_digests" not in self.__dict__:
            self._attempt_digests = [value.input_digest for value in self._attempts]
        if "_event_digests" not in self.__dict__:
            self._event_digests = [value.input_digest for value in self._events]
        self._attempt_sequence_digest = _CanonicalStringSequenceDigest(
            self._attempt_digests
        )
        self._event_sequence_digest = _CanonicalStringSequenceDigest(
            self._event_digests
        )
        self.require_consistent()

    def append(self, value: ShadowClockInput) -> bool:
        if not isinstance(value, ShadowClockInput):
            raise TypeError("shadow journal accepts only ShadowClockInput")
        prior = self._by_feed_id.get(value.feed_event_id)
        if prior is not None:
            if prior.input_digest != value.input_digest:
                raise ShadowLiveError("feed event identity conflicts with immutable input")
            return False
        existing_feed_id = self._by_clock.get(value.bar.end)
        if existing_feed_id is not None:
            raise ShadowLiveError(
                "completed clock is already bound to another feed event identity"
            )
        if self._events and value.bar.end <= self._events[-1].bar.end:
            raise ShadowLiveError("completed inputs are out of causal clock order")
        execution_digest = _digest(
            {
                "execution": value.execution,
                "observed_at": value.execution_observed_at,
                "known_at": value.execution_known_at,
            }
        )
        account_digest = _digest(
            {
                "account": value.account,
                "observed_at": value.account_observed_at,
                "known_at": value.account_known_at,
            }
        )
        prior_execution_digest = self._execution_evidence_digests.get(
            value.execution_source_event_id
        )
        prior_account_digest = self._account_evidence_digests.get(
            value.account_snapshot_id
        )
        if (
            prior_execution_digest is not None
            and prior_execution_digest != execution_digest
        ):
            raise ShadowLiveError(
                "execution evidence identity conflicts with immutable content"
            )
        if prior_account_digest is not None and prior_account_digest != account_digest:
            raise ShadowLiveError(
                "account evidence identity conflicts with immutable content"
            )
        self._events.append(value)
        self._event_digests.append(value.input_digest)
        self._event_sequence_digest.append(value.input_digest)
        self._by_feed_id[value.feed_event_id] = value
        self._by_clock[value.bar.end] = value.feed_event_id
        self._execution_evidence_digests[
            value.execution_source_event_id
        ] = execution_digest
        self._account_evidence_digests[value.account_snapshot_id] = account_digest
        return True


@dataclass(frozen=True)
class ShadowParityRecord:
    sequence: int
    feed_event_id: str
    asof: pd.Timestamp
    input_digest: str
    journal_prefix_fingerprint: str
    observation_fingerprint: str
    audit_event_store_fingerprint: str
    market_snapshot_fingerprint: str
    market_state_fingerprint: str
    relation_fingerprint: str
    session_fingerprint: str
    neutral_market_state_fingerprint: str
    belief_fingerprint: str
    path_state_fingerprint: str
    path_update_fingerprint: str
    dol_probability_fingerprint: str
    signal_assessment_fingerprint: str
    trade_intent_fingerprint: str
    decision_fingerprint: str
    risk_fingerprint: str
    engine_snapshot_fingerprint: str
    runtime_action_policy_fingerprint: str
    runtime_bindings_fingerprint: str
    execution_approval_fingerprint: str
    execution_event_fingerprint: str
    execution_state_fingerprint: str
    protocol_id: str
    external_submission_attempts: int
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", _aware(self.asof, name="parity asof"))
        if (
            type(self.sequence) is not int
            or self.sequence <= 0
            or not self.feed_event_id
            or type(self.external_submission_attempts) is not int
            or self.external_submission_attempts != 0
        ):
            raise ShadowLiveError("shadow parity record identity is invalid")
        for name in SHADOW_RECORD_FIELDS:
            value = getattr(self, name)
            if name == "protocol_id":
                _prefixed_digest(
                    value,
                    prefix="shadow-live-protocol:",
                    name=name,
                )
            elif name == "external_submission_attempts":
                continue
            else:
                _sha256_text(value, name=name)
        payload = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "record_id"
        }
        object.__setattr__(self, "record_id", f"shadow-parity:{_digest(payload)}")


@dataclass(frozen=True)
class ShadowParityMismatch:
    sequence: int
    feed_event_id: str
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.sequence) is not int
            or self.sequence <= 0
            or not isinstance(self.feed_event_id, str)
            or not self.feed_event_id
            or not self.fields
            or len(self.fields) != len(set(self.fields))
            or any(not isinstance(value, str) or not value for value in self.fields)
        ):
            raise ShadowLiveError("shadow parity mismatch is invalid")


@dataclass(frozen=True)
class ShadowFailureRecord:
    sequence: int
    feed_event_id: str
    input_digest: str
    journal_fingerprint: str
    attempt_fingerprint: str
    last_record_id: str | None
    error_type: str
    error_message: str
    failure_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.sequence) is not int
            or self.sequence <= 0
            or not self.feed_event_id
            or not self.error_type
            or not self.error_message
        ):
            raise ShadowLiveError("shadow failure record is invalid")
        _sha256_text(self.input_digest, name="failure input_digest")
        _sha256_text(
            self.journal_fingerprint,
            name="failure journal_fingerprint",
        )
        _sha256_text(
            self.attempt_fingerprint,
            name="failure attempt_fingerprint",
        )
        if self.last_record_id is not None:
            _prefixed_digest(
                self.last_record_id,
                prefix="shadow-parity:",
                name="failure last_record_id",
            )
        payload = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "failure_id"
        }
        object.__setattr__(self, "failure_id", f"shadow-failure:{_digest(payload)}")


@dataclass(frozen=True)
class ShadowParityAudit:
    expected_records: int
    actual_records: int
    mismatches: tuple[ShadowParityMismatch, ...]
    terminal_fields: tuple[str, ...]
    exact_match: bool
    coverage_complete: bool
    gate_pass: bool
    audit_id: str = field(init=False)

    def __post_init__(self) -> None:
        parity_exact = (
            self.expected_records == self.actual_records
            and not self.mismatches
            and not self.terminal_fields
        )
        if (
            type(self.expected_records) is not int
            or type(self.actual_records) is not int
            or self.expected_records < 0
            or self.actual_records < 0
            or any(
                not isinstance(item, ShadowParityMismatch)
                for item in self.mismatches
            )
            or len(self.terminal_fields) != len(set(self.terminal_fields))
            or any(
                not isinstance(value, str) or not value
                for value in self.terminal_fields
            )
            or type(self.coverage_complete) is not bool
            or self.exact_match is not parity_exact
            or self.gate_pass is not (parity_exact and self.coverage_complete)
        ):
            raise ShadowLiveError("shadow parity audit status is inconsistent")
        payload = {
            "expected_records": self.expected_records,
            "actual_records": self.actual_records,
            "mismatches": self.mismatches,
            "terminal_fields": self.terminal_fields,
            "exact_match": self.exact_match,
            "coverage_complete": self.coverage_complete,
            "gate_pass": self.gate_pass,
        }
        object.__setattr__(self, "audit_id", f"shadow-parity-audit:{_digest(payload)}")

    def require_exact(self) -> None:
        if not self.gate_pass:
            raise ShadowLiveError(
                "shadow live/replay gate requires exact healthy non-empty coverage"
            )


class ShadowLiveRunner:
    """Run one no-order stream through the production engine and reducers."""

    def __init__(
        self,
        *,
        engine: ContinuousSMCEngine,
        protocol: ShadowLiveProtocol,
        runtime_bindings: Mapping[str, str] | Sequence[tuple[str, str]],
        gateway: NullExecutionGateway | None = None,
        model_config_path: str | Path = "configs/model.json",
    ) -> None:
        if type(engine) is not ContinuousSMCEngine:
            raise TypeError("shadow runner requires the exact ContinuousSMCEngine")
        if engine.runtime_mode != "development":
            raise ShadowLiveError("shadow runner cannot use the live Engine mode")
        if not isinstance(protocol, ShadowLiveProtocol):
            raise TypeError("shadow runner requires ShadowLiveProtocol")
        self.engine = engine
        self.protocol = protocol
        if isinstance(runtime_bindings, Mapping):
            raw_binding_items = tuple(runtime_bindings.items())
        else:
            raw_binding_items = tuple(runtime_bindings)
        if (
            any(
                not isinstance(item, tuple) or len(item) != 2
                for item in raw_binding_items
            )
            or len({item[0] for item in raw_binding_items})
            != len(raw_binding_items)
        ):
            raise ShadowLiveError("shadow runner runtime bindings are duplicated")
        binding_items = tuple(sorted(raw_binding_items))
        binding_keys = tuple(key for key, _ in binding_items)
        if (
            binding_keys
            not in {
                SHADOW_RUNTIME_BINDING_KEYS,
                _LEGACY_SHADOW_RUNTIME_BINDING_KEYS,
            }
            or any(not isinstance(value, str) or not value for _, value in binding_items)
        ):
            raise ShadowLiveError("shadow runner runtime bindings are incomplete")
        binding_map = dict(binding_items)
        component_digest_version = binding_map.get(
            "shadow_component_digest_version",
            SHADOW_LEGACY_COMPONENT_DIGEST_VERSION,
        )
        if component_digest_version not in {
            SHADOW_COMPONENT_DIGEST_VERSION,
            SHADOW_LEGACY_COMPONENT_DIGEST_VERSION,
        }:
            raise ShadowLiveError("shadow component digest version changed")
        if (
            engine.model_config_sha256 != binding_map["model_config_sha256"]
            or SMC_SEMANTIC_VERSION != binding_map["semantic_version"]
            or engine.brain.path_protocol.fingerprint
            != binding_map["path_protocol_fingerprint"]
            or engine.brain.dol_protocol.fingerprint
            != binding_map["dol_ranking_protocol_fingerprint"]
            or engine.brain.dol_probability_protocol.fingerprint
            != binding_map["dol_probability_protocol_fingerprint"]
            or engine.brain.signal_policy.fingerprint
            != binding_map["signal_policy_protocol_fingerprint"]
            or EXECUTION_PROTOCOL_FINGERPRINT
            != binding_map["execution_protocol_fingerprint"]
        ):
            raise ShadowLiveError("shadow runner and runtime bindings differ")
        self.runtime_bindings = binding_items
        config_source = Path(model_config_path)
        if not config_source.is_absolute() and not config_source.exists():
            config_source = Path(__file__).resolve().parents[1] / config_source
        self.model_config_path = config_source.resolve()
        if (
            hashlib.sha256(self.model_config_path.read_bytes()).hexdigest()
            != binding_map["model_config_sha256"]
        ):
            raise ShadowLiveError("shadow model config bytes differ from bindings")
        self.gateway = gateway or NullExecutionGateway()
        if (
            type(self.gateway) is not NullExecutionGateway
            or NullExecutionGateway.submit is not _NULL_EXECUTION_GATEWAY_SUBMIT
            or getattr(self.gateway.submit, "__func__", None)
            is not _NULL_EXECUTION_GATEWAY_SUBMIT
        ):
            raise ShadowLiveError("shadow runner accepts only NullExecutionGateway")
        self.journal = ShadowInputJournal()
        self.execution_fsms: dict[str, ExecutionFSM] = {}
        self._approval_digests: dict[str, str] = {}
        self._records: list[ShadowParityRecord] = []
        self._record_ids: list[str] = []
        self._record_sequence_digest = _CanonicalStringSequenceDigest()
        self._by_feed_id: dict[str, ShadowParityRecord] = {}
        self._failure: ShadowFailureRecord | None = None
        self._rebuild_component_digest_caches()

    def _rebuild_component_digest_caches(self) -> None:
        binding_map = dict(self.runtime_bindings)
        component_digest_version = binding_map.get(
            "shadow_component_digest_version",
            SHADOW_LEGACY_COMPONENT_DIGEST_VERSION,
        )
        if component_digest_version not in {
            SHADOW_COMPONENT_DIGEST_VERSION,
            SHADOW_LEGACY_COMPONENT_DIGEST_VERSION,
        }:
            raise ShadowLiveError("shadow component digest version changed")
        self._component_digest_version = component_digest_version
        self._runtime_action_policy_fingerprint = _digest(
            self.engine.runtime_action_policy_identity
        )
        self._runtime_bindings_fingerprint = _digest(self.runtime_bindings)

    def _component_digest_bundle(
        self,
        snapshot: EngineSnapshot,
    ) -> ShadowComponentDigestBundle:
        if self._component_digest_version == SHADOW_COMPONENT_DIGEST_VERSION:
            return _current_shadow_component_digest_bundle(self, snapshot)
        if (
            self._component_digest_version
            == SHADOW_LEGACY_COMPONENT_DIGEST_VERSION
        ):
            return _legacy_shadow_component_digest_bundle(self, snapshot)
        raise ShadowLiveError("shadow component digest version is unregistered")

    def _require_runtime_bindings(self) -> None:
        """Reject checkpoint/config/code drift before every clock is consumed."""

        try:
            self.protocol.__post_init__()
        except Exception as exc:
            raise ShadowLiveError("shadow protocol drifted after restore") from exc
        binding_map = dict(self.runtime_bindings)
        current = shadow_runtime_bindings_from_model_config(
            self.model_config_path
        )
        expected_bindings = (
            current
            if self._component_digest_version
            == SHADOW_COMPONENT_DIGEST_VERSION
            else tuple(
                item
                for item in current
                if item[0] != "shadow_component_digest_version"
            )
        )
        try:
            model_payload = json.loads(self.model_config_path.read_text(encoding="utf-8"))
            mapping = dict(self.protocol.instrument_mapping)
            model_contract_matches = math.isclose(
                float(model_payload["tick_size"]),
                float(mapping["tick_size"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ) and math.isclose(
                float(model_payload["point_value"]),
                float(mapping["point_value"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            model_contract_matches = False
        if (
            expected_bindings != self.runtime_bindings
            or not model_contract_matches
            or self.engine.model_config_sha256 != binding_map["model_config_sha256"]
            or SMC_SEMANTIC_VERSION != binding_map["semantic_version"]
            or self.engine.brain.path_protocol.fingerprint
            != binding_map["path_protocol_fingerprint"]
            or self.engine.brain.dol_protocol.fingerprint
            != binding_map["dol_ranking_protocol_fingerprint"]
            or self.engine.brain.dol_probability_protocol.fingerprint
            != binding_map["dol_probability_protocol_fingerprint"]
            or self.engine.brain.signal_policy.fingerprint
            != binding_map["signal_policy_protocol_fingerprint"]
            or EXECUTION_PROTOCOL_FINGERPRINT
            != binding_map["execution_protocol_fingerprint"]
        ):
            raise ShadowLiveError("shadow runtime bindings drifted after restore")

    def _require_internal_consistency(self, *, deep: bool = True) -> None:
        if deep:
            self.journal.require_consistent()
            for record in self._records:
                if not isinstance(record, ShadowParityRecord):
                    raise ShadowLiveError(
                        "shadow parity record history changed type"
                    )
                rebuilt = ShadowParityRecord(
                    **{
                        item.name: getattr(record, item.name)
                        for item in fields(ShadowParityRecord)
                        if item.init
                    }
                )
                if rebuilt != record or rebuilt.record_id != record.record_id:
                    raise ShadowLiveError(
                        "shadow parity record content does not bind record_id"
                    )
        else:
            self.journal.require_incremental_consistent()
        expected_records = (
            {record.feed_event_id: record for record in self._records}
            if deep
            else None
        )
        if (
            len(self._record_ids) != len(self._records)
            or self._record_sequence_digest.count != len(self._records)
            or (
                deep
                and (
                    len(expected_records) != len(self._records)
                    or self._by_feed_id != expected_records
                    or self._record_ids
                    != [record.record_id for record in self._records]
                    or self.record_fingerprint != _digest(tuple(self._record_ids))
                )
            )
            or (
                not deep
                and (
                    len(self._by_feed_id) != len(self._records)
                    or (
                        self._records
                        and (
                            self._record_ids[-1] != self._records[-1].record_id
                            or self._by_feed_id.get(
                                self._records[-1].feed_event_id
                            )
                            is not self._records[-1]
                        )
                    )
                )
            )
            or set(self.execution_fsms) != set(self._approval_digests)
            or type(self.engine) is not ContinuousSMCEngine
            or self.engine.runtime_mode != "development"
            or type(self.gateway) is not NullExecutionGateway
            or NullExecutionGateway.submit is not _NULL_EXECUTION_GATEWAY_SUBMIT
            or getattr(self.gateway.submit, "__func__", None)
            is not _NULL_EXECUTION_GATEWAY_SUBMIT
            or type(self.gateway.submission_attempts) is not int
            or self.gateway.submission_attempts < 0
            or any(
                _digest(fsm.approved) != self._approval_digests[identity]
                for identity, fsm in self.execution_fsms.items()
            )
            or self._runtime_action_policy_fingerprint
            != _digest(self.engine.runtime_action_policy_identity)
            or self._runtime_bindings_fingerprint
            != _digest(self.runtime_bindings)
        ):
            raise ShadowLiveError("shadow runner indexes or approvals drifted")

    def _require_terminal_snapshot_exact(self) -> None:
        """Bind restored runtime bytes to the last immutable parity record.

        Checkpoint restore is deliberately stricter than the hot clock path:
        every retained ``MarketEvent`` occurrence must be the exact event
        committed under the same ID in the audit store, and every terminal
        component is recomputed from the restored Engine snapshot.  This
        prevents an internally valid but mutated snapshot from borrowing the
        original journal/record identities.
        """

        if not self._records:
            return
        snapshot = self.engine.last_snapshot
        if not isinstance(snapshot, EngineSnapshot):
            raise ShadowLiveError(
                "shadow checkpoint terminal state drifted: "
                "Engine snapshot is missing"
            )
        observation = snapshot.observation
        event_sequences: list[Iterable[Any]] = [
            observation.recent_events,
            observation.semantic_events_this_update,
            *observation.retained_entity_timelines.values(),
        ]
        market = snapshot.market_snapshot
        if market is not None:
            event_sequences.append(market.events_this_update)
        store = self.engine.observer.audit_store
        try:
            for sequence in event_sequences:
                for event in sequence:
                    if (
                        store.event_digest(event.event_id)
                        != store.recompute_event_digest(event)
                    ):
                        raise ShadowLiveError(
                            "observation event bytes differ from exact audit history"
                        )
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            if isinstance(error, ShadowLiveError):
                raise
            raise ShadowLiveError(
                "observation event bytes differ from exact audit history"
            ) from error

        terminal_values = self._component_digest_bundle(
            snapshot
        ).as_record_fields()
        final_record = self._records[-1]
        mismatches = tuple(
            name
            for name, value in terminal_values.items()
            if getattr(final_record, name) != value
        )
        if mismatches:
            raise ShadowLiveError(
                "shadow checkpoint terminal parity components differ: "
                + ",".join(mismatches)
            )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_record_sequence_digest", None)
        state.pop("_component_digest_version", None)
        state.pop("_runtime_action_policy_fingerprint", None)
        state.pop("_runtime_bindings_fingerprint", None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(state)
        if "_record_ids" not in self.__dict__:
            self._record_ids = [record.record_id for record in self._records]
        self._record_sequence_digest = _CanonicalStringSequenceDigest(
            self._record_ids
        )
        self._rebuild_component_digest_caches()
        self._require_internal_consistency()
        self._require_runtime_bindings()
        self._require_terminal_snapshot_exact()
        audit = audit_shadow_parity(self, self)
        if any(
            value != "non_independent_runner_alias"
            for value in audit.terminal_fields
        ):
            raise ShadowLiveError("shadow checkpoint terminal state drifted")

    @property
    def records(self) -> tuple[ShadowParityRecord, ...]:
        return tuple(self._records)

    @property
    def record_fingerprint(self) -> str:
        return self._record_sequence_digest.fingerprint

    @property
    def failure(self) -> ShadowFailureRecord | None:
        return self._failure

    def compact_runtime_checkpoint(self) -> dict[str, Any]:
        """Externalize journal/record history while retaining runtime state.

        Phase-9 v3 keeps those immutable histories in its fsynced WAL.  This
        checkpoint therefore serializes the Engine and execution reducers but
        not every prior input and parity record a second time.
        """

        self._require_internal_consistency(deep=False)
        self._require_runtime_bindings()
        if self._failure is not None or self.gateway.submission_attempts != 0:
            raise ShadowLiveError("only a healthy no-order runner can checkpoint")
        if self._component_digest_version != SHADOW_COMPONENT_DIGEST_VERSION:
            raise ShadowLiveError(
                "legacy component digest cannot publish a compact checkpoint"
            )
        return {
            "schema_version": SHADOW_COMPACT_RUNTIME_CHECKPOINT_SCHEMA,
            "engine": self.engine,
            "protocol": self.protocol,
            "runtime_bindings": self.runtime_bindings,
            "model_config_path": str(self.model_config_path),
            "execution_fsms": self.execution_fsms,
            "approval_digests": self._approval_digests,
            "journal_events": len(self.journal),
            "journal_fingerprint": self.journal.fingerprint,
            "attempt_fingerprint": self.journal.attempt_fingerprint,
            "records": len(self._records),
            "record_fingerprint": self.record_fingerprint,
            "last_record_id": (
                None if not self._records else self._records[-1].record_id
            ),
            "external_submission_attempts": 0,
        }

    @classmethod
    def from_compact_runtime_checkpoint(
        cls,
        state: Mapping[str, Any],
        *,
        journal_events: Sequence[ShadowClockInput],
        records: Sequence[ShadowParityRecord],
    ) -> "ShadowLiveRunner":
        """Restore compact runtime state against exact WAL-backed histories."""

        common_fields = {
            "schema_version",
            "engine",
            "protocol",
            "runtime_bindings",
            "model_config_path",
            "execution_fsms",
            "approval_digests",
            "journal_events",
            "journal_fingerprint",
            "attempt_fingerprint",
            "records",
            "record_fingerprint",
            "last_record_id",
            "external_submission_attempts",
        }
        if not isinstance(state, Mapping):
            raise ShadowLiveError("compact shadow runtime checkpoint changed")
        schema_version = state.get("schema_version")
        if schema_version != SHADOW_COMPACT_RUNTIME_CHECKPOINT_SCHEMA:
            raise ShadowLiveError(
                "legacy compact shadow runtime schema is read-only/unsupported"
            )
        if (
            set(state) != common_fields
            or state.get("external_submission_attempts") != 0
            or state.get("journal_events") != len(journal_events)
            or state.get("records") != len(records)
            or len(journal_events) != len(records)
        ):
            raise ShadowLiveError("compact shadow runtime checkpoint changed")
        # ``state`` is a public mapping API and callers may pass a live object
        # graph rather than bytes just decoded by pickle.  Normalize the
        # Engine through its canonical pickle boundary so nested owners drop
        # and rebuild derived indexes before any fingerprint is trusted.
        try:
            restored_engine = pickle.loads(
                pickle.dumps(
                    state["engine"],
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            )
        except Exception as error:
            raise ShadowLiveError(
                "compact checkpoint Engine state cannot be revalidated"
            ) from error
        runner = cls.__new__(cls)
        runner.engine = restored_engine
        runner.protocol = state["protocol"]
        runner.runtime_bindings = tuple(state["runtime_bindings"])
        runner.model_config_path = Path(state["model_config_path"])
        runner.gateway = NullExecutionGateway()
        runner.execution_fsms = dict(state["execution_fsms"])
        runner._approval_digests = dict(state["approval_digests"])
        runner.journal = ShadowInputJournal()
        for value in journal_events:
            runner.journal.record_attempt(value)
            if not runner.journal.append(value):
                raise ShadowLiveError("compact checkpoint WAL contains a duplicate")
        runner._records = list(records)
        runner._record_ids = [record.record_id for record in records]
        runner._record_sequence_digest = _CanonicalStringSequenceDigest(
            runner._record_ids
        )
        runner._by_feed_id = {record.feed_event_id: record for record in records}
        runner._failure = None
        runner._rebuild_component_digest_caches()
        if any(
            record.feed_event_id != value.feed_event_id
            or record.input_digest != value.input_digest
            for record, value in zip(records, journal_events, strict=True)
        ) or state.get("schema_version") != SHADOW_COMPACT_RUNTIME_CHECKPOINT_SCHEMA:
            raise ShadowLiveError("compact checkpoint histories do not align")
        runner._require_internal_consistency()
        runner._require_runtime_bindings()
        runner._require_terminal_snapshot_exact()
        if (
            runner.journal.fingerprint != state["journal_fingerprint"]
            or runner.journal.attempt_fingerprint
            != state["attempt_fingerprint"]
            or runner.record_fingerprint != state["record_fingerprint"]
            or (
                None if not records else records[-1].record_id
            )
            != state["last_record_id"]
        ):
            raise ShadowLiveError("compact checkpoint history fingerprint differs")
        return runner

    def _stage_execution_updates(
        self,
        approvals: Iterable[RiskApprovedTradeIntent],
        events: Iterable[ExecutionEventEnvelope],
        *,
        account: AccountState,
        account_snapshot_id: str,
        account_known_at: pd.Timestamp,
        asof: pd.Timestamp,
    ) -> None:
        next_fsms = dict(self.execution_fsms)
        next_digests = dict(self._approval_digests)
        for approved in approvals:
            identity = approved.approved_intent_id
            digest = _digest(approved)
            previous = next_digests.get(identity)
            if previous is not None:
                if previous != digest:
                    raise ShadowLiveError("approved intent identity conflicts")
                continue
            if (
                approved.approval.approved_at > asof
                or account_known_at > approved.approval.approved_at
                or approved.approval.account_snapshot_id != account_snapshot_id
                or approved.approval.account_snapshot_fingerprint
                != account_state_fingerprint(account)
            ):
                raise ShadowLiveError(
                    "shadow approval is future-known or bound to another account"
                )
            next_fsms[identity] = ExecutionFSM(approved)
            next_digests[identity] = digest
        events_by_approval: dict[str, list[ExecutionEventEnvelope]] = {}
        for event in events:
            identity = event.fact.approved_intent_id
            fsm = next_fsms.get(identity)
            if fsm is None:
                raise ShadowLiveError("execution event has no registered approval")
            events_by_approval.setdefault(identity, []).append(event)
        for identity, incoming in events_by_approval.items():
            source = next_fsms[identity]
            staged = ExecutionFSM(source.approved)
            staged.store.append_batch((*source.store.events, *incoming))
            next_fsms[identity] = staged
        self.execution_fsms = next_fsms
        self._approval_digests = next_digests

    def _validate_clock_admissions(self, value: ShadowClockInput) -> None:
        lower = self._records[-1].asof if self._records else value.bar.start
        first_clock = not self._records
        for approved in value.approved_intents:
            if approved.approved_intent_id in self._approval_digests:
                continue
            approved_at = approved.approval.approved_at
            lower_ok = approved_at >= lower if first_clock else approved_at > lower
            if not lower_ok or approved_at > value.bar.end:
                raise ShadowLiveError(
                    "new approval was delivered outside its causal completed-clock batch"
                )
        for event in value.execution_events:
            lower_ok = event.known_at >= lower if first_clock else event.known_at > lower
            if not lower_ok or event.known_at > value.bar.end:
                raise ShadowLiveError(
                    "execution fact was delivered outside its causal completed-clock batch"
                )

    def _validate_instrument_mapping(self, value: ShadowClockInput) -> None:
        mapping = dict(self.protocol.instrument_mapping)
        tick_size = float(mapping["tick_size"])

        def _on_tick(price: float | None) -> bool:
            return price is None or math.isclose(
                float(price) / tick_size,
                round(float(price) / tick_size),
                rel_tol=0.0,
                abs_tol=1e-9,
            )

        tick_prices = (
            value.bar.open,
            value.bar.high,
            value.bar.low,
            value.bar.close,
            value.execution.bid,
            value.execution.ask,
        )
        if (
            value.bar.instrument_id != mapping["vendor_instrument_id"]
            or value.bar.symbol != mapping["vendor_symbol"]
            or not math.isclose(
                value.account.point_value,
                float(mapping["point_value"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or any(not _on_tick(price) for price in tick_prices)
        ):
            raise ShadowLiveError(
                "shadow market/account differs from frozen vendor instrument mapping"
            )
        for approved in value.approved_intents:
            intent_prices = (
                approved.intent.planned_entry,
                approved.intent.invalidation.price,
                *(target.price for target in approved.intent.targets),
            )
            if (
                approved.intent.instrument_id != mapping["logical_instrument_id"]
                or approved.intent.symbol != mapping["vendor_symbol"]
                or not math.isclose(
                    approved.intent.point_value,
                    float(mapping["point_value"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or any(not _on_tick(price) for price in intent_prices)
            ):
                raise ShadowLiveError(
                    "shadow approval differs from frozen logical/vendor contract mapping"
                )
        for event in value.execution_events:
            fact = event.fact
            fact_prices = (fact.price, fact.limit_price, fact.stop_price)
            if fact.top_of_book is not None:
                fact_prices += (
                    fact.top_of_book.bid,
                    fact.top_of_book.ask,
                )
            if any(not _on_tick(price) for price in fact_prices):
                raise ShadowLiveError(
                    "shadow execution fact differs from frozen tick grid"
                )

    def _record_failure(
        self,
        value: ShadowClockInput,
        exc: Exception,
    ) -> None:
        if self._failure is not None:
            return
        self._failure = ShadowFailureRecord(
            sequence=len(self._records) + 1,
            feed_event_id=value.feed_event_id,
            input_digest=value.input_digest,
            journal_fingerprint=self.journal.fingerprint,
            attempt_fingerprint=self.journal.attempt_fingerprint,
            last_record_id=(None if not self._records else self._records[-1].record_id),
            error_type=type(exc).__name__,
            error_message=(str(exc).strip() or "exception_without_message"),
        )

    def process(self, value: ShadowClockInput) -> ShadowParityRecord:
        if self._failure is not None:
            raise ShadowLiveError(
                "shadow runner is terminal after a recorded processing failure"
            )
        if not isinstance(value, ShadowClockInput):
            raise TypeError("shadow runner requires ShadowClockInput")
        self.journal.record_attempt(value)
        try:
            self._require_internal_consistency(deep=False)
            self._require_runtime_bindings()
            if self.gateway.submission_attempts:
                raise ShadowLiveError(
                    "external submission was attempted before shadow update"
                )
            appended = self.journal.append(value)
        except Exception as exc:
            self._record_failure(value, exc)
            raise
        if not appended:
            try:
                prior = self._by_feed_id.get(value.feed_event_id)
                if prior is None:
                    raise ShadowLiveError(
                        "idempotent input has no committed parity record"
                    )
                return prior
            except Exception as exc:
                self._record_failure(value, exc)
                raise
        try:
            self._validate_instrument_mapping(value)
            self._validate_clock_admissions(value)
            self._stage_execution_updates(
                value.approved_intents,
                value.execution_events,
                account=value.account,
                account_snapshot_id=value.account_snapshot_id,
                account_known_at=value.account_known_at,
                asof=value.bar.end,
            )
            snapshot = self.engine.on_bar(
                value.bar,
                execution=value.execution,
                account=value.account,
                belief_position=value.account.position,
            )
        except Exception as exc:
            self._record_failure(value, exc)
            raise
        try:
            if not isinstance(snapshot, EngineSnapshot):
                raise ShadowLiveError("shadow engine did not publish EngineSnapshot")
            if snapshot.observation.asof != value.bar.end:
                raise ShadowLiveError(
                    "shadow snapshot clock differs from completed input"
                )
            component_fields = self._component_digest_bundle(
                snapshot
            ).as_record_fields()
            record = ShadowParityRecord(
                sequence=len(self._records) + 1,
                feed_event_id=value.feed_event_id,
                asof=snapshot.observation.asof,
                input_digest=value.input_digest,
                journal_prefix_fingerprint=self.journal.fingerprint,
                protocol_id=self.protocol.protocol_id,
                external_submission_attempts=self.gateway.submission_attempts,
                **component_fields,
            )
            if self.gateway.submission_attempts:
                raise ShadowLiveError("external submission occurred during shadow update")
        except Exception as exc:
            self._record_failure(value, exc)
            raise
        self._records.append(record)
        self._record_ids.append(record.record_id)
        self._record_sequence_digest.append(record.record_id)
        self._by_feed_id[value.feed_event_id] = record
        return record


def audit_shadow_parity(
    expected: ShadowLiveRunner | Sequence[ShadowParityRecord],
    actual: ShadowLiveRunner | Sequence[ShadowParityRecord],
) -> ShadowParityAudit:
    """Compare records plus terminal journal/failure/gateway state.

    Record-only inputs remain useful for locating field drift but can never
    establish exact live/replay parity because they omit terminal state.
    """

    expected_runner = expected if isinstance(expected, ShadowLiveRunner) else None
    actual_runner = actual if isinstance(actual, ShadowLiveRunner) else None
    expected_values = tuple(
        expected_runner.records if expected_runner is not None else expected
    )
    actual_values = tuple(actual_runner.records if actual_runner is not None else actual)
    mismatches: list[ShadowParityMismatch] = []
    field_names = tuple(item.name for item in fields(ShadowParityRecord))
    for index in range(max(len(expected_values), len(actual_values))):
        expected_record = expected_values[index] if index < len(expected_values) else None
        actual_record = actual_values[index] if index < len(actual_values) else None
        if expected_record is None or actual_record is None:
            record = expected_record or actual_record
            assert record is not None
            mismatches.append(
                ShadowParityMismatch(
                    sequence=index + 1,
                    feed_event_id=record.feed_event_id,
                    fields=("missing_record",),
                )
            )
            continue
        changed = tuple(
            name
            for name in field_names
            if getattr(expected_record, name) != getattr(actual_record, name)
        )
        if changed:
            mismatches.append(
                ShadowParityMismatch(
                    sequence=index + 1,
                    feed_event_id=expected_record.feed_event_id,
                    fields=changed,
                )
            )
    terminal_fields: list[str] = []
    coverage_complete = False
    if expected_runner is None or actual_runner is None:
        terminal_fields.append("runner_terminal_state_unavailable")
    else:
        def _runner_invariant_violations(
            runner: ShadowLiveRunner,
        ) -> tuple[str, ...]:
            violations: list[str] = []
            try:
                runner._require_internal_consistency()
                runner._require_runtime_bindings()
            except Exception:
                violations.append("journal_or_runtime_consistency")
            events = runner.journal.events
            runtime_fingerprint = runner._runtime_bindings_fingerprint
            prefix_digest = _CanonicalStringSequenceDigest()
            for index, record in enumerate(runner.records):
                if index >= len(events):
                    violations.append("record_without_journal_input")
                    break
                value = events[index]
                prefix_digest.append(value.input_digest)
                expected_prefix = prefix_digest.fingerprint
                if record.sequence != index + 1:
                    violations.append("record_sequence")
                if record.feed_event_id != value.feed_event_id:
                    violations.append("record_feed_event_id")
                if record.asof != value.bar.end:
                    violations.append("record_asof")
                if record.input_digest != value.input_digest:
                    violations.append("record_input_digest")
                if record.journal_prefix_fingerprint != expected_prefix:
                    violations.append("record_journal_prefix")
                if record.protocol_id != runner.protocol.protocol_id:
                    violations.append("record_protocol_id")
                if record.runtime_bindings_fingerprint != runtime_fingerprint:
                    violations.append("record_runtime_bindings")
                if record.external_submission_attempts != 0:
                    violations.append("record_external_submission")
            failure = runner.failure
            if failure is None:
                if len(runner.records) != len(events):
                    violations.append("healthy_record_journal_coverage")
            else:
                if failure.sequence != len(runner.records) + 1:
                    violations.append("failure_sequence")
                expected_last = (
                    None if not runner.records else runner.records[-1].record_id
                )
                if failure.last_record_id != expected_last:
                    violations.append("failure_last_record_id")
                if failure.journal_fingerprint != runner.journal.fingerprint:
                    violations.append("failure_journal_fingerprint")
                if (
                    failure.attempt_fingerprint
                    != runner.journal.attempt_fingerprint
                ):
                    violations.append("failure_attempt_fingerprint")
                if (
                    not runner.journal.attempts
                    or runner.journal.attempts[-1].feed_event_id
                    != failure.feed_event_id
                    or runner.journal.attempts[-1].input_digest
                    != failure.input_digest
                ):
                    violations.append("failure_input_digest")
            if runner.gateway.submission_attempts and failure is None:
                violations.append("gateway_attempt_without_failure")
            if runner.records:
                final_record = runner.records[-1]
                snapshot = runner.engine.last_snapshot
                if not isinstance(snapshot, EngineSnapshot):
                    violations.append("engine_last_snapshot_missing")
                else:
                    market = snapshot.market_snapshot
                    terminal_values = runner._component_digest_bundle(
                        snapshot
                    ).as_record_fields()
                    violations.extend(
                        f"terminal_{name}"
                        for name, value in terminal_values.items()
                        if getattr(final_record, name) != value
                    )
                    if (
                        runner._component_digest_version
                        == SHADOW_COMPONENT_DIGEST_VERSION
                        and market is not None
                    ):
                        try:
                            from .market_state import (
                                replay_atomic_market_snapshot,
                            )

                            replayed_market = replay_atomic_market_snapshot(
                                runner.engine.observer.audit_store.events(),
                                semantic_registry_identity=(
                                    market.semantic_registry_identity
                                ),
                                semantic_version=market.semantic_version,
                                expected_timeframes=(
                                    market.timeframe_states.keys()
                                ),
                            )
                            if (
                                replayed_market.replay_payload()
                                != market.replay_payload()
                            ):
                                violations.append(
                                    "market_full_replay_payload"
                                )
                        except Exception:
                            violations.append("market_full_replay_payload")
            return tuple(dict.fromkeys(violations))

        for side, runner in (
            ("expected", expected_runner),
            ("actual", actual_runner),
        ):
            terminal_fields.extend(
                f"{side}_runner_invariant:{name}"
                for name in _runner_invariant_violations(runner)
            )
        if expected_runner is actual_runner:
            terminal_fields.append("non_independent_runner_alias")
        terminal_pairs = {
            "journal_fingerprint": (
                expected_runner.journal.fingerprint,
                actual_runner.journal.fingerprint,
            ),
            "attempt_fingerprint": (
                expected_runner.journal.attempt_fingerprint,
                actual_runner.journal.attempt_fingerprint,
            ),
            "failure_id": (
                None
                if expected_runner.failure is None
                else expected_runner.failure.failure_id,
                None
                if actual_runner.failure is None
                else actual_runner.failure.failure_id,
            ),
            "external_submission_attempts": (
                expected_runner.gateway.submission_attempts,
                actual_runner.gateway.submission_attempts,
            ),
            "protocol_id": (
                expected_runner.protocol.protocol_id,
                actual_runner.protocol.protocol_id,
            ),
            "runtime_bindings": (
                expected_runner.runtime_bindings,
                actual_runner.runtime_bindings,
            ),
        }
        terminal_fields.extend(
            name for name, (left, right) in terminal_pairs.items() if left != right
        )
        coverage_complete = (
            expected_runner.failure is None
            and actual_runner.failure is None
            and expected_runner.gateway.submission_attempts == 0
            and actual_runner.gateway.submission_attempts == 0
            and len(expected_runner.records) > 0
            and len(actual_runner.records) > 0
            and len(expected_runner.records) == len(expected_runner.journal)
            and len(actual_runner.records) == len(actual_runner.journal)
        )
    terminal_fields = list(dict.fromkeys(terminal_fields))
    parity_exact = (
        len(expected_values) == len(actual_values)
        and not mismatches
        and not terminal_fields
    )
    return ShadowParityAudit(
        expected_records=len(expected_values),
        actual_records=len(actual_values),
        mismatches=tuple(mismatches),
        terminal_fields=tuple(terminal_fields),
        exact_match=parity_exact,
        coverage_complete=coverage_complete,
        gate_pass=(parity_exact and coverage_complete),
    )


def replay_shadow_journal(
    journal: ShadowInputJournal,
    *,
    engine_factory: Callable[[], ContinuousSMCEngine],
    protocol: ShadowLiveProtocol,
    runtime_bindings: Mapping[str, str] | Sequence[tuple[str, str]],
) -> ShadowLiveRunner:
    if not isinstance(journal, ShadowInputJournal):
        raise TypeError("shadow replay requires ShadowInputJournal")
    runner = ShadowLiveRunner(
        engine=engine_factory(),
        protocol=protocol,
        runtime_bindings=runtime_bindings,
    )
    journal.require_consistent()
    for value in journal.attempts:
        try:
            runner.process(value)
        except Exception:
            if runner.failure is None:
                raise
            break
    return runner


__all__ = [
    "NullExecutionGateway",
    "SHADOW_COMPONENT_DIGEST_VERSION",
    "SHADOW_LEGACY_COMPONENT_DIGEST_VERSION",
    "SHADOW_LIVE_AUTHORITY",
    "SHADOW_LIVE_SCHEMA_VERSION",
    "SHADOW_RECORD_FIELDS",
    "SHADOW_RUNTIME_BINDING_KEYS",
    "ShadowClockInput",
    "ShadowComponentDigestBundle",
    "ShadowInputJournal",
    "ShadowFailureRecord",
    "ShadowLiveError",
    "ShadowLiveProtocol",
    "ShadowLiveRunner",
    "ShadowParityAudit",
    "ShadowParityMismatch",
    "ShadowParityRecord",
    "audit_shadow_parity",
    "load_shadow_live_protocol",
    "replay_shadow_journal",
    "shadow_runtime_bindings_from_model_config",
]
