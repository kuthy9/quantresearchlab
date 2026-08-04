"""Minimal, physically separate future evidence for diagnostic primitives.

The recorder stores market evolution only.  It never embeds the causal decision
packet and never records action, outcome/success, PnL, MFE/MAE, target-hit, or
stop-hit fields.  Frozen draw/source lifecycle changes remain valid evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from .decision_trace import (
    decision_packet_payload_sha256,
    decision_packet_sha256,
    read_verified_decision_packet,
)
from .market_clock import scheduled_gap_kind
from .model import (
    Bar,
    Candle,
    DealingRangeState,
    DisplacementTransitionObservation,
    EntryLocationState,
    FairValueGapState,
    ManipulationState,
    MicroBOSReference,
    OrderBlockState,
    PathSequenceState,
    QualifiedReacceptanceState,
    Timeframe,
    aware_timestamp,
    content_hash,
    to_primitive,
)


PATH_EVIDENCE_FORMAT_VERSION = 1
PATH_EVIDENCE_ARTIFACT = "primitive_typed_future_path_evidence"
PATH_EVIDENCE_HARD_BOUNDARIES = frozenset(
    {
        "contract_change_reset",
        "data_gap_reset",
        "data_anomaly",
        "tick_size_mismatch",
    }
)
PATH_EVIDENCE_FINAL_REASONS = frozenset(
    {"deadline", "right_boundary", *PATH_EVIDENCE_HARD_BOUNDARIES}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEYS = frozenset(
    {
        "action",
        "selected_action",
        "final_action",
        "action_utilities",
        "pnl",
        "pnl_r",
        "net_pnl",
        "gross_pnl",
        "mfe",
        "mfe_r",
        "mae",
        "mae_r",
        "success",
        "trade_success",
        "outcome",
        "trade_outcome",
        "target_hit",
        "target_touched",
        "stop_hit",
        "stop_touched",
        "invalidation_hit",
    }
)
_FORBIDDEN_VALUES = frozenset(
    {
        "pnl",
        "mfe",
        "mae",
        "success",
        "outcome",
        "target_hit",
        "target_touched",
        "target_reached",
        "stop_hit",
        "stop_touched",
        "invalidation_hit",
        "trade_won",
        "trade_lost",
    }
)
_REGISTERED_TYPED_OUTCOMES = frozenset(
    {
        "aligned",
        "opposed",
        "simultaneous_unknown",
        "ambiguous_same_clock",
    }
)
_TYPED_TRANSITION_KEYS = frozenset(
    {
        "family",
        "entity_id",
        "revision_id",
        "lifecycle",
        "observed_at",
        "reason",
        "state",
    }
)
_TYPED_TRANSITION_SPECS = {
    "displacement": {
        "state_type": DisplacementTransitionObservation,
        "identity": "entity_id",
        "clock": "observed_at",
        "reason": "reason",
        "lifecycles": frozenset(
            {"started", "active", "exhausted", "censored"}
        ),
        "required": frozenset(
            {
                "transition_id",
                "entity_id",
                "lifecycle",
                "reason",
                "observed_at",
                "direction",
            }
        ),
    },
    "fvg_boundary": {
        "state_type": FairValueGapState,
        "identity": "fvg_id",
        "clock": "last_updated_at",
        "reason": "transition_reason",
        "lifecycles": frozenset(
            {"open", "partial", "mitigated", "invalidated"}
        ),
    },
    "order_block_boundary": {
        "state_type": OrderBlockState,
        "identity": "order_block_id",
        "clock": "last_updated_at",
        "reason": "transition_reason",
        "lifecycles": frozenset(
            {"created", "untested", "mitigated", "failed"}
        ),
    },
    "range_boundary": {
        "state_type": DealingRangeState,
        "identity": "range_id",
        "clock": "last_updated_at",
        "reason": "transition_reason",
        "lifecycles": frozenset({"forming", "mature", "broken"}),
    },
    "manipulation_boundary": {
        "state_type": ManipulationState,
        "identity": "manipulation_id",
        "clock": "last_updated_at",
        "reason": "transition_reason",
        "lifecycles": frozenset(
            {"swept", "reaccepted", "accepted_outside"}
        ),
    },
    "entry_path_boundary": {
        "state_type": PathSequenceState,
        "identity": "sequence_id",
        "clock": "last_updated_at",
        "reason": "transition_reason",
        "lifecycles": frozenset({"active", "closed", "censored"}),
    },
    "reacceptance_boundary": {
        "state_type": QualifiedReacceptanceState,
        "identity": "reacceptance_id",
        "clock": "last_updated_at",
        "reason": "transition_reason",
        "lifecycles": frozenset(
            {"left", "reclaimed", "held", "failed", "censored"}
        ),
    },
    "entry_location": {
        "state_type": EntryLocationState,
        "identity": "location_id",
        "clock": "last_updated_at",
        "reason": "transition_reason",
        "lifecycles": frozenset(
            {"approaching", "in_zone", "rejected", "left"}
        ),
    },
    "qualified_reacceptance": {
        "state_type": QualifiedReacceptanceState,
        "identity": "reacceptance_id",
        "clock": "last_updated_at",
        "reason": "transition_reason",
        "lifecycles": frozenset(
            {"left", "reclaimed", "held", "failed", "censored"}
        ),
    },
    "entry_path": {
        "state_type": PathSequenceState,
        "identity": "sequence_id",
        "clock": "last_updated_at",
        "reason": "transition_reason",
        "lifecycles": frozenset({"active", "closed", "censored"}),
    },
    "micro_bos": {
        "state_type": MicroBOSReference,
        "identity": "reference_id",
        "clock": "resolved_at",
        "reason": "outcome",
        "lifecycles": frozenset({"qualified", "observed"}),
        "required": frozenset(
            {
                "reference_id",
                "resolved_at",
                "outcome",
                "qualified",
            }
        ),
    },
}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _sha(value: Any, name: str) -> str:
    value = _text(value, name)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _texts(values: Sequence[Any], name: str) -> tuple[str, ...]:
    result = tuple(_text(value, name) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicates")
    return result


@dataclass(frozen=True)
class PrimitivePathEvidenceQuery:
    """Generic query identity and transition selectors; not a controller."""

    query_id: str
    issue: str
    primitive_name: str
    formula_version: str
    definition_hash: str
    relevant_transition_families: tuple[str, ...] = ()
    relevant_entity_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("query_id", "issue", "primitive_name", "formula_version"):
            object.__setattr__(self, name, _text(getattr(self, name), f"query.{name}"))
        object.__setattr__(
            self,
            "definition_hash",
            _sha(self.definition_hash, "query.definition_hash"),
        )
        object.__setattr__(
            self,
            "relevant_transition_families",
            _texts(self.relevant_transition_families, "query.transition_family"),
        )
        object.__setattr__(
            self,
            "relevant_entity_ids",
            _texts(self.relevant_entity_ids, "query.entity_id"),
        )

    @classmethod
    def from_proposal(
        cls,
        proposal: Any,
        *,
        relevant_transition_families: Sequence[str] = (),
        relevant_entity_ids: Sequence[str] = (),
    ) -> "PrimitivePathEvidenceQuery":
        issue = getattr(proposal.issue, "value", proposal.issue)
        return cls(
            query_id=proposal.proposal_id,
            issue=str(issue),
            primitive_name=proposal.primitive_name,
            formula_version=proposal.formula_version,
            definition_hash=proposal.definition_hash,
            relevant_transition_families=tuple(relevant_transition_families),
            relevant_entity_ids=tuple(relevant_entity_ids),
        )


@dataclass(frozen=True)
class PrimitivePathEvidenceIdentity:
    decision_hash: str
    decision_packet_hash: str
    decision_packet_sha256: str
    decision_asof: pd.Timestamp
    hypothesis_key: str
    setup_id: str
    entry_location_id: str
    entry_path_id: str
    deadline: pd.Timestamp
    symbol: str
    instrument_id: int
    query: PrimitivePathEvidenceQuery

    def __post_init__(self) -> None:
        for name in ("decision_hash", "decision_packet_hash", "decision_packet_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), f"identity.{name}"))
        for name in ("hypothesis_key", "setup_id", "entry_location_id", "entry_path_id", "symbol"):
            object.__setattr__(self, name, _text(getattr(self, name), f"identity.{name}"))
        object.__setattr__(
            self,
            "decision_asof",
            aware_timestamp(self.decision_asof, name="path_evidence.decision_asof"),
        )
        object.__setattr__(
            self,
            "deadline",
            aware_timestamp(self.deadline, name="path_evidence.deadline"),
        )
        if self.deadline <= self.decision_asof:
            raise ValueError("path-evidence deadline must follow the decision")
        if type(self.instrument_id) is not int or self.instrument_id < 0:
            raise ValueError("path-evidence instrument id is invalid")
        if not isinstance(self.query, PrimitivePathEvidenceQuery):
            raise ValueError("path-evidence query is invalid")


@dataclass(frozen=True)
class PrimitivePathEvidencePoint:
    observed_at: pd.Timestamp
    completed_m1_bar: Mapping[str, Any]
    events_added: tuple[Mapping[str, Any], ...]
    events_ended: tuple[Mapping[str, Any], ...]
    events_invalidated: tuple[Mapping[str, Any], ...]
    typed_state_transitions: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_at",
            aware_timestamp(self.observed_at, name="path_evidence.point.observed_at"),
        )
        object.__setattr__(self, "completed_m1_bar", dict(self.completed_m1_bar))
        for name in ("events_added", "events_ended", "events_invalidated", "typed_state_transitions"):
            object.__setattr__(self, name, tuple(dict(item) for item in getattr(self, name)))


@dataclass(frozen=True)
class PrimitivePathBoundaryDelta:
    """Old-setup terminal transitions without any new-contract OHLC."""

    observed_at: pd.Timestamp
    reason: str
    events_ended: tuple[Mapping[str, Any], ...] = ()
    events_invalidated: tuple[Mapping[str, Any], ...] = ()
    typed_state_transitions: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_at",
            aware_timestamp(self.observed_at, name="path_evidence.boundary.observed_at"),
        )
        if self.reason not in PATH_EVIDENCE_HARD_BOUNDARIES:
            raise ValueError("path-evidence boundary reason is invalid")
        for name in ("events_ended", "events_invalidated", "typed_state_transitions"):
            object.__setattr__(self, name, tuple(dict(item) for item in getattr(self, name)))


@dataclass(frozen=True)
class PrimitivePathEvidence:
    identity: PrimitivePathEvidenceIdentity
    started_at: pd.Timestamp
    finalized_at: pd.Timestamp
    finalization_reason: str
    points: tuple[PrimitivePathEvidencePoint, ...]
    boundary_delta: PrimitivePathBoundaryDelta | None
    evidence_hash: str
    format_version: int = PATH_EVIDENCE_FORMAT_VERSION
    artifact: str = PATH_EVIDENCE_ARTIFACT

    def __post_init__(self) -> None:
        if self.format_version != PATH_EVIDENCE_FORMAT_VERSION or self.artifact != PATH_EVIDENCE_ARTIFACT:
            raise ValueError("unsupported path-evidence artifact")
        if not isinstance(self.identity, PrimitivePathEvidenceIdentity):
            raise ValueError("path-evidence identity is invalid")
        object.__setattr__(self, "started_at", aware_timestamp(self.started_at, name="path_evidence.started_at"))
        object.__setattr__(self, "finalized_at", aware_timestamp(self.finalized_at, name="path_evidence.finalized_at"))
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "evidence_hash", _sha(self.evidence_hash, "path_evidence.evidence_hash"))
        if self.started_at != self.identity.decision_asof:
            raise ValueError("path evidence must start at its frozen decision")
        if self.finalization_reason not in PATH_EVIDENCE_FINAL_REASONS:
            raise ValueError("path-evidence finalization reason is invalid")
        if not self.started_at <= self.finalized_at <= self.identity.deadline:
            raise ValueError("path-evidence final clock is outside its boundary")
        if self.boundary_delta is not None and not isinstance(self.boundary_delta, PrimitivePathBoundaryDelta):
            raise ValueError("path-evidence boundary delta is invalid")


def _reject_forbidden(value: Any, prefix: str = "") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = re.sub(r"[^a-z0-9]+", "_", str(raw_key).casefold()).strip("_")
            path = f"{prefix}.{raw_key}" if prefix else str(raw_key)
            typed_structural_outcome = bool(
                key == "outcome"
                and path.endswith(".state.outcome")
                and item in _REGISTERED_TYPED_OUTCOMES
            )
            if (
                (key in _FORBIDDEN_KEYS and not typed_structural_outcome)
                or key.startswith(
                    (
                        "action_",
                        "pnl_",
                        "mfe_",
                        "mae_",
                        "success_",
                        "target_hit",
                        "target_touched",
                        "stop_hit",
                        "stop_touched",
                        "invalidation_hit",
                    )
                )
                or key.endswith(("_action", "_pnl", "_mfe", "_mae", "_success"))
            ):
                raise ValueError(f"path evidence contains forbidden field: {path}")
            _reject_forbidden(item, path)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_forbidden(item, f"{prefix}[{index}]")
    elif isinstance(value, str):
        normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
        if normalized in _FORBIDDEN_VALUES:
            raise ValueError(
                f"path evidence contains forbidden result value: {prefix}"
            )


def _reject_future_clocks(value: Any, cutoff: pd.Timestamp) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).casefold()
            clock_key = key in {"start", "end", "cutoff", "clock", "deadline"} or key.endswith(
                ("_at", "_time", "_cutoff", "_clock", "_start", "_end")
            )
            if clock_key and item is not None:
                clock = aware_timestamp(item, name=f"path_evidence.{raw_key}")
                if clock > cutoff:
                    raise ValueError("path evidence contains a future clock")
            _reject_future_clocks(item, cutoff)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_future_clocks(item, cutoff)


def _record_clock(record: Mapping[str, Any], bucket: str) -> pd.Timestamp:
    candidates = {
        "events_added": ("observed_at",),
        "events_ended": ("ended_at",),
        "events_invalidated": ("ended_at", "observed_at"),
        "typed_state_transitions": ("observed_at",),
    }[bucket]
    raw = next((record.get(key) for key in candidates if record.get(key) is not None), None)
    if raw is None:
        raise ValueError(f"{bucket} record has no registered clock")
    return aware_timestamp(raw, name=f"path_evidence.{bucket}.clock")


def _validate_typed_transition(record: Mapping[str, Any]) -> None:
    if set(record) != _TYPED_TRANSITION_KEYS:
        raise ValueError("typed transition top-level schema is invalid")
    family = record.get("family")
    spec = _TYPED_TRANSITION_SPECS.get(family)
    if spec is None:
        raise ValueError("typed transition family is not registered")
    entity_id = _text(record.get("entity_id"), "typed_transition.entity_id")
    revision_id = _text(
        record.get("revision_id"),
        "typed_transition.revision_id",
    )
    lifecycle = _text(
        record.get("lifecycle"),
        "typed_transition.lifecycle",
    )
    if lifecycle not in spec["lifecycles"]:
        raise ValueError("typed transition lifecycle is invalid")
    reason = record.get("reason")
    if reason is not None:
        _text(reason, "typed_transition.reason")
    state = record.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("typed transition state is missing")
    expected_state_keys = {
        field.name for field in fields(spec["state_type"])
    }
    if set(state) != expected_state_keys:
        raise ValueError(
            "typed transition state does not match its production schema"
        )
    required = set(
        spec.get(
            "required",
            {
                spec["identity"],
                "lifecycle",
                spec["clock"],
                spec["reason"],
            },
        )
    )
    if not required <= set(state):
        raise ValueError("typed transition state schema is incomplete")
    if state.get(spec["identity"]) != entity_id:
        raise ValueError("typed transition state identity disagrees")
    observed_at = aware_timestamp(
        record.get("observed_at"),
        name="typed_transition.observed_at",
    )
    state_clock = aware_timestamp(
        state.get(spec["clock"]),
        name=f"typed_transition.state.{spec['clock']}",
    )
    if state_clock != observed_at:
        raise ValueError("typed transition state clock disagrees")
    if state.get(spec["reason"]) != reason:
        raise ValueError("typed transition reason disagrees with its state")

    if family == "micro_bos":
        if (
            type(state.get("qualified")) is not bool
            or state.get("outcome") not in _REGISTERED_TYPED_OUTCOMES
            or lifecycle
            != ("qualified" if state["qualified"] else "observed")
            or revision_id != f"micro_bos:{entity_id}"
        ):
            raise ValueError("micro-BOS transition schema is invalid")
        return

    if state.get("lifecycle") != lifecycle:
        raise ValueError("typed transition lifecycle disagrees with its state")
    if family == "displacement":
        if (
            revision_id != state.get("transition_id")
            or state.get("direction") not in {"long", "short"}
        ):
            raise ValueError("displacement transition schema is invalid")
        return

    revision_prefix = f"{family}:{entity_id}:{lifecycle}:"
    if not revision_id.startswith(revision_prefix):
        raise ValueError("typed transition revision identity is invalid")
    revision_clock = aware_timestamp(
        revision_id[len(revision_prefix):],
        name="typed_transition.revision_clock",
    )
    if revision_clock != observed_at:
        raise ValueError("typed transition revision clock disagrees")


def _records(
    values: Sequence[Mapping[str, Any]],
    bucket: str,
    observed_at: pd.Timestamp,
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError(f"{bucket} must contain objects")
        item = to_primitive(value)
        _reject_forbidden(item)
        _reject_future_clocks(item, observed_at)
        if _record_clock(item, bucket) != observed_at:
            raise ValueError(f"{bucket} record belongs to another M1 bar")
        if bucket == "typed_state_transitions":
            _validate_typed_transition(item)
        result.append(dict(item))
    return tuple(result)


def _relevant(transition: Mapping[str, Any], query: PrimitivePathEvidenceQuery) -> bool:
    return bool(
        transition.get("family") in query.relevant_transition_families
        or transition.get("entity_id") in query.relevant_entity_ids
    )


def _boundary_relevant(
    record: Mapping[str, Any],
    identity: PrimitivePathEvidenceIdentity,
) -> bool:
    registered = {
        identity.setup_id,
        identity.entry_location_id,
        identity.entry_path_id,
        *identity.query.relevant_entity_ids,
    }
    direct = {
        value
        for value in (
            record.get("event_id"),
            record.get("entity_id"),
            record.get("revision_id"),
        )
        if isinstance(value, str)
    }
    sources = record.get("source_ids")
    if not isinstance(sources, (tuple, list)):
        sources = ()
    source_ids = {value for value in sources if isinstance(value, str)}
    return bool(registered.intersection(direct | source_ids))


def _bar_payload(value: Bar | Candle) -> dict[str, Any]:
    if isinstance(value, Bar):
        raw = to_primitive(value)
        raw["end"] = value.end.isoformat()
        return raw
    if not isinstance(value, Candle) or value.timeframe is not Timeframe.M1 or not value.complete:
        raise ValueError("path evidence requires a completed 1m Bar or Candle")
    if value.end - value.start != pd.Timedelta(minutes=1):
        raise ValueError("path-evidence candle is not one minute")
    raw = to_primitive(value)
    return {
        key: raw[key]
        for key in ("start", "end", "open", "high", "low", "close", "volume", "symbol", "instrument_id")
    } | {
        "synthetic_no_trade": value.synthetic_minutes == 1,
        "data_gap_before_minutes": 0,
    }


def _packet_identity(
    packet: Mapping[str, Any],
    packet_file_sha256: str,
    hypothesis_key: str,
    query: PrimitivePathEvidenceQuery,
) -> PrimitivePathEvidenceIdentity:
    if packet_file_sha256 != decision_packet_payload_sha256(packet):
        raise ValueError(
            "decision packet does not use its registered serialization"
        )
    if packet.get("future_path") != {
        "included": False,
        "revealed": False,
        "storage": "physically_separate_artifact",
    }:
        raise ValueError("decision packet is not sealed from future evidence")
    decision_asof = aware_timestamp(packet.get("decision_asof"), name="packet.decision_asof")
    maximum_market_time = aware_timestamp(packet.get("maximum_market_time"), name="packet.maximum_market_time")
    if maximum_market_time > decision_asof:
        raise ValueError("decision packet contains post-decision market data")
    if packet.get("audit_hypothesis_key") != hypothesis_key:
        raise ValueError("path query differs from the packet audit hypothesis")
    trace = packet.get("decision_trace")
    if (
        not isinstance(trace, Mapping)
        or trace.get("decision_hash") != packet.get("decision_hash")
        or trace.get("decision_asof") != packet.get("decision_asof")
        or trace.get("selected_hypothesis_key") != hypothesis_key
        or trace.get("future_path_included") is not False
    ):
        raise ValueError("path evidence requires an exact causal decision trace")
    belief = packet.get("belief_t")
    hypotheses = belief.get("hypotheses") if isinstance(belief, Mapping) else None
    hypothesis = hypotheses.get(hypothesis_key) if isinstance(hypotheses, Mapping) else None
    if not isinstance(hypothesis, Mapping):
        raise ValueError("path query hypothesis is absent")
    sequence, plan = hypothesis.get("sequence"), hypothesis.get("plan")
    if not isinstance(sequence, Mapping) or not isinstance(plan, Mapping):
        raise ValueError("path evidence requires a frozen sequence and plan")
    setup_values = (
        sequence.get("setup_id"),
        hypothesis.get("setup_context_id"),
        plan.get("setup_id"),
    )
    location_values = (
        hypothesis.get("entry_location_id"),
        plan.get("entry_location_id"),
    )
    if (
        any(not isinstance(item, str) or not item for item in setup_values)
        or len(set(setup_values)) != 1
        or any(
            not isinstance(item, str) or not item
            for item in location_values
        )
        or len(set(location_values)) != 1
    ):
        raise ValueError("frozen setup/location identities disagree")
    from .ai_review import ReviewIssue, compute_primitive

    try:
        registered_value = compute_primitive(
            ReviewIssue(query.issue),
            packet,
            hypothesis_key=hypothesis_key,
        )
    except ValueError as error:
        raise ValueError(
            "path-evidence query is not a registered causal primitive"
        ) from error
    if (
        registered_value.primitive_name != query.primitive_name
        or registered_value.formula_version != query.formula_version
        or registered_value.definition_hash != query.definition_hash
        or registered_value.setup_id != setup_values[0]
        or registered_value.entry_location_id != location_values[0]
        or registered_value.entry_path_id != plan.get("entry_path_id")
    ):
        raise ValueError(
            "path-evidence query or typed setup identity changed"
        )
    if packet.get("audit_context") != {
        "scenario": "sealed_path_audit",
        "setup_id": setup_values[0],
        "future_present": False,
    }:
        raise ValueError(
            "path evidence requires an exact sealed audit context"
        )
    observation = packet.get("observation_t")
    if not isinstance(observation, Mapping) or aware_timestamp(observation.get("asof"), name="packet.observation.asof") != decision_asof:
        raise ValueError("decision packet observation identity is invalid")
    return PrimitivePathEvidenceIdentity(
        decision_hash=packet.get("decision_hash"),
        decision_packet_hash=packet.get("packet_hash"),
        decision_packet_sha256=packet_file_sha256,
        decision_asof=decision_asof,
        hypothesis_key=hypothesis_key,
        setup_id=setup_values[0],
        entry_location_id=location_values[0],
        entry_path_id=plan.get("entry_path_id"),
        deadline=plan.get("deadline"),
        symbol=observation.get("symbol"),
        instrument_id=observation.get("instrument_id"),
        query=query,
    )


def _payload(evidence: PrimitivePathEvidence) -> dict[str, Any]:
    return to_primitive(
        {
            "format_version": evidence.format_version,
            "artifact": evidence.artifact,
            "identity": evidence.identity,
            "started_at": evidence.started_at,
            "finalized_at": evidence.finalized_at,
            "finalization_reason": evidence.finalization_reason,
            "points": evidence.points,
            "boundary_delta": evidence.boundary_delta,
        }
    )


def _validate_point(
    point: PrimitivePathEvidencePoint,
    identity: PrimitivePathEvidenceIdentity,
    expected_start: pd.Timestamp,
) -> tuple[pd.Timestamp, bool]:
    bar = point.completed_m1_bar
    expected_fields = {
        "start", "end", "open", "high", "low", "close", "volume", "symbol",
        "instrument_id", "synthetic_no_trade", "data_gap_before_minutes",
    }
    if set(bar) != expected_fields:
        raise ValueError("path-evidence M1 bar schema is invalid")
    start = aware_timestamp(bar["start"], name="path_evidence.bar.start")
    end = aware_timestamp(bar["end"], name="path_evidence.bar.end")
    if start < expected_start or end != start + pd.Timedelta(minutes=1) or point.observed_at != end:
        raise ValueError("path-evidence M1 bars are not contiguous and completed")
    if end > identity.deadline or (bar["symbol"], bar["instrument_id"]) != (identity.symbol, identity.instrument_id):
        raise ValueError("path evidence crosses its deadline or contract")
    if (
        type(bar["instrument_id"]) is not int
        or type(bar["synthetic_no_trade"]) is not bool
        or type(bar["data_gap_before_minutes"]) is not int
        or any(
            type(bar[name]) not in {int, float}
            or type(bar[name]) is bool
            for name in ("open", "high", "low", "close", "volume")
        )
    ):
        raise ValueError("path-evidence M1 metadata types are invalid")
    checked = Bar(
        start=start,
        open=bar["open"], high=bar["high"], low=bar["low"], close=bar["close"], volume=bar["volume"],
        symbol=bar["symbol"], instrument_id=bar["instrument_id"],
        synthetic_no_trade=bar["synthetic_no_trade"], data_gap_before_minutes=bar["data_gap_before_minutes"],
    )
    for bucket in ("events_added", "events_ended", "events_invalidated", "typed_state_transitions"):
        for record in getattr(point, bucket):
            _reject_forbidden(record)
            _reject_future_clocks(record, end)
            if _record_clock(record, bucket) != end:
                raise ValueError(f"{bucket} record belongs to another M1 bar")
            if bucket == "typed_state_transitions":
                _validate_typed_transition(record)
    if any(not _relevant(item, identity.query) for item in point.typed_state_transitions):
        raise ValueError("path evidence contains an unrelated typed transition")
    unexplained_gap = bool(
        start > expected_start
        and scheduled_gap_kind(expected_start, start) is None
    )
    return end, bool(unexplained_gap or checked.data_gap_before_minutes)


def _validate_boundary(
    boundary: PrimitivePathBoundaryDelta,
    evidence: PrimitivePathEvidence,
) -> None:
    if (
        boundary.observed_at != evidence.finalized_at
        or boundary.reason != evidence.finalization_reason
    ):
        raise ValueError("path-evidence terminal boundary identity is invalid")
    for bucket in ("events_ended", "events_invalidated", "typed_state_transitions"):
        for record in getattr(boundary, bucket):
            _reject_forbidden(record)
            _reject_future_clocks(record, boundary.observed_at)
            if _record_clock(record, bucket) != boundary.observed_at:
                raise ValueError(f"boundary {bucket} record has the wrong clock")
            if bucket == "typed_state_transitions":
                _validate_typed_transition(record)
    for bucket in ("events_ended", "events_invalidated", "typed_state_transitions"):
        if any(
            not _boundary_relevant(item, evidence.identity)
            for item in getattr(boundary, bucket)
        ):
            raise ValueError("path boundary contains an unrelated transition")


def verify_primitive_path_evidence(
    evidence: PrimitivePathEvidence,
    *,
    decision_packet_path: str | Path | None = None,
) -> PrimitivePathEvidence:
    if not isinstance(evidence, PrimitivePathEvidence):
        raise ValueError("path evidence has the wrong type")
    raw = _payload(evidence)
    _reject_forbidden(raw)
    if content_hash(raw) != evidence.evidence_hash:
        raise ValueError("path-evidence content hash is invalid")
    expected = evidence.identity.decision_asof
    gap_indices: list[int] = []
    for index, point in enumerate(evidence.points):
        expected, crossed_gap = _validate_point(point, evidence.identity, expected)
        if crossed_gap:
            gap_indices.append(index)
    if expected > evidence.finalized_at:
        raise ValueError("path evidence finalizes before its last completed bar")
    if (
        evidence.finalization_reason == "right_boundary"
        and expected != evidence.finalized_at
    ):
        raise ValueError(
            "right-boundary evidence omitted completed market time"
        )
    if gap_indices and (
        gap_indices != [len(evidence.points) - 1]
        or evidence.finalization_reason != "data_gap_reset"
    ):
        raise ValueError("a data-gap bar must be the final recorded path point")
    if evidence.finalization_reason == "deadline" and evidence.finalized_at != evidence.identity.deadline:
        raise ValueError("deadline evidence must end at the frozen deadline")
    if (
        evidence.finalization_reason == "deadline"
        and expected < evidence.identity.deadline
        and scheduled_gap_kind(
            expected,
            evidence.identity.deadline,
        )
        is None
    ):
        raise ValueError(
            "deadline evidence omitted registered trading minutes"
        )
    if evidence.boundary_delta is not None:
        _validate_boundary(evidence.boundary_delta, evidence)
    elif evidence.finalization_reason == "contract_change_reset":
        raise ValueError("contract-change evidence requires a terminal boundary delta")
    if decision_packet_path is not None:
        packet_path = Path(decision_packet_path)
        packet = read_verified_decision_packet(packet_path)
        linked = _packet_identity(
            packet,
            decision_packet_sha256(packet_path),
            evidence.identity.hypothesis_key,
            evidence.identity.query,
        )
        if linked != evidence.identity:
            raise ValueError("path evidence is bound to another decision packet")
    return evidence


def primitive_path_evidence_bytes(evidence: PrimitivePathEvidence) -> bytes:
    verify_primitive_path_evidence(evidence)
    raw = _payload(evidence)
    raw["evidence_hash"] = evidence.evidence_hash
    return json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def write_primitive_path_evidence(
    evidence: PrimitivePathEvidence,
    destination: str | Path,
    *,
    decision_packet_path: str | Path,
) -> Path:
    output, packet_path = Path(destination), Path(decision_packet_path)
    if output.resolve() == packet_path.resolve():
        raise ValueError("future evidence must not overwrite the decision packet")
    verify_primitive_path_evidence(evidence, decision_packet_path=packet_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(primitive_path_evidence_bytes(evidence))
    return output


def _keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} schema is invalid")
    return value


def _query(value: Any) -> PrimitivePathEvidenceQuery:
    value = _keys(
        value,
        {"query_id", "issue", "primitive_name", "formula_version", "definition_hash", "relevant_transition_families", "relevant_entity_ids"},
        "path-evidence query",
    )
    if not isinstance(value["relevant_transition_families"], list) or not isinstance(
        value["relevant_entity_ids"],
        list,
    ):
        raise ValueError("path-evidence query selectors must be arrays")
    return PrimitivePathEvidenceQuery(
        query_id=value["query_id"], issue=value["issue"], primitive_name=value["primitive_name"],
        formula_version=value["formula_version"], definition_hash=value["definition_hash"],
        relevant_transition_families=tuple(value["relevant_transition_families"]),
        relevant_entity_ids=tuple(value["relevant_entity_ids"]),
    )


def _identity(value: Any) -> PrimitivePathEvidenceIdentity:
    value = _keys(
        value,
        {"decision_hash", "decision_packet_hash", "decision_packet_sha256", "decision_asof", "hypothesis_key", "setup_id", "entry_location_id", "entry_path_id", "deadline", "symbol", "instrument_id", "query"},
        "path-evidence identity",
    )
    return PrimitivePathEvidenceIdentity(
        decision_hash=value["decision_hash"], decision_packet_hash=value["decision_packet_hash"],
        decision_packet_sha256=value["decision_packet_sha256"], decision_asof=pd.Timestamp(value["decision_asof"]),
        hypothesis_key=value["hypothesis_key"], setup_id=value["setup_id"], entry_location_id=value["entry_location_id"],
        entry_path_id=value["entry_path_id"], deadline=pd.Timestamp(value["deadline"]), symbol=value["symbol"],
        instrument_id=value["instrument_id"], query=_query(value["query"]),
    )


def _point(value: Any) -> PrimitivePathEvidencePoint:
    value = _keys(
        value,
        {"observed_at", "completed_m1_bar", "events_added", "events_ended", "events_invalidated", "typed_state_transitions"},
        "path-evidence point",
    )
    if not isinstance(value["completed_m1_bar"], Mapping):
        raise ValueError("path-evidence completed bar is invalid")
    names = ("events_added", "events_ended", "events_invalidated", "typed_state_transitions")
    if any(not isinstance(value[name], list) or any(not isinstance(item, Mapping) for item in value[name]) for name in names):
        raise ValueError("path-evidence records must be arrays of objects")
    return PrimitivePathEvidencePoint(
        observed_at=pd.Timestamp(value["observed_at"]), completed_m1_bar=value["completed_m1_bar"],
        events_added=tuple(value["events_added"]), events_ended=tuple(value["events_ended"]),
        events_invalidated=tuple(value["events_invalidated"]), typed_state_transitions=tuple(value["typed_state_transitions"]),
    )


def _boundary(value: Any) -> PrimitivePathBoundaryDelta | None:
    if value is None:
        return None
    value = _keys(
        value,
        {
            "observed_at",
            "reason",
            "events_ended",
            "events_invalidated",
            "typed_state_transitions",
        },
        "path-evidence boundary",
    )
    names = ("events_ended", "events_invalidated", "typed_state_transitions")
    if any(
        not isinstance(value[name], list)
        or any(not isinstance(item, Mapping) for item in value[name])
        for name in names
    ):
        raise ValueError("path-evidence boundary records must be arrays of objects")
    return PrimitivePathBoundaryDelta(
        observed_at=pd.Timestamp(value["observed_at"]),
        reason=value["reason"],
        events_ended=tuple(value["events_ended"]),
        events_invalidated=tuple(value["events_invalidated"]),
        typed_state_transitions=tuple(value["typed_state_transitions"]),
    )


def read_verified_primitive_path_evidence(
    source: str | Path,
    *,
    decision_packet_path: str | Path,
) -> PrimitivePathEvidence:
    raw = json.loads(Path(source).read_text(encoding="utf-8"))
    raw = _keys(
        raw,
        {"format_version", "artifact", "identity", "started_at", "finalized_at", "finalization_reason", "points", "boundary_delta", "evidence_hash"},
        "path-evidence artifact",
    )
    if not isinstance(raw["points"], list):
        raise ValueError("path-evidence points must be an array")
    evidence = PrimitivePathEvidence(
        format_version=raw["format_version"], artifact=raw["artifact"], identity=_identity(raw["identity"]),
        started_at=pd.Timestamp(raw["started_at"]), finalized_at=pd.Timestamp(raw["finalized_at"]),
        finalization_reason=raw["finalization_reason"], points=tuple(_point(item) for item in raw["points"]),
        boundary_delta=_boundary(raw["boundary_delta"]),
        evidence_hash=raw["evidence_hash"],
    )
    return verify_primitive_path_evidence(evidence, decision_packet_path=decision_packet_path)


class PrimitivePathEvidenceRecorder:
    """One incremental recorder shared by all primitive queries."""

    def __init__(self, identity: PrimitivePathEvidenceIdentity, decision_packet_path: str | Path) -> None:
        if not isinstance(identity, PrimitivePathEvidenceIdentity):
            raise ValueError("path-evidence recorder identity is invalid")
        self.identity = identity
        self.decision_packet_path = Path(decision_packet_path)
        self._points: list[PrimitivePathEvidencePoint] = []
        self._boundary: PrimitivePathBoundaryDelta | None = None
        self._final: PrimitivePathEvidence | None = None

    @classmethod
    def from_decision_packet(
        cls,
        decision_packet_path: str | Path,
        *,
        hypothesis_key: str,
        query: PrimitivePathEvidenceQuery,
    ) -> "PrimitivePathEvidenceRecorder":
        packet_path = Path(decision_packet_path)
        packet = read_verified_decision_packet(packet_path)
        identity = _packet_identity(
            packet,
            decision_packet_sha256(packet_path),
            _text(hypothesis_key, "path_evidence.hypothesis_key"),
            query,
        )
        return cls(identity, packet_path)

    @property
    def points(self) -> tuple[PrimitivePathEvidencePoint, ...]:
        return tuple(self._points)

    @property
    def finalized(self) -> bool:
        return self._final is not None

    @property
    def evidence(self) -> PrimitivePathEvidence:
        if self._final is None:
            raise ValueError("path evidence has not reached a final boundary")
        return self._final

    def _expected_start(self) -> pd.Timestamp:
        return self.identity.decision_asof if not self._points else self._points[-1].observed_at

    def _finalize(self, at: pd.Timestamp, reason: str) -> PrimitivePathEvidence:
        if self._final is not None:
            raise ValueError("path evidence is already finalized")
        at = aware_timestamp(at, name="path_evidence.finalized_at")
        if reason not in PATH_EVIDENCE_FINAL_REASONS or not self._expected_start() <= at <= self.identity.deadline:
            raise ValueError("path-evidence final boundary is invalid")
        provisional = PrimitivePathEvidence(
            identity=self.identity, started_at=self.identity.decision_asof, finalized_at=at,
            finalization_reason=reason, points=tuple(self._points), boundary_delta=self._boundary,
            evidence_hash="0" * 64,
        )
        candidate = PrimitivePathEvidence(
            identity=provisional.identity, started_at=provisional.started_at, finalized_at=provisional.finalized_at,
            finalization_reason=provisional.finalization_reason, points=provisional.points,
            boundary_delta=provisional.boundary_delta,
            evidence_hash=content_hash(_payload(provisional)),
        )
        self._final = verify_primitive_path_evidence(
            candidate,
            decision_packet_path=self.decision_packet_path,
        )
        return self._final

    def finalize_hard_boundary(
        self,
        observed_at: pd.Timestamp,
        reason: str,
        *,
        events_ended: Sequence[Mapping[str, Any]] = (),
        events_invalidated: Sequence[Mapping[str, Any]] = (),
        typed_state_transitions: Sequence[Mapping[str, Any]] = (),
    ) -> PrimitivePathEvidence:
        if reason not in PATH_EVIDENCE_HARD_BOUNDARIES:
            raise ValueError("unregistered path-evidence hard boundary")
        observed_at = aware_timestamp(observed_at, name="path_evidence.hard_boundary")
        if observed_at < self.identity.deadline:
            ended = _records(events_ended, "events_ended", observed_at)
            invalidated = _records(
                events_invalidated,
                "events_invalidated",
                observed_at,
            )
            transitions = _records(
                typed_state_transitions,
                "typed_state_transitions",
                observed_at,
            )
            self._boundary = PrimitivePathBoundaryDelta(
                observed_at=observed_at,
                reason=reason,
                events_ended=tuple(
                    item
                    for item in ended
                    if _boundary_relevant(item, self.identity)
                ),
                events_invalidated=tuple(
                    item
                    for item in invalidated
                    if _boundary_relevant(item, self.identity)
                ),
                typed_state_transitions=tuple(
                    item
                    for item in transitions
                    if _boundary_relevant(item, self.identity)
                ),
            )
        return self._finalize(
            self.identity.deadline if observed_at >= self.identity.deadline else observed_at,
            "deadline" if observed_at >= self.identity.deadline else reason,
        )

    def close_right_boundary(self, observed_at: pd.Timestamp) -> PrimitivePathEvidence:
        observed_at = aware_timestamp(observed_at, name="path_evidence.right_boundary")
        return self._finalize(
            self.identity.deadline if observed_at >= self.identity.deadline else observed_at,
            "deadline" if observed_at >= self.identity.deadline else "right_boundary",
        )

    def observe(
        self,
        completed_m1_bar: Bar | Candle,
        *,
        events_added: Sequence[Mapping[str, Any]] = (),
        events_ended: Sequence[Mapping[str, Any]] = (),
        events_invalidated: Sequence[Mapping[str, Any]] = (),
        typed_state_transitions: Sequence[Mapping[str, Any]] = (),
        hard_boundary_reason: str | None = None,
    ) -> PrimitivePathEvidence | None:
        if self._final is not None:
            raise ValueError("path evidence is already finalized")
        bar = _bar_payload(completed_m1_bar)
        start = aware_timestamp(bar["start"], name="path_evidence.bar.start")
        end = aware_timestamp(bar["end"], name="path_evidence.bar.end")
        if start >= self.identity.deadline or end > self.identity.deadline:
            return self._finalize(self.identity.deadline, "deadline")
        if (bar["symbol"], bar["instrument_id"]) != (self.identity.symbol, self.identity.instrument_id):
            return self.finalize_hard_boundary(
                min(end, self.identity.deadline),
                "contract_change_reset",
                events_ended=events_ended,
                events_invalidated=events_invalidated,
                typed_state_transitions=typed_state_transitions,
            )
        expected_start = self._expected_start()
        if start < expected_start:
            raise ValueError("path-evidence bar precedes the next causal clock")
        unexplained_gap = bool(
            start > expected_start
            and scheduled_gap_kind(expected_start, start) is None
        )
        gap_boundary = bool(
            bar["data_gap_before_minutes"]
            or unexplained_gap
        )
        if hard_boundary_reason is not None and hard_boundary_reason not in PATH_EVIDENCE_HARD_BOUNDARIES:
            raise ValueError("unregistered path-evidence hard boundary")
        if hard_boundary_reason == "contract_change_reset":
            raise ValueError(
                "contract-change evidence cannot include same-contract OHLC"
            )
        transitions = tuple(
            item for item in _records(typed_state_transitions, "typed_state_transitions", end)
            if _relevant(item, self.identity.query)
        )
        point = PrimitivePathEvidencePoint(
            observed_at=end,
            completed_m1_bar=bar,
            events_added=_records(events_added, "events_added", end),
            events_ended=_records(events_ended, "events_ended", end),
            events_invalidated=_records(events_invalidated, "events_invalidated", end),
            typed_state_transitions=transitions,
        )
        _validate_point(point, self.identity, self._expected_start())
        self._points.append(point)
        effective_boundary = (
            "data_gap_reset" if gap_boundary else hard_boundary_reason
        )
        if effective_boundary is not None:
            return self._finalize(end, effective_boundary)
        if end == self.identity.deadline:
            return self._finalize(end, "deadline")
        return None

    def write(self, destination: str | Path) -> Path:
        return write_primitive_path_evidence(
            self.evidence,
            destination,
            decision_packet_path=self.decision_packet_path,
        )


__all__ = [
    "PATH_EVIDENCE_ARTIFACT",
    "PATH_EVIDENCE_FINAL_REASONS",
    "PATH_EVIDENCE_FORMAT_VERSION",
    "PATH_EVIDENCE_HARD_BOUNDARIES",
    "PrimitivePathEvidence",
    "PrimitivePathEvidenceIdentity",
    "PrimitivePathBoundaryDelta",
    "PrimitivePathEvidencePoint",
    "PrimitivePathEvidenceQuery",
    "PrimitivePathEvidenceRecorder",
    "primitive_path_evidence_bytes",
    "read_verified_primitive_path_evidence",
    "verify_primitive_path_evidence",
    "write_primitive_path_evidence",
]
