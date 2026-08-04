"""Versioned, immutable playbook protocols loaded before belief updates."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .model import Playbook, PlaybookPhase


class PlaybookRegistryError(ValueError):
    """Raised when a preregistered protocol is incomplete or inconsistent."""


@dataclass(frozen=True)
class SequenceStepProtocol:
    step_id: str
    description: str
    minimum_value: float
    required_inputs: tuple[str, ...]


@dataclass(frozen=True)
class PlaybookProtocol:
    playbook: Playbook
    version: str
    status: str
    thesis: str
    required_sequence: tuple[SequenceStepProtocol, ...]
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    phase_rules: Mapping[str, str]
    invalidation: Mapping[str, Any]
    targets: Mapping[str, Any]
    deadline: Mapping[str, Any]
    path_test: Mapping[str, Any]
    evidence_groups: Mapping[str, tuple[str, ...]]
    hard_gates: tuple[str, ...]


@dataclass(frozen=True)
class PlaybookRegistry:
    registry_version: str
    frozen_at: str
    fingerprint: str
    protocols: tuple[PlaybookProtocol, ...]

    def for_playbook(self, playbook: Playbook) -> PlaybookProtocol:
        for protocol in self.protocols:
            if protocol.playbook is playbook:
                return protocol
        raise KeyError(playbook.value)


def _require_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlaybookRegistryError(f"{path} must be non-empty text")
    return value.strip()


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlaybookRegistryError(f"{path} must be an object")
    return value


def _load_step(raw: Any, path: str) -> SequenceStepProtocol:
    item = _require_mapping(raw, path)
    inputs = item.get("required_inputs")
    if not isinstance(inputs, list) or not inputs:
        raise PlaybookRegistryError(f"{path}.required_inputs must be a non-empty list")
    try:
        minimum = float(item.get("minimum_value"))
    except (TypeError, ValueError) as error:
        raise PlaybookRegistryError(
            f"{path}.minimum_value must be a finite number"
        ) from error
    if not 0.0 <= minimum <= 1.0:
        raise PlaybookRegistryError(f"{path}.minimum_value must be in [0, 1]")
    return SequenceStepProtocol(
        step_id=_require_text(item.get("id"), f"{path}.id"),
        description=_require_text(item.get("description"), f"{path}.description"),
        minimum_value=minimum,
        required_inputs=tuple(
            _require_text(value, f"{path}.required_inputs") for value in inputs
        ),
    )


def _load_protocol(raw: Any, path: str) -> PlaybookProtocol:
    item = _require_mapping(raw, path)
    playbook = Playbook(_require_text(item.get("id"), f"{path}.id"))
    sequence_raw = item.get("required_sequence")
    if not isinstance(sequence_raw, list) or len(sequence_raw) < 2:
        raise PlaybookRegistryError(
            f"{path}.required_sequence must contain at least two ordered steps"
        )
    sequence = tuple(
        _load_step(value, f"{path}.required_sequence[{index}]")
        for index, value in enumerate(sequence_raw)
    )
    step_ids = [step.step_id for step in sequence]
    if len(step_ids) != len(set(step_ids)):
        raise PlaybookRegistryError(f"{path}.required_sequence has duplicate step ids")

    phase_rules = _require_mapping(item.get("phase_rules"), f"{path}.phase_rules")
    all_phases = {phase.value for phase in PlaybookPhase}
    legacy_phases = all_phases - {
        PlaybookPhase.WAITING_LOCATION.value,
        PlaybookPhase.WAITING_TRIGGER.value,
    }
    typed_phases = all_phases - {
        PlaybookPhase.WAITING_PULLBACK.value,
    }
    if frozenset(phase_rules) not in {
        frozenset(legacy_phases),
        frozenset(typed_phases),
    }:
        raise PlaybookRegistryError(
            f"{path}.phase_rules must use either the archived or typed phase set"
        )
    supports = item.get("supporting_evidence")
    against = item.get("contradicting_evidence")
    if not isinstance(supports, list) or not supports:
        raise PlaybookRegistryError(f"{path}.supporting_evidence is required")
    if not isinstance(against, list) or not against:
        raise PlaybookRegistryError(f"{path}.contradicting_evidence is required")
    support_names = tuple(
        _require_text(value, f"{path}.supporting_evidence")
        for value in supports
    )
    against_names = tuple(
        _require_text(value, f"{path}.contradicting_evidence")
        for value in against
    )
    if (
        len(support_names) != len(set(support_names))
        or len(against_names) != len(set(against_names))
        or set(support_names) & set(against_names)
    ):
        raise PlaybookRegistryError(
            f"{path} evidence names must be unique and side-exclusive"
        )

    invalidation = _require_mapping(item.get("invalidation"), f"{path}.invalidation")
    if invalidation.get("retrospective_rewrite") is not False:
        raise PlaybookRegistryError(
            f"{path}.invalidation.retrospective_rewrite must be false"
        )
    targets = _require_mapping(item.get("targets"), f"{path}.targets")
    deadline = _require_mapping(item.get("deadline"), f"{path}.deadline")
    path_test = _require_mapping(item.get("path_test"), f"{path}.path_test")
    _require_text(path_test.get("primary_outcome"), f"{path}.path_test.primary_outcome")
    evidence_groups_raw = item.get("evidence_groups", {})
    evidence_groups_mapping = _require_mapping(
        evidence_groups_raw,
        f"{path}.evidence_groups",
    )
    evidence_groups = {
        str(group): tuple(
            _require_text(
                primitive,
                f"{path}.evidence_groups.{group}",
            )
            for primitive in primitives
        )
        for group, primitives in evidence_groups_mapping.items()
        if isinstance(primitives, list)
    }
    if len(evidence_groups) != len(evidence_groups_mapping):
        raise PlaybookRegistryError(
            f"{path}.evidence_groups values must be lists"
        )
    if evidence_groups and set(evidence_groups) != {
        "structure",
        "displacement",
        "location",
        "liquidity",
        "trigger",
        "execution",
    }:
        raise PlaybookRegistryError(
            f"{path}.evidence_groups must define the six registered groups"
        )
    registered_evidence = set(support_names) | set(against_names)
    grouped_evidence = {
        primitive
        for primitives in evidence_groups.values()
        for primitive in primitives
    }
    if evidence_groups and grouped_evidence != registered_evidence:
        raise PlaybookRegistryError(
            f"{path}.evidence_groups must cover each evidence primitive once"
        )
    if sum(len(values) for values in evidence_groups.values()) != len(
        grouped_evidence
    ):
        raise PlaybookRegistryError(
            f"{path}.evidence_groups cannot repeat a primitive"
        )
    hard_gates_raw = item.get("hard_gates", [])
    if not isinstance(hard_gates_raw, list):
        raise PlaybookRegistryError(f"{path}.hard_gates must be a list")
    hard_gates = tuple(
        _require_text(value, f"{path}.hard_gates")
        for value in hard_gates_raw
    )
    if len(hard_gates) != len(set(hard_gates)) or not set(
        hard_gates
    ).issubset(step_ids):
        raise PlaybookRegistryError(
            f"{path}.hard_gates must reference unique sequence steps"
        )
    typed_phase_set = frozenset(phase_rules) == frozenset(typed_phases)
    if typed_phase_set and (not evidence_groups or not hard_gates):
        raise PlaybookRegistryError(
            f"{path} typed protocol requires six evidence groups "
            "and explicit hard gates"
        )

    return PlaybookProtocol(
        playbook=playbook,
        version=_require_text(item.get("version"), f"{path}.version"),
        status=_require_text(item.get("status"), f"{path}.status"),
        thesis=_require_text(item.get("thesis"), f"{path}.thesis"),
        required_sequence=sequence,
        supporting_evidence=support_names,
        contradicting_evidence=against_names,
        phase_rules={
            str(key): _require_text(value, f"{path}.phase_rules.{key}")
            for key, value in phase_rules.items()
        },
        invalidation=dict(invalidation),
        targets=dict(targets),
        deadline=dict(deadline),
        path_test=dict(path_test),
        evidence_groups=evidence_groups,
        hard_gates=hard_gates,
    )


@lru_cache(maxsize=8)
def load_playbook_registry(
    path: str | Path = "configs/playbooks_v2.json",
) -> PlaybookRegistry:
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = Path(__file__).resolve().parents[1] / source
    raw_bytes = source.read_bytes()
    payload = json.loads(raw_bytes)
    if not isinstance(payload, Mapping):
        raise PlaybookRegistryError("playbook registry root must be an object")
    values = payload.get("playbooks")
    if not isinstance(values, list):
        raise PlaybookRegistryError("playbooks must be a list")
    protocols = tuple(
        _load_protocol(value, f"playbooks[{index}]")
        for index, value in enumerate(values)
    )
    registered = {protocol.playbook for protocol in protocols}
    if registered != set(Playbook) or len(protocols) != len(Playbook):
        raise PlaybookRegistryError(
            "registry must contain exactly the three code-registered playbooks"
        )
    return PlaybookRegistry(
        registry_version=_require_text(
            payload.get("registry_version"), "registry_version"
        ),
        frozen_at=_require_text(payload.get("frozen_at"), "frozen_at"),
        fingerprint=hashlib.sha256(raw_bytes).hexdigest(),
        protocols=protocols,
    )


__all__ = [
    "PlaybookProtocol",
    "PlaybookRegistry",
    "PlaybookRegistryError",
    "SequenceStepProtocol",
    "load_playbook_registry",
]
