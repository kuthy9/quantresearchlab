"""Versioned preregistration contract for SMC semantic events.

The ``.yaml`` registries intentionally use the JSON subset of YAML so the
runtime does not gain a new parser dependency.  Their exact bytes are hashed
and can be bound by experiment manifests and replay checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .model import EventKind, FrozenDict, SMC_SEMANTIC_VERSION

if TYPE_CHECKING:
    from .foundation_registry import FoundationRegistry


class SemanticRegistryError(ValueError):
    """Raised when a semantic or parameter registry is incomplete."""


SEMANTIC_SELECTION_SCHEMA_VERSION = 1
_SEMANTIC_SELECTION_KEYS = frozenset(
    {
        "schema_version",
        "atomic_semantics_version",
        "atomic_registry",
        "atomic_definition_identity",
        "foundation_projection_version",
        "foundation_registry",
        "foundation_registry_identity",
    }
)


_CONCEPT_FIELDS = (
    "domain_meaning",
    "operational_definition",
    "hypothesized_relations",
    "falsification_conditions",
    "oos_pass_criteria",
)

SEMANTIC_EVENT_BINDING_STATUSES = frozenset(
    {
        "canonical_emitted",
        "reserved_not_emitted",
        "compatibility_alias_not_emitted",
        "snapshot_derived",
    }
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PRIMITIVE_PROTOCOL_REFERENCE = re.compile(
    r"configs/primitives_[A-Za-z0-9_.-]+\.json"
)


def _resolve(source: str | Path) -> Path:
    path = Path(source)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parents[1] / path
    return path


def _reject_duplicate_object_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    """Build one JSON object while rejecting identity-erasing duplicate keys."""

    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SemanticRegistryError(
                f"semantic registry contains duplicate JSON key: {key!r}"
            )
        value[key] = item
    return value


def _read_json_yaml(source: str | Path) -> tuple[Path, bytes, Mapping[str, Any]]:
    path = _resolve(source)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SemanticRegistryError(
            f"cannot read semantic registry: {path}"
        ) from error
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_object_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticRegistryError(
            f"semantic registry must use the JSON subset of YAML: {path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise SemanticRegistryError("semantic registry root must be an object")
    return path, raw, payload


def _sha256_file(source: str | Path, *, role: str) -> tuple[Path, str]:
    path = _resolve(source)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SemanticRegistryError(
            f"cannot read referenced {role}: {path}"
        ) from error
    return path, hashlib.sha256(raw).hexdigest()


def _sha256_identity(value: Any, *, name: str) -> str:
    normalized = str(value).lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise SemanticRegistryError(f"{name} must be a lowercase SHA-256 identity")
    return normalized


def _registry_reference(value: Any, *, name: str) -> str:
    if not _nonempty_text(value) or "\\" in str(value):
        raise SemanticRegistryError(f"{name} must be a repository-relative path")
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise SemanticRegistryError(f"{name} must be a repository-relative path")
    return str(path)


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonempty_text_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_nonempty_text(item) for item in value)
    )


@dataclass(frozen=True)
class SemanticConcept:
    name: str
    owner: str
    domain_meaning: str
    operational_definition: str
    hypothesized_relations: tuple[str, ...]
    falsification_conditions: tuple[str, ...]
    oos_pass_criteria: tuple[str, ...]
    existing_binding: str


@dataclass(frozen=True)
class SemanticEventBinding:
    """Machine-checkable ownership and runtime status for one semantic fact."""

    concept: str
    event_kind: EventKind | None
    status: str


@dataclass(frozen=True)
class SemanticParameterRegistry:
    semantic_version: str
    status: str
    data_split_registry: str
    freeze_policy: Mapping[str, Any]
    parameters: Mapping[str, Mapping[str, Any]]
    source_path: Path
    source_sha256: str

    @classmethod
    def from_file(
        cls,
        source: str | Path,
        *,
        expected_version: str | None = None,
    ) -> "SemanticParameterRegistry":
        path, raw, payload = _read_json_yaml(source)
        version = payload.get("semantic_version")
        parameters = payload.get("parameters")
        freeze_policy = payload.get("freeze_policy")
        if (
            payload.get("schema_version") != 1
            or not _nonempty_text(version)
            or not _nonempty_text(payload.get("status"))
            or not _nonempty_text(payload.get("data_split_registry"))
            or not isinstance(freeze_policy, Mapping)
            or not freeze_policy
            or not isinstance(parameters, Mapping)
            or not parameters
        ):
            raise SemanticRegistryError("semantic parameter registry is incomplete")
        if expected_version is not None and version != expected_version:
            raise SemanticRegistryError(
                "semantic registry and parameter versions differ"
            )
        normalized: dict[str, Mapping[str, Any]] = {}
        for name, definition in parameters.items():
            if (
                not _nonempty_text(name)
                or not isinstance(definition, Mapping)
                or not _nonempty_text(definition.get("status"))
                or not _nonempty_text(definition.get("source"))
                or not ({"value", "values", "values_by_timeframe"} & set(definition))
            ):
                raise SemanticRegistryError(
                    f"semantic parameter {name!r} is incomplete"
                )
            normalized[str(name)] = FrozenDict(definition)
        return cls(
            semantic_version=str(version),
            status=str(payload["status"]),
            data_split_registry=str(payload["data_split_registry"]),
            freeze_policy=FrozenDict(freeze_policy),
            parameters=FrozenDict(normalized),
            source_path=path,
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )


@dataclass(frozen=True)
class SemanticDefinitionIdentity:
    """Exact content identity of one preregistered semantic definition.

    A semantic version is a human governance label, not a content address.
    This value additionally binds the exact registry and parameter bytes plus
    every primitive protocol and data-split registry referenced by those
    parameters.  Paths are retained as audit labels; the digest is derived
    from both the labels and exact file hashes and contains no machine-local
    absolute paths.
    """

    semantic_version: str
    registry_sha256: str
    parameters_sha256: str
    data_split_registry: str
    data_split_sha256: str
    primitive_protocol_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if not _nonempty_text(self.semantic_version):
            raise SemanticRegistryError(
                "semantic definition identity requires a semantic version"
            )
        object.__setattr__(
            self,
            "data_split_registry",
            _registry_reference(
                self.data_split_registry,
                name="data split registry",
            ),
        )
        object.__setattr__(
            self,
            "registry_sha256",
            _sha256_identity(self.registry_sha256, name="registry_sha256"),
        )
        object.__setattr__(
            self,
            "parameters_sha256",
            _sha256_identity(self.parameters_sha256, name="parameters_sha256"),
        )
        object.__setattr__(
            self,
            "data_split_sha256",
            _sha256_identity(self.data_split_sha256, name="data_split_sha256"),
        )
        protocols = self.primitive_protocol_sha256
        if not isinstance(protocols, Mapping) or not protocols:
            raise SemanticRegistryError(
                "semantic definition identity requires primitive protocols"
            )
        normalized: dict[str, str] = {}
        for source, digest in protocols.items():
            normalized_source = _registry_reference(
                source,
                name="semantic primitive protocol path",
            )
            normalized[normalized_source] = _sha256_identity(
                digest,
                name=f"primitive protocol {source!r}",
            )
        object.__setattr__(
            self,
            "primitive_protocol_sha256",
            FrozenDict(dict(sorted(normalized.items()))),
        )

    @property
    def identity(self) -> str:
        payload = json.dumps(
            self.to_metadata(include_identity=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_metadata(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "semantic_version": self.semantic_version,
            "registry_sha256": self.registry_sha256,
            "parameters_sha256": self.parameters_sha256,
            "data_split_registry": self.data_split_registry,
            "data_split_sha256": self.data_split_sha256,
            "primitive_protocol_sha256": dict(
                self.primitive_protocol_sha256
            ),
        }
        if include_identity:
            payload["identity"] = self.identity
        return payload

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any]) -> "SemanticDefinitionIdentity":
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise SemanticRegistryError(
                "semantic definition identity metadata is invalid"
            )
        required = {
            "schema_version",
            "semantic_version",
            "registry_sha256",
            "parameters_sha256",
            "data_split_registry",
            "data_split_sha256",
            "primitive_protocol_sha256",
        }
        extra = set(payload) - (required | {"identity"})
        missing = required - set(payload)
        if missing or extra:
            raise SemanticRegistryError(
                "semantic definition identity metadata fields differ: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        identity = cls(
            semantic_version=str(payload["semantic_version"]),
            registry_sha256=str(payload["registry_sha256"]),
            parameters_sha256=str(payload["parameters_sha256"]),
            data_split_registry=str(payload["data_split_registry"]),
            data_split_sha256=str(payload["data_split_sha256"]),
            primitive_protocol_sha256=payload["primitive_protocol_sha256"],
        )
        recorded = payload.get("identity")
        if recorded is not None and recorded != identity.identity:
            raise SemanticRegistryError(
                "semantic definition identity metadata digest is invalid"
            )
        return identity

    def require(
        self,
        expected: "SemanticDefinitionIdentity | Mapping[str, Any] | str",
    ) -> None:
        if isinstance(expected, Mapping):
            expected = SemanticDefinitionIdentity.from_metadata(expected)
        expected_digest = (
            expected.identity
            if isinstance(expected, SemanticDefinitionIdentity)
            else _sha256_identity(
                expected,
                name="expected semantic definition identity",
            )
        )
        if self.identity != expected_digest:
            raise SemanticRegistryError(
                "semantic definition identity drifted without a version change: "
                f"version={self.semantic_version}, expected={expected_digest}, "
                f"actual={self.identity}"
            )


def _definition_identity(
    *,
    semantic_version: str,
    registry_sha256: str,
    parameters: SemanticParameterRegistry,
) -> SemanticDefinitionIdentity:
    _, split_sha256 = _sha256_file(
        parameters.data_split_registry,
        role="data split registry",
    )
    protocol_references: set[str] = set()
    for definition in parameters.parameters.values():
        source = definition.get("source")
        if isinstance(source, str):
            protocol_references.update(
                _PRIMITIVE_PROTOCOL_REFERENCE.findall(source)
            )
    if not protocol_references:
        raise SemanticRegistryError(
            "semantic parameters do not reference any primitive protocols"
        )
    protocol_hashes: dict[str, str] = {}
    for reference in sorted(protocol_references):
        _, digest = _sha256_file(reference, role="primitive protocol")
        protocol_hashes[reference] = digest
    return SemanticDefinitionIdentity(
        semantic_version=semantic_version,
        registry_sha256=registry_sha256,
        parameters_sha256=parameters.source_sha256,
        data_split_registry=parameters.data_split_registry,
        data_split_sha256=split_sha256,
        primitive_protocol_sha256=protocol_hashes,
    )


@dataclass(frozen=True)
class SemanticRegistry:
    semantic_version: str
    title: str
    status: str
    clock_contract: Mapping[str, Any]
    layer_boundaries: Mapping[str, Any]
    concepts: Mapping[str, SemanticConcept]
    event_bindings: tuple[SemanticEventBinding, ...]
    parameters: SemanticParameterRegistry
    source_path: Path
    source_sha256: str
    definition_identity: SemanticDefinitionIdentity

    @classmethod
    def from_file(
        cls,
        source: str | Path = "semantics/registry_v1_2.yaml",
        *,
        required_version: str = SMC_SEMANTIC_VERSION,
        expected_definition_identity: (
            SemanticDefinitionIdentity | Mapping[str, Any] | str | None
        ) = None,
    ) -> "SemanticRegistry":
        path, raw, payload = _read_json_yaml(source)
        version = payload.get("semantic_version")
        concept_payload = payload.get("concepts")
        event_binding_payload = payload.get("event_bindings")
        clock_contract = payload.get("clock_contract")
        boundaries = payload.get("layer_boundaries")
        if (
            payload.get("schema_version") != 1
            or version != required_version
            or not _nonempty_text(payload.get("title"))
            or not _nonempty_text(payload.get("status"))
            or not _nonempty_text(payload.get("parameters_file"))
            or not isinstance(clock_contract, Mapping)
            or not clock_contract
            or not isinstance(boundaries, Mapping)
            or set(boundaries) != {"eye", "brain", "execution"}
            or not isinstance(concept_payload, Mapping)
            or not concept_payload
            or not isinstance(event_binding_payload, Sequence)
            or isinstance(event_binding_payload, (str, bytes))
            or not event_binding_payload
        ):
            raise SemanticRegistryError("semantic registry is incomplete")
        concepts: dict[str, SemanticConcept] = {}
        for name, definition in concept_payload.items():
            if not _nonempty_text(name) or not isinstance(definition, Mapping):
                raise SemanticRegistryError("semantic concept identity is invalid")
            missing = [field for field in _CONCEPT_FIELDS if field not in definition]
            if missing:
                raise SemanticRegistryError(
                    f"semantic concept {name!r} lacks: {', '.join(missing)}"
                )
            if (
                definition.get("owner") not in {"eye", "brain", "execution"}
                or not _nonempty_text(definition["domain_meaning"])
                or not _nonempty_text(definition["operational_definition"])
                or not _nonempty_text_list(definition["hypothesized_relations"])
                or not _nonempty_text_list(definition["falsification_conditions"])
                or not _nonempty_text_list(definition["oos_pass_criteria"])
                or not _nonempty_text(definition.get("existing_binding"))
            ):
                raise SemanticRegistryError(
                    f"semantic concept {name!r} has an invalid preregistration"
                )
            concepts[str(name)] = SemanticConcept(
                name=str(name),
                owner=str(definition["owner"]),
                domain_meaning=str(definition["domain_meaning"]),
                operational_definition=str(definition["operational_definition"]),
                hypothesized_relations=tuple(definition["hypothesized_relations"]),
                falsification_conditions=tuple(definition["falsification_conditions"]),
                oos_pass_criteria=tuple(definition["oos_pass_criteria"]),
                existing_binding=str(definition["existing_binding"]),
            )
        event_bindings: list[SemanticEventBinding] = []
        bound_event_kinds: set[EventKind] = set()
        snapshot_concepts: set[str] = set()
        bound_concepts: set[str] = set()
        for raw_binding in event_binding_payload:
            if (
                not isinstance(raw_binding, Mapping)
                or set(raw_binding) != {"concept", "event_kind", "status"}
            ):
                raise SemanticRegistryError(
                    "semantic event binding must contain exactly concept, "
                    "event_kind, and status"
                )
            concept = raw_binding.get("concept")
            status = raw_binding.get("status")
            raw_event_kind = raw_binding.get("event_kind")
            if concept not in concepts:
                raise SemanticRegistryError(
                    f"semantic event binding references unknown concept: {concept!r}"
                )
            if status not in SEMANTIC_EVENT_BINDING_STATUSES:
                raise SemanticRegistryError(
                    f"semantic event binding has invalid status: {status!r}"
                )
            if status == "snapshot_derived":
                if raw_event_kind is not None:
                    raise SemanticRegistryError(
                        "snapshot-derived semantic binding cannot claim an event kind"
                    )
                if str(concept) in snapshot_concepts:
                    raise SemanticRegistryError(
                        f"duplicate snapshot-derived semantic binding: {concept!r}"
                    )
                event_kind = None
                snapshot_concepts.add(str(concept))
            else:
                if not _nonempty_text(raw_event_kind):
                    raise SemanticRegistryError(
                        "event binding requires a registered EventKind value"
                    )
                try:
                    event_kind = EventKind(str(raw_event_kind))
                except ValueError as error:
                    raise SemanticRegistryError(
                        f"semantic event binding uses unknown EventKind: "
                        f"{raw_event_kind!r}"
                    ) from error
                if event_kind in bound_event_kinds:
                    raise SemanticRegistryError(
                        f"duplicate semantic event binding: {event_kind.value}"
                    )
                bound_event_kinds.add(event_kind)
            bound_concepts.add(str(concept))
            event_bindings.append(
                SemanticEventBinding(
                    concept=str(concept),
                    event_kind=event_kind,
                    status=str(status),
                )
            )
        unbound_concepts = set(concepts) - bound_concepts
        if unbound_concepts:
            raise SemanticRegistryError(
                "semantic concepts lack event/snapshot bindings: "
                + ", ".join(sorted(unbound_concepts))
            )
        parameters = SemanticParameterRegistry.from_file(
            str(payload["parameters_file"]),
            expected_version=str(version),
        )
        source_sha256 = hashlib.sha256(raw).hexdigest()
        definition_identity = _definition_identity(
            semantic_version=str(version),
            registry_sha256=source_sha256,
            parameters=parameters,
        )
        if expected_definition_identity is not None:
            definition_identity.require(expected_definition_identity)
        return cls(
            semantic_version=str(version),
            title=str(payload["title"]),
            status=str(payload["status"]),
            clock_contract=FrozenDict(clock_contract),
            layer_boundaries=FrozenDict(boundaries),
            concepts=FrozenDict(concepts),
            event_bindings=tuple(event_bindings),
            parameters=parameters,
            source_path=path,
            source_sha256=source_sha256,
            definition_identity=definition_identity,
        )

    @property
    def identity(self) -> str:
        """Backward-compatible digest for the complete definition identity."""

        return self.definition_identity.identity


    @property
    def canonical_emitted_event_kinds(self) -> frozenset[EventKind]:
        """Return the exact production-emitted canonical semantic surface."""

        return frozenset(
            binding.event_kind
            for binding in self.event_bindings
            if binding.status == "canonical_emitted"
            and binding.event_kind is not None
        )

    @property
    def event_binding_by_kind(self) -> Mapping[EventKind, SemanticEventBinding]:
        """Return the unique registry binding for every registered EventKind."""

        return FrozenDict(
            {
                binding.event_kind: binding
                for binding in self.event_bindings
                if binding.event_kind is not None
            }
        )


@dataclass(frozen=True)
class SemanticSelection:
    """The selected atomic protocol and its dependent Foundation projection."""

    atomic_registry_path: str
    foundation_registry_path: str
    atomic_registry: SemanticRegistry
    foundation_registry: "FoundationRegistry"

    @property
    def atomic_semantics_version(self) -> str:
        return self.atomic_registry.semantic_version

    @property
    def atomic_definition_identity(self) -> str:
        return self.atomic_registry.identity

    @property
    def foundation_projection_version(self) -> str:
        return self.foundation_registry.foundation_version

    @property
    def foundation_registry_identity(self) -> str:
        return self.foundation_registry.identity

    @property
    def parent_atomic_semantics_version(self) -> str:
        return self.foundation_registry.parent_atomic_semantic_version

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SEMANTIC_SELECTION_SCHEMA_VERSION,
            "atomic_semantics_version": self.atomic_semantics_version,
            "atomic_registry": self.atomic_registry_path,
            "atomic_definition_identity": self.atomic_definition_identity,
            "foundation_projection_version": self.foundation_projection_version,
            "foundation_registry": self.foundation_registry_path,
            "foundation_registry_identity": self.foundation_registry_identity,
            "parent_atomic_semantics_version": (
                self.parent_atomic_semantics_version
            ),
        }


def _selection_file(
    value: Any,
    *,
    name: str,
    root: Path,
) -> tuple[str, Path]:
    reference = _registry_reference(value, name=name)
    candidate = root / reference
    if candidate.is_symlink():
        raise SemanticRegistryError(f"{name} must be a direct regular file")
    path = candidate.resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SemanticRegistryError(f"{name} escapes the repository") from error
    if not path.is_file():
        raise SemanticRegistryError(f"{name} must be a direct regular file")
    return reference, path


def load_semantic_selection(
    payload: Any,
    *,
    root: str | Path | None = None,
) -> SemanticSelection:
    """Strict-load one atomic/Foundation pair without inventing a stack version."""

    from .foundation_registry import (
        FOUNDATION_VERSION,
        FoundationRegistryError,
        load_foundation_registry,
    )

    fields = set(payload) if isinstance(payload, Mapping) else set()
    if not isinstance(payload, Mapping) or fields != _SEMANTIC_SELECTION_KEYS:
        raise SemanticRegistryError(
            "semantic_selection fields differ: "
            f"missing={sorted(_SEMANTIC_SELECTION_KEYS - fields)}, "
            f"extra={sorted(fields - _SEMANTIC_SELECTION_KEYS)}"
        )
    if payload["schema_version"] != SEMANTIC_SELECTION_SCHEMA_VERSION:
        raise SemanticRegistryError("semantic_selection schema is unsupported")
    atomic_version = payload["atomic_semantics_version"]
    foundation_version = payload["foundation_projection_version"]
    if atomic_version != SMC_SEMANTIC_VERSION:
        raise SemanticRegistryError(
            "runtime does not implement the selected atomic semantics"
        )
    if foundation_version != FOUNDATION_VERSION:
        raise SemanticRegistryError(
            "runtime does not implement the selected Foundation projection"
        )
    repository_root = (
        Path(__file__).resolve().parents[1] if root is None else Path(root).resolve()
    )
    atomic_reference, atomic_path = _selection_file(
        payload["atomic_registry"], name="atomic_registry", root=repository_root
    )
    foundation_reference, foundation_path = _selection_file(
        payload["foundation_registry"],
        name="foundation_registry",
        root=repository_root,
    )
    atomic = SemanticRegistry.from_file(
        atomic_path,
        required_version=atomic_version,
        expected_definition_identity=_sha256_identity(
            payload["atomic_definition_identity"],
            name="atomic_definition_identity",
        ),
    )
    try:
        foundation = load_foundation_registry(
            foundation_path,
            expected_identity=_sha256_identity(
                payload["foundation_registry_identity"],
                name="foundation_registry_identity",
            ),
        )
    except FoundationRegistryError as error:
        raise SemanticRegistryError(str(error)) from error
    if foundation.foundation_version != foundation_version:
        raise SemanticRegistryError(
            "Foundation registry version differs from semantic_selection"
        )
    if foundation.parent_atomic_semantic_version != atomic.semantic_version:
        raise SemanticRegistryError(
            "Foundation parent atomic semantics differs from semantic_selection"
        )
    return SemanticSelection(
        atomic_registry_path=atomic_reference,
        foundation_registry_path=foundation_reference,
        atomic_registry=atomic,
        foundation_registry=foundation,
    )


__all__ = [
    "SEMANTIC_SELECTION_SCHEMA_VERSION",
    "SemanticConcept",
    "SemanticDefinitionIdentity",
    "SemanticEventBinding",
    "SemanticParameterRegistry",
    "SemanticRegistry",
    "SemanticRegistryError",
    "SemanticSelection",
    "load_semantic_selection",
    "SEMANTIC_EVENT_BINDING_STATUSES",
]
