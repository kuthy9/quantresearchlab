"""Fail-closed primitives for preregistered semantic event studies.

This module is intentionally independent of the Trading Eye and Trading Brain.
It validates research authority before market data is opened and provides the
small deterministic operations used by the signal-research runner.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import pandas as pd


EXECUTABLE_DIAGNOSTIC_STATUS = (
    "frozen_development_diagnostic_not_oos_not_trading_authority"
)
REQUIRED_RUNTIME_CODE_BINDINGS: Mapping[str, str] = MappingProxyType(
    {
        "runtime_package_init": "smc_trader/__init__.py",
        "runtime_artifact_stream": "smc_trader/artifact_stream.py",
        "runtime_causal": "smc_trader/causal.py",
        "runtime_io": "smc_trader/io.py",
        "runtime_market_clock": "smc_trader/market_clock.py",
        "runtime_model": "smc_trader/model.py",
        "runtime_observation": "smc_trader/observation.py",
        "runtime_semantics": "smc_trader/semantics.py",
        "runtime_event_store": "smc_trader/event_store.py",
        "runtime_market_state": "smc_trader/market_state.py",
        "runtime_structure": "smc_trader/structure.py",
        "runtime_liquidity": "smc_trader/liquidity.py",
        "runtime_displacement": "smc_trader/displacement.py",
        "runtime_displacement_observer": "smc_trader/displacement_observer.py",
        "runtime_group3": "smc_trader/zone.py",
        "runtime_group4": "smc_trader/range_auction.py",
        "runtime_group5": "smc_trader/group5.py",
        "runtime_scene_graph": "smc_trader/scene_graph.py",
        "runtime_signal_research": "smc_trader/signal_research.py",
        "runtime_validation": "smc_trader/validation.py",
    }
)
REQUIRED_IDENTITY_BINDINGS = frozenset(
    {
        "semantic_registry",
        "semantic_parameters",
        "dataset_manifest",
        "split_registry",
        "model_config",
        "structure_protocol",
        "liquidity_protocol",
        "displacement_protocol",
        "group3_protocol",
        "group4_protocol",
        "group5_protocol",
        "runner",
        "pyproject",
        "lockfile",
    }
).union(REQUIRED_RUNTIME_CODE_BINDINGS)
FORBIDDEN_EVALUATION_ROLE_TOKENS = (
    "rolling_oof",
    "sealed",
    "holdout",
)


class ResearchContractError(RuntimeError):
    """Raised before data access when a research contract is not executable."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aware(value: Any, *, name: str) -> pd.Timestamp:
    clock = pd.Timestamp(value)
    if clock.tzinfo is None:
        raise ResearchContractError(f"{name} must be timezone aware")
    return clock.tz_convert("America/New_York")


def _bound_path(root: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ResearchContractError(f"{name}.path is required")
    candidate = (root / value).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ResearchContractError(f"{name}.path escapes the repository") from error
    if not candidate.is_file() or candidate.is_symlink():
        raise ResearchContractError(f"{name}.path is not a regular file")
    return candidate


@dataclass(frozen=True)
class FrozenResearchContract:
    manifest_path: Path
    manifest_sha256: str
    payload: Mapping[str, Any]
    dataset_path: Path
    dataset_relative_path: str
    dataset_sha256: str
    split_registry_path: Path
    split_registry_sha256: str
    split_role: str
    semantic_registry_path: Path
    semantic_registry_identity: str
    model_path: Path
    warmup_start: pd.Timestamp
    diagnostic_start: pd.Timestamp
    diagnostic_end: pd.Timestamp
    allowed_diagnostic_roles: tuple[str, ...]
    allowed_warmup_roles: tuple[str, ...]
    identity_paths: Mapping[str, Path]


def load_frozen_research_contract(
    manifest_path: str | Path,
    *,
    root: str | Path,
    actual_semantic_registry_identity: str,
) -> FrozenResearchContract:
    """Load and verify every immutable binding before opening market data.

    Schema-1 manifests remain historical evidence only. They deliberately
    cannot be silently upgraded into executable contracts because their old
    result hashes bind the original bytes.
    """

    repository = Path(root).resolve()
    source = Path(manifest_path).resolve()
    try:
        source.relative_to(repository)
    except ValueError as error:
        raise ResearchContractError(
            "manifest must live inside the repository"
        ) from error
    if not source.is_file() or source.is_symlink():
        raise ResearchContractError("research manifest is not a regular file")
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResearchContractError(
            "research manifest must use the JSON subset of YAML"
        ) from error
    if not isinstance(payload, Mapping):
        raise ResearchContractError("research manifest root must be an object")
    if payload.get("schema_version") != 2:
        raise ResearchContractError(
            "only a frozen schema-2 research manifest is executable; "
            "schema-1 manifests and results are historical evidence"
        )
    if (
        payload.get("status") != EXECUTABLE_DIAGNOSTIC_STATUS
        or payload.get("frozen_before_run") is not True
        or payload.get("parameter_search_space") != {}
    ):
        raise ResearchContractError("research manifest is not frozen diagnostic-only")

    authority = payload.get("authority")
    if not isinstance(authority, Mapping) or authority != {
        "diagnostic_only": True,
        "artifact_fit_allowed": False,
        "trading_authority": False,
        "oos_opened": False,
    }:
        raise ResearchContractError(
            "research authority must fail closed as diagnostic-only"
        )
    if payload.get("out_of_sample_period") != "not_opened":
        raise ResearchContractError("out-of-sample data is not authorized")

    expected_registry_identity = payload.get("semantic_registry_identity")
    if (
        not isinstance(expected_registry_identity, str)
        or len(expected_registry_identity) != 64
        or expected_registry_identity != actual_semantic_registry_identity
    ):
        raise ResearchContractError("semantic registry identity is not exactly bound")

    bindings = payload.get("identity_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != REQUIRED_IDENTITY_BINDINGS:
        raise ResearchContractError(
            "identity_bindings must contain the exact registered research files"
        )
    identity_paths: dict[str, Path] = {}
    for name in sorted(REQUIRED_IDENTITY_BINDINGS):
        binding = bindings[name]
        if not isinstance(binding, Mapping):
            raise ResearchContractError(f"identity binding {name} is invalid")
        path = _bound_path(repository, binding.get("path"), name=name)
        expected = binding.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ResearchContractError(f"identity binding {name} lacks SHA-256")
        if sha256_file(path) != expected:
            raise ResearchContractError(f"identity binding changed: {name}")
        identity_paths[name] = path
    for name, relative_path in REQUIRED_RUNTIME_CODE_BINDINGS.items():
        expected_path = (repository / relative_path).resolve(strict=False)
        if identity_paths[name] != expected_path:
            raise ResearchContractError(
                f"identity binding {name} must bind {relative_path}"
            )

    dataset = payload.get("dataset_version")
    if not isinstance(dataset, Mapping):
        raise ResearchContractError("dataset_version must be an object")
    dataset_path = _bound_path(repository, dataset.get("path"), name="dataset_version")
    dataset_sha = dataset.get("sha256")
    if not isinstance(dataset_sha, str) or len(dataset_sha) != 64:
        raise ResearchContractError("dataset_version.sha256 is required")
    if sha256_file(dataset_path) != dataset_sha:
        raise ResearchContractError("registered OHLCV identity changed")
    dataset_manifest = identity_paths["dataset_manifest"]
    if dataset.get("manifest_path") != str(dataset_manifest.relative_to(repository)):
        raise ResearchContractError("dataset manifest path is not exactly bound")
    if dataset.get("manifest_sha256") != sha256_file(dataset_manifest):
        raise ResearchContractError("dataset manifest hash is not exactly bound")
    split_role = dataset.get("split_role")
    if split_role != "brain_calibration_trial":
        raise ResearchContractError(
            "only brain_calibration_trial diagnostics are allowed"
        )
    split_registry = identity_paths["split_registry"]
    if dataset.get("split_registry") != str(split_registry.relative_to(repository)):
        raise ResearchContractError("dataset split registry path is not exactly bound")
    if dataset.get("split_registry_sha256") != sha256_file(split_registry):
        raise ResearchContractError("dataset split registry hash is not exactly bound")

    semantic_path = identity_paths["semantic_registry"]
    if payload.get("semantic_registry") != str(semantic_path.relative_to(repository)):
        raise ResearchContractError("semantic registry path is not exactly bound")

    warmup = payload.get("warmup_period")
    diagnostic = payload.get("diagnostic_period")
    if not isinstance(warmup, Mapping) or not isinstance(diagnostic, Mapping):
        raise ResearchContractError("warmup_period and diagnostic_period are required")
    warmup_start = _aware(warmup.get("start"), name="warmup_period.start")
    diagnostic_start = _aware(diagnostic.get("start"), name="diagnostic_period.start")
    diagnostic_end = _aware(
        diagnostic.get("end_exclusive"),
        name="diagnostic_period.end_exclusive",
    )
    if (
        warmup_start >= diagnostic_start
        or diagnostic_end <= diagnostic_start
        or warmup.get("outcomes_opened") is not False
    ):
        raise ResearchContractError("research periods or warmup authority are invalid")

    allowed_roles = payload.get("allowed_split_roles")
    if not isinstance(allowed_roles, Mapping):
        raise ResearchContractError("allowed_split_roles is required")
    diagnostic_roles = tuple(allowed_roles.get("diagnostic", ()))
    warmup_roles = tuple(allowed_roles.get("warmup", ()))
    if diagnostic_roles != ("brain_calibration_trial",) or not warmup_roles:
        raise ResearchContractError("diagnostic split authority is not fail closed")
    if any(
        token in role
        for role in (*diagnostic_roles, *warmup_roles)
        for token in FORBIDDEN_EVALUATION_ROLE_TOKENS
    ):
        raise ResearchContractError("OOF or holdout role cannot enter this diagnostic")

    return FrozenResearchContract(
        manifest_path=source,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        payload=payload,
        dataset_path=dataset_path,
        dataset_relative_path=str(dataset_path.relative_to(repository)),
        dataset_sha256=dataset_sha,
        split_registry_path=split_registry,
        split_registry_sha256=sha256_file(split_registry),
        split_role=str(split_role),
        semantic_registry_path=semantic_path,
        semantic_registry_identity=expected_registry_identity,
        model_path=identity_paths["model_config"],
        warmup_start=warmup_start,
        diagnostic_start=diagnostic_start,
        diagnostic_end=diagnostic_end,
        allowed_diagnostic_roles=diagnostic_roles,
        allowed_warmup_roles=warmup_roles,
        identity_paths=identity_paths,
    )


def validate_split_authority(contract: FrozenResearchContract, validation: Any) -> None:
    """Cross-check the manifest against the canonical split registry."""

    if validation.fingerprint != contract.split_registry_sha256:
        raise ResearchContractError("loaded split registry identity changed")
    source = validation.causal_source
    if (
        source.path != contract.dataset_relative_path
        or source.sha256 != contract.dataset_sha256
    ):
        raise ResearchContractError("split registry and experiment dataset disagree")
    dataset_manifest = contract.identity_paths["dataset_manifest"]
    manifest_label = contract.payload["dataset_version"]["manifest_path"]
    if source.manifest_path != manifest_label or source.manifest_sha256 != sha256_file(
        dataset_manifest
    ):
        raise ResearchContractError(
            "split registry and experiment dataset manifest disagree"
        )
    diagnostic = validation.classify_ohlcv(
        contract.diagnostic_start,
        contract.diagnostic_end,
    )
    if (
        diagnostic.role != contract.split_role
        or diagnostic.role not in contract.allowed_diagnostic_roles
    ):
        raise ResearchContractError(
            "diagnostic period does not have registered authority"
        )
    warmup = validation.classify_ohlcv(
        contract.warmup_start,
        contract.diagnostic_start,
    )
    if warmup.role not in contract.allowed_warmup_roles:
        raise ResearchContractError("warmup period crosses an unauthorized split")
    if any(token in diagnostic.role for token in FORBIDDEN_EVALUATION_ROLE_TOKENS):
        raise ResearchContractError("OOF or holdout evaluation is forbidden")


def completed_bar_distance(
    earlier: pd.Timestamp,
    later: pd.Timestamp,
    completed_index: Mapping[pd.Timestamp, int],
) -> int | None:
    """Return strictly later completed-real-bar distance, never wall time."""

    left = completed_index.get(earlier)
    right = completed_index.get(later)
    if left is None or right is None or right <= left:
        return None
    return int(right - left)


_ATOMIC_ORIGIN = "semantic_atomic"
_EXPLICIT_EVENT_ORIGINS = frozenset(
    {
        "normalized_data",
        "semantic_atomic",
        "state_projection",
        "legacy_transport",
    }
)


def _value(value: Any, name: str, default: Any = None) -> Any:
    return (
        value.get(name, default)
        if isinstance(value, Mapping)
        else getattr(value, name, default)
    )


def _origin(value: Any) -> str | None:
    origin = _value(value, "origin")
    if origin is None:
        return None
    raw = getattr(origin, "value", origin)
    return raw if isinstance(raw, str) else None


def _identity_values(value: Any, name: str) -> tuple[str, ...]:
    raw = _value(value, name, ())
    if isinstance(raw, (str, bytes)):
        raise ResearchContractError(f"{name} must be an explicit identity sequence")
    try:
        identities = tuple(raw or ())
    except TypeError as error:
        raise ResearchContractError(
            f"{name} must be an explicit identity sequence"
        ) from error
    if any(
        not isinstance(identity, str) or not identity.strip() for identity in identities
    ):
        raise ResearchContractError(f"{name} contains an invalid identity")
    if len(identities) != len(set(identities)):
        raise ResearchContractError(f"{name} contains duplicate identities")
    return identities


def direct_lineage_tokens(value: Any) -> frozenset[str]:
    """Extract typed provenance tokens without inferring event ancestry.

    ``source_ids`` and evidence fields are intentionally ignored.  They may
    contain historical candle, entity, or mixed transport identities and
    therefore cannot become research-lineage edges.
    """

    tokens: set[str] = set()
    event_id = _value(value, "event_id")
    if event_id:
        tokens.add(f"event:{event_id}")
    entity_id = _value(value, "entity_id")
    if entity_id:
        tokens.add(f"entity:{entity_id}")
    if _origin(value) == _ATOMIC_ORIGIN:
        for field in ("source_event_ids", "context_event_ids"):
            tokens.update(
                f"event:{identity}" for identity in _identity_values(value, field)
            )
    tokens.update(
        f"data:{identity}" for identity in _identity_values(value, "source_data_ids")
    )
    tokens.update(
        f"entity:{identity}"
        for identity in _identity_values(value, "source_entity_ids")
    )
    return frozenset(tokens)


def resolve_lineage_tokens(
    event_id: str,
    lookup: Callable[[str], Any | None],
    *,
    memo: dict[str, frozenset[str]] | None = None,
    maximum_depth: int = 64,
) -> frozenset[str]:
    """Resolve the closed atomic event DAG from explicit event namespaces.

    Only ``SEMANTIC_ATOMIC`` nodes are recursively traversed.  Their explicit
    ``source_event_ids`` and ``context_event_ids`` must resolve in the
    canonical store.  Normalized-data, state-projection, and legacy parents
    may be referenced as terminal context, but their opaque transports are
    never reinterpreted as event ancestry.
    """

    cache = {} if memo is None else memo
    visiting: set[str] = set()

    root = lookup(str(event_id))
    if root is None:
        raise ResearchContractError(
            f"canonical research event is unresolved: {event_id}"
        )
    if _origin(root) != _ATOMIC_ORIGIN:
        raise ResearchContractError(
            f"research lineage root is not semantic_atomic: {event_id}"
        )

    def visit(identity: str, depth: int) -> frozenset[str]:
        if identity in cache:
            return cache[identity]
        if depth > maximum_depth or identity in visiting:
            raise ResearchContractError("semantic source lineage is cyclic or too deep")
        event = lookup(identity)
        if event is None:
            raise ResearchContractError(
                f"canonical event lineage parent is unresolved: {identity}"
            )
        visiting.add(identity)
        tokens = set(direct_lineage_tokens(event))
        if _origin(event) == _ATOMIC_ORIGIN:
            references = (
                *_identity_values(event, "source_event_ids"),
                *_identity_values(event, "context_event_ids"),
            )
            for parent_id in references:
                parent = lookup(parent_id)
                if parent is None:
                    raise ResearchContractError(
                        "canonical event lineage parent is unresolved: " f"{parent_id}"
                    )
                parent_origin = _origin(parent)
                if parent_origin not in _EXPLICIT_EVENT_ORIGINS:
                    raise ResearchContractError(
                        "canonical event lineage parent lacks explicit "
                        f"EventOrigin: {parent_id}"
                    )
                tokens.add(f"event:{parent_id}")
                if parent_origin == _ATOMIC_ORIGIN:
                    tokens.update(visit(parent_id, depth + 1))
        visiting.remove(identity)
        result = frozenset(tokens)
        cache[identity] = result
        return result

    return visit(str(event_id), 0)


def resolve_source_lineage_tokens(
    event_id: str,
    lookup: Callable[[str], Any | None],
    *,
    memo: dict[str, frozenset[str]] | None = None,
    maximum_depth: int = 64,
) -> frozenset[str]:
    """Resolve directed semantic ancestry using ``source_event_ids`` only.

    ``context_event_ids`` remain available through :func:`resolve_lineage_tokens`
    for audit and interpretation, but can never prove source ancestry or
    constituent-BAR composition.
    """

    cache = {} if memo is None else memo
    visiting: set[str] = set()

    root = lookup(str(event_id))
    if root is None:
        raise ResearchContractError(
            f"canonical research event is unresolved: {event_id}"
        )
    if _origin(root) != _ATOMIC_ORIGIN:
        raise ResearchContractError(
            f"research source-lineage root is not semantic_atomic: {event_id}"
        )

    def visit(identity: str, depth: int) -> frozenset[str]:
        if identity in cache:
            return cache[identity]
        if depth > maximum_depth or identity in visiting:
            raise ResearchContractError("semantic source lineage is cyclic or too deep")
        event = lookup(identity)
        if event is None:
            raise ResearchContractError(
                f"canonical source-lineage parent is unresolved: {identity}"
            )
        visiting.add(identity)
        tokens = {f"event:{identity}"}
        if _origin(event) == _ATOMIC_ORIGIN:
            for parent_id in _identity_values(event, "source_event_ids"):
                parent = lookup(parent_id)
                if parent is None:
                    raise ResearchContractError(
                        "canonical source-lineage parent is unresolved: " f"{parent_id}"
                    )
                parent_origin = _origin(parent)
                if parent_origin not in _EXPLICIT_EVENT_ORIGINS:
                    raise ResearchContractError(
                        "canonical source-lineage parent lacks explicit "
                        f"EventOrigin: {parent_id}"
                    )
                tokens.add(f"event:{parent_id}")
                if parent_origin == _ATOMIC_ORIGIN:
                    tokens.update(visit(parent_id, depth + 1))
        visiting.remove(identity)
        result = frozenset(tokens)
        cache[identity] = result
        return result

    return visit(str(event_id), 0)


@dataclass(frozen=True)
class SourceLink:
    prior: Mapping[str, Any]
    completed_bars: int
    shared_tokens: tuple[str, ...]


class ResearchLinkMode(str, Enum):
    """Mutually exclusive evidence modes for a registered research edge.

    A temporal episode is deliberately not a weaker spelling of source
    ancestry.  Callers must inspect ``source_ancestry_proven`` and
    ``composition_proven`` before assigning meaning to a returned edge.
    """

    STRICT_SOURCE_ANCESTRY = "strict_source_ancestry"
    CROSS_TIMEFRAME_CONSTITUENT_BAR = "cross_timeframe_constituent_bar_composition"
    REGISTERED_TEMPORAL_EPISODE = "registered_temporal_episode"


@dataclass(frozen=True)
class TypedLinkSpec:
    previous_kind: str
    previous_timeframe: str
    current_kind: str
    current_timeframe: str
    maximum_completed_bars: int
    mode: ResearchLinkMode = ResearchLinkMode.STRICT_SOURCE_ANCESTRY
    constituent_bar_timeframe: str | None = None
    registered_episode_definition: str | None = None

    def __post_init__(self) -> None:
        try:
            mode = ResearchLinkMode(self.mode)
        except ValueError as error:
            raise ResearchContractError(
                "research link mode is not registered"
            ) from error
        object.__setattr__(self, "mode", mode)
        for name in (
            "previous_kind",
            "previous_timeframe",
            "current_kind",
            "current_timeframe",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ResearchContractError(f"typed link {name} is required")
        if (
            isinstance(self.maximum_completed_bars, bool)
            or not isinstance(self.maximum_completed_bars, int)
            or self.maximum_completed_bars < 1
        ):
            raise ResearchContractError(
                "typed link maximum_completed_bars must be positive"
            )
        if mode is ResearchLinkMode.CROSS_TIMEFRAME_CONSTITUENT_BAR:
            if (
                not isinstance(self.constituent_bar_timeframe, str)
                or not self.constituent_bar_timeframe
                or self.registered_episode_definition is not None
            ):
                raise ResearchContractError(
                    "constituent-BAR links require one registered BAR timeframe"
                )
        elif mode is ResearchLinkMode.REGISTERED_TEMPORAL_EPISODE:
            if (
                not isinstance(self.registered_episode_definition, str)
                or not self.registered_episode_definition
                or self.constituent_bar_timeframe is not None
            ):
                raise ResearchContractError(
                    "temporal episode links require an explicit registered definition"
                )
        elif (
            self.constituent_bar_timeframe is not None
            or self.registered_episode_definition is not None
        ):
            raise ResearchContractError(
                "strict ancestry links cannot carry composition or episode fields"
            )


@dataclass(frozen=True)
class TypedResearchLink:
    prior: Mapping[str, Any]
    completed_bars: int
    mode: ResearchLinkMode
    shared_event_ids: tuple[str, ...]
    source_ancestry_proven: bool
    registered_episode_definition: str | None = None

    @property
    def composition_proven(self) -> bool:
        """Whether this link proves registered constituent-BAR composition."""

        return (
            self.mode is ResearchLinkMode.CROSS_TIMEFRAME_CONSTITUENT_BAR
            and bool(self.shared_event_ids)
            and not self.source_ancestry_proven
        )

    @property
    def shared_tokens(self) -> tuple[str, ...]:
        """Compatibility projection; temporal episodes always return empty."""

        return tuple(f"event:{identity}" for identity in self.shared_event_ids)


def _explicit_text(value: Any, name: str) -> str:
    raw = _value(value, name)
    enum_value = getattr(raw, "value", raw)
    if not isinstance(enum_value, str) or not enum_value:
        raise ResearchContractError(f"{name} must be explicit non-empty text")
    return enum_value


def _constituent_bar_ids(value: Any) -> frozenset[str]:
    identities = frozenset(_identity_values(value, "constituent_bar_event_ids"))
    lineage_event_ids = frozenset(
        token.removeprefix("event:")
        for token in _value(value, "lineage_tokens", ())
        if isinstance(token, str) and token.startswith("event:")
    )
    if not identities.issubset(lineage_event_ids):
        raise ResearchContractError(
            "constituent BAR identities must already belong to canonical lineage"
        )
    return identities


def _validate_constituent_bar(
    event_id: str,
    lookup: Callable[[str], Any | None],
    *,
    timeframe: str,
    consumer_known_at: pd.Timestamp,
) -> None:
    event = lookup(event_id)
    if event is None:
        raise ResearchContractError(f"constituent BAR event is unresolved: {event_id}")
    if (
        _explicit_text(event, "kind") != "bar_completed"
        or _explicit_text(event, "timeframe") != timeframe
        or _origin(event) != "normalized_data"
    ):
        raise ResearchContractError(
            "constituent BAR identity must resolve to the registered normalized "
            f"{timeframe} BAR: {event_id}"
        )
    evidence_value = _value(event, "evidence", {})
    evidence = evidence_value if isinstance(evidence_value, Mapping) else {}

    def has_top_level(name: str) -> bool:
        return name in event if isinstance(event, Mapping) else hasattr(event, name)

    representations: list[tuple[object, object]] = []
    top_presence = tuple(
        has_top_level(name) for name in ("real_completed", "clock_only")
    )
    if any(top_presence):
        if not all(top_presence):
            representations.append((None, None))
        else:
            representations.append(
                (
                    _value(event, "real_completed"),
                    _value(event, "clock_only"),
                )
            )
    evidence_presence = tuple(
        name in evidence for name in ("real_completed", "clock_only")
    )
    if any(evidence_presence):
        if not all(evidence_presence):
            representations.append((None, None))
        else:
            representations.append(
                (evidence["real_completed"], evidence["clock_only"])
            )
    valid_representations = bool(representations) and all(
        type(real_completed) is bool
        and type(clock_only) is bool
        and clock_only is (not real_completed)
        for real_completed, clock_only in representations
    )
    if (
        not valid_representations
        or len(set(representations)) != 1
        or representations[0] != (True, False)
    ):
        raise ResearchContractError(
            "constituent BAR identity must resolve to an exact real "
            f"normalized BAR: {event_id}"
        )
    bar_known_at = pd.Timestamp(_value(event, "known_at"))
    if (
        bar_known_at.tzinfo is None
        or consumer_known_at.tzinfo is None
        or bar_known_at > consumer_known_at
    ):
        raise ResearchContractError(
            f"constituent BAR is not known by its consumer: {event_id}"
        )


def _canonical_typed_scope(
    event: Any,
    *,
    event_lookup: Callable[[str], Any | None] | None,
    source_lineage: frozenset[str] | None,
) -> tuple[str, int]:
    """Resolve market scope without reinterpreting semantic-event evidence.

    ``MarketEvent`` deliberately has no top-level symbol or instrument field.
    Canonical semantic facts inherit that scope from their source-only
    normalized BAR ancestry.  Context ancestry is explanatory and must never
    establish a typed research edge's market identity.  Lightweight external
    canonical records may expose the two fields directly.
    """

    direct_symbol = _value(event, "symbol")
    direct_instrument = _value(event, "instrument_id")
    if direct_symbol is not None or direct_instrument is not None:
        symbol = _explicit_text(event, "symbol")
        if isinstance(direct_instrument, bool) or not isinstance(
            direct_instrument, Integral
        ):
            raise ResearchContractError(
                "canonical typed link instrument_id must be an integer"
            )
        return symbol, int(direct_instrument)

    if event_lookup is None:
        raise ResearchContractError(
            "canonical typed link market scope requires event lookup"
        )
    event_id = _explicit_text(event, "event_id")
    lineage = (
        resolve_source_lineage_tokens(event_id, event_lookup)
        if source_lineage is None
        else source_lineage
    )
    scopes: set[tuple[str, int]] = set()
    for token in lineage:
        if not token.startswith("event:"):
            continue
        ancestor = event_lookup(token.removeprefix("event:"))
        if (
            ancestor is None
            or _origin(ancestor) != "normalized_data"
            or _explicit_text(ancestor, "kind") != "bar_completed"
        ):
            continue
        evidence = _value(ancestor, "evidence")
        if not isinstance(evidence, Mapping):
            raise ResearchContractError(
                "canonical normalized BAR scope evidence is missing"
            )
        symbol = _explicit_text(evidence, "symbol")
        instrument_id = _value(evidence, "instrument_id")
        if isinstance(instrument_id, bool) or not isinstance(
            instrument_id, Integral
        ):
            raise ResearchContractError(
                "canonical normalized BAR instrument_id must be an integer"
            )
        scopes.add((symbol, int(instrument_id)))
    if len(scopes) != 1:
        raise ResearchContractError(
            "canonical typed link source-only BAR scope must resolve uniquely"
        )
    return next(iter(scopes))


def _canonical_typed_direction(event: Any) -> str:
    """Resolve the sole registered derived research direction."""

    direction = _value(event, "direction")
    if direction is not None:
        return _explicit_text(event, "direction")
    if _explicit_text(event, "kind") == "level_touched":
        side = _explicit_text(event, "side")
        if side == "above":
            return "short"
        if side == "below":
            return "long"
        raise ResearchContractError(
            "canonical level_touched side cannot derive research direction"
        )
    raise ResearchContractError(
        "canonical typed link direction must be explicit or registered"
    )


def _validate_canonical_typed_endpoint(
    record: Mapping[str, Any],
    event: Any,
    *,
    event_lookup: Callable[[str], Any | None] | None = None,
    source_lineage: frozenset[str] | None = None,
) -> None:
    """Reject an enriched research row that disagrees with its source event."""

    for field in ("event_id", "kind", "timeframe"):
        if _explicit_text(record, field) != _explicit_text(event, field):
            raise ResearchContractError(
                f"typed link {field} disagrees with the canonical event"
            )
    if _explicit_text(record, "direction") != _canonical_typed_direction(event):
        raise ResearchContractError(
            "typed link direction disagrees with the canonical event"
        )
    record_symbol = _explicit_text(record, "symbol")
    record_instrument = _value(record, "instrument_id")
    if isinstance(record_instrument, bool) or not isinstance(
        record_instrument, Integral
    ):
        raise ResearchContractError("typed link instrument_id must be an integer")
    event_symbol, event_instrument = _canonical_typed_scope(
        event,
        event_lookup=event_lookup,
        source_lineage=source_lineage,
    )
    if record_symbol != event_symbol:
        raise ResearchContractError(
            "typed link symbol disagrees with the canonical event source scope"
        )
    if int(record_instrument) != event_instrument:
        raise ResearchContractError(
            "typed link instrument_id disagrees with the canonical event source scope"
        )
    record_known_at = pd.Timestamp(_value(record, "known_at"))
    event_known_at = pd.Timestamp(_value(event, "known_at"))
    if (
        record_known_at.tzinfo is None
        or event_known_at.tzinfo is None
        or record_known_at != event_known_at
    ):
        raise ResearchContractError(
            "typed link known_at disagrees with the canonical event"
        )


def _find_prior_typed_links(
    records: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any],
    *,
    completed_index: Mapping[pd.Timestamp, int],
    spec: TypedLinkSpec,
    event_lookup: Callable[[str], Any | None] | None = None,
    maximum_matches: int | None,
) -> tuple[TypedResearchLink, ...]:
    """Resolve typed edges without conflating time with provenance.

    Strict ancestry intersects explicit event-lineage tokens.  Cross-timeframe
    composition intersects a separately declared set of constituent normalized
    BAR event identities and verifies every shared BAR in the canonical store.
    A registered temporal episode may use the same causal clock filters, but it
    is returned with ``source_ancestry_proven=False`` and no shared identities.
    """

    if (
        current.get("kind") != spec.current_kind
        or current.get("timeframe") != spec.current_timeframe
    ):
        raise ResearchContractError(
            "typed link current event is outside its registered population"
        )
    current_direction = _explicit_text(current, "direction")
    current_symbol = _explicit_text(current, "symbol")
    current_instrument = _value(current, "instrument_id")
    if isinstance(current_instrument, bool) or not isinstance(
        current_instrument, Integral
    ):
        raise ResearchContractError("typed link instrument_id must be an integer")
    current_instrument = int(current_instrument)
    current_known_at = pd.Timestamp(current["known_at"])
    if current_known_at.tzinfo is None:
        raise ResearchContractError(
            "typed link current known_at must be timezone aware"
        )

    matches: list[TypedResearchLink] = []
    ordered: list[tuple[pd.Timestamp, Mapping[str, Any]]] = []
    for prior in records:
        prior_known_at = pd.Timestamp(prior["known_at"])
        if prior_known_at.tzinfo is None:
            raise ResearchContractError(
                "typed link prior known_at must be timezone aware"
            )
        ordered.append((prior_known_at, prior))
    ordered.sort(
        key=lambda item: (item[0], str(item[1].get("event_id", ""))),
        reverse=True,
    )
    for prior_known_at, prior in ordered:
        distance = completed_bar_distance(
            prior_known_at,
            current_known_at,
            completed_index,
        )
        if distance is None or distance > spec.maximum_completed_bars:
            continue
        if (
            prior.get("kind") != spec.previous_kind
            or prior.get("timeframe") != spec.previous_timeframe
            or prior.get("direction") != current_direction
            or prior.get("symbol") != current_symbol
            or prior.get("instrument_id") != current_instrument
        ):
            continue

        if spec.mode is ResearchLinkMode.STRICT_SOURCE_ANCESTRY:
            current_id = _explicit_text(current, "event_id")
            prior_id = _explicit_text(prior, "event_id")
            if f"event:{prior_id}" not in frozenset(current.get("lineage_tokens", ())):
                # A missing directional edge is simply outside the strict
                # population.  Canonical lookup is required only when an
                # enriched row actually claims that edge.
                continue
            if event_lookup is None:
                raise ResearchContractError(
                    "strict source ancestry requires canonical event lookup"
                )
            canonical_current = event_lookup(current_id)
            canonical_prior = event_lookup(prior_id)
            if (
                canonical_current is None
                or canonical_prior is None
                or _origin(canonical_current) != _ATOMIC_ORIGIN
                or _origin(canonical_prior) != _ATOMIC_ORIGIN
            ):
                raise ResearchContractError(
                    "strict source ancestry endpoints must resolve to canonical "
                    "semantic_atomic events"
                )
            current_lineage = resolve_source_lineage_tokens(current_id, event_lookup)
            prior_lineage = resolve_source_lineage_tokens(prior_id, event_lookup)
            _validate_canonical_typed_endpoint(
                current,
                canonical_current,
                event_lookup=event_lookup,
                source_lineage=current_lineage,
            )
            _validate_canonical_typed_endpoint(
                prior,
                canonical_prior,
                event_lookup=event_lookup,
                source_lineage=prior_lineage,
            )
            if f"event:{prior_id}" not in current_lineage:
                # Sharing some older BAR or semantic ancestor would only prove
                # sibling composition.  It cannot become a directional source
                # edge between these two semantic facts.
                continue
            matches.append(
                TypedResearchLink(
                    prior=prior,
                    completed_bars=distance,
                    mode=spec.mode,
                    shared_event_ids=(prior_id,),
                    source_ancestry_proven=True,
                )
            )
            if maximum_matches is not None and len(matches) >= maximum_matches:
                return tuple(matches)
            continue

        if spec.mode is ResearchLinkMode.CROSS_TIMEFRAME_CONSTITUENT_BAR:
            if event_lookup is None:
                raise ResearchContractError(
                    "constituent-BAR composition requires canonical event lookup"
                )
            current_id = _explicit_text(current, "event_id")
            prior_id = _explicit_text(prior, "event_id")
            canonical_current = event_lookup(current_id)
            canonical_prior = event_lookup(prior_id)
            if (
                canonical_current is None
                or canonical_prior is None
                or _origin(canonical_current) != _ATOMIC_ORIGIN
                or _origin(canonical_prior) != _ATOMIC_ORIGIN
            ):
                raise ResearchContractError(
                    "constituent-BAR endpoints must resolve to canonical "
                    "semantic_atomic events"
                )
            current_source_lineage = resolve_source_lineage_tokens(
                current_id, event_lookup
            )
            prior_source_lineage = resolve_source_lineage_tokens(prior_id, event_lookup)
            _validate_canonical_typed_endpoint(
                current,
                canonical_current,
                event_lookup=event_lookup,
                source_lineage=current_source_lineage,
            )
            _validate_canonical_typed_endpoint(
                prior,
                canonical_prior,
                event_lookup=event_lookup,
                source_lineage=prior_source_lineage,
            )
            current_constituents = _constituent_bar_ids(current)
            prior_constituents = _constituent_bar_ids(prior)
            current_source_event_ids = {
                token.removeprefix("event:")
                for token in current_source_lineage
                if token.startswith("event:")
            }
            prior_source_event_ids = {
                token.removeprefix("event:")
                for token in prior_source_lineage
                if token.startswith("event:")
            }
            if not current_constituents.issubset(
                current_source_event_ids
            ) or not prior_constituents.issubset(prior_source_event_ids):
                raise ResearchContractError(
                    "constituent BAR identities must belong to canonical "
                    "source-only lineage"
                )
            shared = tuple(
                sorted(current_constituents.intersection(prior_constituents))
            )
            if not shared:
                continue
            for event_id in shared:
                _validate_constituent_bar(
                    event_id,
                    event_lookup,
                    timeframe=str(spec.constituent_bar_timeframe),
                    consumer_known_at=pd.Timestamp(prior["known_at"]),
                )
            matches.append(
                TypedResearchLink(
                    prior=prior,
                    completed_bars=distance,
                    mode=spec.mode,
                    shared_event_ids=shared,
                    source_ancestry_proven=False,
                )
            )
            if maximum_matches is not None and len(matches) >= maximum_matches:
                return tuple(matches)
            continue

        matches.append(
            TypedResearchLink(
                prior=prior,
                completed_bars=distance,
                mode=spec.mode,
                shared_event_ids=(),
                source_ancestry_proven=False,
                registered_episode_definition=spec.registered_episode_definition,
            )
        )
        if maximum_matches is not None and len(matches) >= maximum_matches:
            return tuple(matches)
    return tuple(matches)


def find_prior_typed_links(
    records: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any],
    *,
    completed_index: Mapping[pd.Timestamp, int],
    spec: TypedLinkSpec,
    event_lookup: Callable[[str], Any | None] | None = None,
) -> tuple[TypedResearchLink, ...]:
    """Return every registered predecessor instead of choosing one chain.

    The returned order is deterministic (latest eligible predecessor first).
    This is a research projection only: multiple temporal candidates remain
    candidate relations and gain neither ancestry nor causal authority.
    """

    return _find_prior_typed_links(
        records,
        current,
        completed_index=completed_index,
        spec=spec,
        event_lookup=event_lookup,
        maximum_matches=None,
    )


def find_prior_typed_link(
    records: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any],
    *,
    completed_index: Mapping[pd.Timestamp, int],
    spec: TypedLinkSpec,
    event_lookup: Callable[[str], Any | None] | None = None,
) -> TypedResearchLink | None:
    """Resolve the latest eligible typed edge for legacy linear callers."""

    matches = _find_prior_typed_links(
        records,
        current,
        completed_index=completed_index,
        spec=spec,
        event_lookup=event_lookup,
        maximum_matches=1,
    )
    return matches[0] if matches else None


def find_prior_source_link(
    records: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any],
    *,
    completed_index: Mapping[pd.Timestamp, int],
    kinds: frozenset[str],
    maximum_completed_bars: int,
    timeframe: str | None = None,
) -> SourceLink | None:
    """Find a causal predecessor with explicit shared immutable lineage."""

    current_tokens = frozenset(current.get("lineage_tokens", ()))
    if not current_tokens:
        return None
    for prior in reversed(records):
        distance = completed_bar_distance(
            prior["known_at"],
            current["known_at"],
            completed_index,
        )
        if distance is None:
            continue
        if distance > maximum_completed_bars:
            break
        if (
            prior.get("kind") not in kinds
            or prior.get("direction") != current.get("direction")
            or prior.get("symbol") != current.get("symbol")
            or prior.get("instrument_id") != current.get("instrument_id")
            or (timeframe is not None and prior.get("timeframe") != timeframe)
        ):
            continue
        shared = tuple(
            sorted(
                token
                for token in current_tokens.intersection(
                    prior.get("lineage_tokens", ())
                )
                if token.startswith("event:")
            )
        )
        if shared:
            return SourceLink(
                prior=prior, completed_bars=distance, shared_tokens=shared
            )
    return None


_EPISODE_IDENTITY_FIELDS = (
    "kind",
    "symbol",
    "instrument_id",
    "known_at",
    "direction",
    "timeframe",
)


def _identity_json(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            raise ResearchContractError(
                "research identity clock must be timezone aware"
            )
        return value.isoformat()
    enum_value = getattr(value, "value", value)
    if isinstance(enum_value, (str, int, float, bool)) or enum_value is None:
        return enum_value
    raise ResearchContractError("research identity contains a non-canonical value")


def canonical_treatment_episodes(
    events: Sequence[Mapping[str, Any]],
    *,
    identity_fields: Sequence[str] = _EPISODE_IDENTITY_FIELDS,
) -> tuple[dict[str, Any], ...]:
    """Collapse duplicate path observations into deterministic signal episodes.

    The constituent events remain auditable.  The returned episode is the
    analysis unit for path outcomes; it must not be substituted for a semantic
    market event in the Trading Eye.
    """

    fields = tuple(identity_fields)
    if (
        not fields
        or len(fields) != len(set(fields))
        or any(not isinstance(name, str) or not name for name in fields)
        or "known_at" not in fields
        or "direction" not in fields
        or "symbol" not in fields
        or "instrument_id" not in fields
    ):
        raise ResearchContractError("treatment episode identity fields are invalid")
    event_ids: set[str] = set()
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        event_id = _explicit_text(event, "event_id")
        if event_id in event_ids:
            raise ResearchContractError(
                f"treatment episode contains duplicate event_id: {event_id}"
            )
        event_ids.add(event_id)
        key = tuple(_identity_json(_value(event, name)) for name in fields)
        if any(value is None or value == "" for value in key):
            raise ResearchContractError("treatment episode identity is incomplete")
        grouped[key].append(event)

    episodes: list[dict[str, Any]] = []
    for key, members in grouped.items():
        canonical_key = dict(zip(fields, key, strict=True))
        payload = json.dumps(
            canonical_key,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        episode_id = hashlib.sha256(
            f"research_treatment_episode_v1|{payload}".encode("utf-8")
        ).hexdigest()
        constituent_ids = tuple(
            sorted(_explicit_text(member, "event_id") for member in members)
        )
        episode = {
            **canonical_key,
            "known_at": pd.Timestamp(canonical_key["known_at"]),
            "event_id": f"episode:{episode_id[:24]}",
            "episode_id": episode_id,
            "constituent_event_ids": constituent_ids,
            "constituent_event_count": len(constituent_ids),
            "analysis_origin": "canonical_treatment_episode_v1",
        }
        episodes.append(episode)
    episodes.sort(key=lambda item: (item["known_at"], item["event_id"]))
    return tuple(episodes)


@dataclass(frozen=True)
class BranchingResearchEdge:
    """One auditable relation in a research-only branching episode."""

    edge_id: str
    relation_id: str
    prior_event_id: str
    current_event_id: str
    prior_known_at: pd.Timestamp
    current_known_at: pd.Timestamp
    completed_bars: int
    mode: ResearchLinkMode
    shared_event_ids: tuple[str, ...]
    source_ancestry_proven: bool
    composition_proven: bool
    registered_episode_definition: str | None


@dataclass(frozen=True)
class BranchingResearchEpisode:
    """Rooted typed projection; it is not an Eye event or a probability model."""

    episode_id: str
    root_event_id: str
    event_ids: tuple[str, ...]
    terminal_event_ids: tuple[str, ...]
    edge_ledger: tuple[BranchingResearchEdge, ...]


def project_branching_research_episode(
    records: Sequence[Mapping[str, Any]],
    *,
    root_event_id: str,
    completed_index: Mapping[pd.Timestamp, int],
    link_specs: Mapping[str, TypedLinkSpec],
    event_lookup: Callable[[str], Any | None] | None = None,
) -> BranchingResearchEpisode:
    """Project every reachable registered relation without requiring a chain.

    Relation names and all time windows come from ``link_specs``.  The
    projection assigns no probability, does not fill missing stages, and does
    not reinterpret temporal proximity as semantic ancestry.  Disconnected
    records remain outside the rooted episode, while every eligible incoming
    and outgoing relation inside its reachable closure is retained.
    """

    if not isinstance(root_event_id, str) or not root_event_id:
        raise ResearchContractError("branching episode root_event_id is required")
    if not isinstance(link_specs, Mapping) or not link_specs:
        raise ResearchContractError(
            "branching episode requires explicit typed link specs"
        )

    by_id: dict[str, Mapping[str, Any]] = {}
    known_at_by_id: dict[str, pd.Timestamp] = {}
    for record in records:
        event_id = _explicit_text(record, "event_id")
        if event_id in by_id:
            raise ResearchContractError(
                f"branching episode contains duplicate event_id: {event_id}"
            )
        known_at = pd.Timestamp(_value(record, "known_at"))
        if known_at.tzinfo is None:
            raise ResearchContractError(
                "branching episode event known_at must be timezone aware"
            )
        by_id[event_id] = record
        known_at_by_id[event_id] = known_at
    if root_event_id not in by_id:
        raise ResearchContractError("branching episode root event is unresolved")

    registered_specs: list[tuple[str, TypedLinkSpec]] = []
    for relation_id, spec in link_specs.items():
        if not isinstance(relation_id, str) or not relation_id:
            raise ResearchContractError(
                "branching episode relation_id must be explicit text"
            )
        if not isinstance(spec, TypedLinkSpec):
            raise ResearchContractError(
                f"branching episode relation {relation_id} lacks TypedLinkSpec"
            )
        registered_specs.append((relation_id, spec))
    registered_specs.sort(key=lambda item: item[0])

    ordered_records = tuple(
        sorted(
            records,
            key=lambda item: (
                pd.Timestamp(item["known_at"]),
                str(item["event_id"]),
            ),
        )
    )
    candidates: list[BranchingResearchEdge] = []
    for relation_id, spec in registered_specs:
        current_records = (
            record
            for record in ordered_records
            if record.get("kind") == spec.current_kind
            and record.get("timeframe") == spec.current_timeframe
        )
        for current in current_records:
            current_id = _explicit_text(current, "event_id")
            links = find_prior_typed_links(
                ordered_records,
                current,
                completed_index=completed_index,
                spec=spec,
                event_lookup=event_lookup,
            )
            for link in links:
                prior_id = _explicit_text(link.prior, "event_id")
                edge_payload = json.dumps(
                    {
                        "relation_id": relation_id,
                        "prior_event_id": prior_id,
                        "current_event_id": current_id,
                        "completed_bars": link.completed_bars,
                        "mode": link.mode.value,
                        "shared_event_ids": link.shared_event_ids,
                        "source_ancestry_proven": link.source_ancestry_proven,
                        "composition_proven": link.composition_proven,
                        "registered_episode_definition": (
                            link.registered_episode_definition
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                edge_id = hashlib.sha256(
                    f"branching_research_edge_v1|{edge_payload}".encode("utf-8")
                ).hexdigest()
                candidates.append(
                    BranchingResearchEdge(
                        edge_id=edge_id,
                        relation_id=relation_id,
                        prior_event_id=prior_id,
                        current_event_id=current_id,
                        prior_known_at=known_at_by_id[prior_id],
                        current_known_at=known_at_by_id[current_id],
                        completed_bars=link.completed_bars,
                        mode=link.mode,
                        shared_event_ids=link.shared_event_ids,
                        source_ancestry_proven=link.source_ancestry_proven,
                        composition_proven=link.composition_proven,
                        registered_episode_definition=(
                            link.registered_episode_definition
                        ),
                    )
                )

    candidates.sort(
        key=lambda edge: (
            edge.prior_known_at,
            edge.current_known_at,
            edge.relation_id,
            edge.prior_event_id,
            edge.current_event_id,
        )
    )
    outgoing: dict[str, list[BranchingResearchEdge]] = defaultdict(list)
    for edge in candidates:
        outgoing[edge.prior_event_id].append(edge)

    reachable = {root_event_id}
    selected: dict[str, BranchingResearchEdge] = {}
    pending = deque((root_event_id,))
    while pending:
        prior_id = pending.popleft()
        for edge in outgoing.get(prior_id, ()):
            selected[edge.edge_id] = edge
            if edge.current_event_id not in reachable:
                reachable.add(edge.current_event_id)
                pending.append(edge.current_event_id)

    ledger = tuple(
        sorted(
            selected.values(),
            key=lambda edge: (
                edge.prior_known_at,
                edge.current_known_at,
                edge.relation_id,
                edge.prior_event_id,
                edge.current_event_id,
            ),
        )
    )
    event_ids = tuple(
        sorted(
            reachable,
            key=lambda event_id: (known_at_by_id[event_id], event_id),
        )
    )
    source_ids = frozenset(edge.prior_event_id for edge in ledger)
    terminal_ids = tuple(
        event_id for event_id in event_ids if event_id not in source_ids
    )
    episode_payload = json.dumps(
        {
            "root_event_id": root_event_id,
            "event_ids": event_ids,
            "edge_ids": tuple(edge.edge_id for edge in ledger),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    episode_id = hashlib.sha256(
        f"branching_research_episode_v1|{episode_payload}".encode("utf-8")
    ).hexdigest()
    return BranchingResearchEpisode(
        episode_id=episode_id,
        root_event_id=root_event_id,
        event_ids=event_ids,
        terminal_event_ids=terminal_ids,
        edge_ledger=ledger,
    )


class ControlDirectionPolicy(str, Enum):
    INHERIT_TREATMENT_AFTER_KNOWN_AT = "inherit_treatment_after_known_at"
    CANDIDATE_LOCAL = "candidate_local"


@dataclass(frozen=True)
class MatchSpec:
    exact_fields: tuple[str, ...]
    maximum_completed_bar_offset: int
    outcome_horizon_completed_bars: int
    embargo_completed_bars: int = 0
    forward_only: bool = True
    replacement_limit: int = 1
    direction_policy: ControlDirectionPolicy = (
        ControlDirectionPolicy.INHERIT_TREATMENT_AFTER_KNOWN_AT
    )
    treatment_clock_field: str = "known_at"
    candidate_clock_field: str = "known_at"
    treatment_id_field: str = "event_id"
    candidate_id_field: str = "candidate_id"

    def __post_init__(self) -> None:
        fields = tuple(self.exact_fields)
        object.__setattr__(self, "exact_fields", fields)
        try:
            policy = ControlDirectionPolicy(self.direction_policy)
        except ValueError as error:
            raise ResearchContractError(
                "control direction policy is invalid"
            ) from error
        object.__setattr__(self, "direction_policy", policy)
        if (
            len(fields) != len(set(fields))
            or not {"symbol", "instrument_id"}.issubset(fields)
            or any(not isinstance(name, str) or not name for name in fields)
        ):
            raise ResearchContractError(
                "matching exact fields must include symbol and instrument_id"
            )
        integers = (
            self.maximum_completed_bar_offset,
            self.outcome_horizon_completed_bars,
            self.embargo_completed_bars,
            self.replacement_limit,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in integers
        ):
            raise ResearchContractError(
                "matching distances and capacity must be integers"
            )
        if (
            self.maximum_completed_bar_offset < 1
            or self.outcome_horizon_completed_bars < 0
            or self.embargo_completed_bars < 0
            or self.replacement_limit < 1
        ):
            raise ResearchContractError("matching distances or capacity are invalid")
        if not self.forward_only:
            raise ResearchContractError(
                "registered causal controls must be forward-only"
            )
        minimum_offset = (
            self.outcome_horizon_completed_bars + self.embargo_completed_bars + 1
        )
        if self.maximum_completed_bar_offset < minimum_offset:
            raise ResearchContractError(
                "matching caliper cannot satisfy outcome-horizon embargo"
            )
        for name in (
            "treatment_clock_field",
            "candidate_clock_field",
            "treatment_id_field",
            "candidate_id_field",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ResearchContractError(f"MatchSpec.{name} is required")


@dataclass(frozen=True)
class MatchedControlPair:
    treatment_id: str
    candidate_id: str
    treatment: Mapping[str, Any]
    candidate: Mapping[str, Any]
    completed_bar_offset: int
    control_direction: str
    direction_known_at: pd.Timestamp


@dataclass(frozen=True)
class MatchResult:
    pairs: tuple[MatchedControlPair, ...]
    unmatched: Mapping[str, str]
    requested: int
    eligible_candidates: int

    @property
    def matched(self) -> int:
        return len(self.pairs)


def _validated_completed_index(
    completed_index: Mapping[pd.Timestamp, int],
    *,
    role: str,
) -> int:
    positions: list[int] = []
    for clock, position in completed_index.items():
        timestamp = pd.Timestamp(clock)
        if timestamp.tzinfo is None:
            raise ResearchContractError(f"{role} clock must be timezone aware")
        if isinstance(position, bool) or not isinstance(position, Integral):
            raise ResearchContractError(f"{role} position must be an integer")
        positions.append(int(position))
    if len(positions) != len(set(positions)):
        raise ResearchContractError(f"{role} positions must be unique")
    if positions:
        ordered = sorted(positions)
        if ordered != list(range(ordered[0], ordered[-1] + 1)):
            raise ResearchContractError(f"{role} positions must be contiguous")
        return ordered[-1]
    return -1


def _matching_stratum(
    value: Mapping[str, Any],
    fields: Sequence[str],
    *,
    role: str,
) -> tuple[Any, ...]:
    stratum: list[Any] = []
    for field in fields:
        field_value = _value(value, field)
        if field_value is None or (isinstance(field_value, str) and not field_value):
            raise ResearchContractError(
                f"{role} matching stratum field is missing: {field}"
            )
        if field == "symbol" and not isinstance(field_value, str):
            raise ResearchContractError(f"{role} symbol must be text")
        if field == "instrument_id" and (
            isinstance(field_value, bool) or not isinstance(field_value, Integral)
        ):
            raise ResearchContractError(f"{role} instrument_id must be an integer")
        try:
            hash(field_value)
        except TypeError as error:
            raise ResearchContractError(
                f"{role} matching stratum field is not canonical: {field}"
            ) from error
        stratum.append(field_value)
    return tuple(stratum)


def deterministic_maximum_cardinality_match(
    treatments: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    completed_index: Mapping[pd.Timestamp, int],
    spec: MatchSpec,
) -> MatchResult:
    """Build a deterministic maximum-cardinality causal control matching.

    Candidate edges are sparse because the exact stratum, forward-only clock,
    full outcome horizon, embargo, and completed-bar caliper are applied before
    matching.  The augmenting-path algorithm maximizes cardinality; sorted
    identities make the selected maximum deterministic rather than input-order
    dependent.
    """

    max_index = _validated_completed_index(
        completed_index,
        role="matching completed index",
    )
    treatment_by_id: dict[str, Mapping[str, Any]] = {}
    candidate_by_id: dict[str, Mapping[str, Any]] = {}
    for value, field, target, label in (
        (treatments, spec.treatment_id_field, treatment_by_id, "treatment"),
        (candidates, spec.candidate_id_field, candidate_by_id, "candidate"),
    ):
        for item in value:
            identity = _explicit_text(item, field)
            if identity in target:
                raise ResearchContractError(
                    f"duplicate {label} matching identity: {identity}"
                )
            target[identity] = item

    minimum_offset = (
        spec.outcome_horizon_completed_bars + spec.embargo_completed_bars + 1
    )
    candidate_strata: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    candidate_clocks: dict[str, pd.Timestamp] = {}
    candidate_indices: dict[str, int | None] = {}
    candidate_direction_known_at: dict[str, pd.Timestamp] = {}
    for candidate_id, candidate in candidate_by_id.items():
        candidate_clock = pd.Timestamp(candidate[spec.candidate_clock_field])
        if candidate_clock.tzinfo is None:
            raise ResearchContractError(
                "matching candidate clock must be timezone aware"
            )
        candidate_clocks[candidate_id] = candidate_clock
        candidate_indices[candidate_id] = completed_index.get(candidate_clock)
        if spec.direction_policy is ControlDirectionPolicy.CANDIDATE_LOCAL:
            if "direction_known_at" not in candidate:
                raise ResearchContractError(
                    "candidate-local control direction requires direction_known_at"
                )
            direction_known_at = pd.Timestamp(candidate["direction_known_at"])
            if (
                direction_known_at.tzinfo is None
                or direction_known_at > candidate_clock
            ):
                raise ResearchContractError(
                    "candidate-local control direction is future-known"
                )
            candidate_direction_known_at[candidate_id] = direction_known_at
        candidate_strata[
            _matching_stratum(
                candidate,
                spec.exact_fields,
                role="candidate",
            )
        ].append(candidate_id)
    for candidate_ids in candidate_strata.values():
        candidate_ids.sort()

    adjacency: dict[str, tuple[tuple[str, int], ...]] = {}
    preliminary_reason: dict[str, str] = {}
    for treatment_id, treatment in treatment_by_id.items():
        treatment_clock = pd.Timestamp(treatment[spec.treatment_clock_field])
        if treatment_clock.tzinfo is None:
            raise ResearchContractError(
                "matching treatment clock must be timezone aware"
            )
        treatment_index = completed_index.get(treatment_clock)
        if treatment_index is None:
            preliminary_reason[treatment_id] = "treatment_clock_not_completed"
            adjacency[treatment_id] = ()
            continue
        if treatment_index + spec.outcome_horizon_completed_bars > max_index:
            preliminary_reason[treatment_id] = "treatment_outcome_horizon_incomplete"
            adjacency[treatment_id] = ()
            continue
        treatment_direction = _explicit_text(treatment, "direction")
        edges: list[tuple[int, str, int]] = []
        stratum = _matching_stratum(
            treatment,
            spec.exact_fields,
            role="treatment",
        )
        stratum_candidates = candidate_strata.get(stratum, ())
        for candidate_id in stratum_candidates:
            candidate = candidate_by_id[candidate_id]
            candidate_index = candidate_indices[candidate_id]
            if candidate_index is None:
                continue
            offset = int(candidate_index - treatment_index)
            if (
                offset < minimum_offset
                or offset > spec.maximum_completed_bar_offset
                or candidate_index + spec.outcome_horizon_completed_bars > max_index
            ):
                continue
            if spec.direction_policy is ControlDirectionPolicy.CANDIDATE_LOCAL:
                if _value(candidate, "direction") != treatment_direction:
                    continue
            for slot in range(spec.replacement_limit):
                edges.append((offset, candidate_id, slot))
        if not edges:
            preliminary_reason[treatment_id] = (
                "no_exact_stratum"
                if not stratum_candidates
                else "no_causal_caliper_edge"
            )
        adjacency[treatment_id] = tuple(
            (candidate_id, slot) for _, candidate_id, slot in sorted(edges)
        )

    slot_owner: dict[tuple[str, int], str] = {}
    chosen_slot: dict[str, tuple[str, int]] = {}

    def augment(root_treatment_id: str) -> bool:
        """Find and flip one augmenting path without recursion depth risk."""

        queue: deque[str] = deque((root_treatment_id,))
        seen_treatments = {root_treatment_id}
        seen_slots: set[tuple[str, int]] = set()
        parent_by_slot: dict[tuple[str, int], str] = {}
        free_slot: tuple[str, int] | None = None
        while queue and free_slot is None:
            treatment_id = queue.popleft()
            for slot in adjacency[treatment_id]:
                if slot in seen_slots:
                    continue
                seen_slots.add(slot)
                parent_by_slot[slot] = treatment_id
                owner = slot_owner.get(slot)
                if owner is None:
                    free_slot = slot
                    break
                if owner not in seen_treatments:
                    seen_treatments.add(owner)
                    queue.append(owner)
        if free_slot is None:
            return False
        slot = free_slot
        while True:
            treatment_id = parent_by_slot[slot]
            previous_slot = chosen_slot.get(treatment_id)
            slot_owner[slot] = treatment_id
            chosen_slot[treatment_id] = slot
            if previous_slot is None:
                break
            slot = previous_slot
        return True

    matching_order = sorted(
        treatment_by_id,
        key=lambda identity: (
            len(adjacency[identity]),
            pd.Timestamp(treatment_by_id[identity][spec.treatment_clock_field]),
            identity,
        ),
    )
    for treatment_id in matching_order:
        if adjacency[treatment_id]:
            augment(treatment_id)

    pairs: list[MatchedControlPair] = []
    unmatched: dict[str, str] = {}
    for treatment_id, treatment in treatment_by_id.items():
        slot = chosen_slot.get(treatment_id)
        if slot is None:
            unmatched[treatment_id] = preliminary_reason.get(
                treatment_id,
                "candidate_capacity_exhausted",
            )
            continue
        candidate = candidate_by_id[slot[0]]
        treatment_clock = pd.Timestamp(treatment[spec.treatment_clock_field])
        candidate_clock = candidate_clocks[slot[0]]
        direction = (
            _explicit_text(candidate, "direction")
            if spec.direction_policy is ControlDirectionPolicy.CANDIDATE_LOCAL
            else _explicit_text(treatment, "direction")
        )
        direction_known_at = (
            candidate_direction_known_at[slot[0]]
            if spec.direction_policy is ControlDirectionPolicy.CANDIDATE_LOCAL
            else treatment_clock
        )
        if direction_known_at > candidate_clock:
            raise ResearchContractError(
                "matched control direction is not known at its entry clock"
            )
        pairs.append(
            MatchedControlPair(
                treatment_id=treatment_id,
                candidate_id=slot[0],
                treatment=treatment,
                candidate=candidate,
                completed_bar_offset=int(
                    completed_index[candidate_clock] - completed_index[treatment_clock]
                ),
                control_direction=direction,
                direction_known_at=direction_known_at,
            )
        )
    pairs.sort(
        key=lambda pair: (
            pd.Timestamp(pair.treatment[spec.treatment_clock_field]),
            pair.treatment_id,
            pair.candidate_id,
        )
    )
    return MatchResult(
        pairs=tuple(pairs),
        unmatched=MappingProxyType(dict(sorted(unmatched.items()))),
        requested=len(treatment_by_id),
        eligible_candidates=len(candidate_by_id),
    )


@dataclass(frozen=True)
class PseudoLevelSpec:
    protocol_id: str
    relative_locations: tuple[float, ...]
    tick_size: float
    minimum_separation_ticks: int = 1

    def __post_init__(self) -> None:
        locations = tuple(float(value) for value in self.relative_locations)
        object.__setattr__(self, "relative_locations", locations)
        if not isinstance(self.protocol_id, str) or not self.protocol_id:
            raise ResearchContractError("pseudo-level protocol_id is required")
        if (
            not locations
            or len(locations) != len(set(locations))
            or any(
                not math.isfinite(value) or not 0.0 < value < 1.0 for value in locations
            )
            or not math.isfinite(float(self.tick_size))
            or self.tick_size <= 0.0
            or isinstance(self.minimum_separation_ticks, bool)
            or not isinstance(self.minimum_separation_ticks, int)
            or self.minimum_separation_ticks < 1
        ):
            raise ResearchContractError("pseudo-level construction spec is invalid")


@dataclass(frozen=True)
class PseudoLevel:
    pseudo_level_id: str
    anchor_event_id: str
    anchor_snapshot_id: str
    construction_known_at: pd.Timestamp
    symbol: str
    instrument_id: int
    price: float
    normalized_location: float
    side: str
    direction: str


@dataclass(frozen=True)
class PseudoLevelBuildResult:
    levels: tuple[PseudoLevel, ...]
    exclusions: tuple[Mapping[str, Any], ...]


def construct_pseudo_levels(
    candidate_creation: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    known_real_levels: Sequence[Mapping[str, Any]],
    *,
    spec: PseudoLevelSpec,
) -> PseudoLevelBuildResult:
    """Construct deterministic placebo levels from one contemporaneous range.

    Every input level must already be known at construction.  The helper never
    reads later bars and returns research controls only; it does not publish a
    semantic event or mutate Eye state.
    """

    anchor_event_id = _explicit_text(candidate_creation, "event_id")
    symbol = _explicit_text(candidate_creation, "symbol")
    instrument_id = _value(candidate_creation, "instrument_id")
    if isinstance(instrument_id, bool) or not isinstance(instrument_id, Integral):
        raise ResearchContractError(
            "pseudo-level candidate instrument_id must be an integer"
        )
    instrument_id = int(instrument_id)
    known_at = pd.Timestamp(candidate_creation["known_at"])
    snapshot_asof = pd.Timestamp(snapshot["asof"])
    if (
        known_at.tzinfo is None
        or snapshot_asof.tzinfo is None
        or snapshot_asof != known_at
    ):
        raise ResearchContractError(
            "pseudo-level snapshot must be the exact timezone-aware creation clock"
        )
    snapshot_id = _explicit_text(snapshot, "snapshot_id")
    snapshot_instrument = _value(snapshot, "instrument_id")
    if (
        _explicit_text(snapshot, "symbol") != symbol
        or isinstance(snapshot_instrument, bool)
        or not isinstance(snapshot_instrument, Integral)
        or int(snapshot_instrument) != instrument_id
    ):
        raise ResearchContractError("pseudo-level snapshot contract identity changed")
    low = float(snapshot["range_low"])
    high = float(snapshot["range_high"])
    current_price = float(snapshot["current_price"])
    if (
        not all(math.isfinite(value) for value in (low, high, current_price))
        or high <= low
        or not low <= current_price <= high
    ):
        raise ResearchContractError("pseudo-level snapshot range is invalid")
    real_prices: list[float] = []
    for level in known_real_levels:
        level_instrument = _value(level, "instrument_id")
        if (
            _explicit_text(level, "symbol") != symbol
            or isinstance(level_instrument, bool)
            or not isinstance(level_instrument, Integral)
            or int(level_instrument) != instrument_id
        ):
            raise ResearchContractError(
                "pseudo-level real-level contract identity changed"
            )
        level_known_at = pd.Timestamp(level["known_at"])
        if level_known_at.tzinfo is None or level_known_at > known_at:
            raise ResearchContractError(
                "pseudo-level construction received a future-known real level"
            )
        price = float(level["price"])
        if not math.isfinite(price):
            raise ResearchContractError("real level price must be finite")
        real_prices.append(price)

    minimum_separation = spec.minimum_separation_ticks * spec.tick_size
    emitted_prices: set[float] = set()
    levels: list[PseudoLevel] = []
    exclusions: list[Mapping[str, Any]] = []
    for location in spec.relative_locations:
        raw_price = low + location * (high - low)
        ticks = math.floor(raw_price / spec.tick_size + 0.5)
        price = round(ticks * spec.tick_size, 12)
        reason = None
        if price in emitted_prices:
            reason = "duplicate_tick_price"
        elif any(
            abs(price - real_price) < minimum_separation for real_price in real_prices
        ):
            reason = "too_close_to_known_real_level"
        elif price == current_price:
            reason = "equal_to_current_price"
        if reason is not None:
            exclusions.append(
                MappingProxyType(
                    {
                        "normalized_location": location,
                        "price": price,
                        "reason": reason,
                    }
                )
            )
            continue
        emitted_prices.add(price)
        side = "above" if price > current_price else "below"
        direction = "short" if side == "above" else "long"
        raw_identity = (
            f"{spec.protocol_id}|{anchor_event_id}|{snapshot_id}|"
            f"{known_at.isoformat()}|{location:.17g}|{price:.12g}"
        )
        levels.append(
            PseudoLevel(
                pseudo_level_id=(
                    "pseudo:"
                    + hashlib.sha256(raw_identity.encode("utf-8")).hexdigest()[:24]
                ),
                anchor_event_id=anchor_event_id,
                anchor_snapshot_id=snapshot_id,
                construction_known_at=known_at,
                symbol=symbol,
                instrument_id=instrument_id,
                price=price,
                normalized_location=location,
                side=side,
                direction=direction,
            )
        )
    return PseudoLevelBuildResult(
        levels=tuple(levels),
        exclusions=tuple(exclusions),
    )


@dataclass(frozen=True)
class TimeShiftSpec:
    protocol_id: str
    forward_offsets_completed_bars: tuple[int, ...]
    outcome_horizon_completed_bars: int
    embargo_completed_bars: int = 0
    require_same_session_phase: bool = True
    exclude_other_treatment_windows: bool = True

    def __post_init__(self) -> None:
        offsets = tuple(self.forward_offsets_completed_bars)
        object.__setattr__(self, "forward_offsets_completed_bars", offsets)
        if not isinstance(self.protocol_id, str) or not self.protocol_id:
            raise ResearchContractError("time-shift protocol_id is required")
        if (
            not offsets
            or len(offsets) != len(set(offsets))
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offsets
            )
            or isinstance(self.outcome_horizon_completed_bars, bool)
            or not isinstance(self.outcome_horizon_completed_bars, int)
            or self.outcome_horizon_completed_bars < 1
            or isinstance(self.embargo_completed_bars, bool)
            or not isinstance(self.embargo_completed_bars, int)
            or self.embargo_completed_bars < 0
        ):
            raise ResearchContractError("time-shift specification is invalid")
        minimum = self.outcome_horizon_completed_bars + self.embargo_completed_bars + 1
        if any(value < minimum for value in offsets):
            raise ResearchContractError(
                "time-shift offsets must be forward of the outcome horizon and embargo"
            )


@dataclass(frozen=True)
class TimeShiftBuildResult:
    controls: tuple[Mapping[str, Any], ...]
    exclusions: tuple[Mapping[str, Any], ...]


def build_forward_time_shift_controls(
    events: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    completed_index: Mapping[pd.Timestamp, int],
    spec: TimeShiftSpec,
) -> TimeShiftBuildResult:
    """Create preregistered forward placebos without future-known direction."""

    max_index = _validated_completed_index(
        completed_index,
        role="time-shift completed index",
    )
    rows_by_index: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        clock = pd.Timestamp(row["asof"])
        if clock.tzinfo is None:
            raise ResearchContractError(
                "time-shift completed row clock must be timezone aware"
            )
        index = completed_index.get(clock)
        if index is None:
            continue
        if index in rows_by_index:
            raise ResearchContractError(
                "time-shift completed index contains duplicate rows"
            )
        rows_by_index[index] = row
    if rows_by_index and max(rows_by_index) != max_index:
        raise ResearchContractError(
            "time-shift rows do not reach the registered completed index"
        )
    treatment_windows: list[tuple[str, int, int, str, int]] = []
    for event in events:
        event_id = _explicit_text(event, "event_id")
        clock = pd.Timestamp(event["known_at"])
        if clock.tzinfo is None:
            raise ResearchContractError(
                "time-shift event known_at must be timezone aware"
            )
        index = completed_index.get(clock)
        if index is None:
            continue
        instrument_id = _value(event, "instrument_id")
        if isinstance(instrument_id, bool) or not isinstance(instrument_id, Integral):
            raise ResearchContractError("time-shift instrument_id must be an integer")
        treatment_windows.append(
            (
                event_id,
                index + 1,
                index + spec.outcome_horizon_completed_bars,
                _explicit_text(event, "symbol"),
                int(instrument_id),
            )
        )

    controls: list[Mapping[str, Any]] = []
    exclusions: list[Mapping[str, Any]] = []
    for event in sorted(
        events,
        key=lambda item: (pd.Timestamp(item["known_at"]), str(item["event_id"])),
    ):
        event_id = _explicit_text(event, "event_id")
        event_clock = pd.Timestamp(event["known_at"])
        event_index = completed_index.get(event_clock)
        direction = _explicit_text(event, "direction")
        symbol = _explicit_text(event, "symbol")
        event_session_phase = (
            _explicit_text(event, "session_phase")
            if spec.require_same_session_phase
            else None
        )
        instrument_id = _value(event, "instrument_id")
        if isinstance(instrument_id, bool) or not isinstance(instrument_id, Integral):
            raise ResearchContractError("time-shift instrument_id must be an integer")
        instrument_id = int(instrument_id)
        for offset in sorted(spec.forward_offsets_completed_bars):
            reason = None
            target_index = None if event_index is None else event_index + offset
            row = None if target_index is None else rows_by_index.get(target_index)
            if event_index is None:
                reason = "event_clock_not_completed"
            elif row is None:
                reason = "shifted_entry_unavailable"
            elif target_index + spec.outcome_horizon_completed_bars > max_index:
                reason = "shifted_outcome_horizon_incomplete"
            elif (
                row.get("symbol") != symbol or row.get("instrument_id") != instrument_id
            ):
                reason = "shifted_contract_identity_changed"
            elif any(
                (path_row := rows_by_index.get(path_index)) is None
                or path_row.get("symbol") != symbol
                or path_row.get("instrument_id") != instrument_id
                for path_index in range(
                    target_index + 1,
                    target_index + spec.outcome_horizon_completed_bars + 1,
                )
            ):
                reason = "shifted_outcome_contract_changed_or_incomplete"
            elif (
                spec.require_same_session_phase
                and row.get("session_phase") != event_session_phase
            ):
                reason = "shifted_session_phase_changed"
            elif spec.exclude_other_treatment_windows:
                control_start = target_index + 1
                control_end = target_index + spec.outcome_horizon_completed_bars
                overlaps = any(
                    other_id != event_id
                    and other_symbol == symbol
                    and other_instrument == instrument_id
                    and max(control_start, other_start) <= min(control_end, other_end)
                    for (
                        other_id,
                        other_start,
                        other_end,
                        other_symbol,
                        other_instrument,
                    ) in treatment_windows
                )
                if overlaps:
                    reason = "shifted_outcome_overlaps_other_treatment"
            if reason is not None:
                exclusions.append(
                    MappingProxyType(
                        {
                            "source_event_id": event_id,
                            "completed_bar_offset": offset,
                            "reason": reason,
                        }
                    )
                )
                continue
            assert row is not None and target_index is not None
            shifted_clock = pd.Timestamp(row["asof"])
            if event_clock >= shifted_clock:
                raise ResearchContractError(
                    "time-shift control is not strictly forward"
                )
            raw_identity = (
                f"{spec.protocol_id}|{event_id}|{offset}|{shifted_clock.isoformat()}"
            )
            controls.append(
                MappingProxyType(
                    {
                        "event_id": (
                            "time_shift:"
                            + hashlib.sha256(raw_identity.encode("utf-8")).hexdigest()[
                                :24
                            ]
                        ),
                        "kind": "forward_time_shift_control",
                        "source_event_id": event_id,
                        "construction_known_at": event_clock,
                        "known_at": shifted_clock,
                        "direction_known_at": event_clock,
                        "direction": direction,
                        "symbol": symbol,
                        "instrument_id": instrument_id,
                        "session_phase": row.get("session_phase"),
                        "completed_bar_offset": offset,
                        "control_protocol_id": spec.protocol_id,
                    }
                )
            )
    return TimeShiftBuildResult(
        controls=tuple(controls),
        exclusions=tuple(exclusions),
    )


@dataclass(frozen=True)
class McNemarResult:
    paired_n: int
    excluded_n: int
    treatment_only_successes: int
    control_only_successes: int
    discordant_n: int
    p_value: float


def exact_mcnemar(
    pairs: Sequence[tuple[bool | None, bool | None]],
) -> McNemarResult:
    """Two-sided exact McNemar test for resolved paired binary outcomes."""

    treatment_only = 0
    control_only = 0
    paired = 0
    excluded = 0
    for treatment, control in pairs:
        if treatment is None or control is None:
            excluded += 1
            continue
        if not isinstance(treatment, bool) or not isinstance(control, bool):
            raise ResearchContractError("McNemar outcomes must be bool or None")
        paired += 1
        if treatment and not control:
            treatment_only += 1
        elif control and not treatment:
            control_only += 1
    discordant = treatment_only + control_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(treatment_only, control_only)
        numerator = sum(math.comb(discordant, value) for value in range(tail + 1))
        p_value = min(1.0, 2.0 * (numerator / (1 << discordant)))
    return McNemarResult(
        paired_n=paired,
        excluded_n=excluded,
        treatment_only_successes=treatment_only,
        control_only_successes=control_only,
        discordant_n=discordant,
        p_value=p_value,
    )


@dataclass(frozen=True)
class HolmResult:
    family_order: tuple[str, ...]
    raw_p_values: Mapping[str, float]
    adjusted_p_values: Mapping[str, float]
    rejected: Mapping[str, bool]
    missing_as_one: tuple[str, ...]


def holm_adjust_fixed_family(
    p_values: Mapping[str, float | None],
    *,
    family_order: Sequence[str],
    alpha: float = 0.05,
) -> HolmResult:
    """Holm family-wise adjustment with a frozen, non-droppable family."""

    order = tuple(family_order)
    if (
        not order
        or len(order) != len(set(order))
        or any(not isinstance(name, str) or not name for name in order)
        or set(p_values) != set(order)
    ):
        raise ResearchContractError(
            "Holm p-values must match the exact registered family"
        )
    if (
        not isinstance(alpha, (int, float))
        or isinstance(alpha, bool)
        or not 0.0 < alpha < 1.0
    ):
        raise ResearchContractError("Holm alpha must be between zero and one")
    missing = tuple(name for name in order if p_values[name] is None)
    raw: dict[str, float] = {}
    for name in order:
        value = 1.0 if p_values[name] is None else float(p_values[name])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ResearchContractError(f"Holm p-value is invalid: {name}")
        raw[name] = value
    rank = {name: index for index, name in enumerate(order)}
    sorted_names = sorted(order, key=lambda name: (raw[name], rank[name]))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(order)
    for index, name in enumerate(sorted_names):
        running = max(running, (family_size - index) * raw[name])
        adjusted[name] = min(1.0, running)
    rejected = {name: adjusted[name] <= float(alpha) for name in order}
    return HolmResult(
        family_order=order,
        raw_p_values=MappingProxyType({name: raw[name] for name in order}),
        adjusted_p_values=MappingProxyType({name: adjusted[name] for name in order}),
        rejected=MappingProxyType({name: rejected[name] for name in order}),
        missing_as_one=missing,
    )


def canonical_result_identity(value: Mapping[str, Any]) -> str:
    """Hash substantive output while excluding runtime-only elapsed time."""

    def clean(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): clean(child)
                for key, child in item.items()
                if key not in {"result_identity", "elapsed_seconds"}
            }
        if isinstance(item, (tuple, list)):
            return [clean(child) for child in item]
        if isinstance(item, pd.Timestamp):
            return item.isoformat()
        return item

    payload = json.dumps(
        clean(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "BranchingResearchEdge",
    "BranchingResearchEpisode",
    "ControlDirectionPolicy",
    "EXECUTABLE_DIAGNOSTIC_STATUS",
    "FrozenResearchContract",
    "HolmResult",
    "MatchResult",
    "MatchSpec",
    "MatchedControlPair",
    "McNemarResult",
    "PseudoLevel",
    "PseudoLevelBuildResult",
    "PseudoLevelSpec",
    "REQUIRED_IDENTITY_BINDINGS",
    "REQUIRED_RUNTIME_CODE_BINDINGS",
    "ResearchContractError",
    "ResearchLinkMode",
    "SourceLink",
    "TimeShiftBuildResult",
    "TimeShiftSpec",
    "TypedLinkSpec",
    "TypedResearchLink",
    "build_forward_time_shift_controls",
    "canonical_treatment_episodes",
    "canonical_result_identity",
    "completed_bar_distance",
    "construct_pseudo_levels",
    "deterministic_maximum_cardinality_match",
    "direct_lineage_tokens",
    "find_prior_source_link",
    "find_prior_typed_link",
    "find_prior_typed_links",
    "exact_mcnemar",
    "holm_adjust_fixed_family",
    "load_frozen_research_contract",
    "project_branching_research_episode",
    "resolve_lineage_tokens",
    "resolve_source_lineage_tokens",
    "sha256_file",
    "validate_split_authority",
]
