"""Strict loader for the versioned canonical semantic foundation contract.

The v1.2 atomic event protocol remains immutable.  This registry freezes the
v2 object/lifecycle projection layered over those facts; it is not a second
detector and it carries no trading authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


FOUNDATION_VERSION = "smc_semantic_foundation_v2.1"
PARENT_ATOMIC_VERSION = "smc_semantics_v1.3"
FOUNDATION_CANONICAL_IDENTITY = (
    "69428dbfd2a9b2aa19f0254391fca2da17aedb8d0206829572e69c0cc212a715"
)
_DEFAULT_PATH = Path("semantics/foundation_v2_1.yaml")
_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "foundation_version",
        "parent_atomic_semantic_version",
        "status",
        "purpose",
        "vocabulary_policy",
        "common_generation_contract",
        "structural_leg",
        "swing_geometry",
        "liquidity_cluster",
        "liquidity_level_lifecycle",
        "liquidity_interaction",
        "structure_generation",
        "structure_transition",
        "relation_generation",
        "delivery_phase_generation",
        "ranges",
        "origin_zone",
        "first_reinteraction",
        "fvg_expiry",
        "multi_bar_ancestry",
        "structural_outcome",
        "authority",
    }
)
_CANONICAL_OBJECTS = (
    "confirmed_swing",
    "swing_geometry_nesting",
    "swing_role_assignment",
    "structural_leg",
    "candidate_liquidity_level",
    "liquidity_cluster",
    "liquidity_interaction_generation",
    "touch",
    "penetration_or_boundary_attack",
    "sweep_or_rejection",
    "acceptance",
    "raw_boundary_break",
    "qualified_bos",
    "mss_core",
    "protected_swing_assignment",
    "structure_generation",
    "structure_transition",
    "displacement_episode",
    "fair_value_gap",
    "base_origin_core_or_qualified_order_block",
    "structural_range",
    "balance_range",
    "delivery_phase_generation",
    "cross_timeframe_relation_generation",
)


class FoundationRegistryError(ValueError):
    """Raised when the frozen foundation contract drifts or is incomplete."""


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise FoundationRegistryError(
                f"foundation registry repeats key: {key}"
            )
        output[key] = value
    return output


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class FoundationRegistry:
    foundation_version: str
    parent_atomic_semantic_version: str
    identity: str
    canonical_objects: tuple[str, ...]
    payload: Mapping[str, Any]
    source_path: Path

    def __post_init__(self) -> None:
        if (
            self.foundation_version != FOUNDATION_VERSION
            or self.parent_atomic_semantic_version != PARENT_ATOMIC_VERSION
            or len(self.identity) != 64
            or set(self.identity) - set("0123456789abcdef")
            or self.canonical_objects != _CANONICAL_OBJECTS
        ):
            raise FoundationRegistryError(
                "foundation registry identity or vocabulary is invalid"
            )


def _require_contract(payload: Mapping[str, Any]) -> None:
    if set(payload) != _ROOT_KEYS:
        missing = sorted(_ROOT_KEYS - set(payload))
        extra = sorted(set(payload) - _ROOT_KEYS)
        raise FoundationRegistryError(
            f"foundation registry root drift: missing={missing}, extra={extra}"
        )
    if (
        payload["schema_version"] != 1
        or payload["foundation_version"] != FOUNDATION_VERSION
        or payload["parent_atomic_semantic_version"]
        != PARENT_ATOMIC_VERSION
        or payload["status"]
        != "canonical_preregistered_not_empirically_validated_not_trading_authority"
    ):
        raise FoundationRegistryError(
            "foundation registry version or authority drifted"
        )
    vocabulary = payload["vocabulary_policy"]
    if tuple(vocabulary.get("canonical_objects", ())) != _CANONICAL_OBJECTS:
        raise FoundationRegistryError(
            "foundation canonical vocabulary drifted"
        )
    authority = payload["authority"]
    if set(authority.values()) != {False}:
        raise FoundationRegistryError(
            "foundation registry cannot grant empirical or action authority"
        )
    if payload["fvg_expiry"].get("bar_ttl") is not None:
        raise FoundationRegistryError("foundation v2 forbids an FVG bar TTL")
    if payload["liquidity_cluster"].get("merge_tolerance_ticks") != 1:
        raise FoundationRegistryError(
            "foundation v2 liquidity-cluster tolerance drifted"
        )
    if payload["liquidity_cluster"].get("linkage") != "complete_link":
        raise FoundationRegistryError(
            "foundation v2 requires complete-link liquidity clusters"
        )
    if payload["structure_generation"].get(
        "mss_confirms_opposite_generation"
    ) is not False:
        raise FoundationRegistryError(
            "MSS must remain transition evidence, not regime confirmation"
        )
    if payload["multi_bar_ancestry"].get(
        "temporal_relation_is_ancestry"
    ) is not False:
        raise FoundationRegistryError(
            "temporal relations cannot be promoted to ancestry"
        )


def load_foundation_registry(
    source: str | Path = _DEFAULT_PATH,
    *,
    expected_identity: str | None = None,
) -> FoundationRegistry:
    """Load and strictly validate the frozen foundation JSON-subset YAML."""

    path = Path(source)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FoundationRegistryError(
            f"cannot load foundation registry: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise FoundationRegistryError("foundation registry must be an object")
    _require_contract(payload)
    identity = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if identity != FOUNDATION_CANONICAL_IDENTITY:
        raise FoundationRegistryError(
            "foundation registry canonical identity mismatch"
        )
    if expected_identity is not None and identity != expected_identity:
        raise FoundationRegistryError(
            "foundation registry canonical identity mismatch"
        )
    return FoundationRegistry(
        foundation_version=str(payload["foundation_version"]),
        parent_atomic_semantic_version=str(
            payload["parent_atomic_semantic_version"]
        ),
        identity=identity,
        canonical_objects=_CANONICAL_OBJECTS,
        payload=MappingProxyType(payload),
        source_path=path,
    )


__all__ = [
    "FOUNDATION_CANONICAL_IDENTITY",
    "FOUNDATION_VERSION",
    "FoundationRegistry",
    "FoundationRegistryError",
    "load_foundation_registry",
]
