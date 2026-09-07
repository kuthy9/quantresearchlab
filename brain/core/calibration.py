"""Small monotone reliability maps for preregistered playbook beliefs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from shares.core.model import Playbook, clamp


class CalibrationError(ValueError):
    """Raised when a probability calibration artifact is unusable."""


TYPED_CALIBRATION_DIMENSIONS = (
    "thesis_strength",
    "location_quality",
    "entry_readiness",
    "delivery_quality",
    "uncertainty",
)
TYPED_SEQUENCE_DIMENSION = "sequence_progress"
_ACTION_CALIBRATION_DIMENSIONS = frozenset(
    {
        "thesis_strength",
        "location_quality",
        "entry_readiness",
        "delivery_quality",
    }
)
TYPED_ACTIVE_PLAYBOOKS = (
    Playbook.DISPLACEMENT_FIRST_PULLBACK,
    Playbook.LIQUIDITY_SWEEP_REVERSAL,
)
TYPED_PARKED_PLAYBOOKS = (Playbook.FAILED_AUCTION_VALUE_RETURN,)


@dataclass(frozen=True)
class ReliabilityPoint:
    raw_probability: float
    calibrated_probability: float
    episodes: int


@dataclass(frozen=True)
class DimensionReliabilityPoint:
    """One point in a monotone map for a typed belief dimension."""

    raw_value: float
    calibrated_value: float
    episodes: int


@dataclass(frozen=True)
class DimensionReliabilityMap:
    """A playbook-specific map for exactly one typed belief dimension."""

    playbook: Playbook
    dimension: str
    episodes: int
    points: tuple[DimensionReliabilityPoint, ...]

    def apply(self, value: float) -> float:
        raw = clamp(value)
        x = np.asarray([point.raw_value for point in self.points], dtype=float)
        y = np.asarray(
            [point.calibrated_value for point in self.points],
            dtype=float,
        )
        return clamp(float(np.interp(raw, x, y, left=y[0], right=y[-1])))


def _typed_point_value(
    point: Mapping[str, Any],
    *,
    name: str,
    path: str,
) -> float:
    """Read one current typed calibration coordinate."""

    if name not in point:
        raise CalibrationError(f"{path}.{name} is required")
    try:
        value = float(point[name])
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"{path}.{name} must be a finite number") from error
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise CalibrationError(f"{path}.{name} must lie in [0, 1]")
    return value


def _load_dimension_map(
    playbook: Playbook,
    dimension: str,
    value: Any,
) -> DimensionReliabilityMap:
    path = f"playbooks.{playbook.value}.dimensions.{dimension}"
    if not isinstance(value, Mapping):
        raise CalibrationError(f"{path} must be an object")
    raw_points = value.get("points")
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        raise CalibrationError(f"{path} requires at least two points")
    points: list[DimensionReliabilityPoint] = []
    for index, raw_point in enumerate(raw_points):
        point_path = f"{path}.points[{index}]"
        if not isinstance(raw_point, Mapping):
            raise CalibrationError(f"{point_path} must be an object")
        raw_value = _typed_point_value(
            raw_point,
            name="raw_value",
            path=point_path,
        )
        calibrated_value = _typed_point_value(
            raw_point,
            name="calibrated_value",
            path=point_path,
        )
        try:
            episodes = int(raw_point.get("episodes", 0))
        except (TypeError, ValueError) as error:
            raise CalibrationError(
                f"{point_path}.episodes must be a non-negative integer"
            ) from error
        if episodes < 0:
            raise CalibrationError(
                f"{point_path}.episodes must be a non-negative integer"
            )
        points.append(
            DimensionReliabilityPoint(
                raw_value=raw_value,
                calibrated_value=calibrated_value,
                episodes=episodes,
            )
        )
    if any(
        right.raw_value <= left.raw_value
        or right.calibrated_value < left.calibrated_value
        for left, right in zip(points[:-1], points[1:])
    ):
        raise CalibrationError(f"{path} points are not monotone")
    try:
        episodes = int(value.get("episodes", 0))
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"{path}.episodes must be a non-negative integer") from error
    if episodes < 0:
        raise CalibrationError(f"{path}.episodes must be a non-negative integer")
    return DimensionReliabilityMap(
        playbook=playbook,
        dimension=dimension,
        episodes=episodes,
        points=tuple(points),
    )


@dataclass(frozen=True)
class TypedBrainCalibrator:
    """Maps for typed causal belief dimensions.

    ``sequence_progress`` is a deterministic state-machine projection and is
    deliberately not calibrated.  FAVR remains observable but parked, so a
    ready artifact may not carry any FAVR map.
    """

    version: str
    registry_hash: str | None
    maps: Mapping[Playbook, Mapping[str, DimensionReliabilityMap]]
    status: str

    @classmethod
    def identity(cls) -> "TypedBrainCalibrator":
        return cls(
            version="identity-unvalidated",
            registry_hash=None,
            maps={},
            status="identity_unvalidated",
        )

    @property
    def is_ready(self) -> bool:
        """Whether action-facing typed maps are explicitly usable.

        ``uncertainty`` remains a descriptive conflict/missing-evidence
        formula in the Brain.  It is intentionally not required for Decision
        readiness even though the current artifact schema still carries its
        map.  Sequence progress is deterministic and is never calibrated.
        """

        if (
            self.status != "ready"
            or not self.version
            or self.version == "identity-unvalidated"
            or not self.registry_hash
        ):
            return False
        return all(
            (maps := self.maps.get(playbook)) is not None
            and _ACTION_CALIBRATION_DIMENSIONS.issubset(maps)
            for playbook in TYPED_ACTIVE_PLAYBOOKS
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_registry_hash: str | None,
    ) -> "TypedBrainCalibrator":
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = Path(__file__).resolve().parents[2] / source
        raw = source.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise CalibrationError("typed calibration artifact root must be an object")
        status = str(payload.get("status", ""))
        if status != "ready":
            raise CalibrationError(
                "only a ready typed calibration artifact may affect beliefs"
            )
        version = str(payload.get("calibration_version", "")).strip()
        if not version or version == "identity-unvalidated":
            raise CalibrationError("typed calibration version must be explicit")

        if not expected_registry_hash:
            raise CalibrationError("expected playbook registry hash is required")
        registry_hash = str(payload.get("playbook_registry_hash", ""))
        if registry_hash != expected_registry_hash:
            raise CalibrationError("typed calibration artifact registry hash is stale")
        raw_playbooks = payload.get("playbooks")
        if not isinstance(raw_playbooks, Mapping):
            raise CalibrationError("typed calibration artifact has no playbook maps")
        if set(raw_playbooks) != {playbook.value for playbook in Playbook}:
            raise CalibrationError(
                "typed calibration artifact must declare exactly all registered playbooks"
            )

        maps: dict[Playbook, Mapping[str, DimensionReliabilityMap]] = {}
        required_dimensions = set(TYPED_CALIBRATION_DIMENSIONS)
        for playbook in TYPED_ACTIVE_PLAYBOOKS:
            value = raw_playbooks.get(playbook.value)
            if not isinstance(value, Mapping):
                raise CalibrationError(
                    f"typed calibration artifact omits {playbook.value}"
                )
            if value.get("status") != "active":
                raise CalibrationError(
                    f"{playbook.value} typed calibration must be active"
                )
            dimensions = value.get("dimensions")
            if not isinstance(dimensions, Mapping) or set(dimensions) != required_dimensions:
                raise CalibrationError(
                    f"{playbook.value} must map exactly the five calibrated dimensions"
                )
            maps[playbook] = {
                dimension: _load_dimension_map(
                    playbook,
                    dimension,
                    dimensions[dimension],
                )
                for dimension in TYPED_CALIBRATION_DIMENSIONS
            }

        for playbook in TYPED_PARKED_PLAYBOOKS:
            value = raw_playbooks.get(playbook.value)
            if not isinstance(value, Mapping):
                raise CalibrationError(
                    f"typed calibration artifact must explicitly park {playbook.value}"
                )
            parked_status = str(value.get("status", ""))
            dimensions = value.get("dimensions", {})
            if not parked_status.startswith("parked") or dimensions not in ({}, None):
                raise CalibrationError(
                    f"{playbook.value} must be parked and excluded from calibration"
                )

        return cls(
            version=version,
            registry_hash=registry_hash,
            maps=maps,
            status=status,
        )

    def apply(
        self,
        playbook: Playbook,
        dimension: str,
        raw_value: float,
    ) -> float:
        """Apply one typed map; identity and sequence progression pass through."""

        raw = clamp(raw_value)
        if self.status == "identity_unvalidated":
            return raw
        if playbook in TYPED_PARKED_PLAYBOOKS:
            raise CalibrationError(
                f"{playbook.value} is parked and has no calibrated dimensions"
            )
        if dimension == TYPED_SEQUENCE_DIMENSION:
            return raw
        if dimension not in TYPED_CALIBRATION_DIMENSIONS:
            raise CalibrationError(f"unsupported typed calibration dimension: {dimension}")
        playbook_maps = self.maps.get(playbook)
        if playbook_maps is None or dimension not in playbook_maps:
            raise CalibrationError(
                f"missing typed calibration map for {playbook.value}.{dimension}"
            )
        return playbook_maps[dimension].apply(raw)

def monotone_reliability_points(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int | float],
    *,
    bins: int = 10,
    minimum_bin_episodes: int = 20,
) -> tuple[ReliabilityPoint, ...]:
    """Fit tie-preserving quantile reliability with weighted PAVA.

    Equal raw values are one atomic support level.  They may make a bin larger
    than the nominal quantile size, but are never split across bins; splitting
    a large zero-mass tie would create duplicate x coordinates and falsely
    report that otherwise varied raw support has no calibration levels.
    """

    x = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(x) != len(y) or len(x) == 0:
        raise CalibrationError("calibration inputs must be non-empty and aligned")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise CalibrationError("calibration inputs contain non-finite values")
    if ((x < 0) | (x > 1)).any() or ((y < 0) | (y > 1)).any():
        raise CalibrationError("calibration inputs must lie in [0, 1]")
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    run_starts = np.r_[0, np.flatnonzero(np.diff(sorted_x)) + 1]
    runs = tuple(np.split(order, run_starts[1:]))
    target_bins = min(
        int(bins),
        len(runs),
        len(order) // int(minimum_bin_episodes),
    )
    if target_bins < 2:
        raise CalibrationError("too few populated preregistered reliability bins")
    run_counts = np.asarray([len(run) for run in runs], dtype=int)
    cumulative = np.cumsum(run_counts)
    legal_boundaries = tuple(
        (run_index + 1, int(cumulative[run_index]))
        for run_index in range(len(runs) - 1)
    )
    boundaries = sorted(
        {
            min(
                legal_boundaries,
                key=lambda candidate: (
                    abs(candidate[1] - cut),
                    candidate[0],
                ),
            )[0]
            for cut in (
                index * len(order) / target_bins
                for index in range(1, target_bins)
            )
        }
    )
    run_groups = [
        list(values)
        for values in np.split(np.arange(len(runs)), boundaries)
        if len(values)
    ]
    index = 0
    while index < len(run_groups):
        count = int(sum(run_counts[item] for item in run_groups[index]))
        if count >= int(minimum_bin_episodes):
            index += 1
            continue
        if len(run_groups) == 1:
            break
        if index == 0:
            run_groups[1] = run_groups[0] + run_groups[1]
            del run_groups[0]
        else:
            run_groups[index - 1].extend(run_groups[index])
            del run_groups[index]
            index -= 1
    groups = [
        np.concatenate([runs[item] for item in group])
        for group in run_groups
        if sum(len(runs[item]) for item in group)
        >= int(minimum_bin_episodes)
    ]
    if len(groups) < 2:
        raise CalibrationError("too few populated preregistered reliability bins")
    blocks = [
        {
            "x": float(np.mean(x[index])),
            "y": float((np.sum(y[index]) + 1.0) / (len(index) + 2.0)),
            "n": int(len(index)),
            "members": ((float(np.mean(x[index])), int(len(index))),),
        }
        for index in groups
    ]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index]["y"] <= blocks[index + 1]["y"]:
            index += 1
            continue
        left = blocks[index]
        right = blocks[index + 1]
        count = left["n"] + right["n"]
        blocks[index : index + 2] = [
            {
                "x": (left["x"] * left["n"] + right["x"] * right["n"]) / count,
                "y": (left["y"] * left["n"] + right["y"] * right["n"]) / count,
                "n": count,
                "members": left["members"] + right["members"],
            }
        ]
        index = max(0, index - 1)
    if len(blocks) < 2:
        raise CalibrationError(
            "calibration collapsed to one level; retain identity and gather more episodes"
        )
    points = tuple(
        ReliabilityPoint(
            raw_probability=clamp(raw_value),
            calibrated_probability=clamp(block["y"]),
            episodes=int(episodes),
        )
        for block in blocks
        for raw_value, episodes in block["members"]
    )
    if any(
        right.raw_probability <= left.raw_probability
        for left, right in zip(points, points[1:])
    ):
        raise CalibrationError(
            "raw probabilities do not provide distinct calibration levels; "
            "retain identity and gather more varied episodes"
        )
    return points


__all__ = [
    "CalibrationError",
    "DimensionReliabilityMap",
    "DimensionReliabilityPoint",
    "ReliabilityPoint",
    "TYPED_ACTIVE_PLAYBOOKS",
    "TYPED_CALIBRATION_DIMENSIONS",
    "TYPED_PARKED_PLAYBOOKS",
    "TYPED_SEQUENCE_DIMENSION",
    "TypedBrainCalibrator",
    "monotone_reliability_points",
]
