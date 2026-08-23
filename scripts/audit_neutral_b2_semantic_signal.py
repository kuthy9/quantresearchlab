#!/usr/bin/env python3
"""Train-only decision-time semantic audit before any neutral B2 preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from smc_trader.market_representation import (  # noqa: E402
    NEUTRAL_MARKET_LIFECYCLE_TARGETS,
    NEUTRAL_MARKET_TRANSITION_KINDS,
    RepresentationDataError,
    _neutral_scale_direction_alignment_class,
    build_neutral_market_revision_targets,
    representation_case_from_market_case_input_row,
)
from train_market_representation import (  # noqa: E402
    _load_neutral_market_dataset,
    _neutral_profile_sha256,
    _neutral_split_registry,
)


PROTOCOL_PATH = ROOT / "configs" / "neutral_b2_semantic_audit.json"
PROTOCOL_SHA256 = "512287428d86aa975df9c558e005a686d135bcedb1696e9a0ad224526a734a44"
TIMEFRAMES = ("4H", "1H", "15m", "5m", "1m")
NEXT_CLASSES = tuple(sorted(NEUTRAL_MARKET_LIFECYCLE_TARGETS.values()))
SCALE_CLASSES = (0, 1, 2)
FORBIDDEN_MARKERS = (
    "action", "deadline", "decision", "entry", "invalidation", "mae",
    "mfe", "outcome", "playbook", "pnl", "profit", "risk", "shadow",
    "stop", "target",
)


@dataclass(frozen=True)
class MaterialRecord:
    window: str
    run_sha256: str
    market_epoch_id: str
    market_episode_id: str
    revision_id: str
    et_date: str
    material_kind: str
    base: tuple[Any, ...]
    signature: tuple[Any, ...]
    signature_suffix: tuple[Any, ...]
    structural_tokens: frozenset[str]
    next_lifecycle: int | None
    scale_direction_alignment: int

    @property
    def joint_label(self) -> tuple[int, int] | None:
        if self.next_lifecycle is None:
            return None
        return (self.next_lifecycle, self.scale_direction_alignment)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _load_protocol() -> Mapping[str, Any]:
    if _sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RepresentationDataError("neutral B2 semantic audit protocol changed")
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (
        payload.get("protocol_version")
        != "neutral-b2-train-semantic-audit-1.0.0"
        or payload.get("population", {}).get("split_role") != "train"
        or payload.get("population", {}).get("validation_opened") is not False
        or payload.get("population", {}).get("holdout_opened") is not False
    ):
        raise RepresentationDataError("neutral B2 semantic audit population changed")
    return payload


def _preflight_roots(
    roots: Sequence[str], protocol: Mapping[str, Any]
) -> tuple[tuple[Path, Mapping[str, Any]], ...]:
    expected_profiles = protocol["population"]["profiles"]
    if len(roots) != len(expected_profiles) or len(set(roots)) != len(roots):
        raise RepresentationDataError("semantic audit requires six unique train roots")
    expected = {item["name"]: item for item in expected_profiles}
    supplied: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for raw_root in roots:
        root = Path(raw_root).resolve()
        run_path = root / "run_manifest.json"
        input_path = root / "market_case_input_shards.manifest.json"
        if root.is_symlink() or not run_path.is_file() or not input_path.is_file():
            raise RepresentationDataError("semantic audit input root is incomplete")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        profile = run.get("profile", {}).get("name")
        if not isinstance(profile, str) or profile not in expected:
            raise RepresentationDataError("semantic audit opened a non-train profile")
        item = expected[profile]
        input_manifest = json.loads(input_path.read_text(encoding="utf-8"))
        if (
            _sha256(run_path) != item["run_manifest_sha256"]
            or _sha256(input_path) != item["input_manifest_sha256"]
            or input_manifest.get("rows") != item["rows"]
            or profile in supplied
        ):
            raise RepresentationDataError("semantic audit input identity changed")
        supplied[profile] = (root, item)
    ordered = tuple(supplied[item["name"]] for item in expected_profiles)
    if tuple(item[1]["name"] for item in ordered) != tuple(
        item["name"] for item in expected_profiles
    ):
        raise RepresentationDataError("semantic audit train profile set changed")
    return ordered


def _load_train_datasets(
    roots: Sequence[str], protocol: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    ordered = _preflight_roots(roots, protocol)
    registry, profiles, _ = _neutral_split_registry()
    registered_train = {
        name
        for name, window in registry.windows.items()
        if window.representation_split_role == "train"
    }
    expected_names = {item[1]["name"] for item in ordered}
    if expected_names != registered_train:
        raise RepresentationDataError("semantic audit does not bind every train window")
    cache: dict[tuple[str, str, str], Any] = {}
    datasets: list[Mapping[str, Any]] = []
    for root, expected in ordered:
        loaded = dict(
            _load_neutral_market_dataset(
                input_manifest_path=str(
                    root / "market_case_input_shards.manifest.json"
                ),
                run_manifest_path=str(root / "run_manifest.json"),
                identity_cache=cache,
            )
        )
        run = loaded["run_manifest"]
        profile = profiles.get(expected["name"])
        window = registry.windows.get(expected["name"])
        if (
            not isinstance(profile, Mapping)
            or window is None
            or window.representation_split_role != "train"
            or run["profile"]["identity"] != _neutral_profile_sha256(profile)
            or len(loaded["rows"]) != expected["rows"]
            or loaded["input_sha"] != expected["input_manifest_sha256"]
            or loaded["run_sha"] != expected["run_manifest_sha256"]
        ):
            raise RepresentationDataError("semantic audit train registry binding changed")
        loaded["profile_name"] = expected["name"]
        datasets.append(loaded)
    return tuple(datasets)


def _session_phase(asof: Any) -> tuple[str, str]:
    stamp = pd.Timestamp(asof)
    if stamp.tzinfo is None:
        raise RepresentationDataError("semantic audit clock must be timezone-aware")
    local = stamp.tz_convert("America/New_York")
    minute = local.hour * 60 + local.minute
    if 570 <= minute < 960:
        phase = "rth"
    elif 960 <= minute < 1020:
        phase = "post_rth"
    elif minute >= 1080:
        phase = "evening"
    else:
        phase = "overnight"
    return phase, str(local.date())


def _scale_state(context: Mapping[str, Any], timeframe: str) -> tuple[str, str]:
    details = context.get("scale_relation_details")
    if not isinstance(details, Mapping):
        raise RepresentationDataError("semantic audit scale details are missing")
    raw = details.get(timeframe)
    if raw is None:
        raw = details.get(timeframe.lower())
    if not isinstance(raw, Mapping):
        raise RepresentationDataError("semantic audit scale detail changed")
    relation = str(raw.get("relation", "unknown")).strip().lower()
    connected = raw.get("graph_connected")
    ambiguous = raw.get("ambiguous")
    if type(connected) is not bool or type(ambiguous) is not bool:
        raise RepresentationDataError("semantic audit scale flags changed")
    direction = (
        str(raw.get("direction", "none")).strip().lower()
        if connected and not ambiguous
        else "unknown"
    )
    return relation, direction


def _mapping_events(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for nested in value.values():
            if isinstance(nested, (Mapping, list, tuple)):
                yield from _mapping_events(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            yield from _mapping_events(nested)


def _safe_value(value: Any) -> str | None:
    if value is None:
        return "none"
    if type(value) is bool:
        return str(value).lower()
    if not isinstance(value, str):
        return None
    output = value.strip().lower()
    if not output or len(output) > 80:
        return None
    if any(marker in output for marker in FORBIDDEN_MARKERS):
        return None
    return output


def _structural_tokens(
    observation: Mapping[str, Any],
    graph: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[frozenset[str], Mapping[str, Any]]:
    collections = observation.get("collections")
    if not isinstance(collections, Mapping):
        raise RepresentationDataError("semantic audit observation collections changed")
    allowed = protocol["candidate_structural_tokens"]
    tokens: set[str] = set()
    field_audit: dict[str, Any] = {}
    for collection, raw in sorted(collections.items()):
        observed_keys: set[str] = set()
        used_keys: set[str] = set()
        if not isinstance(collection, str):
            raise RepresentationDataError("semantic audit collection identity changed")
        fields = allowed.get(collection, ())
        if collection == "group5_entry_location_transitions_this_update":
            fields = ()
        for event in _mapping_events(raw):
            observed_keys.update(str(key) for key in event)
            normalized = dict(event)
            if collection == "group5_micro_bos_transitions_this_update" and (
                "outcome" in normalized
            ):
                alignment = normalized.pop("outcome")
                if alignment not in {
                    "aligned", "opposed", "simultaneous_unknown",
                    "ambiguous_same_clock",
                }:
                    raise RepresentationDataError(
                        "semantic audit MicroBOS reference alignment changed"
                    )
                normalized["reference_alignment"] = alignment
            for field in fields:
                if field not in normalized:
                    continue
                value = _safe_value(normalized[field])
                if value is None:
                    continue
                if any(marker in field.lower() for marker in FORBIDDEN_MARKERS):
                    raise RepresentationDataError(
                        "semantic audit allowlist contains a forbidden field"
                    )
                tokens.add(f"event:{collection}:{field}={value}")
                used_keys.add(field)
        field_audit[collection] = {
            "event_objects": len(raw) if isinstance(raw, list) else 0,
            "observed_keys": sorted(observed_keys),
            "allowlisted_keys": sorted(fields),
            "used_keys": sorted(used_keys),
            "excluded_keys": sorted(observed_keys - set(fields)),
        }

    descriptors = graph.get("relation_descriptors")
    if not isinstance(descriptors, list):
        raise RepresentationDataError("semantic audit Scene relation descriptors changed")
    excluded_scene = 0
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise RepresentationDataError("semantic audit Scene relation changed")
        source = descriptor.get("source")
        destination = descriptor.get("target")
        if not isinstance(source, Mapping) or not isinstance(destination, Mapping):
            raise RepresentationDataError("semantic audit Scene endpoints changed")
        endpoint_values = [
            str(endpoint.get(field, "")).strip().lower()
            for endpoint in (source, destination)
            for field in ("kind", "role")
        ]
        if any(
            marker in value
            for value in endpoint_values
            for marker in ("entry", "playbook", "target")
        ):
            excluded_scene += 1
            continue
        parts: list[str] = []
        for field in ("change_kind", "lifecycle", "relation"):
            value = _safe_value(descriptor.get(field))
            if value is not None:
                parts.append(f"{field}={value}")
                tokens.add(f"graph:{field}={value}")
        for prefix, endpoint in (("src", source), ("dst", destination)):
            for field in ("kind", "lifecycle", "role", "structural_scale", "timeframe"):
                value = _safe_value(endpoint.get(field))
                if value is not None:
                    parts.append(f"{prefix}_{field}={value}")
                    tokens.add(f"graph:{prefix}_{field}={value}")
        if parts:
            tokens.add("graph_motif:" + "|".join(parts))
    if any(marker in token.lower() for token in tokens for marker in FORBIDDEN_MARKERS):
        raise RepresentationDataError("semantic audit emitted a forbidden model token")
    return frozenset(tokens), {
        "event_fields": field_audit,
        "scene_descriptors": len(descriptors),
        "scene_descriptors_excluded_by_endpoint": excluded_scene,
    }


def _five_prefixes_nonempty(row: Mapping[str, Any]) -> bool:
    try:
        prefixes = json.loads(str(row["ohlcv_prefix_refs_json"]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RepresentationDataError("semantic audit prefix JSON changed") from exc
    if (
        not isinstance(prefixes, list)
        or {str(item.get("timeframe")) for item in prefixes if isinstance(item, Mapping)}
        != set(TIMEFRAMES)
    ):
        raise RepresentationDataError("semantic audit five-scale prefixes changed")
    return all(
        isinstance(item, Mapping)
        and type(item.get("frame_row_start")) is int
        and type(item.get("frame_row_end_exclusive")) is int
        and item["frame_row_start"] < item["frame_row_end_exclusive"]
        for item in prefixes
    )


def _raw_sparse_targets(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, tuple[int | None, int]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["market_epoch_id"]), str(row["market_episode_id"]))].append(row)
    output: dict[str, tuple[int | None, int]] = {}
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda row: (
                int(row["revision_index"]), pd.Timestamp(row["asof"]),
                str(row["revision_id"]),
            ),
        )
        if [int(row["revision_index"]) for row in ordered] != list(range(len(ordered))):
            raise RepresentationDataError("semantic audit revision indexes changed")
        for index, row in enumerate(ordered):
            lifecycle = str(row["lifecycle"])
            if lifecycle not in NEUTRAL_MARKET_LIFECYCLE_TARGETS:
                raise RepresentationDataError("semantic audit lifecycle target changed")
            next_label = (
                None
                if index + 1 == len(ordered)
                else NEUTRAL_MARKET_LIFECYCLE_TARGETS[
                    str(ordered[index + 1]["lifecycle"])
                ]
            )
            context = json.loads(str(row["neutral_global_context_json"]))
            if not isinstance(context, Mapping):
                raise RepresentationDataError("semantic audit target context changed")
            output[str(row["revision_id"])] = (
                next_label, _neutral_scale_direction_alignment_class(context)
            )
    return output


def _material_records(
    datasets: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[tuple[MaterialRecord, ...], Mapping[str, Any]]:
    records: list[MaterialRecord] = []
    grains: set[tuple[str, str, str, str]] = set()
    field_audit: dict[str, Any] = {}
    scene_descriptors = 0
    scene_excluded = 0
    excluded_rows: Counter[str] = Counter()
    excluded_materials: Counter[str] = Counter()
    production_target_equivalence_rows = 0
    for dataset in datasets:
        raw_targets = _raw_sparse_targets(dataset["rows"])
        if all(_five_prefixes_nonempty(row) for row in dataset["rows"]):
            production_targets = build_neutral_market_revision_targets(
                dataset["rows"], dataset["run_manifest"]
            )
            for revision_id, (next_label, scale_label) in raw_targets.items():
                production = production_targets[revision_id].target
                if (
                    (None if production.next_lifecycle < 0 else int(production.next_lifecycle))
                    != next_label
                    or int(production.scale_direction_alignment) != scale_label
                ):
                    raise RepresentationDataError(
                        "semantic audit raw target builder differs from production"
                    )
            production_target_equivalence_rows += len(dataset["rows"])
        for row in dataset["rows"]:
            context = json.loads(str(row["neutral_global_context_json"]))
            observation = json.loads(str(row["observation_transition_json"]))
            graph = json.loads(str(row["scene_graph_delta_json"]))
            if not all(isinstance(value, Mapping) for value in (context, observation, graph)):
                raise RepresentationDataError("semantic audit row JSON changed")
            event_tokens, audit = _structural_tokens(observation, graph, protocol)
            for collection, values in audit["event_fields"].items():
                prior = field_audit.setdefault(
                    collection,
                    {"event_objects": 0, "observed_keys": set(), "allowlisted_keys": set(),
                     "used_keys": set(), "excluded_keys": set()},
                )
                prior["event_objects"] += values["event_objects"]
                for key in (
                    "observed_keys", "allowlisted_keys", "used_keys", "excluded_keys"
                ):
                    prior[key].update(values[key])
            scene_descriptors += audit["scene_descriptors"]
            scene_excluded += audit["scene_descriptors_excluded_by_endpoint"]
            transition_kinds = json.loads(str(row["transition_kinds_json"]))
            if (
                not isinstance(transition_kinds, list)
                or not transition_kinds
                or any(kind not in NEUTRAL_MARKET_TRANSITION_KINDS for kind in transition_kinds)
            ):
                raise RepresentationDataError("semantic audit material kinds changed")
            if not _five_prefixes_nonempty(row):
                excluded_rows[str(dataset["profile_name"])] += 1
                excluded_materials[str(dataset["profile_name"])] += len(transition_kinds)
                continue
            representation_case_from_market_case_input_row(
                row, dataset["run_manifest"]
            )
            phase, et_date = _session_phase(row["asof"])
            balance = context.get("balance_context") or {}
            if not isinstance(balance, Mapping):
                raise RepresentationDataError("semantic audit balance context changed")
            scale_states = tuple(
                _scale_state(context, timeframe) for timeframe in TIMEFRAMES
            )
            base_prefix = (
                str(row["direction"]), str(row["lifecycle"]),
            )
            suffix = (
                str(context.get("market_mode", "unknown")), phase,
                str(balance.get("status", "none")),
                bool(balance.get("internal_crossing", False)),
                bool(balance.get("accepted_external_break", False)),
                scale_states,
            )
            next_label, scale_label = raw_targets[str(row["revision_id"])]
            if scale_label not in SCALE_CLASSES:
                raise RepresentationDataError("semantic audit scale label changed")
            for material_kind in transition_kinds:
                grain = (
                    str(dataset["run_sha"]), str(row["market_epoch_id"]),
                    str(row["market_episode_id"]), material_kind,
                )
                if grain in grains:
                    raise RepresentationDataError("semantic audit material grain duplicated")
                grains.add(grain)
                base = (material_kind,) + base_prefix
                core_tokens = {
                    f"base:material_kind={material_kind}",
                    f"base:case_direction={base_prefix[0]}",
                    f"base:current_lifecycle={base_prefix[1]}",
                    f"context:market_mode={suffix[0]}",
                    f"context:session_phase={phase}",
                    f"context:balance_status={suffix[2]}",
                    f"context:balance_internal_crossing={str(suffix[3]).lower()}",
                    f"context:balance_accepted_external_break={str(suffix[4]).lower()}",
                }
                for timeframe, (relation, direction) in zip(
                    TIMEFRAMES, scale_states, strict=True
                ):
                    core_tokens.add(f"scale:{timeframe}:relation={relation}")
                    core_tokens.add(f"scale:{timeframe}:direction={direction}")
                records.append(
                    MaterialRecord(
                        window=str(dataset["profile_name"]),
                        run_sha256=str(dataset["run_sha"]),
                        market_epoch_id=str(row["market_epoch_id"]),
                        market_episode_id=str(row["market_episode_id"]),
                        revision_id=str(row["revision_id"]), et_date=et_date,
                        material_kind=material_kind, base=base,
                        signature=base + suffix, signature_suffix=suffix,
                        structural_tokens=frozenset(core_tokens) | event_tokens,
                        next_lifecycle=next_label,
                        scale_direction_alignment=scale_label,
                    )
                )
    return tuple(records), {
        "event_fields": {
            collection: {
                key: value if key == "event_objects" else sorted(value)
                for key, value in audit.items()
            }
            for collection, audit in sorted(field_audit.items())
        },
        "scene_relation_descriptors_observed": scene_descriptors,
        "scene_relation_descriptors_excluded_by_endpoint": scene_excluded,
        "five_scale_feature_eligibility": {
            "excluded_rows_by_profile": dict(sorted(excluded_rows.items())),
            "excluded_material_records_by_profile": dict(sorted(excluded_materials.items())),
            "excluded_rows_total": sum(excluded_rows.values()),
            "excluded_material_records_total": sum(excluded_materials.values()),
            "production_target_equivalence_verified_rows": production_target_equivalence_rows,
            "raw_target_sequence_uses_full_validated_episode": True,
        },
        "emitted_tokens_contain_forbidden_markers": False,
        "labels_used_as_encoder_or_signature_inputs": False,
    }


def _entropy(labels: Iterable[Any]) -> float:
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total) for count in counts.values()
    )


def _conditional_entropy(records: Sequence[MaterialRecord], key: Any, label: Any) -> float:
    groups: dict[Any, list[Any]] = defaultdict(list)
    for record in records:
        value = label(record)
        if value is not None:
            groups[key(record)].append(value)
    total = sum(len(values) for values in groups.values())
    return sum(len(values) / total * _entropy(values) for values in groups.values())


def _label_summary(values: Sequence[Any]) -> Mapping[str, Any]:
    counts = Counter(values)
    total = sum(counts.values())
    return {
        "rows": total,
        "classes": {
            json.dumps(key, separators=(",", ":")): count
            for key, count in sorted(counts.items(), key=lambda item: str(item[0]))
        },
        "entropy_bits": _entropy(values),
        "majority_share": max(counts.values()) / total if total else 0.0,
    }


def _probabilities(counts: Counter[Any], classes: Sequence[Any], alpha: float) -> np.ndarray:
    return np.asarray(
        [(counts[value] + alpha) / (sum(counts.values()) + alpha * len(classes))
         for value in classes],
        dtype=np.float64,
    )


def _classification_metrics(
    probabilities: Sequence[np.ndarray], labels: Sequence[Any], classes: Sequence[Any]
) -> Mapping[str, float]:
    indexes = {value: index for index, value in enumerate(classes)}
    actual = np.asarray([indexes[value] for value in labels], dtype=np.int64)
    matrix = np.stack(probabilities)
    predicted = matrix.argmax(axis=1)
    nll = float(-np.log(np.maximum(matrix[np.arange(len(actual)), actual], 1e-12)).mean())
    recalls = [float((predicted[actual == index] == index).mean())
               for index in sorted(set(actual.tolist()))]
    return {
        "nll": nll,
        "accuracy": float((predicted == actual).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
    }


def _oof_exact_lookup(
    records: Sequence[MaterialRecord], *, label: Any, classes: Sequence[Any]
) -> Mapping[str, Any]:
    labelled = [record for record in records if label(record) is not None]
    base_probabilities: list[np.ndarray] = []
    signature_probabilities: list[np.ndarray] = []
    labels: list[Any] = []
    seen = 0
    for window in sorted({record.window for record in labelled}):
        train = [record for record in labelled if record.window != window]
        test = [record for record in labelled if record.window == window]
        global_counts = Counter(label(record) for record in train)
        base_counts: dict[Any, Counter[Any]] = defaultdict(Counter)
        signature_counts: dict[Any, Counter[Any]] = defaultdict(Counter)
        for record in train:
            base_counts[record.base][label(record)] += 1
            signature_counts[record.signature][label(record)] += 1
        for record in test:
            base_count = base_counts.get(record.base, global_counts)
            full_count = signature_counts.get(record.signature)
            base_probability = _probabilities(base_count, classes, 1.0)
            base_probabilities.append(base_probability)
            if full_count is None:
                signature_probabilities.append(base_probability)
            else:
                signature_probabilities.append(_probabilities(full_count, classes, 1.0))
                seen += 1
            labels.append(label(record))
    base_metrics = _classification_metrics(base_probabilities, labels, classes)
    signature_metrics = _classification_metrics(signature_probabilities, labels, classes)
    return {
        "rows": len(labels), "folds": 6,
        "seen_signature_coverage": seen / len(labels),
        "stage_prior": base_metrics, "visible_signature": signature_metrics,
        "nll_relative_improvement": 1.0 - signature_metrics["nll"] / base_metrics["nll"],
        "balanced_accuracy_lift": (
            signature_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"]
        ),
    }


def _oof_bernoulli_nb_without_scale(
    records: Sequence[MaterialRecord], *, label: Any, classes: Sequence[Any],
    minimum_document_frequency: int,
) -> Mapping[str, Any]:
    labelled = [record for record in records if label(record) is not None]
    base_probabilities: list[np.ndarray] = []
    model_probabilities: list[np.ndarray] = []
    labels: list[Any] = []
    for window in sorted({record.window for record in labelled}):
        train = [record for record in labelled if record.window != window]
        test = [record for record in labelled if record.window == window]
        global_counts = Counter(label(record) for record in train)
        base_counts: dict[Any, Counter[Any]] = defaultdict(Counter)
        for record in train:
            base_counts[record.base][label(record)] += 1
        train_token_counts = Counter(
            token for record in train for token in record.structural_tokens
            if not token.startswith("scale:")
        )
        vocabulary = sorted(
            token for token, count in train_token_counts.items()
            if count >= minimum_document_frequency
        )
        vocabulary_set = set(vocabulary)
        class_counts = Counter(label(record) for record in train)
        document_counts = {value: Counter() for value in classes}
        for record in train:
            value = label(record)
            document_counts[value].update(
                token for token in record.structural_tokens
                if not token.startswith("scale:")
            )
        bases: dict[Any, float] = {}
        deltas: dict[Any, dict[str, float]] = {}
        total = len(train)
        for value in classes:
            class_total = class_counts[value]
            prior = (class_total + 1.0) / (total + len(classes))
            base_log = math.log(prior)
            delta: dict[str, float] = {}
            for token in vocabulary:
                probability = (document_counts[value][token] + 1.0) / (class_total + 2.0)
                base_log += math.log(1.0 - probability)
                delta[token] = math.log(probability) - math.log(1.0 - probability)
            bases[value] = base_log
            deltas[value] = delta
        for record in test:
            base_probabilities.append(
                _probabilities(base_counts.get(record.base, global_counts), classes, 1.0)
            )
            present = {
                token for token in record.structural_tokens
                if not token.startswith("scale:") and token in vocabulary_set
            }
            logits = np.asarray([
                bases[value] + sum(deltas[value][token] for token in present)
                for value in classes
            ])
            logits -= logits.max()
            probability = np.exp(logits)
            model_probabilities.append(probability / probability.sum())
            labels.append(label(record))
    base_metrics = _classification_metrics(base_probabilities, labels, classes)
    model_metrics = _classification_metrics(model_probabilities, labels, classes)
    return {
        "rows": len(labels), "folds": 6,
        "scale_relation_fields_removed": True,
        "stage_prior": base_metrics, "structural_token_naive_bayes": model_metrics,
        "nll_relative_improvement": 1.0 - model_metrics["nll"] / base_metrics["nll"],
        "balanced_accuracy_lift": (
            model_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"]
        ),
    }


def _signature_support(records: Sequence[MaterialRecord]) -> Mapping[str, Any]:
    groups: dict[Any, list[MaterialRecord]] = defaultdict(list)
    for record in records:
        groups[record.signature].append(record)
    eligible = [
        record for values in groups.values()
        if len({item.market_episode_id for item in values}) >= 2
        and len({item.et_date for item in values}) >= 2
        for record in values
    ]
    counts = Counter(len(values) for values in groups.values())
    return {
        "records": len(records), "unique_signatures": len(groups),
        "signature_entropy_bits": _entropy(
            signature for signature, values in groups.items() for _ in values
        ),
        "singleton_signatures": counts[1],
        "cross_episode_cross_date_records": len(eligible),
        "cross_episode_cross_date_coverage": len(eligible) / len(records),
        "support_distribution": {str(key): value for key, value in sorted(counts.items())},
    }


def _signature_conditional_mi(
    records: Sequence[MaterialRecord], *, label: Any, shuffle_seed: int
) -> Mapping[str, Any]:
    labelled = [record for record in records if label(record) is not None]
    base_entropy = _conditional_entropy(
        labelled, lambda record: record.base, label
    )
    signature_entropy = _conditional_entropy(
        labelled, lambda record: record.signature, label
    )
    observed = base_entropy - signature_entropy
    rng = random.Random(shuffle_seed)
    strata: dict[Any, list[MaterialRecord]] = defaultdict(list)
    for record in labelled:
        strata[(record.window, record.base)].append(record)
    shuffled_values: list[float] = []
    for _ in range(100):
        assigned: dict[int, tuple[Any, ...]] = {}
        for values in strata.values():
            suffixes = [record.signature_suffix for record in values]
            rng.shuffle(suffixes)
            assigned.update({id(record): suffix for record, suffix in zip(values, suffixes, strict=True)})
        shuffled_entropy = _conditional_entropy(
            labelled,
            lambda record: record.base + assigned[id(record)],
            label,
        )
        shuffled_values.append(base_entropy - shuffled_entropy)
    p95 = float(np.quantile(np.asarray(shuffled_values), 0.95))
    return {
        "label_entropy_bits": _entropy(label(record) for record in labelled),
        "stage_conditional_entropy_bits": base_entropy,
        "signature_conditional_entropy_bits": signature_entropy,
        "conditional_mutual_information_bits": observed,
        "within_stage_window_shuffle_repetitions": 100,
        "shuffle_mean_bits": float(np.mean(shuffled_values)),
        "shuffle_p95_bits": p95,
        "above_shuffle_p95": observed > p95,
    }


def _cross_date_joint_stability(records: Sequence[MaterialRecord]) -> Mapping[str, Any]:
    labelled = [record for record in records if record.joint_label is not None]
    signatures: dict[Any, list[MaterialRecord]] = defaultdict(list)
    bases: dict[Any, list[MaterialRecord]] = defaultdict(list)
    for record in labelled:
        signatures[record.signature].append(record)
        bases[record.base].append(record)
    purities: list[float] = []
    chances: list[float] = []
    for record in labelled:
        candidates = [
            item for item in signatures[record.signature]
            if item.market_episode_id != record.market_episode_id
            and item.et_date != record.et_date
        ]
        if not candidates:
            continue
        pool = [
            item for item in bases[record.base]
            if item.market_episode_id != record.market_episode_id
            and item.et_date != record.et_date
        ]
        purities.append(sum(item.joint_label == record.joint_label for item in candidates) / len(candidates))
        chances.append(sum(item.joint_label == record.joint_label for item in pool) / len(pool))
    return {
        "labelled_queries": len(labelled), "covered_queries": len(purities),
        "coverage": len(purities) / len(labelled),
        "mean_same_signature_purity": float(np.mean(purities)) if purities else 0.0,
        "mean_stage_chance": float(np.mean(chances)) if chances else 0.0,
        "purity_lift_over_stage_chance": (
            float(np.mean(purities) - np.mean(chances)) if purities else 0.0
        ),
    }


def _binary_mutual_information(records: Sequence[MaterialRecord], token: str, label: Any) -> float:
    counts: Counter[tuple[bool, Any]] = Counter()
    for record in records:
        value = label(record)
        if value is not None:
            counts[(token in record.structural_tokens, value)] += 1
    total = sum(counts.values())
    x_counts = Counter({present: sum(count for (x, _), count in counts.items() if x == present)
                        for present in (False, True)})
    y_counts = Counter()
    for (_, value), count in counts.items():
        y_counts[value] += count
    output = 0.0
    for (present, value), count in counts.items():
        probability = count / total
        output += probability * math.log2(
            probability / ((x_counts[present] / total) * (y_counts[value] / total))
        )
    return output


def _token_audit(
    records: Sequence[MaterialRecord], *, minimum_frequency: int
) -> Mapping[str, Any]:
    frequencies = Counter(token for record in records for token in record.structural_tokens)
    candidate_tokens = [
        token for token in frequencies
        if token.startswith("event:") or token.startswith("graph:")
        or token.startswith("graph_motif:")
        if frequencies[token] >= minimum_frequency
    ]
    next_mi = sorted(
        ((token, _binary_mutual_information(records, token, lambda item: item.next_lifecycle))
         for token in candidate_tokens), key=lambda item: item[1], reverse=True,
    )[:40]
    scale_mi = sorted(
        ((token, _binary_mutual_information(
            records, token, lambda item: item.scale_direction_alignment
        )) for token in candidate_tokens), key=lambda item: item[1], reverse=True,
    )[:40]
    return {
        "unique_structural_tokens": len(frequencies),
        "frequencies": dict(sorted(frequencies.items())),
        "top_next_lifecycle_binary_mi_bits": [
            {"token": token, "mi_bits": value} for token, value in next_mi
        ],
        "top_scale_direction_binary_mi_bits_without_scale_fields": [
            {"token": token, "mi_bits": value} for token, value in scale_mi
        ],
    }


def _event_collection_summary(field_audit: Mapping[str, Any]) -> Mapping[str, Any]:
    counts = Counter({
        collection: values["event_objects"]
        for collection, values in field_audit["event_fields"].items()
    })
    total = sum(counts.values())
    return {
        "events": total,
        "classes": dict(sorted(counts.items())),
        "entropy_bits": _entropy(
            collection for collection, count in counts.items() for _ in range(count)
        ),
        "majority_share": max(counts.values()) / total if total else 0.0,
    }


def _gate_report(
    signature: Mapping[str, Any], mi: Mapping[str, Any],
    next_oof: Mapping[str, Any], scale_oof: Mapping[str, Any],
    stability: Mapping[str, Any], protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    thresholds = protocol["go_no_go_gates"]
    gates = {
        "signature_coverage": signature["cross_episode_cross_date_coverage"]
        >= thresholds["cross_episode_cross_date_signature_coverage_min"],
        "next_nll": next_oof["nll_relative_improvement"]
        >= thresholds["next_lifecycle_oof_nll_relative_improvement_over_stage_prior_min"],
        "next_balanced_accuracy": next_oof["balanced_accuracy_lift"]
        >= thresholds["next_lifecycle_oof_balanced_accuracy_lift_over_stage_prior_min"],
        "next_conditional_mi": mi["conditional_mutual_information_bits"]
        >= thresholds["next_lifecycle_conditional_mi_bits_min"],
        "next_conditional_mi_shuffle": mi["above_shuffle_p95"] is True,
        "non_tautological_scale_nll": scale_oof["nll_relative_improvement"]
        >= thresholds["non_tautological_scale_oof_nll_relative_improvement_over_stage_prior_min"],
        "non_tautological_scale_balanced_accuracy": scale_oof["balanced_accuracy_lift"]
        >= thresholds["non_tautological_scale_oof_balanced_accuracy_lift_over_stage_prior_min"],
        "joint_cross_date_purity_lift": stability["purity_lift_over_stage_chance"]
        >= thresholds["joint_cross_date_same_signature_purity_lift_over_stage_chance_min"],
    }
    return {"gates": gates, "all_pass": all(gates.values())}


def _report(
    datasets: Sequence[Mapping[str, Any]], records: Sequence[MaterialRecord],
    field_audit: Mapping[str, Any], protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    signature = _signature_support(records)
    mi = _signature_conditional_mi(
        records, label=lambda record: record.next_lifecycle,
        shuffle_seed=protocol["evaluation"]["shuffle_seed"],
    )
    scale_signature_mi = _signature_conditional_mi(
        records, label=lambda record: record.scale_direction_alignment,
        shuffle_seed=protocol["evaluation"]["shuffle_seed"],
    )
    next_oof = _oof_exact_lookup(
        records, label=lambda record: record.next_lifecycle, classes=NEXT_CLASSES
    )
    scale_oof = _oof_bernoulli_nb_without_scale(
        records, label=lambda record: record.scale_direction_alignment,
        classes=SCALE_CLASSES,
        minimum_document_frequency=protocol["evaluation"][
            "bernoulli_nb_min_train_document_frequency"
        ],
    )
    stability = _cross_date_joint_stability(records)
    decision = _gate_report(signature, mi, next_oof, scale_oof, stability, protocol)
    windows = [
        {
            "profile": dataset["profile_name"], "rows": len(dataset["rows"]),
            "input_manifest_sha256": dataset["input_sha"],
            "run_manifest_sha256": dataset["run_sha"],
            "representation_split_role": "train",
        }
        for dataset in datasets
    ]
    return {
        "schema": "neutral-b2-train-semantic-audit-report-1.0.0",
        "status": "complete", "protocol": {
            "path": str(PROTOCOL_PATH.relative_to(ROOT)), "sha256": PROTOCOL_SHA256,
            "protocol_version": protocol["protocol_version"],
        },
        "analysis_implementation": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "population": {
            "windows": windows, "market_case_rows": sum(item["rows"] for item in windows),
            "material_records": len(records), "validation_opened": False,
            "holdout_opened": False, "outcome_fields_used": False,
        },
        "feature_and_eye_field_audit": field_audit,
        "frequencies_and_entropy": {
            "event_collection": _event_collection_summary(field_audit),
            "material_kind": _label_summary([record.material_kind for record in records]),
            "next_lifecycle": _label_summary([
                record.next_lifecycle for record in records if record.next_lifecycle is not None
            ]),
            "scale_direction_alignment": _label_summary([
                record.scale_direction_alignment for record in records
            ]),
            "joint_label": _label_summary([
                record.joint_label for record in records if record.joint_label is not None
            ]),
        },
        "visible_signature": signature,
        "next_lifecycle_conditional_information": mi,
        "scale_direction_signature_information": {
            **scale_signature_mi,
            "definitionally_dependent_on_signature_scale_fields": True,
            "used_as_go_no_go_evidence": False,
        },
        "leave_one_train_window_out": {
            "next_lifecycle_exact_visible_signature": next_oof,
            "scale_direction_non_tautological_structural_tokens": scale_oof,
        },
        "cross_date_signature_stability": stability,
        "structural_token_audit": _token_audit(
            records,
            minimum_frequency=protocol["evaluation"][
                "binary_mi_min_material_frequency"
            ],
        ),
        "go_no_go": decision,
        "decision": {
            "b2_preregistration_allowed": decision["all_pass"],
            "validation_may_be_opened": False,
            "holdout_may_be_opened": False,
            "existing_input_artifacts_sufficient_for_audited_fields": True,
            "rematerialization_required_for_audited_fields": False,
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    decision = report["go_no_go"]
    signature = report["visible_signature"]
    mi = report["next_lifecycle_conditional_information"]
    oof = report["leave_one_train_window_out"]
    stability = report["cross_date_signature_stability"]
    lines = [
        "# Neutral B2 train-only semantic signal audit", "",
        "## Verdict", "",
        ("GO: the frozen train-only information gates all passed."
         if decision["all_pass"] else
         "NO-GO: at least one frozen train-only information gate failed; B2 is not preregistered and validation remains closed."),
        "", "## Population and isolation", "",
        f"- Six train windows only; {report['population']['market_case_rows']} MarketCase rows and {report['population']['material_records']} material records.",
        "- Validation opened: false; holdout opened: false; outcome fields used: false.",
        "- Existing input artifacts contain the audited Eye/Scene/scale facts; no rematerialization is required for this audit.",
        "", "## Key evidence", "",
        f"- Cross-episode/cross-date exact signature coverage: {signature['cross_episode_cross_date_coverage']:.4f}.",
        f"- Next-lifecycle conditional MI beyond material stage: {mi['conditional_mutual_information_bits']:.6f} bits; shuffled p95 {mi['shuffle_p95_bits']:.6f}.",
        f"- Next-lifecycle OOF NLL improvement: {oof['next_lifecycle_exact_visible_signature']['nll_relative_improvement']:.4f}; balanced-accuracy lift: {oof['next_lifecycle_exact_visible_signature']['balanced_accuracy_lift']:.4f}.",
        f"- Non-tautological scale OOF NLL improvement: {oof['scale_direction_non_tautological_structural_tokens']['nll_relative_improvement']:.4f}; balanced-accuracy lift: {oof['scale_direction_non_tautological_structural_tokens']['balanced_accuracy_lift']:.4f}.",
        f"- Cross-date joint-label purity lift over stage chance: {stability['purity_lift_over_stage_chance']:.4f}.",
        "", "## Frozen gates", "",
    ]
    lines.extend(
        f"- {name}: {'PASS' if value else 'FAIL'}"
        for name, value in decision["gates"].items()
    )
    lines.extend([
        "", "## Safety interpretation", "",
        "The scale-direction label is definitionally computed from same-clock scale-relation fields. Its direct mutual information is not counted as semantic evidence; the reported scale classifier removes all five scale relation/direction fields.",
        "MicroBOS `outcome` is accepted only as the frozen four-value same-clock reference-alignment enum and is renamed before tokenization. Economic outcome/action/playbook/entry/target/invalidation/deadline fields remain excluded.",
        "",
    ])
    return "\n".join(lines)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    protocol = _load_protocol()
    datasets = _load_train_datasets(args.input_root, protocol)
    records, field_audit = _material_records(datasets, protocol)
    report = _report(datasets, records, field_audit, protocol)
    output = Path(args.output).resolve()
    _atomic_write(output, _canonical_bytes(report) + b"\n")
    _atomic_write(output.with_suffix(".md"), _markdown(report).encode("utf-8"))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
