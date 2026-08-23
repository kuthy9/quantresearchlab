#!/usr/bin/env python3
"""Audit causal-role mechanism semantics on the six frozen train windows only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from smc_trader.market_representation import RepresentationDataError  # noqa: E402
from audit_neutral_b2_semantic_signal import (  # noqa: E402
    _five_prefixes_nonempty,
    _load_train_datasets,
    _session_phase,
)


PROTOCOL_PATH = ROOT / "configs" / "neutral_b2_mechanism_semantic_audit.json"
PROTOCOL_SHA256 = "a77e04ed2113f3cec9e612e0b8c6ab8b7dcadf9000fc7ecc4c5572d651bf71a6"
TIMEFRAMES = ("4H", "1H", "15m", "5m", "1m")
RESOLUTION_CONFIRM = frozenset({"micro_bos_confirmed"})
RESOLUTION_FAIL = frozenset(
    {"location_left", "micro_bos_opposed", "reacceptance_failed"}
)
RETURN_ROLES = frozenset({"reference_reclaimed", "reacceptance_held"})
TRAJECTORY_CLASSES = (
    "confirmed_continuation",
    "failed_or_opposed",
    "returned_to_range_unconfirmed",
    "continued_unresolved",
)


@dataclass(frozen=True)
class MechanismAnchor:
    window: str
    run_sha256: str
    market_epoch_id: str
    market_episode_id: str
    revision_id: str
    entry_path_id: str
    et_date: str
    session_phase: str
    absolute_direction: str
    source_ordinal: int
    template: str
    higher_timeframe_role: str
    local_scale_role: str
    path_roles: tuple[str, ...]
    token_groups: tuple[tuple[str, frozenset[str]], ...]
    trajectory: str | None
    right_censored: bool

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            "|".join(
                (
                    self.run_sha256,
                    self.market_epoch_id,
                    self.market_episode_id,
                    self.revision_id,
                )
            ).encode("utf-8")
        ).hexdigest()

    @property
    def mechanism_key(self) -> tuple[str, str, str]:
        return (self.template, self.higher_timeframe_role, self.local_scale_role)

    def tokens(self, excluded_group: str | None = None) -> frozenset[str]:
        return frozenset(
            token
            for group, values in self.token_groups
            if group != excluded_group
            for token in values
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _load_protocol() -> Mapping[str, Any]:
    if _sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RepresentationDataError("neutral B2 mechanism audit protocol changed")
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (
        payload.get("protocol_version")
        != "neutral-b2-mechanism-semantic-audit-1.0.0"
        or payload.get("population", {}).get("split_role") != "train"
        or payload.get("population", {}).get("validation_opened") is not False
        or payload.get("population", {}).get("holdout_opened") is not False
        or payload.get("decision_time_feature_policy", {}).get(
            "absolute_long_short_used_as_model_feature"
        )
        is not False
    ):
        raise RepresentationDataError("neutral B2 mechanism audit contract changed")
    return payload


def _entropy(values: Iterable[Any]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total) for count in counts.values()
    )


def _safe_value(value: Any, forbidden: Sequence[str]) -> str | None:
    if value is None:
        return "none"
    if type(value) is bool:
        return str(value).lower()
    if not isinstance(value, str):
        return None
    result = value.strip().lower()
    if not result or len(result) > 96:
        return None
    if any(marker in result for marker in forbidden):
        return None
    return result


def _relative_direction(value: Any, mechanism: str) -> str:
    raw = str(value).strip().lower()
    if raw == mechanism:
        return "with_mechanism"
    if {raw, mechanism} == {"long", "short"}:
        return "against_mechanism"
    return "unknown"


def _path_snapshot(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    observation = json.loads(str(row["observation_transition_json"]))
    collections = observation.get("collections")
    if not isinstance(collections, Mapping):
        raise RepresentationDataError("mechanism audit observation collections changed")
    paths = collections.get("group5_path_transitions_this_update")
    if not isinstance(paths, list):
        raise RepresentationDataError("mechanism audit path collection changed")
    matches = [
        item
        for item in paths
        if isinstance(item, Mapping)
        and item.get("sequence_id") == row["entry_path_id"]
    ]
    if not matches:
        raise RepresentationDataError("MarketEpisode entry path is not transported")
    selected = max(
        matches,
        key=lambda item: (len(item.get("steps", ())), str(item.get("last_updated_at"))),
    )
    selected_steps = tuple(
        str(item.get("kind"))
        for item in selected.get("steps", ())
        if isinstance(item, Mapping)
    )
    if (
        selected.get("context_kind") != "zone_return"
        or selected.get("direction") != row["direction"]
        or not selected_steps
    ):
        raise RepresentationDataError("MarketEpisode entry path binding changed")
    for candidate in matches:
        steps = tuple(
            str(item.get("kind"))
            for item in candidate.get("steps", ())
            if isinstance(item, Mapping)
        )
        if steps != selected_steps[: len(steps)]:
            raise RepresentationDataError("same-row path snapshots are not monotone")
    return selected, observation


def _relation_details(context: Mapping[str, Any]) -> Mapping[str, str]:
    details = context.get("scale_relation_details")
    if not isinstance(details, Mapping):
        raise RepresentationDataError("mechanism audit scale relations changed")
    output: dict[str, str] = {}
    for timeframe in TIMEFRAMES:
        raw = details.get(timeframe)
        if raw is None:
            raw = details.get(timeframe.lower())
        if not isinstance(raw, Mapping):
            raise RepresentationDataError("mechanism audit scale relation changed")
        relation = str(raw.get("relation", "unknown")).strip().lower()
        if relation not in {
            "aligned",
            "normal_pullback",
            "material_opposition",
            "unknown",
        }:
            raise RepresentationDataError("mechanism audit scale enum changed")
        output[timeframe] = relation
    return output


def _local_scale_role(relations: Mapping[str, str]) -> str:
    local = [relations[timeframe] for timeframe in ("15m", "5m", "1m")]
    known = [value for value in local if value != "unknown"]
    if "material_opposition" in known:
        return "material_opposition"
    if known.count("aligned") >= 2:
        return "aligned"
    if known.count("normal_pullback") >= 2:
        return "normal_pullback"
    return "unknown"


def _template(roles: Sequence[str]) -> str:
    role_set = set(roles)
    rejection = "wick_rejection" in role_set
    left = "reference_left" in role_set
    if rejection and left:
        return "pullback_rejection_reference_left"
    if rejection:
        return "pullback_rejection"
    if left:
        return "pullback_reference_left"
    return "pullback_plain"


def _linked_graph_tokens(
    graph: Mapping[str, Any], *, path_id: str, location_id: str,
    protocol: Mapping[str, Any],
) -> frozenset[str]:
    descriptors = graph.get("relation_descriptors")
    if not isinstance(descriptors, list):
        raise RepresentationDataError("mechanism audit graph descriptors changed")
    parsed: list[tuple[Mapping[str, Any], str, str]] = []
    seed_nodes: set[str] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise RepresentationDataError("mechanism audit graph descriptor changed")
        source = descriptor.get("source")
        target = descriptor.get("target")
        if not isinstance(source, Mapping) or not isinstance(target, Mapping):
            raise RepresentationDataError("mechanism audit graph endpoint changed")
        source_id = str(source.get("node_id", ""))
        target_id = str(target.get("node_id", ""))
        parsed.append((descriptor, source_id, target_id))
        if any(seed in source_id or seed in target_id for seed in (path_id, location_id)):
            seed_nodes.update((source_id, target_id))
    connected = {
        node
        for _, source_id, target_id in parsed
        if source_id in seed_nodes or target_id in seed_nodes
        for node in (source_id, target_id)
    }
    forbidden = tuple(protocol["decision_time_feature_policy"]["forbidden_key_markers"])
    tokens: set[str] = set()
    for descriptor, source_id, target_id in parsed:
        if source_id not in connected and target_id not in connected:
            continue
        source = descriptor["source"]
        target = descriptor["target"]
        endpoint_values = [
            str(endpoint.get(field, "")).lower()
            for endpoint in (source, target)
            for field in ("kind", "role")
        ]
        if any(marker in value for marker in forbidden for value in endpoint_values):
            continue
        parts: list[str] = []
        for field in protocol["decision_time_feature_policy"][
            "allowed_graph_descriptor_fields"
        ]:
            value = _safe_value(descriptor.get(field), forbidden)
            if value is not None:
                parts.append(f"{field}={value}")
        for prefix, endpoint in (("src", source), ("dst", target)):
            for field in protocol["decision_time_feature_policy"][
                "allowed_graph_endpoint_fields"
            ]:
                value = _safe_value(endpoint.get(field), forbidden)
                if value is not None:
                    parts.append(f"{prefix}_{field}={value}")
        if parts:
            tokens.add("graph:" + "|".join(parts))
    return frozenset(tokens)


def _feature_tokens(
    row: Mapping[str, Any], path: Mapping[str, Any], observation: Mapping[str, Any],
    relations: Mapping[str, str], template: str, phase: str,
    protocol: Mapping[str, Any], field_audit: dict[str, Any],
) -> tuple[tuple[str, frozenset[str]], ...]:
    forbidden = tuple(protocol["decision_time_feature_policy"]["forbidden_key_markers"])
    direction = str(path["direction"])
    groups: dict[str, set[str]] = {
        "path_roles": {f"path:template={template}"},
        "linked_eye": set(),
        "connected_graph": set(),
        "scale_relations": set(),
        "observable_context": {f"context:session_phase={phase}"},
    }
    for step in path.get("steps", ()):
        if not isinstance(step, Mapping):
            raise RepresentationDataError("mechanism audit path step changed")
        for field in protocol["decision_time_feature_policy"]["allowed_eye_fields"][
            "group5_path_steps"
        ]:
            if field not in step:
                continue
            value = (
                _relative_direction(step[field], direction)
                if field == "direction"
                else _safe_value(step[field], forbidden)
            )
            if value is not None:
                groups["path_roles"].add(f"path:{field}={value}")
    collections = observation["collections"]
    location_id = str(row["entry_location_id"])
    for collection in (
        "group5_micro_bos_transitions_this_update",
        "group5_reacceptance_transitions_this_update",
    ):
        raw = collections.get(collection)
        if not isinstance(raw, list):
            raise RepresentationDataError("mechanism audit linked Eye collection changed")
        audit = field_audit.setdefault(
            collection,
            {"objects": 0, "linked_objects": 0, "observed_keys": set(), "used_keys": set()},
        )
        for event in raw:
            if not isinstance(event, Mapping):
                raise RepresentationDataError("mechanism audit Eye event changed")
            audit["objects"] += 1
            audit["observed_keys"].update(str(key) for key in event)
            if event.get("context_id") != location_id:
                continue
            audit["linked_objects"] += 1
            normalized = dict(event)
            normalized.pop("outcome", None)
            for field in protocol["decision_time_feature_policy"]["allowed_eye_fields"][
                collection
            ]:
                if field not in normalized:
                    continue
                value = (
                    _relative_direction(normalized[field], direction)
                    if field in {"direction", "bos_direction", "expected_direction"}
                    else _safe_value(normalized[field], forbidden)
                )
                if value is not None:
                    groups["linked_eye"].add(f"eye:{collection}:{field}={value}")
                    audit["used_keys"].add(field)
    for timeframe, relation in relations.items():
        groups["scale_relations"].add(f"scale:{timeframe}:relation={relation}")
    context = json.loads(str(row["neutral_global_context_json"]))
    balance = context.get("balance_context") or {}
    if not isinstance(balance, Mapping):
        raise RepresentationDataError("mechanism audit balance context changed")
    for name, value in (
        ("market_mode", context.get("market_mode", "unknown")),
        ("balance_status", balance.get("status", "none")),
        ("balance_internal_crossing", balance.get("internal_crossing", False)),
        ("balance_accepted_external_break", balance.get("accepted_external_break", False)),
    ):
        safe = _safe_value(value, forbidden)
        if safe is not None:
            groups["observable_context"].add(f"context:{name}={safe}")
    graph = json.loads(str(row["scene_graph_delta_json"]))
    groups["connected_graph"].update(
        _linked_graph_tokens(
            graph,
            path_id=str(row["entry_path_id"]),
            location_id=location_id,
            protocol=protocol,
        )
    )
    emitted = {token for values in groups.values() for token in values}
    if any(marker in token.lower() for marker in forbidden for token in emitted):
        raise RepresentationDataError("mechanism audit emitted a forbidden feature")
    return tuple((group, frozenset(groups[group])) for group in sorted(groups))


def _epoch_ends(
    rows: Sequence[Mapping[str, Any]], run_manifest: Mapping[str, Any]
) -> Mapping[str, int]:
    starts: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        prefixes = json.loads(str(row["ohlcv_prefix_refs_json"]))
        one_minute = [item for item in prefixes if item.get("timeframe") == "1m"]
        if len(one_minute) != 1 or type(one_minute[0].get("replay_view_1m_row_start")) is not int:
            raise RepresentationDataError("mechanism audit epoch prefix boundary changed")
        starts[str(row["market_epoch_id"])].add(
            int(one_minute[0]["replay_view_1m_row_start"])
        )
    exact = {epoch: next(iter(values)) for epoch, values in starts.items() if len(values) == 1}
    if len(exact) != len(starts):
        raise RepresentationDataError("mechanism audit epoch starts are ambiguous")
    ordered = sorted(exact.items(), key=lambda item: item[1])
    source_end = run_manifest.get("source", {}).get("rows")
    if type(source_end) is not int or source_end <= 0:
        raise RepresentationDataError("mechanism audit source boundary changed")
    return {
        epoch: ordered[index + 1][1] if index + 1 < len(ordered) else source_end
        for index, (epoch, _) in enumerate(ordered)
    }


def _trajectory(
    anchor_index: int,
    ordered: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]],
    *, horizon: int, epoch_end: int,
) -> tuple[str | None, bool]:
    anchor_row, anchor_path, _ = ordered[anchor_index]
    anchor_ordinal = int(anchor_row["source_replay_ordinal"])
    anchor_roles = tuple(str(item["kind"]) for item in anchor_path["steps"])
    latest_roles = anchor_roles
    for row, path, _ in ordered[anchor_index + 1 :]:
        ordinal = int(row["source_replay_ordinal"])
        if ordinal > anchor_ordinal + horizon:
            break
        roles = tuple(str(item["kind"]) for item in path["steps"])
        if roles[: len(anchor_roles)] != anchor_roles:
            raise RepresentationDataError("mechanism path roles are not monotone after anchor")
        latest_roles = roles
    added = latest_roles[len(anchor_roles) :]
    for role in added:
        if role in RESOLUTION_CONFIRM:
            return "confirmed_continuation", False
        if role in RESOLUTION_FAIL:
            return "failed_or_opposed", False
    full_horizon = anchor_ordinal + horizon < epoch_end
    if not full_horizon:
        return None, True
    if RETURN_ROLES.issubset(set(added)):
        return "returned_to_range_unconfirmed", False
    return "continued_unresolved", False


def _extract_anchors(
    datasets: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[tuple[MechanismAnchor, ...], Mapping[str, Any]]:
    policy = protocol["decision_time_feature_policy"]
    required_prefix = tuple(policy["required_path_prefix"])
    excluded_steps = set(policy["anchor_excludes_steps"])
    core_templates = set(protocol["mechanism_templates"]["core_templates"])
    horizon = int(protocol["trajectory_target"]["horizon_completed_real_source_1m_bars"])
    anchors: list[MechanismAnchor] = []
    counts: Counter[str] = Counter()
    templates_by_window: dict[str, Counter[str]] = defaultdict(Counter)
    field_audit: dict[str, Any] = {}
    for dataset in datasets:
        rows = dataset["rows"]
        ends = _epoch_ends(rows, dataset["run_manifest"])
        groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(str(row["market_epoch_id"]), str(row["market_episode_id"]))].append(row)
        for (epoch, _), episode_rows in groups.items():
            counts["episodes"] += 1
            episode_rows.sort(
                key=lambda row: (int(row["revision_index"]), int(row["source_replay_ordinal"]))
            )
            if [int(row["revision_index"]) for row in episode_rows] != list(
                range(len(episode_rows))
            ):
                raise RepresentationDataError("mechanism audit episode revisions changed")
            if len({str(row["entry_path_id"]) for row in episode_rows}) != 1:
                raise RepresentationDataError("MarketEpisode entry path changed within episode")
            ordered: list[
                tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]
            ] = []
            previous_roles: tuple[str, ...] = ()
            for row in episode_rows:
                path, observation = _path_snapshot(row)
                counts[f"entry_path_context_kind:{path['context_kind']}"] += 1
                roles = tuple(str(item["kind"]) for item in path["steps"])
                if roles[: len(previous_roles)] != previous_roles:
                    raise RepresentationDataError("MarketEpisode path is not monotone")
                previous_roles = roles
                ordered.append((row, path, observation))
                counts["path_bound_rows"] += 1
            anchor_index: int | None = None
            exclusion: str | None = None
            for index, (row, path, _) in enumerate(ordered):
                roles = tuple(str(item["kind"]) for item in path["steps"])
                if not all(role in roles for role in required_prefix):
                    continue
                if path.get("lifecycle") != "active":
                    exclusion = "first_pullback_already_closed"
                    continue
                if set(roles) & excluded_steps:
                    exclusion = "ambiguous_or_resolved_at_anchor"
                    continue
                if not _five_prefixes_nonempty(row):
                    exclusion = "incomplete_five_scale_prefix"
                    continue
                anchor_index = index
                break
            if anchor_index is None:
                counts[exclusion or "no_unresolved_pullback_anchor"] += 1
                continue
            row, path, observation = ordered[anchor_index]
            roles = tuple(str(item["kind"]) for item in path["steps"])
            template = _template(roles)
            counts["unresolved_pullback_anchors"] += 1
            counts[f"template_seen:{template}"] += 1
            templates_by_window[str(dataset["profile_name"])][template] += 1
            if template not in core_templates:
                counts["non_core_template"] += 1
                continue
            context = json.loads(str(row["neutral_global_context_json"]))
            relations = _relation_details(context)
            higher = relations["1H"]
            local = _local_scale_role(relations)
            if higher == "unknown" or local == "unknown":
                counts["unknown_cross_scale_relation"] += 1
                continue
            phase, et_date = _session_phase(row["asof"])
            token_groups = _feature_tokens(
                row,
                path,
                observation,
                relations,
                template,
                phase,
                protocol,
                field_audit,
            )
            trajectory, censored = _trajectory(
                anchor_index,
                ordered,
                horizon=horizon,
                epoch_end=ends[epoch],
            )
            anchors.append(
                MechanismAnchor(
                    window=str(dataset["profile_name"]),
                    run_sha256=str(dataset["run_sha"]),
                    market_epoch_id=epoch,
                    market_episode_id=str(row["market_episode_id"]),
                    revision_id=str(row["revision_id"]),
                    entry_path_id=str(row["entry_path_id"]),
                    et_date=et_date,
                    session_phase=phase,
                    absolute_direction=str(path["direction"]),
                    source_ordinal=int(row["source_replay_ordinal"]),
                    template=template,
                    higher_timeframe_role=higher,
                    local_scale_role=local,
                    path_roles=roles,
                    token_groups=token_groups,
                    trajectory=trajectory,
                    right_censored=censored,
                )
            )
            counts["feature_eligible_core_anchors"] += 1
            counts["right_censored"] += censored
    if counts["path_bound_rows"] != sum(len(dataset["rows"]) for dataset in datasets):
        raise RepresentationDataError("not every train row retained its entry path")
    normalized_field_audit = {
        collection: {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in values.items()
        }
        for collection, values in sorted(field_audit.items())
    }
    return tuple(anchors), {
        "counts": dict(sorted(counts.items())),
        "templates_by_window": {
            window: dict(sorted(values.items()))
            for window, values in sorted(templates_by_window.items())
        },
        "linked_eye_field_audit": normalized_field_audit,
        "entry_path_binding_coverage": counts["path_bound_rows"]
        / sum(len(dataset["rows"]) for dataset in datasets),
        "labels_used_as_features": False,
        "absolute_direction_used_as_feature": False,
        "identities_used_as_features": False,
        "forbidden_fields_used": False,
    }


def _hash_choice(anchor: MechanismAnchor, candidates: Sequence[MechanismAnchor], salt: str) -> MechanismAnchor:
    return min(
        candidates,
        key=lambda candidate: hashlib.sha256(
            f"{salt}|{anchor.identity}|{candidate.identity}".encode("utf-8")
        ).hexdigest(),
    )


def _pairs(
    anchors: Sequence[MechanismAnchor],
) -> tuple[tuple[tuple[str, MechanismAnchor, MechanismAnchor], ...], Mapping[str, Any]]:
    positive: list[tuple[str, MechanismAnchor, MechanismAnchor]] = []
    negative: list[tuple[str, MechanismAnchor, MechanismAnchor]] = []
    positive_queries = 0
    negative_queries = 0
    for anchor in anchors:
        candidates = [
            item
            for item in anchors
            if item.mechanism_key == anchor.mechanism_key
            and item.market_episode_id != anchor.market_episode_id
            and item.et_date != anchor.et_date
        ]
        if candidates:
            positive_queries += 1
            positive.append(("positive", anchor, _hash_choice(anchor, candidates, "positive")))
        hard: list[tuple[str, MechanismAnchor]] = []
        for item in anchors:
            if item.market_episode_id == anchor.market_episode_id:
                continue
            common = (
                item.window == anchor.window
                and item.session_phase == anchor.session_phase
                and item.absolute_direction == anchor.absolute_direction
            )
            if not common:
                continue
            if (
                item.template != anchor.template
                and {item.template, anchor.template}
                == {"pullback_rejection", "pullback_reference_left"}
                and item.higher_timeframe_role == anchor.higher_timeframe_role
                and item.local_scale_role == anchor.local_scale_role
            ):
                hard.append(("hard_negative_template_conflict", item))
            if (
                item.template == anchor.template
                and item.local_scale_role == anchor.local_scale_role
                and {item.higher_timeframe_role, anchor.higher_timeframe_role}
                & {"material_opposition"}
                and item.higher_timeframe_role != anchor.higher_timeframe_role
                and "unknown" not in {item.higher_timeframe_role, anchor.higher_timeframe_role}
            ):
                hard.append(("hard_negative_higher_timeframe_conflict", item))
            if (
                item.template == anchor.template
                and item.higher_timeframe_role == anchor.higher_timeframe_role
                and {item.local_scale_role, anchor.local_scale_role}
                & {"material_opposition"}
                and item.local_scale_role != anchor.local_scale_role
                and "unknown" not in {item.local_scale_role, anchor.local_scale_role}
            ):
                hard.append(("hard_negative_local_scale_conflict", item))
        if hard:
            negative_queries += 1
            for kind in sorted({kind for kind, _ in hard}):
                values = [item for candidate_kind, item in hard if candidate_kind == kind]
                negative.append((kind, anchor, _hash_choice(anchor, values, kind)))
    deduplicated: dict[tuple[str, str, str], tuple[str, MechanismAnchor, MechanismAnchor]] = {}
    for kind, left, right in positive + negative:
        key = (kind, *sorted((left.identity, right.identity)))
        deduplicated[key] = (kind, left, right)
    pairs = tuple(deduplicated[key] for key in sorted(deduplicated))
    per_window: dict[str, Any] = {}
    for window in sorted({anchor.window for anchor in anchors}):
        values = [anchor for anchor in anchors if anchor.window == window]
        positive_covered = 0
        negative_covered = 0
        for anchor in values:
            if any(
                item.mechanism_key == anchor.mechanism_key
                and item.market_episode_id != anchor.market_episode_id
                and item.et_date != anchor.et_date
                for item in values
            ):
                positive_covered += 1
            if any(
                kind != "positive" and left.identity == anchor.identity
                for kind, left, _ in negative
            ):
                negative_covered += 1
        per_window[window] = {
            "queries": len(values),
            "within_window_positive_query_coverage": positive_covered / len(values),
            "hard_negative_query_coverage": negative_covered / len(values),
        }
    return pairs, {
        "eligible_queries": len(anchors),
        "positive_query_coverage": positive_queries / len(anchors) if anchors else 0.0,
        "hard_negative_query_coverage": negative_queries / len(anchors) if anchors else 0.0,
        "per_window": per_window,
        "unique_pairs_by_type": dict(Counter(kind for kind, _, _ in pairs)),
    }


def _probabilities(counts: Counter[str], alpha: float) -> np.ndarray:
    total = sum(counts.values()) + alpha * len(TRAJECTORY_CLASSES)
    return np.asarray(
        [(counts[value] + alpha) / total for value in TRAJECTORY_CLASSES],
        dtype=np.float64,
    )


def _metrics(probabilities: Sequence[np.ndarray], labels: Sequence[str]) -> Mapping[str, Any]:
    index = {value: position for position, value in enumerate(TRAJECTORY_CLASSES)}
    actual = np.asarray([index[value] for value in labels], dtype=np.int64)
    matrix = np.stack(probabilities)
    predicted = matrix.argmax(axis=1)
    observed = sorted(set(actual.tolist()))
    recalls = [float((predicted[actual == value] == value).mean()) for value in observed]
    return {
        "rows": len(labels),
        "nll": float(-np.log(np.maximum(matrix[np.arange(len(actual)), actual], 1e-12)).mean()),
        "accuracy": float((predicted == actual).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "observed_classes": [TRAJECTORY_CLASSES[value] for value in observed],
    }


def _oof_nb(
    anchors: Sequence[MechanismAnchor], *, excluded_group: str | None = None,
    token_override: Mapping[str, frozenset[str]] | None = None,
    minimum_document_frequency: int = 5,
) -> Mapping[str, Any]:
    labelled = [anchor for anchor in anchors if anchor.trajectory is not None]
    base_probabilities: list[np.ndarray] = []
    model_probabilities: list[np.ndarray] = []
    labels: list[str] = []
    folds: list[Mapping[str, Any]] = []
    for window in sorted({anchor.window for anchor in labelled}):
        train = [anchor for anchor in labelled if anchor.window != window]
        test = [anchor for anchor in labelled if anchor.window == window]
        global_counts = Counter(anchor.trajectory for anchor in train)
        base_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for anchor in train:
            base_counts[anchor.template][str(anchor.trajectory)] += 1
        def tokens(anchor: MechanismAnchor) -> frozenset[str]:
            if token_override is not None:
                return token_override[anchor.identity]
            return anchor.tokens(excluded_group)
        document_frequency = Counter(token for anchor in train for token in tokens(anchor))
        vocabulary = sorted(
            token for token, count in document_frequency.items()
            if count >= minimum_document_frequency
        )
        vocabulary_set = set(vocabulary)
        class_counts = Counter(str(anchor.trajectory) for anchor in train)
        token_counts = {value: Counter() for value in TRAJECTORY_CLASSES}
        for anchor in train:
            token_counts[str(anchor.trajectory)].update(tokens(anchor) & vocabulary_set)
        bases: dict[str, float] = {}
        deltas: dict[str, dict[str, float]] = {}
        for value in TRAJECTORY_CLASSES:
            class_total = class_counts[value]
            prior = (class_total + 1.0) / (len(train) + len(TRAJECTORY_CLASSES))
            base_log = math.log(prior)
            delta: dict[str, float] = {}
            for token in vocabulary:
                probability = (token_counts[value][token] + 1.0) / (class_total + 2.0)
                base_log += math.log(1.0 - probability)
                delta[token] = math.log(probability) - math.log(1.0 - probability)
            bases[value] = base_log
            deltas[value] = delta
        fold_labels: list[str] = []
        fold_base_probabilities: list[np.ndarray] = []
        fold_model_probabilities: list[np.ndarray] = []
        for anchor in test:
            base_probability = _probabilities(
                base_counts.get(anchor.template, global_counts), 1.0
            )
            base_probabilities.append(base_probability)
            fold_base_probabilities.append(base_probability)
            present = tokens(anchor) & vocabulary_set
            logits = np.asarray(
                [bases[value] + sum(deltas[value][token] for token in present) for value in TRAJECTORY_CLASSES]
            )
            logits -= logits.max()
            probability = np.exp(logits)
            model_probability = probability / probability.sum()
            model_probabilities.append(model_probability)
            fold_model_probabilities.append(model_probability)
            labels.append(str(anchor.trajectory))
            fold_labels.append(str(anchor.trajectory))
        fold_base = _metrics(fold_base_probabilities, fold_labels)
        fold_model = _metrics(fold_model_probabilities, fold_labels)
        folds.append(
            {
                "window": window,
                "rows": len(test),
                "train_rows": len(train),
                "vocabulary": len(vocabulary),
                "test_class_counts": dict(sorted(Counter(fold_labels).items())),
                "template_prior": fold_base,
                "semantic_naive_bayes": fold_model,
                "nll_relative_improvement": 1.0
                - fold_model["nll"] / fold_base["nll"],
                "balanced_accuracy_lift": fold_model["balanced_accuracy"]
                - fold_base["balanced_accuracy"],
            }
        )
    base = _metrics(base_probabilities, labels)
    model = _metrics(model_probabilities, labels)
    return {
        "folds": folds,
        "template_prior": base,
        "semantic_naive_bayes": model,
        "nll_relative_improvement": 1.0 - model["nll"] / base["nll"],
        "balanced_accuracy_lift": model["balanced_accuracy"] - base["balanced_accuracy"],
    }


def _shuffle_baseline(
    anchors: Sequence[MechanismAnchor], protocol: Mapping[str, Any]
) -> Mapping[str, Any]:
    rng = random.Random(protocol["evaluation"]["shuffle_seed"])
    strata: dict[tuple[str, str, str], list[MechanismAnchor]] = defaultdict(list)
    for anchor in anchors:
        strata[(anchor.window, anchor.template, anchor.session_phase)].append(anchor)
    improvements: list[float] = []
    repetitions = int(protocol["evaluation"]["shuffle_repetitions"])
    for _ in range(repetitions):
        override: dict[str, frozenset[str]] = {}
        for values in strata.values():
            bundles = [anchor.tokens() for anchor in values]
            rng.shuffle(bundles)
            override.update(
                (anchor.identity, bundle)
                for anchor, bundle in zip(values, bundles, strict=True)
            )
        result = _oof_nb(
            anchors,
            token_override=override,
            minimum_document_frequency=int(
                protocol["evaluation"]["minimum_train_document_frequency"]
            ),
        )
        improvements.append(float(result["nll_relative_improvement"]))
    return {
        "repetitions": repetitions,
        "mean_nll_relative_improvement": float(np.mean(improvements)),
        "p95_nll_relative_improvement": float(np.quantile(improvements, 0.95)),
    }


def _pair_stability(
    anchors: Sequence[MechanismAnchor],
    pairs: Sequence[tuple[str, MechanismAnchor, MechanismAnchor]],
) -> Mapping[str, Any]:
    labelled = [anchor for anchor in anchors if anchor.trajectory is not None]
    positive: list[float] = []
    negative: list[float] = []
    negative_by_type: dict[str, list[float]] = defaultdict(list)
    positive_chance: list[float] = []
    negative_chance: list[float] = []
    for kind, left, right in pairs:
        if left.trajectory is None or right.trajectory is None:
            continue
        pool = [
            item
            for item in labelled
            if item.template == left.template
            and item.market_episode_id != left.market_episode_id
            and item.et_date != left.et_date
        ]
        same_chance = (
            sum(item.trajectory == left.trajectory for item in pool) / len(pool)
            if pool else 0.0
        )
        if kind == "positive":
            positive.append(float(left.trajectory == right.trajectory))
            positive_chance.append(same_chance)
        else:
            disagreement = float(left.trajectory != right.trajectory)
            negative.append(disagreement)
            negative_by_type[kind].append(disagreement)
            negative_chance.append(1.0 - same_chance)
    positive_purity = float(np.mean(positive)) if positive else 0.0
    positive_base = float(np.mean(positive_chance)) if positive_chance else 0.0
    negative_disagreement = float(np.mean(negative)) if negative else 0.0
    negative_base = float(np.mean(negative_chance)) if negative_chance else 0.0
    return {
        "labelled_positive_pairs": len(positive),
        "positive_trajectory_purity": positive_purity,
        "positive_template_chance": positive_base,
        "positive_purity_lift": positive_purity - positive_base,
        "labelled_hard_negative_pairs": len(negative),
        "hard_negative_trajectory_disagreement": negative_disagreement,
        "hard_negative_template_chance": negative_base,
        "hard_negative_disagreement_lift": negative_disagreement - negative_base,
        "hard_negative_disagreement_by_type": {
            kind: {
                "pairs": len(values),
                "disagreement": float(np.mean(values)) if values else 0.0,
            }
            for kind, values in sorted(negative_by_type.items())
        },
    }


def _support(anchors: Sequence[MechanismAnchor]) -> Mapping[str, Any]:
    by_template = Counter(anchor.template for anchor in anchors)
    by_window_template: dict[str, Counter[str]] = defaultdict(Counter)
    by_key = Counter(anchor.mechanism_key for anchor in anchors)
    dates: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    episodes: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for anchor in anchors:
        by_window_template[anchor.window][anchor.template] += 1
        dates[anchor.mechanism_key].add(anchor.et_date)
        episodes[anchor.mechanism_key].add(anchor.market_episode_id)
    return {
        "feature_eligible_core_anchors": len(anchors),
        "unique_mechanism_keys": len(by_key),
        "templates": dict(sorted(by_template.items())),
        "templates_by_window": {
            window: dict(sorted(counts.items()))
            for window, counts in sorted(by_window_template.items())
        },
        "mechanism_keys": [
            {
                "template": key[0],
                "higher_timeframe_role": key[1],
                "local_scale_role": key[2],
                "anchors": count,
                "dates": len(dates[key]),
                "episodes": len(episodes[key]),
                "windows": len({anchor.window for anchor in anchors if anchor.mechanism_key == key}),
            }
            for key, count in sorted(by_key.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def _trajectory_summary(anchors: Sequence[MechanismAnchor]) -> Mapping[str, Any]:
    labelled = [anchor for anchor in anchors if anchor.trajectory is not None]
    counts = Counter(str(anchor.trajectory) for anchor in labelled)
    windows = {
        value: len({anchor.window for anchor in labelled if anchor.trajectory == value})
        for value in TRAJECTORY_CLASSES
    }
    total = len(labelled)
    def group_summary(values: Sequence[MechanismAnchor]) -> Mapping[str, Any]:
        group_counts = Counter(str(anchor.trajectory) for anchor in values)
        return {
            "anchors": len(values),
            "class_counts": {
                value: group_counts[value] for value in TRAJECTORY_CLASSES
            },
            "entropy_bits": _entropy(anchor.trajectory for anchor in values),
        }
    by_template = {
        template: group_summary(
            [anchor for anchor in labelled if anchor.template == template]
        )
        for template in sorted({anchor.template for anchor in labelled})
    }
    by_window = {
        window: group_summary(
            [anchor for anchor in labelled if anchor.window == window]
        )
        for window in sorted({anchor.window for anchor in labelled})
    }
    by_key = []
    for key in sorted({anchor.mechanism_key for anchor in labelled}):
        summary = group_summary(
            [anchor for anchor in labelled if anchor.mechanism_key == key]
        )
        by_key.append(
            {
                "template": key[0],
                "higher_timeframe_role": key[1],
                "local_scale_role": key[2],
                **summary,
            }
        )
    return {
        "anchors": len(anchors),
        "right_censored": len(anchors) - total,
        "uncensored_fraction": total / len(anchors) if anchors else 0.0,
        "class_counts": {value: counts[value] for value in TRAJECTORY_CLASSES},
        "class_shares": {
            value: counts[value] / total if total else 0.0 for value in TRAJECTORY_CLASSES
        },
        "class_windows": windows,
        "entropy_bits": _entropy(anchor.trajectory for anchor in labelled),
        "by_template": by_template,
        "by_window": by_window,
        "by_mechanism_key": by_key,
    }


def _blind_summary(anchor: MechanismAnchor) -> Mapping[str, Any]:
    groups = dict(anchor.token_groups)
    return {
        "sample_id": anchor.identity[:16],
        "session_phase": anchor.session_phase,
        "path_roles": list(anchor.path_roles),
        "higher_timeframe_relation": anchor.higher_timeframe_role,
        "local_scale_relation": anchor.local_scale_role,
        "linked_eye_facts": sorted(groups["linked_eye"]),
        "connected_graph_motifs": sorted(groups["connected_graph"])[:12],
        "observable_context": sorted(groups["observable_context"]),
    }


def _blind_packets(
    pairs: Sequence[tuple[str, MechanismAnchor, MechanismAnchor]],
    protocol: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    seed = str(protocol["evaluation"]["blind_review_seed"])
    positive = [pair for pair in pairs if pair[0] == "positive"]
    negative = [pair for pair in pairs if pair[0] != "positive"]
    key = lambda pair: hashlib.sha256(
        f"{seed}|{pair[0]}|{pair[1].identity}|{pair[2].identity}".encode("utf-8")
    ).hexdigest()
    positive = sorted(positive, key=key)[: int(protocol["evaluation"]["blind_review_positive_pairs"])]
    negative = sorted(negative, key=key)[: int(protocol["evaluation"]["blind_review_hard_negative_pairs"])]
    selected = sorted(positive + negative, key=key)
    packet: list[Mapping[str, Any]] = []
    answer: dict[str, Any] = {}
    for index, (kind, left, right) in enumerate(selected):
        pair_id = hashlib.sha256(f"blind|{key((kind, left, right))}".encode()).hexdigest()[:16]
        sides = [_blind_summary(left), _blind_summary(right)]
        if int(key((kind, left, right)), 16) % 2:
            sides.reverse()
        packet.append(
            {
                "pair_id": pair_id,
                "left": sides[0],
                "right": sides[1],
                "review_question": "Do these two decision-time states express the same causal market mechanism?",
                "reviewer_answer": None,
            }
        )
        answer[pair_id] = {
            "frozen_rule_relation": kind,
            "expected_same_mechanism": kind == "positive",
            "left_internal_id": left.identity,
            "right_internal_id": right.identity,
        }
    return packet, {
        "schema": "neutral-b2-mechanism-blind-review-key-1.0.0",
        "review_complete": False,
        "pairs": answer,
    }


def _gates(
    anchors: Sequence[MechanismAnchor], extraction: Mapping[str, Any],
    support: Mapping[str, Any], pairs: Mapping[str, Any],
    trajectory: Mapping[str, Any], oof: Mapping[str, Any],
    shuffle: Mapping[str, Any], stability: Mapping[str, Any],
    ablations: Mapping[str, Any], protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    thresholds = protocol["go_no_go_gates"]
    template_counts = support["templates"]
    template_windows = support["templates_by_window"]
    core = protocol["mechanism_templates"]["core_templates"]
    unresolved = extraction["counts"]["unresolved_pullback_anchors"]
    eligible_fraction = len(anchors) / unresolved if unresolved else 0.0
    minority = min(trajectory["class_shares"].values())
    effective = sum(item["effective"] for item in ablations.values())
    gates = {
        "entry_path_binding": extraction["entry_path_binding_coverage"] == 1.0,
        "feature_eligible_anchor_fraction": eligible_fraction
        >= thresholds["feature_eligible_anchor_fraction_min"],
        "core_template_support": all(
            template_counts.get(template, 0) >= thresholds["core_template_anchors_min"]
            and sum(
                values.get(template, 0) > 0 for values in template_windows.values()
            )
            >= thresholds["core_template_windows_min"]
            and all(
                values.get(template, 0) >= thresholds["core_template_each_window_min"]
                for values in template_windows.values()
            )
            for template in core
        ),
        "positive_pair_coverage": pairs["positive_query_coverage"]
        >= thresholds["positive_pair_query_coverage_min"],
        "positive_pair_coverage_each_window": all(
            item["within_window_positive_query_coverage"]
            >= thresholds["positive_pair_query_coverage_each_window_min"]
            for item in pairs["per_window"].values()
        ),
        "hard_negative_coverage": pairs["hard_negative_query_coverage"]
        >= thresholds["hard_negative_query_coverage_min"],
        "hard_negative_coverage_each_window": all(
            item["hard_negative_query_coverage"]
            >= thresholds["hard_negative_query_coverage_each_window_min"]
            for item in pairs["per_window"].values()
        ),
        "trajectory_uncensored": trajectory["uncensored_fraction"]
        >= thresholds["uncensored_trajectory_fraction_min"],
        "trajectory_entropy": trajectory["entropy_bits"]
        >= thresholds["trajectory_entropy_bits_min"],
        "trajectory_minority_share": minority
        >= thresholds["trajectory_minority_share_min"],
        "trajectory_cross_window_classes": all(
            count >= thresholds["trajectory_each_class_windows_min"]
            for count in trajectory["class_windows"].values()
        ),
        "oof_nll": oof["nll_relative_improvement"]
        >= thresholds["oof_nll_relative_improvement_over_template_prior_min"],
        "oof_balanced_accuracy": oof["balanced_accuracy_lift"]
        >= thresholds["oof_balanced_accuracy_lift_over_template_prior_min"],
        "oof_above_shuffle": oof["nll_relative_improvement"]
        > shuffle["p95_nll_relative_improvement"],
        "positive_trajectory_stability": stability["positive_purity_lift"]
        >= thresholds["positive_trajectory_purity_lift_over_template_chance_min"],
        "hard_negative_trajectory_separation": stability[
            "hard_negative_disagreement_lift"
        ]
        >= thresholds["hard_negative_trajectory_disagreement_lift_over_template_chance_min"],
        "field_ablation_effective_groups": effective
        >= thresholds["effective_ablation_groups_min"],
    }
    return {
        "feature_eligible_anchor_fraction": eligible_fraction,
        "trajectory_minority_share": minority,
        "effective_ablation_groups": effective,
        "automated_gates": gates,
        "automated_all_pass": all(gates.values()),
        "human_blind_review_complete": False,
        "human_blind_review_pass": False,
        "all_pass": False,
    }


def _report(
    datasets: Sequence[Mapping[str, Any]], anchors: Sequence[MechanismAnchor],
    extraction: Mapping[str, Any], protocol: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], Mapping[str, Any]]:
    support = _support(anchors)
    pairs, pair_summary = _pairs(anchors)
    trajectory = _trajectory_summary(anchors)
    minimum_frequency = int(protocol["evaluation"]["minimum_train_document_frequency"])
    oof = _oof_nb(anchors, minimum_document_frequency=minimum_frequency)
    shuffle = _shuffle_baseline(anchors, protocol)
    stability = _pair_stability(anchors, pairs)
    ablations: dict[str, Any] = {}
    thresholds = protocol["go_no_go_gates"]
    for group in protocol["evaluation"]["field_ablation_groups"]:
        result = _oof_nb(
            anchors,
            excluded_group=group,
            minimum_document_frequency=minimum_frequency,
        )
        nll_degradation = (
            result["semantic_naive_bayes"]["nll"]
            - oof["semantic_naive_bayes"]["nll"]
        )
        balanced_accuracy_drop = (
            oof["semantic_naive_bayes"]["balanced_accuracy"]
            - result["semantic_naive_bayes"]["balanced_accuracy"]
        )
        ablations[group] = {
            "nll_degradation_when_removed": nll_degradation,
            "balanced_accuracy_drop_when_removed": balanced_accuracy_drop,
            "effective": nll_degradation
            >= thresholds["ablation_effective_nll_degradation_min"]
            or balanced_accuracy_drop
            >= thresholds["ablation_effective_balanced_accuracy_drop_min"],
            "beneficial_when_present_on_nll": nll_degradation > 0.0,
            "beneficial_when_present_on_balanced_accuracy": balanced_accuracy_drop > 0.0,
            "ablated_oof": result,
        }
    gates = _gates(
        anchors,
        extraction,
        support,
        pair_summary,
        trajectory,
        oof,
        shuffle,
        stability,
        ablations,
        protocol,
    )
    blind_packet, blind_key = _blind_packets(pairs, protocol)
    report = {
        "schema": "neutral-b2-mechanism-semantic-audit-report-1.0.0",
        "status": "complete",
        "protocol": {
            "path": str(PROTOCOL_PATH.relative_to(ROOT)),
            "sha256": PROTOCOL_SHA256,
            "protocol_version": protocol["protocol_version"],
        },
        "analysis_implementation": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "loader_dependency": {
            "path": "scripts/audit_neutral_b2_semantic_signal.py",
            "sha256": _sha256(ROOT / "scripts" / "audit_neutral_b2_semantic_signal.py"),
        },
        "population": {
            "windows": [
                {
                    "profile": dataset["profile_name"],
                    "rows": len(dataset["rows"]),
                    "input_manifest_sha256": dataset["input_sha"],
                    "run_manifest_sha256": dataset["run_sha"],
                }
                for dataset in datasets
            ],
            "market_case_rows": sum(len(dataset["rows"]) for dataset in datasets),
            "validation_opened": False,
            "holdout_opened": False,
            "outcome_fields_used": False,
        },
        "eye_semantic_and_transport_audit": extraction,
        "mechanism_support": support,
        "pair_construction": pair_summary,
        "fixed_horizon_trajectory": trajectory,
        "pair_trajectory_stability": stability,
        "leave_one_train_window_out": oof,
        "shuffled_semantic_baseline": shuffle,
        "field_group_ablation": ablations,
        "go_no_go": gates,
        "decision": {
            "technical_signal_eligible_for_blind_review": gates["automated_all_pass"],
            "b2_preregistration_allowed": False,
            "blocker": (
                "external_blind_review_incomplete"
                if gates["automated_all_pass"]
                else "one_or_more_train_only_automated_semantic_gates_failed"
            ),
            "validation_may_be_opened": False,
            "holdout_may_be_opened": False,
            "existing_input_artifacts_sufficient_for_current_entry_path_audit": True,
            "existing_input_artifacts_sufficient_for_full_requested_mechanism_vocabulary": False,
            "full_vocabulary_limitation": "MarketEpisode entry_path binds only zone_return; global pool_reversal paths are not proven to belong to the current episode",
            "rematerialization_required_for_current_entry_path_audit": False,
            "full_b2_rematerialization_decision_deferred": True,
            "next_lifecycle_is_primary_semantic_target": False,
        },
    }
    return report, blind_packet, blind_key


def _markdown(report: Mapping[str, Any]) -> str:
    gate = report["go_no_go"]
    trajectory = report["fixed_horizon_trajectory"]
    oof = report["leave_one_train_window_out"]
    stability = report["pair_trajectory_stability"]
    lines = [
        "# Neutral B2 causal-mechanism train-only audit",
        "",
        "## Verdict",
        "",
        (
            "Automated signal gates passed; B2 remains blocked until the separate human blind review passes."
            if gate["automated_all_pass"]
            else "NO-GO: at least one frozen train-only causal-mechanism gate failed; B2 is not preregistered."
        ),
        "",
        "## Isolation",
        "",
        f"- Six train windows only; {report['population']['market_case_rows']} MarketCase rows.",
        "- Validation opened: false; holdout opened: false; outcome fields used: false.",
        "- Only the current episode's entry_path_id was used; unrelated global paths were excluded.",
        "- The bound entry paths are zone_return only. A broader pool-sweep mechanism vocabulary still needs an explicit association contract before deciding whether rematerialization is required.",
        "",
        "## Key evidence",
        "",
        f"- Feature-eligible core anchors: {report['mechanism_support']['feature_eligible_core_anchors']}.",
        f"- Trajectory counts: {trajectory['class_counts']}; right-censored {trajectory['right_censored']}.",
        f"- OOF NLL improvement over template prior: {oof['nll_relative_improvement']:.4f}; balanced-accuracy lift {oof['balanced_accuracy_lift']:.4f}.",
        f"- Positive trajectory-purity lift: {stability['positive_purity_lift']:.4f}.",
        f"- Hard-negative trajectory-disagreement lift: {stability['hard_negative_disagreement_lift']:.4f}.",
        "",
        "## Frozen automated gates",
        "",
    ]
    lines.extend(
        f"- {name}: {'PASS' if value else 'FAIL'}"
        for name, value in gate["automated_gates"].items()
    )
    lines.extend(
        [
            "",
            "## Blind review",
            "",
            "The audit generated a label-free pair packet and a separate frozen rule key. No independent human judgment is claimed here; preregistration remains blocked until that review is completed.",
            "",
        ]
    )
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
    anchors, extraction = _extract_anchors(datasets, protocol)
    report, blind_packet, blind_key = _report(datasets, anchors, extraction, protocol)
    output = Path(args.output).resolve()
    _atomic_write(output, _canonical_bytes(report) + b"\n")
    _atomic_write(output.with_suffix(".md"), _markdown(report).encode("utf-8"))
    blind_path = output.with_name("blind_pairs.jsonl")
    _atomic_write(
        blind_path,
        b"".join(_canonical_bytes(item) + b"\n" for item in blind_packet),
    )
    _atomic_write(
        output.with_name("blind_pair_key.json"), _canonical_bytes(blind_key) + b"\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
