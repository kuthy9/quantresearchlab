#!/usr/bin/env python3
"""Compare raw causal Eye paths with frozen train MarketEpisode artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_neutral_b2_semantic_signal import _load_train_datasets  # noqa: E402
from scripts.run_eye_authority_scan import (  # noqa: E402
    _build_eye,
    _registered_payload,
)
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.market_representation import RepresentationDataError  # noqa: E402
from smc_trader.model import to_primitive  # noqa: E402


PROTOCOL_PATH = (
    ROOT / "configs" / "neutral_b2_eye_market_episode_transport_audit.json"
)
PROTOCOL_SHA256 = "0f5e441243e1327d45acebcfde2b2e99b9df2b26ca8e37755f589edfa419eda9"
TIMEFRAMES = ("4H", "1H", "15m", "5m", "1m")


@dataclass(frozen=True)
class PathSnapshot:
    source_ordinal: int
    replay_update_ordinal: int
    asof: pd.Timestamp
    epoch: int
    path: Mapping[str, Any]

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(str(item["kind"]) for item in self.path["steps"])

    @property
    def step_ids(self) -> tuple[str, ...]:
        return tuple(str(item["step_id"]) for item in self.path["steps"])


@dataclass(frozen=True)
class TrajectoryAnchor:
    surface: str
    profile: str
    path_id: str
    context_id: str
    epoch: int | str
    anchor_source_ordinal: int
    anchor_clock: str
    anchor_roles: tuple[str, ...]
    horizon_step_ids: tuple[str, ...]
    horizon_step_kinds: tuple[str, ...]
    trajectory: str | None
    right_censored: bool
    market_episode_id: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _protocol() -> Mapping[str, Any]:
    if _sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RepresentationDataError("Eye transport audit protocol changed")
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    population = value.get("population")
    runtime = value.get("raw_eye_runtime")
    if (
        value.get("protocol_version")
        != "neutral-b2-eye-market-episode-transport-audit-1.0.0"
        or not isinstance(population, Mapping)
        or population.get("split_role") != "train"
        or population.get("validation_opened") is not False
        or population.get("holdout_opened") is not False
        or population.get("outcome_fields_used") is not False
        or not isinstance(runtime, Mapping)
        or any(
            runtime.get(key) is not False
            for key in (
                "brain_used",
                "decision_used",
                "risk_used",
                "execution_used",
                "outcome_used",
                "scene_graph_used",
                "market_episode_used",
            )
        )
    ):
        raise RepresentationDataError("Eye transport audit isolation changed")
    for relative, expected in runtime["runtime_file_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise RepresentationDataError(f"raw Eye runtime changed: {relative}")
    if _sha256(ROOT / "configs" / "model.json") != runtime["model_sha256"]:
        raise RepresentationDataError("raw Eye model configuration changed")
    protocol_files = {
        "group12": "configs/primitives_structure_liquidity.json",
        "displacement": "configs/primitives_displacement.json",
        "group3": "configs/primitives_zones.json",
        "group4": "configs/primitives_range.json",
        "group5": "configs/primitives_entry.json",
    }
    if any(
        _sha256(ROOT / relative) != runtime["protocol_sha256"][name]
        for name, relative in protocol_files.items()
    ):
        raise RepresentationDataError("raw Eye primitive protocol changed")
    return value


def _steps(path: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = path.get("steps")
    if not isinstance(raw, list) or not raw or any(
        not isinstance(item, Mapping) for item in raw
    ):
        raise RepresentationDataError("Eye path steps changed")
    output = tuple(raw)
    kinds = tuple(item.get("kind") for item in output)
    ids = tuple(item.get("step_id") for item in output)
    if (
        any(not isinstance(value, str) or not value for value in (*kinds, *ids))
        or len(ids) != len(set(ids))
    ):
        raise RepresentationDataError("Eye path step identity changed")
    return output


def _validate_path_prefix(
    previous: PathSnapshot | None, current: PathSnapshot
) -> None:
    _steps(current.path)
    if previous is None:
        return
    if (
        current.path.get("sequence_id") != previous.path.get("sequence_id")
        or current.path.get("context_id") != previous.path.get("context_id")
        or current.path.get("context_kind") != previous.path.get("context_kind")
        or current.path.get("direction") != previous.path.get("direction")
        or current.step_ids[: len(previous.step_ids)] != previous.step_ids
    ):
        raise RepresentationDataError("Eye path history is not monotone")


def _anchor_index(
    history: Sequence[PathSnapshot], contract: Mapping[str, Any], start: pd.Timestamp,
    end: pd.Timestamp,
) -> int | None:
    required = tuple(contract["required_path_prefix"])
    excluded = set(contract["anchor_excludes_steps"])
    for index, snapshot in enumerate(history):
        if snapshot.path.get("context_kind") != contract["context_kind"]:
            continue
        roles = snapshot.roles
        first_pullback = next(
            (
                pd.Timestamp(step["observed_at"])
                for step in snapshot.path["steps"]
                if step.get("kind") == "first_pullback"
            ),
            None,
        )
        if (
            all(role in roles for role in required)
            and snapshot.path.get("lifecycle") == "active"
            and not excluded.intersection(roles)
            and first_pullback is not None
            and start <= first_pullback < end
        ):
            return index
    return None


def _trajectory(
    history: Sequence[PathSnapshot], anchor_index: int, *, horizon: int,
    epoch_end: int, contract: Mapping[str, Any],
) -> tuple[str | None, bool, tuple[str, ...], tuple[str, ...]]:
    anchor = history[anchor_index]
    anchor_roles = anchor.roles
    latest = anchor
    for snapshot in history[anchor_index + 1 :]:
        if snapshot.source_ordinal > anchor.source_ordinal + horizon:
            break
        if snapshot.step_ids[: len(anchor.step_ids)] != anchor.step_ids:
            raise RepresentationDataError("Eye path changed before trajectory horizon")
        latest = snapshot
    added = latest.roles[len(anchor_roles) :]
    confirm = set(contract["confirmed_continuation_first_roles"])
    failed = set(contract["failed_or_opposed_first_roles"])
    for role in added:
        if role in confirm:
            return (
                "confirmed_continuation",
                False,
                latest.step_ids,
                latest.roles,
            )
        if role in failed:
            return "failed_or_opposed", False, latest.step_ids, latest.roles
    if anchor.source_ordinal + horizon >= epoch_end:
        return None, True, latest.step_ids, latest.roles
    returned = set(contract["returned_to_range_unconfirmed_requires_roles"])
    label = (
        "returned_to_range_unconfirmed"
        if returned.issubset(set(added))
        else "continued_unresolved"
    )
    return label, False, latest.step_ids, latest.roles


def _raw_anchors(
    profile: str, histories: Mapping[str, Sequence[PathSnapshot]],
    epoch_ends: Mapping[int, int], start: pd.Timestamp, end: pd.Timestamp,
    contract: Mapping[str, Any],
) -> tuple[TrajectoryAnchor, ...]:
    horizon = int(contract["horizon_completed_real_source_1m_bars"])
    output: list[TrajectoryAnchor] = []
    for path_id, history in sorted(histories.items()):
        index = _anchor_index(history, contract, start, end)
        if index is None:
            continue
        anchor = history[index]
        label, censored, step_ids, step_kinds = _trajectory(
            history,
            index,
            horizon=horizon,
            epoch_end=epoch_ends[anchor.epoch],
            contract=contract,
        )
        output.append(
            TrajectoryAnchor(
                surface="raw_eye",
                profile=profile,
                path_id=path_id,
                context_id=str(anchor.path["context_id"]),
                epoch=anchor.epoch,
                anchor_source_ordinal=anchor.source_ordinal,
                anchor_clock=str(
                    next(
                        step["observed_at"]
                        for step in anchor.path["steps"]
                        if step["kind"] == "first_pullback"
                    )
                ),
                anchor_roles=anchor.roles,
                horizon_step_ids=step_ids,
                horizon_step_kinds=step_kinds,
                trajectory=label,
                right_censored=censored,
            )
        )
    return tuple(output)


def _scan_raw_eye(
    dataset: Mapping[str, Any], protocol: Mapping[str, Any]
) -> tuple[tuple[TrajectoryAnchor, ...], Mapping[str, Any]]:
    run = dataset["run_manifest"]
    profile = str(dataset["profile_name"])
    source = Path(str(run["source"]["path"]))
    start = pd.Timestamp(run["window"]["start"])
    end = pd.Timestamp(run["window"]["end_exclusive"])
    first = pd.Timestamp(run["source"]["first"])
    loaded = load_ohlcv(source, start=first, end=end)
    # Match the market-input runner's observation-clock interval exactly: a
    # source bar belongs only when its completed-bar end is strictly before
    # the registered end_exclusive clock.
    frame = loaded.frame.loc[
        loaded.frame.index + pd.Timedelta(1, unit="min") < end
    ]
    if (
        len(frame) != run["source"]["rows"]
        or loaded.source.resolve() != source.resolve()
        or loaded.warnings
        or set(frame["symbol"].astype(str)) != {run["source"]["symbol"]}
        or set(frame["instrument_id"].astype(int))
        != {int(run["source"]["instrument_id"])}
    ):
        raise RepresentationDataError(f"raw Eye source differs for {profile}")
    reader, observer = _build_eye(_registered_payload())
    histories: dict[str, list[PathSnapshot]] = defaultdict(list)
    path_epoch: dict[str, int] = {}
    unique_steps: dict[str, str] = {}
    epoch = 0
    epoch_ends: dict[int, int] = {}
    source_ordinal = -1
    updates = 0
    reset_counts: Counter[str] = Counter()
    for bar in iter_completed_bars(frame, allow_data_gap_reset=True):
        if not bar.synthetic_no_trade:
            source_ordinal += 1
        observation = observer.observe(reader.on_bar(bar))
        reset = next(
            (
                value
                for value in observation.anomalies
                if value
                in {"data_gap_history_reset", "contract_change_history_reset"}
            ),
            None,
        )
        if reset is not None:
            epoch_ends[epoch] = source_ordinal
            epoch += 1
            reset_counts[reset] += 1
        for state in observation.group5_path_transitions_this_update:
            path = to_primitive(state)
            if not isinstance(path, Mapping):
                raise RepresentationDataError("raw Eye path serialization changed")
            path_id = str(path.get("sequence_id", ""))
            if not path_id:
                raise RepresentationDataError("raw Eye path identity is missing")
            assigned_epoch = path_epoch.setdefault(path_id, epoch)
            snapshot = PathSnapshot(
                source_ordinal=source_ordinal,
                replay_update_ordinal=updates,
                asof=observation.asof,
                epoch=assigned_epoch,
                path=path,
            )
            prior = histories[path_id][-1] if histories[path_id] else None
            _validate_path_prefix(prior, snapshot)
            histories[path_id].append(snapshot)
            for step in _steps(path):
                step_id = str(step["step_id"])
                kind = str(step["kind"])
                if step_id in unique_steps and unique_steps[step_id] != kind:
                    raise RepresentationDataError("raw Eye step identity was reused")
                unique_steps[step_id] = kind
        updates += 1
        if updates % 5000 == 0:
            print(
                json.dumps(
                    {
                        "profile": profile,
                        "raw_eye_updates": updates,
                        "source_rows": source_ordinal + 1,
                        "paths": len(histories),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if source_ordinal + 1 != int(run["source"]["rows"]):
        raise RepresentationDataError("raw Eye real-source count changed")
    epoch_ends[epoch] = source_ordinal + 1
    contract = protocol["trajectory_definition_under_test"]
    anchors = _raw_anchors(profile, histories, epoch_ends, start, end, contract)
    class_counts = Counter(
        anchor.trajectory for anchor in anchors if anchor.trajectory is not None
    )
    return anchors, {
        "profile": profile,
        "source_rows": source_ordinal + 1,
        "replay_updates": updates,
        "reset_counts": dict(sorted(reset_counts.items())),
        "raw_path_sequences": len(histories),
        "raw_zone_return_paths": sum(
            history[0].path.get("context_kind") == "zone_return"
            for history in histories.values()
        ),
        "raw_pool_reversal_paths": sum(
            history[0].path.get("context_kind") == "pool_reversal"
            for history in histories.values()
        ),
        "trajectory_anchors": len(anchors),
        "right_censored": sum(anchor.right_censored for anchor in anchors),
        "trajectory_class_counts": {
            value: class_counts[value] for value in contract["classes"]
        },
        "unique_step_kind_counts": dict(sorted(Counter(unique_steps.values()).items())),
    }


def _scan_raw_eye_worker(
    arguments: tuple[Mapping[str, Any], Mapping[str, Any]]
) -> tuple[tuple[TrajectoryAnchor, ...], Mapping[str, Any]]:
    return _scan_raw_eye(*arguments)


def _market_path(row: Mapping[str, Any]) -> Mapping[str, Any]:
    observation = json.loads(str(row["observation_transition_json"]))
    collections = observation.get("collections")
    paths = (
        None
        if not isinstance(collections, Mapping)
        else collections.get("group5_path_transitions_this_update")
    )
    if not isinstance(paths, list):
        raise RepresentationDataError("MarketEpisode Eye transport changed")
    matches = [
        item
        for item in paths
        if isinstance(item, Mapping)
        and item.get("sequence_id") == row["entry_path_id"]
    ]
    if not matches:
        raise RepresentationDataError("MarketEpisode entry path was not transported")
    selected = max(
        matches,
        key=lambda item: (len(item.get("steps", ())), str(item.get("last_updated_at"))),
    )
    _steps(selected)
    return selected


def _market_epoch_ends(
    rows: Sequence[Mapping[str, Any]], run: Mapping[str, Any]
) -> Mapping[str, int]:
    starts: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        prefixes = json.loads(str(row["ohlcv_prefix_refs_json"]))
        one = [item for item in prefixes if item.get("timeframe") == "1m"]
        if (
            not isinstance(prefixes, list)
            or len(one) != 1
            or type(one[0].get("replay_view_1m_row_start")) is not int
        ):
            raise RepresentationDataError("MarketEpisode 1m prefix changed")
        starts[str(row["market_epoch_id"])].add(
            int(one[0]["replay_view_1m_row_start"])
        )
    exact = {key: next(iter(value)) for key, value in starts.items() if len(value) == 1}
    if len(exact) != len(starts):
        raise RepresentationDataError("MarketEpisode epoch start is ambiguous")
    ordered = sorted(exact.items(), key=lambda item: item[1])
    end = int(run["source"]["rows"])
    return {
        key: ordered[index + 1][1] if index + 1 < len(ordered) else end
        for index, (key, _) in enumerate(ordered)
    }


def _market_anchors(
    dataset: Mapping[str, Any], protocol: Mapping[str, Any]
) -> tuple[tuple[TrajectoryAnchor, ...], Mapping[str, list[PathSnapshot]]]:
    profile = str(dataset["profile_name"])
    run = dataset["run_manifest"]
    start = pd.Timestamp(run["window"]["start"])
    end = pd.Timestamp(run["window"]["end_exclusive"])
    contract = protocol["trajectory_definition_under_test"]
    horizon = int(contract["horizon_completed_real_source_1m_bars"])
    epoch_ends = _market_epoch_ends(dataset["rows"], run)
    episodes: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_path: dict[str, list[PathSnapshot]] = defaultdict(list)
    for row in dataset["rows"]:
        episodes[(str(row["market_epoch_id"]), str(row["market_episode_id"]))].append(row)
    output: list[TrajectoryAnchor] = []
    for (epoch, episode), rows in sorted(episodes.items()):
        rows.sort(key=lambda row: (int(row["revision_index"]), int(row["source_replay_ordinal"])))
        if [int(row["revision_index"]) for row in rows] != list(range(len(rows))):
            raise RepresentationDataError("MarketEpisode revision sequence changed")
        history: list[PathSnapshot] = []
        for row in rows:
            path = _market_path(row)
            snapshot = PathSnapshot(
                source_ordinal=int(row["source_replay_ordinal"]),
                replay_update_ordinal=int(row["replay_update_ordinal"]),
                asof=pd.Timestamp(row["asof"]),
                epoch=0,
                path=path,
            )
            _validate_path_prefix(history[-1] if history else None, snapshot)
            history.append(snapshot)
            by_path[str(path["sequence_id"])].append(snapshot)
        index = _anchor_index(history, contract, start, end)
        if index is None:
            continue
        anchor = history[index]
        label, censored, step_ids, step_kinds = _trajectory(
            history,
            index,
            horizon=horizon,
            epoch_end=epoch_ends[epoch],
            contract=contract,
        )
        output.append(
            TrajectoryAnchor(
                surface="market_episode",
                profile=profile,
                path_id=str(anchor.path["sequence_id"]),
                context_id=str(anchor.path["context_id"]),
                epoch=epoch,
                anchor_source_ordinal=anchor.source_ordinal,
                anchor_clock=str(
                    next(
                        step["observed_at"]
                        for step in anchor.path["steps"]
                        if step["kind"] == "first_pullback"
                    )
                ),
                anchor_roles=anchor.roles,
                horizon_step_ids=step_ids,
                horizon_step_kinds=step_kinds,
                trajectory=label,
                right_censored=censored,
                market_episode_id=episode,
            )
        )
    for values in by_path.values():
        values.sort(key=lambda value: (value.source_ordinal, value.replay_update_ordinal))
    return tuple(output), by_path


def _scale_diagnostics(
    datasets: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    detail_counts: Counter[tuple[str, str]] = Counter()
    aggregation_patterns: Counter[tuple[str, str, str]] = Counter()
    rows: list[Mapping[str, Any]] = []
    required = {
        "relation", "graph_connected", "ambiguous", "evidence_ids",
        "evidence_kind", "authority_layer_id", "direction",
    }
    for dataset in datasets:
        profile = str(dataset["profile_name"])
        for row in dataset["rows"]:
            context = json.loads(str(row["neutral_global_context_json"]))
            details = context.get("scale_relation_details")
            unknown = context.get("unknown_evidence")
            if not isinstance(details, Mapping) or not isinstance(unknown, list):
                raise RepresentationDataError("MarketEpisode scale context changed")
            relations: dict[str, str] = {}
            for timeframe in TIMEFRAMES:
                detail = details.get(timeframe, details.get(timeframe.lower()))
                base = {
                    "profile": profile,
                    "market_epoch_id": row["market_epoch_id"],
                    "market_episode_id": row["market_episode_id"],
                    "revision_id": row["revision_id"],
                    "decision_at": str(row["asof"]),
                    "timeframe": timeframe,
                }
                if not isinstance(detail, Mapping) or required - set(detail):
                    detail_counts[(timeframe, "field_missing")] += 1
                    rows.append({**base, "reason_category": "field_missing"})
                    relations[timeframe] = "unknown"
                    continue
                relation = str(detail["relation"])
                relations[timeframe] = relation
                if relation != "unknown":
                    detail_counts[(timeframe, "known")] += 1
                    continue
                tags = [
                    value.split(":", 2)[2]
                    for value in unknown
                    if isinstance(value, str)
                    and value.startswith(f"scale:{timeframe}:")
                ]
                if detail["ambiguous"] is True:
                    category = "genuinely_uncertain_ambiguous"
                elif "graph_disconnected" in tags:
                    category = "graph_unconnected"
                elif set(tags).intersection(
                    {"not_ready", "authority_unresolved", "no_structural_evidence"}
                ):
                    category = "genuinely_uncertain"
                else:
                    category = "field_missing"
                detail_counts[(timeframe, category)] += 1
                rows.append(
                    {
                        **base,
                        "reason_category": category,
                        "reason_codes": sorted(tags),
                        "graph_connected": detail["graph_connected"],
                        "ambiguous": detail["ambiguous"],
                        "evidence_kind": detail["evidence_kind"],
                        "evidence_count": len(detail["evidence_ids"]),
                    }
                )
            local = tuple(relations[value] for value in ("15m", "5m", "1m"))
            known = [value for value in local if value != "unknown"]
            aggregate = (
                "material_opposition"
                if "material_opposition" in known
                else "aligned"
                if known.count("aligned") >= 2
                else "normal_pullback"
                if known.count("normal_pullback") >= 2
                else "unknown"
            )
            if aggregate == "unknown" and known:
                aggregation_patterns[local] += 1
                rows.append(
                    {
                        "profile": profile,
                        "market_epoch_id": row["market_epoch_id"],
                        "market_episode_id": row["market_episode_id"],
                        "revision_id": row["revision_id"],
                        "decision_at": str(row["asof"]),
                        "timeframe": "15m+5m+1m",
                        "reason_category": "audit_aggregation_loss",
                        "input_relations": list(local),
                    }
                )
    return {
        "detail_counts_by_timeframe_and_reason": [
            {"timeframe": key[0], "reason": key[1], "rows": count}
            for key, count in sorted(detail_counts.items())
        ],
        "unknown_detail_rows": sum(
            count for (timeframe, reason), count in detail_counts.items()
            if reason != "known"
        ),
        "field_missing_rows": sum(
            count for (_, reason), count in detail_counts.items()
            if reason == "field_missing"
        ),
        "graph_unconnected_rows": sum(
            count for (_, reason), count in detail_counts.items()
            if reason == "graph_unconnected"
        ),
        "genuinely_uncertain_rows": sum(
            count for (_, reason), count in detail_counts.items()
            if reason.startswith("genuinely_uncertain")
        ),
        "audit_aggregation_loss_rows": sum(aggregation_patterns.values()),
        "audit_aggregation_loss_patterns": [
            {"relations_15m_5m_1m": list(key), "rows": count}
            for key, count in aggregation_patterns.most_common()
        ],
    }, tuple(rows)


def _counts(anchors: Sequence[TrajectoryAnchor], classes: Sequence[str]) -> Mapping[str, int]:
    values = Counter(anchor.trajectory for anchor in anchors if anchor.trajectory)
    return {value: values[value] for value in classes}


def _transport(
    raw: Sequence[TrajectoryAnchor], market: Sequence[TrajectoryAnchor],
    market_histories: Mapping[tuple[str, str], Sequence[PathSnapshot]], horizon: int,
    classes: Sequence[str],
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    market_by_path: dict[tuple[str, str], list[TrajectoryAnchor]] = defaultdict(list)
    for anchor in market:
        market_by_path[(anchor.profile, anchor.path_id)].append(anchor)
    raw_by_path = {(anchor.profile, anchor.path_id): anchor for anchor in raw}
    details: list[Mapping[str, Any]] = []
    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    linked = evaluable = complete = comparable = matches = 0
    multiple = 0
    for anchor in raw:
        identity = (anchor.profile, anchor.path_id)
        candidates = market_by_path.get(identity, [])
        if len(candidates) > 1:
            multiple += 1
        selected = min(
            candidates,
            key=lambda item: (
                abs(item.anchor_source_ordinal - anchor.anchor_source_ordinal),
                str(item.market_episode_id),
            ),
            default=None,
        )
        linked_now = selected is not None
        linked += linked_now
        me_steps = {
            step_id
            for snapshot in market_histories.get(identity, ())
            if snapshot.source_ordinal <= anchor.anchor_source_ordinal + horizon
            for step_id in snapshot.step_ids
        }
        missing_steps = tuple(
            step_id for step_id in anchor.horizon_step_ids if step_id not in me_steps
        )
        evaluable_now = not anchor.right_censored
        evaluable += evaluable_now
        complete_now = evaluable_now and linked_now and not missing_steps
        complete += complete_now
        comparable_now = (
            selected is not None
            and not anchor.right_censored
            and not selected.right_censored
            and anchor.trajectory is not None
            and selected.trajectory is not None
        )
        comparable += comparable_now
        match_now = comparable_now and selected.trajectory == anchor.trajectory
        matches += match_now
        if anchor.trajectory is not None:
            counter = per_class[anchor.trajectory]
            counter["raw"] += 1
            counter["linked"] += linked_now
            counter["step_complete"] += complete_now
            counter["trajectory_match"] += match_now
        details.append(
            {
                "surface": "raw_eye_anchor",
                "profile": anchor.profile,
                "path_id": anchor.path_id,
                "context_id": anchor.context_id,
                "raw_epoch": anchor.epoch,
                "raw_anchor_source_ordinal": anchor.anchor_source_ordinal,
                "raw_anchor_clock": anchor.anchor_clock,
                "raw_trajectory": anchor.trajectory,
                "raw_right_censored": anchor.right_censored,
                "market_episode_ids": [
                    value.market_episode_id for value in candidates
                ],
                "market_episode_anchor_source_ordinal": (
                    None if selected is None else selected.anchor_source_ordinal
                ),
                "market_episode_anchor_clock": (
                    None if selected is None else selected.anchor_clock
                ),
                "market_episode_trajectory": (
                    None if selected is None else selected.trajectory
                ),
                "path_linked": linked_now,
                "steps_evaluable_by_horizon": evaluable_now,
                "steps_complete_by_horizon": complete_now,
                "missing_raw_step_ids": list(missing_steps),
                "missing_raw_step_kinds": [
                    kind
                    for step_id, kind in zip(
                        anchor.horizon_step_ids, anchor.horizon_step_kinds
                    )
                    if step_id in missing_steps
                ],
                "trajectory_comparable": comparable_now,
                "trajectory_match": match_now,
            }
        )
    market_only = [
        anchor
        for anchor in market
        if (anchor.profile, anchor.path_id) not in raw_by_path
    ]
    details.extend(
        {
            "surface": "market_episode_only_anchor",
            "profile": anchor.profile,
            "path_id": anchor.path_id,
            "market_episode_id": anchor.market_episode_id,
            "market_episode_trajectory": anchor.trajectory,
        }
        for anchor in market_only
    )
    return {
        "raw_anchors": len(raw),
        "market_episode_anchors": len(market),
        "raw_path_linked": linked,
        "raw_path_link_rate": 0.0 if not raw else linked / len(raw),
        "raw_steps_evaluable_by_horizon": evaluable,
        "raw_steps_complete_by_horizon": complete,
        "raw_step_complete_rate": 0.0 if not evaluable else complete / evaluable,
        "trajectory_comparable": comparable,
        "trajectory_matches": matches,
        "trajectory_match_rate": 0.0 if not comparable else matches / comparable,
        "raw_paths_with_multiple_market_episodes": multiple,
        "market_episode_only_anchors": len(market_only),
        "per_raw_class": {
            value: dict(per_class[value]) for value in classes
        },
    }, tuple(details)


def _verdict(
    raw_counts: Mapping[str, int], market_counts: Mapping[str, int],
    transport: Mapping[str, Any], raw_step_counts: Mapping[str, int],
    classes: Sequence[str],
) -> Mapping[str, Any]:
    raw_missing = [value for value in classes if raw_counts[value] == 0]
    market_missing = [value for value in classes if market_counts[value] == 0]
    connection_gap = bool(
        any(raw_counts[value] > 0 and market_counts[value] == 0 for value in classes)
        or transport["raw_path_linked"] != transport["raw_anchors"]
        or transport["raw_steps_complete_by_horizon"]
        != transport["raw_steps_evaluable_by_horizon"]
        or transport["market_episode_only_anchors"]
        or transport["trajectory_matches"] != transport["trajectory_comparable"]
    )
    required_for_missing = {
        "returned_to_range_unconfirmed": ("reference_reclaimed", "reacceptance_held"),
        "continued_unresolved": (),
        "confirmed_continuation": ("micro_bos_confirmed",),
        "failed_or_opposed": (
            "location_left", "micro_bos_opposed", "reacceptance_failed"
        ),
    }
    absent_roles = {
        label: [role for role in required_for_missing[label] if raw_step_counts.get(role, 0) == 0]
        for label in raw_missing
    }
    if raw_missing:
        primary = "trajectory_definition_or_eye_expression_gap"
    elif connection_gap:
        primary = "eye_to_market_episode_connection_gap"
    else:
        primary = "raw_eye_and_transport_complete"
    return {
        "primary_localization": primary,
        "raw_eye_missing_classes": raw_missing,
        "market_episode_missing_classes": market_missing,
        "eye_to_market_episode_connection_gap": connection_gap,
        "required_roles_absent_for_missing_raw_classes": absent_roles,
        "eye_semantic_sufficiency_resolved": False,
        "eye_semantic_sufficiency_note": (
            "Absence of a raw event is only a candidate Eye semantic gap until blinded price-structure review establishes that the event should have been emitted."
        ),
        "b2_model_in_scope": primary == "raw_eye_and_transport_complete",
        "b2_training_authorized": False,
    }


def _transport_summary_from_rows(
    rows: Sequence[Mapping[str, Any]], market_episode_anchors: int,
    classes: Sequence[str],
) -> Mapping[str, Any]:
    raw = [row for row in rows if row.get("surface") == "raw_eye_anchor"]
    market_only = [
        row for row in rows if row.get("surface") == "market_episode_only_anchor"
    ]
    evaluable = [row for row in raw if row.get("raw_right_censored") is False]
    comparable = [row for row in raw if row.get("trajectory_comparable") is True]
    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    for row in raw:
        label = row.get("raw_trajectory")
        if label is None:
            continue
        if label not in classes:
            raise RepresentationDataError("saved raw trajectory class changed")
        counter = per_class[str(label)]
        counter["raw"] += 1
        counter["linked"] += row.get("path_linked") is True
        counter["step_complete"] += row.get("steps_complete_by_horizon") is True
        counter["trajectory_match"] += row.get("trajectory_match") is True
    linked = sum(row.get("path_linked") is True for row in raw)
    complete = sum(row.get("steps_complete_by_horizon") is True for row in evaluable)
    matches = sum(row.get("trajectory_match") is True for row in comparable)
    return {
        "raw_anchors": len(raw),
        "market_episode_anchors": market_episode_anchors,
        "raw_path_linked": linked,
        "raw_path_link_rate": 0.0 if not raw else linked / len(raw),
        "raw_steps_evaluable_by_horizon": len(evaluable),
        "raw_steps_complete_by_horizon": complete,
        "raw_step_complete_rate": 0.0 if not evaluable else complete / len(evaluable),
        "trajectory_comparable": len(comparable),
        "trajectory_matches": matches,
        "trajectory_match_rate": 0.0 if not comparable else matches / len(comparable),
        "raw_paths_with_multiple_market_episodes": sum(
            len(row.get("market_episode_ids", ())) > 1 for row in raw
        ),
        "market_episode_only_anchors": len(market_only),
        "per_raw_class": {value: dict(per_class[value]) for value in classes},
    }


def _publish(
    output: Path, report: Mapping[str, Any],
    transport_rows: Sequence[Mapping[str, Any]],
    scale_rows: Sequence[Mapping[str, Any]],
) -> None:
    payloads = {
        output: _canonical(report) + b"\n",
        output.with_suffix(".md"): _markdown(report).encode("utf-8"),
        output.with_name("trajectory_transport.jsonl"): b"".join(
            _canonical(row) + b"\n" for row in transport_rows
        ),
        output.with_name("scale_relation_diagnostics.jsonl"): b"".join(
            _canonical(row) + b"\n" for row in scale_rows
        ),
    }
    for path, payload in payloads.items():
        _atomic_write(path, payload)


def _markdown(report: Mapping[str, Any]) -> str:
    verdict = report["verdict"]
    raw = report["raw_eye"]["trajectory_class_counts"]
    market = report["market_episode"]["trajectory_class_counts"]
    transport = report["transport"]
    unknown = report["unknown_scale_relation_taxonomy"]
    return "\n".join(
        (
            "# Raw Eye → MarketEpisode layered transport audit",
            "",
            "## Technical summary",
            "",
            f"Primary localization: **{verdict['primary_localization']}**. B2 training remains unauthorized.",
            "",
            f"Raw Eye trajectory counts: `{raw}`.",
            f"MarketEpisode trajectory counts: `{market}`.",
            f"Raw path link rate: {transport['raw_path_link_rate']:.2%}; step-complete rate: {transport['raw_step_complete_rate']:.2%}; comparable trajectory match: {transport['trajectory_match_rate']:.2%}.",
            "",
            "## Scale-relation unknowns",
            "",
            f"Graph-unconnected scale details: {unknown['graph_unconnected_rows']}; genuinely uncertain: {unknown['genuinely_uncertain_rows']}; field missing: {unknown['field_missing_rows']}; local audit aggregation loss: {unknown['audit_aggregation_loss_rows']}.",
            "",
            "## Scope and interpretation",
            "",
            "The raw layer is a replay of CausalMarketReader → CausalObserver in Eye-authority mode. SceneGraph, MarketEpisode, Brain, Decision, Risk, execution, outcomes, validation, holdout and all B2 training are absent.",
            "",
            "The four-class rule is intentionally unchanged from the failed mechanism audit. This run diagnoses where that rule loses coverage; it does not redefine the mechanism or tune a model.",
            "",
            "Eye semantic sufficiency remains unresolved: a missing raw event becomes an Eye issue only after blinded structure review shows that Eye should have emitted it.",
            "",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args(argv)
    protocol = _protocol()
    datasets = _load_train_datasets(args.input_root, protocol)
    all_market: list[TrajectoryAnchor] = []
    all_market_histories: dict[tuple[str, str], list[PathSnapshot]] = defaultdict(list)
    raw_specs: list[Mapping[str, Any]] = []
    profile_rows: list[tuple[str, int]] = []
    for dataset in datasets:
        market, market_histories = _market_anchors(dataset, protocol)
        all_market.extend(market)
        profile = str(dataset["profile_name"])
        profile_rows.append((profile, len(dataset["rows"])))
        raw_specs.append(
            {
                "profile_name": profile,
                "run_manifest": dataset["run_manifest"],
            }
        )
        for path_id, values in market_histories.items():
            all_market_histories[(profile, path_id)].extend(values)
    scale, scale_rows = _scale_diagnostics(datasets, protocol)
    del datasets
    gc.collect()
    output = Path(args.output).resolve()
    classes = tuple(protocol["trajectory_definition_under_test"]["classes"])
    if args.finalize_existing:
        transport_path = output.with_name("trajectory_transport.jsonl")
        if not output.is_file() or not transport_path.is_file():
            raise RepresentationDataError("saved raw audit artifacts are missing")
        saved = json.loads(output.read_text(encoding="utf-8"))
        if (
            saved.get("protocol", {}).get("sha256") != PROTOCOL_SHA256
            or saved.get("population", {}).get("profiles")
            != [profile for profile, _ in profile_rows]
            or saved.get("market_episode", {}).get("trajectory_anchors")
            != len(all_market)
            or saved.get("market_episode", {}).get("trajectory_class_counts")
            != _counts(all_market, classes)
        ):
            raise RepresentationDataError("saved raw audit report binding changed")
        transport_rows = tuple(
            json.loads(line)
            for line in transport_path.read_text(encoding="utf-8").splitlines()
            if line
        )
        transport = _transport_summary_from_rows(
            transport_rows, len(all_market), classes
        )
        if transport["raw_anchors"] != saved.get("raw_eye", {}).get(
            "trajectory_anchors"
        ):
            raise RepresentationDataError("saved raw audit row count changed")
        prior_implementation = saved.get("analysis_implementation")
        saved["raw_scan_implementation"] = prior_implementation
        saved["analysis_implementation"] = {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        }
        saved["transport"] = transport
        saved["unknown_scale_relation_taxonomy"] = scale
        saved["verdict"] = _verdict(
            saved["raw_eye"]["trajectory_class_counts"],
            saved["market_episode"]["trajectory_class_counts"],
            transport,
            saved["raw_eye"]["unique_step_kind_counts"],
            classes,
        )
        saved["finalization"] = {
            "mode": "reuse_saved_raw_transport_after_publication_failure",
            "raw_eye_replayed": False,
            "scale_diagnostics_recomputed": True,
        }
        _publish(output, saved, transport_rows, scale_rows)
        print(json.dumps(saved["verdict"], sort_keys=True), flush=True)
        return 0
    raw_arguments = [(spec, protocol) for spec in raw_specs]
    if args.workers == 1:
        raw_results = [_scan_raw_eye_worker(value) for value in raw_arguments]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            raw_results = list(executor.map(_scan_raw_eye_worker, raw_arguments))
    all_raw: list[TrajectoryAnchor] = []
    raw_windows: list[Mapping[str, Any]] = []
    raw_step_counts: Counter[str] = Counter()
    for raw, window in raw_results:
        all_raw.extend(raw)
        raw_windows.append(window)
        raw_step_counts.update(window["unique_step_kind_counts"])
    raw_counts = _counts(all_raw, classes)
    market_counts = _counts(all_market, classes)
    transport, transport_rows = _transport(
        all_raw,
        all_market,
        all_market_histories,
        int(protocol["trajectory_definition_under_test"]["horizon_completed_real_source_1m_bars"]),
        classes,
    )
    report = {
        "schema": "neutral-b2-eye-market-episode-transport-audit-report-1.0.0",
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
        "population": {
            "profiles": [profile for profile, _ in profile_rows],
            "market_episode_rows": sum(rows for _, rows in profile_rows),
            "validation_opened": False,
            "holdout_opened": False,
            "outcome_fields_used": False,
        },
        "raw_eye": {
            "windows": raw_windows,
            "trajectory_anchors": len(all_raw),
            "trajectory_class_counts": raw_counts,
            "right_censored": sum(value.right_censored for value in all_raw),
            "unique_step_kind_counts": dict(sorted(raw_step_counts.items())),
        },
        "market_episode": {
            "trajectory_anchors": len(all_market),
            "trajectory_class_counts": market_counts,
            "right_censored": sum(value.right_censored for value in all_market),
        },
        "transport": transport,
        "unknown_scale_relation_taxonomy": scale,
        "verdict": _verdict(
            raw_counts,
            market_counts,
            transport,
            raw_step_counts,
            classes,
        ),
    }
    _publish(output, report, transport_rows, scale_rows)
    print(json.dumps(report["verdict"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
