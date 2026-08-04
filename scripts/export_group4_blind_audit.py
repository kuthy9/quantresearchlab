#!/usr/bin/env python3
"""Export a small outcome-blind Group 4 primitive audit batch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.artifact_stream import atomic_bytes, sha256_file  # noqa: E402
from smc_trader.calibration import model_code_fingerprint  # noqa: E402
from smc_trader.causal import CausalMarketReader  # noqa: E402
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.model import (  # noqa: E402
    CORE_TIMEFRAMES,
    Timeframe,
    content_hash,
    to_primitive,
)
from smc_trader.semantic_audit import SemanticCaseVisualizer  # noqa: E402


STRATA = (
    "range_forming",
    "range_broken",
    "manipulation_swept_above",
    "manipulation_swept_below",
    "manipulation_reaccepted_above",
    "manipulation_reaccepted_below",
    "manipulation_accepted_outside_above",
    "manipulation_accepted_outside_below",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(
            to_primitive(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    atomic_bytes(
        path,
        encoded,
    )


def _selection_key(
    protocol_hash: str,
    stratum: str,
    entity_id: str,
    focus_clock: pd.Timestamp,
) -> str:
    raw = (
        f"{protocol_hash}|{stratum}|{entity_id}|"
        f"{focus_clock.isoformat()}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _candidate(
    *,
    protocol_hash: str,
    stratum: str,
    entity_kind: str,
    entity_id: str,
    lifecycle: str,
    focus_clock: pd.Timestamp,
    state: dict[str, Any],
) -> dict[str, Any]:
    key = _selection_key(
        protocol_hash,
        stratum,
        entity_id,
        focus_clock,
    )
    opaque = hashlib.sha256(
        f"group4-r3-blind|{key}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "opaque_case_id": opaque,
        "stratum": stratum,
        "case_clock": focus_clock,
        "entity_kind": entity_kind,
        "entity_id": entity_id,
        "expected_lifecycle": lifecycle,
        "selection_key": key,
        "target_state_sha256": content_hash(state),
        "target_state": state,
    }


def _exact_candidates(
    replay_root: Path,
    protocol_hash: str,
) -> tuple[dict[str, Any], ...]:
    candidates: dict[str, dict[str, Any]] = {}
    manifest = _json(replay_root / "decision_shards.manifest.json")
    for shard in manifest["shards"]:
        shard_path = replay_root / shard["path"]
        if sha256_file(shard_path) != shard["sha256"]:
            raise RuntimeError(f"decision shard SHA differs: {shard_path}")
        table = pq.read_table(
            shard_path,
            columns=["asof", "group4_state"],
        )
        for asof_raw, state_raw in zip(
            table["asof"].to_pylist(),
            table["group4_state"].to_pylist(),
        ):
            asof = pd.Timestamp(asof_raw)
            state = json.loads(state_raw)
            for item in state.get("dealing_ranges", ()):
                lifecycle = str(item["lifecycle"])
                clock_name = (
                    "formed_at"
                    if lifecycle == "forming"
                    else "broken_at"
                    if lifecycle == "broken"
                    else None
                )
                if clock_name is None:
                    continue
                focus_clock = pd.Timestamp(item[clock_name])
                if focus_clock != asof:
                    continue
                stratum = f"range_{lifecycle}"
                candidate = _candidate(
                    protocol_hash=protocol_hash,
                    stratum=stratum,
                    entity_kind="range",
                    entity_id=str(item["range_id"]),
                    lifecycle=lifecycle,
                    focus_clock=focus_clock,
                    state=item,
                )
                prior = candidates.get(stratum)
                if (
                    prior is None
                    or candidate["selection_key"] < prior["selection_key"]
                ):
                    candidates[stratum] = candidate
            for item in state.get("manipulations", ()):
                lifecycle = str(item["lifecycle"])
                if lifecycle not in {
                    "swept",
                    "reaccepted",
                    "accepted_outside",
                }:
                    continue
                clock_name = (
                    "swept_at"
                    if lifecycle == "swept"
                    else "resolved_at"
                )
                focus_clock = pd.Timestamp(item[clock_name])
                if focus_clock != asof:
                    continue
                side = str(item["side"])
                stratum = f"manipulation_{lifecycle}_{side}"
                candidate = _candidate(
                    protocol_hash=protocol_hash,
                    stratum=stratum,
                    entity_kind="manipulation",
                    entity_id=str(item["manipulation_id"]),
                    lifecycle=lifecycle,
                    focus_clock=focus_clock,
                    state=item,
                )
                prior = candidates.get(stratum)
                if (
                    prior is None
                    or candidate["selection_key"] < prior["selection_key"]
                ):
                    candidates[stratum] = candidate
    missing = tuple(stratum for stratum in STRATA if stratum not in candidates)
    if missing:
        raise RuntimeError(f"Group 4 blind strata missing: {missing}")
    selected = tuple(candidates[stratum] for stratum in STRATA)
    opaque_ids = tuple(item["opaque_case_id"] for item in selected)
    if len(set(opaque_ids)) != len(opaque_ids):
        raise RuntimeError("Group 4 opaque case identity collision")
    return selected


def _validate_reviews(
    path: Path,
    cases: tuple[dict[str, Any], ...],
    output: Path,
    authority_output: Path,
) -> dict[str, Any]:
    reviews = _json(path)
    if (
        reviews.get("future_visible") is not False
        or reviews.get("model_overlay_visible") is not False
    ):
        raise ValueError("review file does not attest a blind first pass")
    expected = {case["opaque_case_id"] for case in cases}
    reviewed_ids = tuple(
        str(item.get("opaque_case_id"))
        for item in reviews.get("cases", ())
    )
    if (
        len(reviewed_ids) != len(expected)
        or len(set(reviewed_ids)) != len(reviewed_ids)
        or set(reviewed_ids) != expected
    ):
        raise ValueError("review case identities differ from selection")
    if str(reviews.get("reviewer_id", "")).strip() in {"", "pending"}:
        raise ValueError("blind reviews lack a completed reviewer identity")
    for item in reviews["cases"]:
        if type(item.get("chart_sufficient")) is not bool:
            raise ValueError("blind review lacks chart sufficiency judgment")
        if not item.get("observations") and not item.get("issues"):
            raise ValueError("blind review lacks a frozen observation")
    authority_path = authority_output / "selection.json"
    blind_manifest_path = output / "blind" / "manifest.json"
    if not authority_path.is_file() or not blind_manifest_path.is_file():
        raise ValueError("reveal requires the completed blind package")
    if (
        reviews.get("selection_authority_sha256")
        != sha256_file(authority_path)
        or reviews.get("blind_manifest_sha256")
        != sha256_file(blind_manifest_path)
    ):
        raise ValueError("blind review package binding differs")
    manifest = _json(blind_manifest_path)
    image_rows = tuple(
        sorted(
            (
                str(item["opaque_case_id"]),
                str(item["image_sha256"]),
            )
            for item in manifest.get("cases", ())
        )
    )
    if {item[0] for item in image_rows} != expected:
        raise ValueError("blind manifest cases differ from selection")
    for item in manifest["cases"]:
        if sha256_file(item["image"]) != item["image_sha256"]:
            raise ValueError("blind image SHA differs from manifest")
    image_commitment = content_hash(image_rows)
    if reviews.get("blind_image_commitment") != image_commitment:
        raise ValueError("blind review image commitment differs")
    return reviews


def _bindings(
    replay_root: Path,
    source: Path,
    config: Path,
    completed: dict[str, Any],
) -> dict[str, Any]:
    registered = completed["bindings"]
    decision_manifest = replay_root / "decision_shards.manifest.json"
    if (
        sha256_file(decision_manifest)
        != completed["stream_manifest_sha256"]["decision_shards"]
    ):
        raise RuntimeError("decision manifest SHA differs from completion")
    if sha256_file(source) != registered["source_sha256"]:
        raise RuntimeError("source SHA differs from completed replay")
    if sha256_file(config) != registered["config_sha256"]:
        raise RuntimeError("config SHA differs from completed replay")
    export_code_hash = model_code_fingerprint()
    if export_code_hash != registered["model_code_hash"]:
        raise RuntimeError("model code SHA differs from completed replay")
    protocol_path = Path(
        _json(config)["observer"]["group4_protocol"]
    )
    if sha256_file(protocol_path) != registered["group4_protocol_sha256"]:
        raise RuntimeError("Group 4 protocol SHA differs from replay")
    return {
        "source": str(source),
        "source_sha256": registered["source_sha256"],
        "config": str(config),
        "config_sha256": registered["config_sha256"],
        "model_code_hash_at_replay": registered["model_code_hash"],
        "model_code_hash_at_export": export_code_hash,
        "group4_protocol_sha256": registered["group4_protocol_sha256"],
        "completed_sha256": sha256_file(replay_root / "COMPLETED.json"),
        "decision_manifest_sha256": sha256_file(decision_manifest),
        "source_first": registered["source_first"],
        "visual_reconstruction": (
            "causal_completed-bar reader only; frozen typed target is read "
            "from the hash-verified decision shard"
        ),
        "target_state_recomputed": False,
        "selection_rule": (
            "minimum sha256(group4_protocol_sha256|stratum|"
            "entity_id|exact_event_clock)"
        ),
        "future_path_used": False,
        "pnl_used": False,
        "mbo_used": False,
    }


def _clock_index(candles: tuple[Any, ...], clock: Any) -> int | None:
    if clock is None:
        return None
    timestamp = pd.Timestamp(clock)
    return next(
        (
            index
            for index, candle in enumerate(candles)
            if candle.start < timestamp <= candle.end
        ),
        None,
    )


def _target_level(
    axis: Any,
    *,
    price: float,
    label: str,
    color: str,
    linestyle: str,
) -> None:
    lower, upper = axis.get_ylim()
    if lower <= price <= upper:
        axis.axhline(
            price,
            color=color,
            linestyle=linestyle,
            linewidth=0.9,
            alpha=0.95,
            zorder=5,
        )
        axis.text(
            0.995,
            price,
            f"{label} {price:.2f}",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color=color,
            fontsize=6,
        )
    else:
        axis.text(
            0.995,
            0.985 if price > upper else 0.015,
            f"{label} {price:.2f} ({'above' if price > upper else 'below'})",
            transform=axis.transAxes,
            ha="right",
            va="top" if price > upper else "bottom",
            color=color,
            fontsize=6,
        )


def _render_target_reveal(
    asof: pd.Timestamp,
    histories: dict[Timeframe, tuple[Any, ...]],
    destination: Path,
    *,
    case: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = SemanticCaseVisualizer._panels(
        histories,
        asof,
    )
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(15, 12),
        dpi=110,
        constrained_layout=True,
    )
    axis_checks: dict[str, dict[str, Any]] = {}
    for axis, timeframe in zip(axes, CORE_TIMEFRAMES):
        first_index, candles = panels[timeframe]
        SemanticCaseVisualizer._candles(
            axis,
            candles,
            first_history_index=first_index,
            tick_size=0.25,
        )
        before = tuple(float(value) for value in axis.get_ylim())
        if case["entity_kind"] == "range" and timeframe is Timeframe.H1:
            for label, price, style in (
                ("frozen lower", target["lower_bound"], "--"),
                ("frozen midpoint", target["midpoint"], ":"),
                ("frozen upper", target["upper_bound"], "--"),
            ):
                _target_level(
                    axis,
                    price=float(price),
                    label=label,
                    color="#a16207",
                    linestyle=style,
                )
            for label, clock, color in (
                ("formed", target["formed_at"], "#0369a1"),
                ("mature", target["mature_at"], "#15803d"),
                ("broken", target["broken_at"], "#b91c1c"),
            ):
                index = _clock_index(candles, clock)
                if index is not None:
                    axis.axvline(
                        index,
                        color=color,
                        linestyle="-.",
                        linewidth=0.8,
                    )
                    axis.text(
                        index,
                        0.98,
                        label,
                        transform=axis.get_xaxis_transform(),
                        ha="right",
                        va="top",
                        color=color,
                        fontsize=6,
                    )
        if (
            case["entity_kind"] == "manipulation"
            and timeframe is Timeframe.M1
        ):
            for label, price in (
                ("source lower", target["source_lower_bound"]),
                ("source upper", target["source_upper_bound"]),
            ):
                _target_level(
                    axis,
                    price=float(price),
                    label=label,
                    color="#7c3aed",
                    linestyle="--",
                )
            sweep_index = _clock_index(candles, target["swept_at"])
            if sweep_index is not None:
                axis.scatter(
                    [sweep_index],
                    [target["sweep_extreme"]],
                    marker="x",
                    s=42,
                    color="#c2410c",
                    linewidths=1.1,
                    zorder=7,
                )
                axis.axvline(
                    sweep_index,
                    color="#c2410c",
                    linestyle=":",
                    linewidth=0.8,
                )
            resolution_index = _clock_index(
                candles,
                target["resolved_at"],
            )
            if resolution_index is not None:
                axis.axvline(
                    resolution_index,
                    color="#15803d",
                    linestyle="-.",
                    linewidth=0.8,
                )
                axis.text(
                    resolution_index,
                    0.98,
                    target["lifecycle"],
                    transform=axis.get_xaxis_transform(),
                    ha="right",
                    va="top",
                    color="#15803d",
                    fontsize=6,
                )
        axis.set_ylim(*before)
        after = tuple(float(value) for value in axis.get_ylim())
        axis_checks[timeframe.value] = {
            "ylim_before": before,
            "ylim_after": after,
            "preserved": before == after,
        }
        axis.axvline(
            len(candles) - 0.5,
            color="#111827",
            linewidth=1.0,
        )
        axis.set_title(
            f"{timeframe.value} · completed through "
            f"{candles[-1].end:%Y-%m-%d %H:%M %Z}",
            loc="left",
            fontsize=9,
        )
    figure.suptitle(
        "GROUP 4 TARGET REVEAL — "
        f"{case['entity_kind']} / {target['lifecycle']} / "
        f"{case['entity_id']}",
        fontsize=11,
        weight="bold",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    maximum = max(
        candle.end
        for _, candles in panels.values()
        for candle in candles
    )
    if maximum > asof:
        raise AssertionError("target reveal contains a future candle")
    if not all(item["preserved"] for item in axis_checks.values()):
        raise AssertionError("target reveal changed a candle y-axis")
    return {
        "path": destination,
        "sha256": sha256_file(destination),
        "maximum_market_time": maximum,
        "axis_checks": axis_checks,
    }


def _render_raw_blind(
    asof: pd.Timestamp,
    histories: dict[Timeframe, tuple[Any, ...]],
    destination: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = SemanticCaseVisualizer._panels(histories, asof)
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(15, 12),
        dpi=110,
        constrained_layout=True,
    )
    for axis, timeframe in zip(axes, CORE_TIMEFRAMES):
        first_index, candles = panels[timeframe]
        SemanticCaseVisualizer._candles(
            axis,
            candles,
            first_history_index=first_index,
            tick_size=0.25,
        )
        axis.axvline(
            len(candles) - 0.5,
            color="#111827",
            linewidth=1.0,
        )
        axis.set_title(
            f"{timeframe.value} · raw completed candles through "
            f"{candles[-1].end:%Y-%m-%d %H:%M %Z}",
            loc="left",
            fontsize=9,
        )
    figure.suptitle(
        "BLIND STRUCTURE CASE — model labels and later market path absent",
        fontsize=13,
        weight="bold",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    maximum = max(
        candle.end
        for _, candles in panels.values()
        for candle in candles
    )
    if maximum > asof:
        raise AssertionError("blind image contains a future candle")
    return {
        "path": destination,
        "sha256": sha256_file(destination),
        "maximum_market_time": maximum,
    }


def _replay_and_render(
    *,
    phase: str,
    cases: tuple[dict[str, Any], ...],
    source: Path,
    source_first: pd.Timestamp,
    output: Path,
) -> tuple[dict[str, Any], ...]:
    selected: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for case in cases:
        selected.setdefault(
            pd.Timestamp(case["case_clock"]),
            [],
        ).append(case)
    final_clock = max(selected)
    loaded = load_ohlcv(
        source,
        start=source_first,
        end=final_clock,
    )
    if not loaded.contract_selection_causal:
        raise RuntimeError("blind audit requires causal contract selection")
    reader = CausalMarketReader()
    artifacts: list[dict[str, Any]] = []
    for bar in iter_completed_bars(loaded.frame):
        if bar.end > final_clock:
            break
        update = reader.on_bar(bar)
        for case in selected.get(update.asof, ()):
            target = case["target_state"]
            if content_hash(target) != case["target_state_sha256"]:
                raise RuntimeError("frozen target state hash changed")
            histories = {
                timeframe: reader.window(timeframe, 80)
                for timeframe in CORE_TIMEFRAMES
            }
            if any(
                not candle.complete
                or candle.end > update.asof
                for values in histories.values()
                for candle in values
            ):
                raise RuntimeError("render histories are not causally complete")
            if phase == "blind":
                artifact = _render_raw_blind(
                    update.asof,
                    histories,
                    output
                    / "blind"
                    / "cases"
                    / f"{case['opaque_case_id']}.png",
                )
                record = {
                    "opaque_case_id": case["opaque_case_id"],
                    "case_clock": case["case_clock"],
                    "image": str(artifact["path"]),
                    "image_sha256": artifact["sha256"],
                    "maximum_market_time": artifact[
                        "maximum_market_time"
                    ],
                    "future_visible": False,
                    "model_overlay_visible": False,
                }
            else:
                artifact = _render_target_reveal(
                    update.asof,
                    histories,
                    output
                    / "revealed"
                    / "cases"
                    / f"{case['opaque_case_id']}.png",
                    case=case,
                    target=target,
                )
                record = {
                    "opaque_case_id": case["opaque_case_id"],
                    "case_clock": case["case_clock"],
                    "image": str(artifact["path"]),
                    "image_sha256": artifact["sha256"],
                    "maximum_market_time": artifact[
                        "maximum_market_time"
                    ],
                    "axis_checks": artifact["axis_checks"],
                    "target_state": target,
                    "target_state_sha256": content_hash(target),
                    "future_visible": False,
                }
            artifacts.append(record)
    if len(artifacts) != len(cases):
        raise RuntimeError(
            f"rendered {len(artifacts)} of {len(cases)} selected cases"
        )
    return tuple(
        sorted(artifacts, key=lambda item: item["opaque_case_id"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("blind", "reveal"))
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--config",
        default="configs/model_v3_development.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--authority-output", required=True)
    parser.add_argument("--reviews")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_root = Path(args.replay_root)
    source = Path(args.source)
    config = Path(args.config)
    output = Path(args.output)
    authority_output = Path(args.authority_output)
    resolved_output = output.resolve()
    resolved_authority = authority_output.resolve()
    if (
        resolved_output == resolved_authority
        or resolved_output.is_relative_to(resolved_authority)
        or resolved_authority.is_relative_to(resolved_output)
    ):
        raise ValueError(
            "reviewer output and selection authority must not overlap"
        )
    if args.phase == "blind" and (output / "revealed").exists():
        raise ValueError(
            "blind output already contains a reveal; use a fresh reviewer root"
        )
    completed = _json(replay_root / "COMPLETED.json")
    if completed.get("status") != "complete":
        raise RuntimeError("Group 4 source replay is not complete")
    bindings = _bindings(
        replay_root,
        source,
        config,
        completed,
    )
    cases = _exact_candidates(
        replay_root,
        bindings["group4_protocol_sha256"],
    )
    if args.phase == "reveal":
        if not args.reviews:
            raise ValueError("reveal requires --reviews")
        reviews = _validate_reviews(
            Path(args.reviews),
            cases,
            output,
            authority_output,
        )
        review_binding = {
            "path": str(Path(args.reviews)),
            "sha256": sha256_file(args.reviews),
            "reviewer_id": reviews.get("reviewer_id"),
        }
    else:
        review_binding = None
    artifacts = _replay_and_render(
        phase=args.phase,
        cases=cases,
        source=source,
        source_first=pd.Timestamp(bindings["source_first"]),
        output=output,
    )
    authority = {
        "format_version": 1,
        "artifact": "group4_blind_selection_authority",
        "bindings": bindings,
        "registered_strata": STRATA,
        "missing_registered_strata": [],
        "known_uncovered_semantics": [
            "mature_dealing_range",
            "mature_range_boundary_inventory",
            "mature_range_sourced_manipulation",
        ],
        "cases": cases,
    }
    authority_path = authority_output / "selection.json"
    if args.phase == "blind":
        _write_json(authority_path, authority)
    else:
        if not authority_path.is_file():
            raise RuntimeError("reveal lacks blind selection authority")
        if content_hash(_json(authority_path)) != content_hash(authority):
            raise RuntimeError("reveal selection authority changed")
    manifest = {
        "format_version": 1,
        "artifact": f"group4_{args.phase}_audit",
        "phase": args.phase,
        "bindings": bindings,
        "review_binding": review_binding,
        "cases": artifacts,
    }
    _write_json(
        (
            output
            / ("blind" if args.phase == "blind" else "revealed")
            / "manifest.json"
        ),
        manifest,
    )
    if args.phase == "blind":
        blind_manifest_path = output / "blind" / "manifest.json"
        image_rows = tuple(
            sorted(
                (
                    str(item["opaque_case_id"]),
                    str(item["image_sha256"]),
                )
                for item in artifacts
            )
        )
        _write_json(
            output / "blind" / "review_template.json",
            {
                "format_version": 1,
                "artifact": "group4_blind_reviews",
                "reviewer_id": "pending",
                "future_visible": False,
                "model_overlay_visible": False,
                "selection_authority_sha256": sha256_file(
                    authority_path
                ),
                "blind_manifest_sha256": sha256_file(
                    blind_manifest_path
                ),
                "blind_image_commitment": content_hash(image_rows),
                "cases": [
                    {
                        "opaque_case_id": item["opaque_case_id"],
                        "chart_sufficient": None,
                        "observations": [],
                        "issues": [],
                    }
                    for item in artifacts
                ],
            },
        )
    print(
        json.dumps(
            {
                "phase": args.phase,
                "cases": len(artifacts),
                "output": (
                    str(output / "blind")
                    if args.phase == "blind"
                    else str(output / "revealed")
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
