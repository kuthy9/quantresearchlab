#!/usr/bin/env python3
"""Render outcome-blind raw candles for selected mature-range coverage cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.artifact_stream import atomic_bytes, sha256_file  # noqa: E402
from smc_trader.causal import CausalMarketReader  # noqa: E402
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.model import CORE_TIMEFRAMES, Timeframe  # noqa: E402
from smc_trader.semantic_audit import SemanticCaseVisualizer  # noqa: E402


DEFAULT_SUMMARY = (
    ROOT
    / "outputs/development/group4_mature_range_coverage_v1/summary.json"
)
DEFAULT_CONFIG = ROOT / "configs/group4_mature_range_coverage_v1.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    # Deliberately keep the blind manifest free of strata and model state.
    encoded = (
        json.dumps(
            {"cases": rows},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    atomic_bytes(path, encoded)


def _render(
    histories: dict[Timeframe, tuple[Any, ...]],
    focus_clock: pd.Timestamp,
    destination: Path,
    *,
    tick_size: float,
) -> tuple[str, pd.Timestamp]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = SemanticCaseVisualizer._panels(histories, focus_clock)
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
            tick_size=tick_size,
        )
        axis.axvline(len(candles) - 0.5, color="#111827", linewidth=1.0)
        axis.set_title(
            f"{timeframe.value} · raw completed candles through "
            f"{candles[-1].end:%Y-%m-%d %H:%M %Z}",
            loc="left",
            fontsize=9,
        )
    figure.suptitle(
        "BLIND RANGE CASE — raw completed candles only",
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
    if maximum > focus_clock:
        raise AssertionError("blind image contains a future candle")
    return sha256_file(destination), maximum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    config_path = Path(args.config)
    summary = _json(summary_path)
    config = _json(config_path)
    selected = summary.get("selected_cases")
    if not isinstance(selected, dict):
        raise ValueError("summary lacks selected_cases")

    cases: list[dict[str, Any]] = []
    for value in selected.values():
        if value is None:
            continue
        cases.append(
            {
                "opaque_case_id": str(value["opaque_case_id"]),
                "focus_clock": pd.Timestamp(value["focus_clock"]),
                "window_id": str(value["window_id"]),
            }
        )
    if not cases or len({case["opaque_case_id"] for case in cases}) != len(cases):
        raise ValueError("selected cases are empty or opaque identities collide")
    if any(case["focus_clock"].tzinfo is None for case in cases):
        raise ValueError("case clocks must be timezone aware")

    windows = {str(item["id"]): item for item in config["windows"]}
    source = ROOT / str(config["source"])
    output = summary_path.parent / "blind"
    warmup_days = int(config["warmup_calendar_days"])
    tick_size = 0.25
    artifacts: list[dict[str, str]] = []

    for window_id in sorted({case["window_id"] for case in cases}):
        window_cases = [case for case in cases if case["window_id"] == window_id]
        window = windows[window_id]
        replay_start = pd.Timestamp(window["start"]) - pd.Timedelta(
            warmup_days,
            unit="D",
        )
        final_clock = max(case["focus_clock"] for case in window_cases)
        loaded = load_ohlcv(source, start=replay_start, end=final_clock)
        if not loaded.contract_selection_causal:
            raise RuntimeError("blind rendering requires causal contract selection")
        reader = CausalMarketReader()
        by_clock: dict[pd.Timestamp, list[dict[str, Any]]] = {}
        for case in window_cases:
            by_clock.setdefault(case["focus_clock"], []).append(case)
        for bar in iter_completed_bars(
            loaded.frame,
            allow_data_gap_reset=True,
        ):
            update = reader.on_bar(bar)
            for case in by_clock.get(update.asof, ()):
                histories = {
                    timeframe: reader.window(
                        timeframe,
                        SemanticCaseVisualizer.PANEL_BARS[timeframe],
                    )
                    for timeframe in CORE_TIMEFRAMES
                }
                destination = output / f"{case['opaque_case_id']}.png"
                image_sha, maximum = _render(
                    histories,
                    update.asof,
                    destination,
                    tick_size=tick_size,
                )
                artifacts.append(
                    {
                        "case_clock": update.asof.isoformat(),
                        "image_sha256": image_sha,
                        "maximum_market_time": maximum.isoformat(),
                        "opaque_case_id": case["opaque_case_id"],
                    }
                )

    if len(artifacts) != len(cases):
        rendered = {row["opaque_case_id"] for row in artifacts}
        missing = sorted(
            case["opaque_case_id"]
            for case in cases
            if case["opaque_case_id"] not in rendered
        )
        raise RuntimeError(f"selected case clocks were not rendered: {missing}")
    artifacts.sort(key=lambda row: row["opaque_case_id"])
    _write_manifest(output / "manifest.json", artifacts)
    print(
        json.dumps(
            {
                "cases": len(artifacts),
                "manifest": str(output / "manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
