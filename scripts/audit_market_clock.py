#!/usr/bin/env python3
"""Reconcile every causal OHLCV minute against the registered market clock."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.causal import CausalMarketReader  # noqa: E402
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.model import CORE_TIMEFRAMES  # noqa: E402
from smc_trader.validation import load_validation_protocol  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet",
    )
    parser.add_argument(
        "--validation-protocol",
        default="configs/data_splits.json",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite clock audit: {destination}")
    protocol = load_validation_protocol(args.validation_protocol)
    source_hash = _sha256_file(args.source)
    if source_hash != protocol.causal_source.sha256:
        raise RuntimeError("clock audit source is not the preregistered causal front")
    loaded = load_ohlcv(args.source)
    reader = CausalMarketReader()
    anomaly_counts: Counter[str] = Counter()
    candle_counts: Counter[str] = Counter()
    contracts: set[tuple[str, int]] = set()
    transitions = 0
    prior_contract: tuple[str, int] | None = None
    processed = 0
    synthetic = 0
    data_gap_resets = 0
    data_gap_open_minutes = 0
    first_asof = None
    last_asof = None
    for bar in iter_completed_bars(
        loaded.frame,
        allow_data_gap_reset=True,
    ):
        contract = (bar.symbol, int(bar.instrument_id))
        if prior_contract is not None and contract != prior_contract:
            transitions += 1
        prior_contract = contract
        contracts.add(contract)
        update = reader.on_bar(bar)
        anomaly_counts.update(update.anomalies)
        for timeframe in CORE_TIMEFRAMES:
            candle_counts[timeframe.value] += len(update.newly_completed[timeframe])
        processed += 1
        synthetic += int(bar.synthetic_no_trade)
        data_gap_resets += int(bar.data_gap_before_minutes > 0)
        data_gap_open_minutes += int(bar.data_gap_before_minutes)
        first_asof = update.asof if first_asof is None else first_asof
        last_asof = update.asof
        if processed % 500_000 == 0:
            print(json.dumps({"clock_rows_checked": processed}), flush=True)
    if processed != len(loaded.frame) + synthetic:
        raise AssertionError("clock audit row accounting is inconsistent")
    if first_asof is None or last_asof is None:
        raise ValueError("clock audit source is empty")
    payload = {
        "version": 1,
        "source": str(loaded.source),
        "source_sha256": source_hash,
        "source_rows": len(loaded.frame),
        "processed_rows": processed,
        "synthetic_no_trade_rows": synthetic,
        "data_gap_resets": data_gap_resets,
        "data_gap_open_minutes": data_gap_open_minutes,
        "first_asof": first_asof.isoformat(),
        "last_asof": last_asof.isoformat(),
        "contracts": len(contracts),
        "contract_transitions": transitions,
        "registered_gap_counts": dict(sorted(anomaly_counts.items())),
        "completed_candle_counts": dict(sorted(candle_counts.items())),
        "passed": True,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
