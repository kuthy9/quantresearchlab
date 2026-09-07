#!/usr/bin/env python3
"""Materialize NQ 1m using only the prior completed session's contract volume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shares.core.io import (  # noqa: E402
    build_previous_session_front,
    load_raw_multicontract,
)
from shares.core.market_clock import is_registered_trading_minute  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument(
        "--out",
        default="data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet",
    )
    parser.add_argument(
        "--roll-out",
        default="data/processed/nq_previous_session_front_roll_map_v2_3.parquet",
    )
    args = parser.parse_args()
    out = Path(args.out)
    roll_out = Path(args.roll_out)
    for target in (out, roll_out, out.with_suffix(".manifest.json")):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
    raw = pd.concat(
        [load_raw_multicontract(path) for path in args.source],
        ignore_index=True,
    ).sort_values("ts", kind="stable")
    off_session_rows_removed = int(
        (~raw["ts"].map(is_registered_trading_minute)).sum()
    )
    bars, roll = build_previous_session_front(raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    roll_out.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(out)
    roll.to_parquet(roll_out, index=False)
    manifest = {
        "version": "2.0.0",
        "sources": list(args.source),
        "selection": "highest total volume from the strictly prior completed Globex session",
        "current_session_volume_used": False,
        "off_session_rows_removed_before_contract_selection": (
            off_session_rows_removed
        ),
        "rows": len(bars),
        "start": bars.index.min().isoformat(),
        "end": bars.index.max().isoformat(),
        "roll_rows": len(roll),
        "output": str(out),
        "roll_output": str(roll_out),
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
