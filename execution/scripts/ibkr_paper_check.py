"""Read-only check of the IBKR paper session the executor would use.

    .venv/bin/python -m execution.scripts.ibkr_paper_check

Connects to TWS / IB Gateway with ``execution/configs/ibkr.json``, applies the
same guards as ``IBKRBroker`` (paper account only, ``live_execution_allowed``
false) and prints the account snapshot the Risk gate would read.  It never
places, modifies or cancels an order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from execution.core.ibkr_broker import IBKRBroker, IBKRConfig, IBKRRefused

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="execution/configs/ibkr.json")
    parser.add_argument("--model-path", default="configs/model.json")
    args = parser.parse_args(argv)
    config = IBKRConfig.from_json(ROOT / args.config)
    live = bool(json.loads((ROOT / args.model_path).read_text(encoding="utf-8"))["release_readiness"]["live_execution_allowed"])
    try:
        broker = IBKRBroker.connect(config, live_execution_allowed=live)
    except IBKRRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # connection problems: TWS not running, API not enabled, wrong port
        print(f"could not connect to {config.host}:{config.port}: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    snapshot = broker.snapshot(pd.Timestamp.now(tz="UTC"))
    print(json.dumps(snapshot.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
