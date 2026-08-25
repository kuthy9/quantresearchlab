"""Legacy import shim for pre-rename Range Auction protocol pickles.

New code must import :mod:`smc_trader.range_auction`. These aliases exist only
for historical class lookup before normal checkpoint validation runs.
"""
from __future__ import annotations

import warnings

from .range_auction import (
    CausalRangeAuctionTracker,
    RangeAuctionProtocol,
    RangeAuctionUpdate,
)

warnings.warn(
    "smc_trader.group4 is legacy-only; import smc_trader.range_auction",
    DeprecationWarning,
    stacklevel=2,
)

CausalGroup4Tracker = CausalRangeAuctionTracker
Group4Protocol = RangeAuctionProtocol
Group4Update = RangeAuctionUpdate

__all__: tuple[str, ...] = ()
