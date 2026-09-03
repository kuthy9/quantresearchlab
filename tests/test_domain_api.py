from __future__ import annotations

import importlib
import sys

import pytest

import smc_trader
from smc_trader.range_auction import RangeAuctionProtocol
from smc_trader.zone import ZoneProtocol


def test_public_domain_api_contains_no_numbered_group_types() -> None:
    assert "ZoneProtocol" not in smc_trader.__all__
    assert "CausalZoneTracker" not in smc_trader.__all__
    assert "RangeAuctionProtocol" not in smc_trader.__all__
    assert "CausalRangeAuctionTracker" not in smc_trader.__all__
    assert not any(name.startswith(("Group3", "Group4")) for name in smc_trader.__all__)
