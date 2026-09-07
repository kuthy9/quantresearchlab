from __future__ import annotations

import importlib
import sys

import pytest

import shares
from eyes.core.range_auction import RangeAuctionProtocol
from eyes.core.zone import ZoneProtocol


def test_public_domain_api_contains_no_numbered_group_types() -> None:
    assert "ZoneProtocol" not in shares.__all__
    assert "CausalZoneTracker" not in shares.__all__
    assert "RangeAuctionProtocol" not in shares.__all__
    assert "CausalRangeAuctionTracker" not in shares.__all__
    assert not any(name.startswith(("Group3", "Group4")) for name in shares.__all__)
