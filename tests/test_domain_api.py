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


@pytest.mark.parametrize(
    ("module_name", "legacy_name", "canonical"),
    (
        ("smc_trader.group3", "Group3Protocol", ZoneProtocol),
        ("smc_trader.group4", "Group4Protocol", RangeAuctionProtocol),
    ),
)
def test_legacy_numbered_module_is_warning_only_pickle_lookup_shim(
    module_name: str,
    legacy_name: str,
    canonical: type[object],
) -> None:
    sys.modules.pop(module_name, None)
    with pytest.warns(DeprecationWarning, match="legacy-only"):
        module = importlib.import_module(module_name)

    assert module.__all__ == ()
    assert getattr(module, legacy_name) is canonical
