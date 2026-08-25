"""Legacy import shim for pre-rename Zone protocol pickles.

New code must import :mod:`smc_trader.zone`. These aliases exist only so an
older pickle that names ``smc_trader.group3`` can still resolve its class before
the enclosing checkpoint's normal schema and identity checks run.
"""
from __future__ import annotations

import warnings

from .zone import (
    CausalZoneTracker,
    FVG_BOUNDARY_REASONS,
    ORDER_BLOCK_BOUNDARY_REASONS,
    WINDOW_RESET_REASONS,
    ZoneBOSSource,
    ZoneProtocol,
    ZoneRawOnlyStructureDisposition,
    ZoneUpdate,
)

warnings.warn(
    "smc_trader.group3 is legacy-only; import smc_trader.zone",
    DeprecationWarning,
    stacklevel=2,
)

CausalGroup3Tracker = CausalZoneTracker
Group3BOSSource = ZoneBOSSource
Group3Protocol = ZoneProtocol
Group3RawOnlyStructureDisposition = ZoneRawOnlyStructureDisposition
Group3Update = ZoneUpdate

__all__: tuple[str, ...] = ()
