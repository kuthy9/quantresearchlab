"""M₀ is geometry only; M₁ adds the Setup; the analytic driftless probability
is the zero-parameter reference."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from brain.core.hypothesis_proposer import FEATURE_NAMES
from brain.research.setup_features import (
    GEOMETRY_COLUMNS,
    analytic_target_probability,
    eye_state_matrix,
    geometry_matrix,
    setup_matrix,
)
from contract.eye.vocabulary import INTERACTION_PHYSICAL_PATH_STEP_KINDS


def _instances() -> pd.DataFrame:
    rows = []
    for i, (kind, zone, direction, ext) in enumerate(
        [("zone_return", "fvg", 1.0, 1.0), ("pool_reversal", None, -1.0, 1.0), ("zone_return", "ob", 1.0, -1.0)]
    ):
        row = {
            "known_at": pd.Timestamp("2022-01-04T15:00", tz="UTC") + pd.Timedelta(minutes=i),
            "path_formed_at": pd.Timestamp("2022-01-04T14:30", tz="UTC"),
            "context_kind": kind, "source_zone_kind": zone, "entry_mode": "touch" if zone else None,
            "source_timeframe": None if zone else "5m", "direction": direction,
            "d_target_atr": 2.0, "d_failure_atr": 1.0, "minutes_to_session_end": 300, "rv_30": 1.1, "rv_60": 1.5,
            "minutes_since_open": 690.0,
            "steps_so_far": json.dumps([["zone_visible", "typed_entry_zone_registered", 0.6], ["reacceptance_held", "held", 0.8]]),
            "lower_bound": 99.0 if zone else np.nan, "upper_bound": 99.8 if zone else np.nan, "atr_1m": 1.0, "first_penetration_fraction": 0.2,
            "penetration_atr": np.nan, "reclaim_margin_atr": np.nan, "hold_margin_atr": np.nan,
            "ext_dir_5m": ext, "int_dir_5m": 0.0, "last_bos_dir_5m": 1.0,
            "ext_dir_15m": 1.0, "int_dir_15m": 1.0, "last_bos_dir_15m": 1.0,
            "ext_dir_1h": -1.0, "int_dir_1h": -1.0, "last_bos_dir_1h": -1.0,
        }
        row.update({name: float(j) for j, name in enumerate(FEATURE_NAMES)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_geometry_matrix_is_the_nine_registered_columns() -> None:
    x = geometry_matrix(_instances())
    assert x.shape == (3, len(GEOMETRY_COLUMNS)) and len(GEOMETRY_COLUMNS) == 9
    assert np.isclose(x[0, 0], np.log(2.0)) and x[0, 1] == 3.0 and x[0, 4] == 300
    # 690 minutes into a 1380-minute session is half a turn: sin ≈ 0, cos ≈ -1
    assert np.isclose(x[0, 7], 0.0, atol=1e-9) and np.isclose(x[0, 8], -1.0)


def test_analytic_probability_is_the_driftless_ratio() -> None:
    assert np.allclose(analytic_target_probability(_instances()), 1.0 / 3.0)


def test_setup_matrix_columns_and_values() -> None:
    x, names = setup_matrix(_instances())
    assert x.shape == (3, len(names))
    col = {name: i for i, name in enumerate(names)}
    assert x[0, col["context_kind=zone_return"]] == 1.0 and x[1, col["context_kind=pool_reversal"]] == 1.0
    assert x[0, col["zone_kind=fvg"]] == 1.0 and x[2, col["zone_kind=ob"]] == 1.0 and x[1, col["zone_kind=fvg"]] == 0.0
    assert x[1, col["source_tf=5m"]] == 1.0 and "source_tf=1H" in col and "context_kind=pool_reversal" in col
    assert "reason=typed_entry_zone_registered" in col and "reason=held" not in col  # reasons come from the contract
    assert x[0, col["step_strength:reacceptance_held"]] == 0.8 and x[0, col["step_strength:micro_break_observed"]] == 0.0
    assert all(f"step_strength:{kind}" in col for kind in INTERACTION_PHYSICAL_PATH_STEP_KINDS)
    assert x[0, col["reason=typed_entry_zone_registered"]] == 1.0 and x[0, col["reason=formation_close_on_delivery_side"]] == 0.0
    assert x[0, col["path_age"]] == 30.0 and x[1, col["path_age"]] == 31.0
    assert np.isclose(x[0, col["zone_width_atr"]], 0.8 / np.sqrt(60))
    assert np.isnan(x[1, col["zone_width_atr"]])  # no zone on a pool path: NaN, imputed later
    # alignment flips with direction: ext 5m is +1 on row 0 (long) and +1 on row 1 (short)
    assert x[0, col["align_ext_5m"]] == 1.0 and x[1, col["align_ext_5m"]] == -1.0 and x[2, col["align_ext_5m"]] == -1.0
    assert len(names) == len(set(names))


def test_eye_state_matrix_is_the_150_components_in_order() -> None:
    x = eye_state_matrix(_instances())
    assert x.shape == (3, len(FEATURE_NAMES)) and x[0, 5] == 5.0
