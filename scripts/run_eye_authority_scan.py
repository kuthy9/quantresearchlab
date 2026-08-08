#!/usr/bin/env python3
"""Run the registered 2023 Group1-5 eye-only authority scan.

The runtime is intentionally narrow:

    CausalMarketReader -> CausalObserver -> EyeAuthorityStatistics

It never constructs the Brain, Decision, Risk, execution simulation, MBO,
PnL, future-path, Scene Graph, image, or per-minute trace facilities.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any, Iterator, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.artifact_stream import atomic_bytes  # noqa: E402
from smc_trader.causal import CausalMarketReader  # noqa: E402
from smc_trader.eye_statistics import EyeAuthorityStatistics  # noqa: E402
from smc_trader.group4 import Group4Protocol  # noqa: E402
from smc_trader.io import LoadedOHLCV, iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.market_clock import is_registered_trading_minute  # noqa: E402
from smc_trader.model import MarketObservation, Timeframe, to_primitive  # noqa: E402
from smc_trader.observation import CausalObserver, ObserverConfig  # noqa: E402
from smc_trader.scene_graph import parse_scale_specs  # noqa: E402
from smc_trader.validation import load_validation_protocol  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/data_splits.json"
DEFAULT_OUTPUT = ROOT / "outputs/development/eye_authority"
CANONICAL_PROFILE = "eye_group1_5_natural_authority_2023_full_year"
EXPECTED_WINDOW = {
    "id": "2023-full-year-eye",
    "start": "2023-01-01T00:00:00-05:00",
    "end_exclusive": "2024-01-01T00:00:00-05:00",
}
EXPECTED_PROTOCOLS = {
    "group12": "configs/primitives_structure_liquidity.json",
    "displacement": "configs/primitives_displacement.json",
    "group3": "configs/primitives_zones.json",
    "group4": "configs/primitives_range.json",
    "group5": "configs/primitives_entry.json",
}
EXPECTED_POOL_TIMEFRAMES = {"4H", "1H", "15m", "5m", "1m"}
EXPECTED_CASE_CATEGORIES = (
    "all_recognized_mature",
    "obvious_mature_looking_but_rejected",
    "near_mature_single_gate",
    "multiple_gate_rejected",
    "forming_reasonably_broken",
    "source_identity_or_reset_failure",
    "mature_range_manipulation",
    "group5_complete_path",
    "group5_interrupted_path",
)
EXPECTED_CASE_SELECTION_METHOD = (
    "all_recognized_mature_first_smallest_sha256_then_registered_"
    "strata_round_robin_with_global_episode_uniqueness"
)
EXPECTED_MATURE_RANGE_TARGET = {
    "semantic_name": "MatureBalanceRange",
    "definition_class": "A",
    "meaning": (
        "rare high-quality persistent H1 balance or accumulation, not a "
        "generic H1 dealing range"
    ),
    "systematic_miss_requires": [
        "stable bilateral boundaries are visible",
        "multiple internal rotations are visible",
        "no sustained one-way delivery dominates the interval",
        "the balance persists for multiple completed H1 bars",
        (
            "different-month cases are repeatedly rejected for the same gate "
            "or source-identity reason"
        ),
    ],
}
EXPECTED_STOPPING_RULES = (
    "isolated anomalies do not change a definition",
    (
        "one repeated semantic miss across different months permits at most "
        "one conceptual repair"
    ),
    (
        "a repair is checked on a different frozen window rather than retuning "
        "2023"
    ),
    (
        "a sparse but visually accurate MatureBalanceRange remains a rare "
        "context"
    ),
    (
        "an unstable MatureBalanceRange remains parked and FAVR may only become "
        "an optional LSR range context"
    ),
)
RUNTIME_CODE_FILES = (
    "scripts/run_eye_authority_scan.py",
    "smc_trader/artifact_stream.py",
    "smc_trader/causal.py",
    "smc_trader/displacement.py",
    "smc_trader/displacement_observer.py",
    "smc_trader/eye_statistics.py",
    "smc_trader/group3.py",
    "smc_trader/group4.py",
    "smc_trader/group5.py",
    "smc_trader/io.py",
    "smc_trader/liquidity.py",
    "smc_trader/market_clock.py",
    "smc_trader/model.py",
    "smc_trader/observation.py",
    "smc_trader/scene_graph.py",
    "smc_trader/structure.py",
    "smc_trader/validation.py",
)
_CHECKPOINT_MAGIC = b"SMC-EYE-AUTHORITY-CHECKPOINT-v1\n"
RUNTIME_SWITCHES = {
    "brain_used": False,
    "decision_used": False,
    "risk_used": False,
    "execution_used": False,
    "mbo_used": False,
    "pnl_used": False,
    "future_path_used": False,
    "project_scene_graph": False,
    "materialize_event_view": False,
    "per_minute_trace": False,
    "per_minute_snapshot": False,
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _aware(value: Any, *, name: str) -> pd.Timestamp:
    clock = pd.Timestamp(value)
    if clock.tzinfo is None:
        raise ValueError(f"{name} must be timezone aware")
    return clock


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("eye scan could not resolve one Git commit")
    return commit


def _git_worktree_status() -> str:
    completed = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_identity(*, require_clean: bool) -> dict[str, Any]:
    status = _git_worktree_status()
    clean = not status
    if require_clean and not clean:
        raise RuntimeError(
            "formal full-year eye scan requires a clean Git worktree; "
            "commit the frozen runner, reducers, model and protocols first"
        )
    return {
        "commit": _git_commit(),
        "clean": clean,
        # Do not persist the dirty file list in evidence.  Runtime source hashes
        # below bind every reducer/core file needed for checkpoint equivalence.
        "dirty_runtime_allowed": bool(not clean and not require_clean),
    }


def _registered_payload(
    config: Path = DEFAULT_CONFIG,
    *,
    profile: str = CANONICAL_PROFILE,
) -> dict[str, Any]:
    if profile != CANONICAL_PROFILE:
        raise ValueError(
            "eye authority runner accepts only the registered 2023 profile"
        )
    splits = _json(config)
    profiles = splits.get("authority_scan_profiles")
    sources = splits.get("sources")
    registered = (
        profiles.get(profile) if isinstance(profiles, Mapping) else None
    )
    source = sources.get("ohlcv") if isinstance(sources, Mapping) else None
    if not isinstance(registered, Mapping) or not isinstance(source, Mapping):
        raise ValueError("registered eye profile or canonical OHLCV source is missing")
    return {
        "schema_version": splits.get("schema_version"),
        "profile_name": profile,
        "profile": deepcopy(dict(registered)),
        "source": deepcopy(dict(source)),
        "validation_protocol": str(config.relative_to(ROOT)),
    }


def _validate_registered_payload(payload: Mapping[str, Any]) -> None:
    canonical = _registered_payload()
    if payload != canonical:
        raise ValueError(
            "eye authority payload differs from the canonical registered profile"
        )
    profile = payload["profile"]
    source = payload["source"]
    false_flags = {
        "threshold_search",
        "outcome_fields_used",
        "pnl_used",
        "future_path_used",
        "mbo_used",
        "brain_used",
        "decision_used",
        "risk_used",
        "execution_used",
        "project_scene_graph",
        "materialize_event_view",
    }
    if (
        payload.get("schema_version") != 1
        or profile.get("runner_mode") != "eye_only_all_registered_semantics"
        or any(profile.get(name) is not False for name in false_flags)
        or profile.get("execution_reality_status") != "not_evaluated"
        or profile.get("timezone") != "America/New_York"
        or int(profile.get("warmup_calendar_days", 0)) != 7
        or profile.get("allowed_ohlcv_role") != "calibration"
        or profile.get("registered_calibration_exception")
        != "outcome_blind_natural_authority_only"
        or profile.get("allow_data_gap_reset") is not True
        or profile.get("include_all_pool_source_timeframes") is not True
        or set(profile.get("pool_source_timeframes") or ())
        != EXPECTED_POOL_TIMEFRAMES
        or profile.get("protocols") != EXPECTED_PROTOCOLS
        or profile.get("windows") != [EXPECTED_WINDOW]
        or int(profile.get("checkpoint_every_completed_1m", 0)) < 1
    ):
        raise ValueError("registered eye profile violates the frozen scan contract")
    if (
        source.get("path")
        != "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
        or source.get("current_session_volume_used") is not False
        or source.get("selection")
        != "highest total volume from the strictly prior completed Globex session"
        or not isinstance(source.get("sha256"), str)
        or len(source["sha256"]) != 64
    ):
        raise ValueError("eye scan requires the canonical previous-session source")
    sampling = profile.get("blind_case_sampling")
    if (
        not isinstance(sampling, Mapping)
        or int(sampling.get("minimum_cases", 0)) != 20
        or int(sampling.get("maximum_cases", 0)) != 40
        or sampling.get("selection_method")
        != EXPECTED_CASE_SELECTION_METHOD
        or sampling.get("future_hidden_on_first_review") is not True
        or tuple(sampling.get("categories") or ())
        != EXPECTED_CASE_CATEGORIES
        or profile.get("mature_range_target")
        != EXPECTED_MATURE_RANGE_TARGET
        or tuple(profile.get("stopping_rules") or ())
        != EXPECTED_STOPPING_RULES
    ):
        raise ValueError("eye scan blind case sampling is not frozen")


def _calendar_warmup_start(
    start: pd.Timestamp,
    *,
    days: int,
    timezone: str,
) -> pd.Timestamp:
    return start.tz_convert(timezone) - pd.DateOffset(days=days)


def _run_identity(
    payload: Mapping[str, Any],
    *,
    require_clean: bool = False,
) -> dict[str, Any]:
    profile = payload["profile"]
    source = ROOT / str(payload["source"]["path"])
    source_hash = _sha256_file(source)
    if source_hash != payload["source"]["sha256"]:
        raise RuntimeError("eye scan OHLCV source hash mismatch")
    protocol_identity: dict[str, Any] = {}
    for name, relative in sorted(profile["protocols"].items()):
        path = ROOT / str(relative)
        protocol = _json(path)
        protocol_identity[str(name)] = {
            "path": str(relative),
            "protocol_version": protocol.get("protocol_version"),
            "sha256": _sha256_file(path),
        }
    model_path = ROOT / str(profile["model_config"])
    code_identity = {
        relative: _sha256_file(ROOT / relative)
        for relative in RUNTIME_CODE_FILES
    }
    return {
        "git": _git_identity(require_clean=require_clean),
        "source": {
            "path": str(payload["source"]["path"]),
            "sha256": source_hash,
        },
        "data_splits": {
            "path": payload["validation_protocol"],
            "sha256": _sha256_file(ROOT / payload["validation_protocol"]),
        },
        "model": {
            "path": str(profile["model_config"]),
            "sha256": _sha256_file(model_path),
        },
        "protocols": protocol_identity,
        "code": code_identity,
    }


def _scan_identity(
    payload: Mapping[str, Any],
    run_identity: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        to_primitive(
            {
                "profile_name": payload["profile_name"],
                "profile": payload["profile"],
                "identity": run_identity,
                "runtime_switches": RUNTIME_SWITCHES,
            }
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_eye(
    payload: Mapping[str, Any],
) -> tuple[CausalMarketReader, CausalObserver]:
    profile = payload["profile"]
    model = _json(ROOT / str(profile["model_config"]))
    specs = parse_scale_specs(model.get("scales"))
    observer_raw = model.get("observer")
    if not isinstance(observer_raw, Mapping):
        raise ValueError("model observer configuration is missing")
    expected_model_bindings = {
        "structure_protocol": EXPECTED_PROTOCOLS["group12"],
        "liquidity_protocol": EXPECTED_PROTOCOLS["group12"],
        "displacement_protocol": EXPECTED_PROTOCOLS["displacement"],
        "group3_protocol": EXPECTED_PROTOCOLS["group3"],
        "group4_protocol": EXPECTED_PROTOCOLS["group4"],
        "group5_protocol": EXPECTED_PROTOCOLS["group5"],
    }
    if any(
        str(observer_raw.get(name)) != value
        for name, value in expected_model_bindings.items()
    ):
        raise ValueError("model and eye profile protocol bindings disagree")
    minimum = observer_raw.get("minimum_bars")
    if not isinstance(minimum, Mapping):
        raise ValueError("model observer minimum bars are invalid")
    observer = CausalObserver(
        ObserverConfig(
            atr_period=int(observer_raw.get("atr_period", 14)),
            memory_events=int(observer_raw.get("memory_events", 512)),
            minimum_bars={
                timeframe: int(minimum.get(timeframe.value, default))
                for timeframe, default in {
                    Timeframe.H4: 16,
                    Timeframe.H1: 24,
                    Timeframe.M15: 24,
                    Timeframe.M5: 24,
                    Timeframe.M1: 30,
                }.items()
                if any(
                    spec.enabled and spec.native_timeframe is timeframe
                    for spec in specs
                )
            },
            tick_size=float(model.get("tick_size", 0.25)),
            point_value=float(model.get("point_value", 20.0)),
            structure_protocol=str(ROOT / EXPECTED_PROTOCOLS["group12"]),
            liquidity_protocol=str(ROOT / EXPECTED_PROTOCOLS["group12"]),
            displacement_protocol=str(
                ROOT / EXPECTED_PROTOCOLS["displacement"]
            ),
            group3_protocol=str(ROOT / EXPECTED_PROTOCOLS["group3"]),
            group4_protocol=str(ROOT / EXPECTED_PROTOCOLS["group4"]),
            group5_protocol=str(ROOT / EXPECTED_PROTOCOLS["group5"]),
            scale_specs=specs,
            project_scene_graph=False,
            materialize_event_view=False,
            group4_projection_only=False,
            eye_authority_mode=True,
        )
    )
    return CausalMarketReader(scale_specs=specs), observer


def _write_json(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(
            to_primitive(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    atomic_bytes(path, encoded)


def _write_checkpoint(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    payload = pickle.dumps(dict(value), protocol=pickle.HIGHEST_PROTOCOL)
    checksum = hashlib.sha256(payload).hexdigest().encode("ascii")
    atomic_bytes(
        path,
        _CHECKPOINT_MAGIC + checksum + b"\n" + payload,
    )


def _checkpoint_binding(
    *,
    payload: Mapping[str, Any],
    scan_identity: str,
    max_bars: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile_name": payload["profile_name"],
        "window": payload["profile"]["windows"][0],
        "scan_identity": scan_identity,
        "runtime_switches": RUNTIME_SWITCHES,
        "max_in_window_bars": max_bars,
    }


def _load_checkpoint(
    path: Path,
    *,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    encoded = path.read_bytes()
    if not encoded.startswith(_CHECKPOINT_MAGIC):
        raise RuntimeError("eye scan checkpoint envelope is invalid")
    remainder = encoded[len(_CHECKPOINT_MAGIC) :]
    checksum, separator, payload = remainder.partition(b"\n")
    if (
        separator != b"\n"
        or len(checksum) != 64
        or any(value not in b"0123456789abcdef" for value in checksum)
    ):
        raise RuntimeError("eye scan checkpoint envelope is invalid")
    if hashlib.sha256(payload).hexdigest().encode("ascii") != checksum:
        raise RuntimeError("eye scan checkpoint pickle integrity failed")
    try:
        value = pickle.loads(payload)
    except Exception as exc:
        raise RuntimeError("eye scan checkpoint payload is invalid") from exc
    if not isinstance(value, dict) or value.get("binding") != binding:
        raise RuntimeError("eye scan checkpoint identity or switches disagree")
    if not isinstance(value.get("reader"), CausalMarketReader):
        raise RuntimeError("eye scan checkpoint lacks its causal reader")
    if not isinstance(value.get("observer"), CausalObserver):
        raise RuntimeError("eye scan checkpoint lacks its causal observer")
    if not isinstance(value.get("statistics"), EyeAuthorityStatistics):
        raise RuntimeError("eye scan checkpoint lacks its statistics state")
    if type(value.get("emitted_bars")) is not int or value["emitted_bars"] < 0:
        raise RuntimeError("eye scan checkpoint bar position is invalid")
    if (
        type(value.get("in_window_bars")) is not int
        or value["in_window_bars"] < 0
        or value["in_window_bars"] > value["emitted_bars"]
    ):
        raise RuntimeError("eye scan checkpoint window position is invalid")
    prior = value.get("previous_observation")
    if prior is not None and not isinstance(prior, MarketObservation):
        raise RuntimeError("eye scan checkpoint prior observation is invalid")
    return value


def _skip_emitted(bars: Iterator[Any], count: int) -> None:
    for _ in range(count):
        try:
            next(bars)
        except StopIteration as exc:
            raise RuntimeError(
                "eye scan checkpoint lies beyond the current source"
            ) from exc


def _validate_loaded_source(
    loaded: LoadedOHLCV,
    *,
    expected_path: Path,
) -> None:
    if (
        loaded.source.resolve() != expected_path.resolve()
        or loaded.source_role != "processed_continuous_front"
        or loaded.contract_selection_causal is not True
        or loaded.warnings
    ):
        raise RuntimeError(
            "eye scan source is not the canonical causal previous-session front"
        )
    if loaded.frame.empty:
        raise RuntimeError("eye scan source interval contains no OHLCV rows")


def _expected_last_observation_asof(
    *,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    timezone: str,
) -> pd.Timestamp:
    """Return the close clock of the last registered 1m bar before ``end``."""

    cursor = end_exclusive.tz_convert(timezone) - pd.Timedelta(minutes=1)
    lower = start.tz_convert(timezone)
    while cursor >= lower:
        if is_registered_trading_minute(cursor):
            return cursor + pd.Timedelta(minutes=1)
        cursor -= pd.Timedelta(minutes=1)
    raise RuntimeError("registered eye window contains no trading minute")


def _evidence_integrity_checks(
    raw_summary: Mapping[str, Any],
    *,
    in_window_bars: int,
    last_observation_asof: pd.Timestamp | None,
    expected_last_observation_asof: pd.Timestamp,
) -> dict[str, bool]:
    """Validate the small set of contracts required before permanent output."""

    window = raw_summary.get("window")
    group3 = raw_summary.get("group3")
    group4 = raw_summary.get("group4")
    group5 = raw_summary.get("group5")
    funnels = raw_summary.get("funnels")
    case_selection = raw_summary.get("case_selection")
    case_index = raw_summary.get("case_index")
    if not isinstance(window, Mapping):
        window = {}
    if not isinstance(group3, Mapping):
        group3 = {}
    if not isinstance(group4, Mapping):
        group4 = {}
    if not isinstance(group5, Mapping):
        group5 = {}
    if not isinstance(case_selection, Mapping):
        case_selection = {}
    if not isinstance(case_index, list):
        case_index = []
    ob = group3.get("order_block_admission")
    range_funnel = group4.get("range_formation_funnel")
    source_disposition = group4.get("source_disposition_conservation")
    manipulation = group4.get("manipulation_conservation")
    selected_primary_join = group4.get("selected_primary_to_episode_join")
    group5_manipulation_funnel = group5.get(
        "manipulation_to_path_identity_funnel"
    )
    group5_zone_funnel = group5.get(
        "qualified_zone_to_terminal_identity_funnel"
    )
    group5_identity = group5.get("exact_identity_conservation")
    favr_chain = group5.get("favr_observation_chain")
    ob_conservation = ob.get("conservation") if isinstance(ob, Mapping) else None
    range_conservation = (
        range_funnel.get("conservation")
        if isinstance(range_funnel, Mapping)
        else None
    )
    manipulation_conservation = (
        manipulation.get("conservation")
        if isinstance(manipulation, Mapping)
        else None
    )
    funnel_rows = funnels if isinstance(funnels, list) else []
    return {
        "tail_registered_minute_reached": (
            last_observation_asof is not None
            and last_observation_asof == expected_last_observation_asof
        ),
        "in_window_observation_count_matches": (
            type(window.get("observations")) is int
            and int(window["observations"]) == in_window_bars
        ),
        "order_block_producer_exposed": (
            isinstance(ob, Mapping)
            and ob.get("denominator_status") == "producer_exposed"
        ),
        "order_block_conserved": (
            isinstance(ob_conservation, Mapping)
            and ob_conservation.get("balanced") is True
        ),
        "range_producer_exposed": (
            isinstance(range_funnel, Mapping)
            and range_funnel.get("denominator_status") == "producer_exposed"
        ),
        "range_conserved": (
            isinstance(range_conservation, Mapping)
            and range_conservation.get("balanced") is True
        ),
        "source_disposition_producer_exposed": any(
            isinstance(row, Mapping)
            and row.get("group") == "group4"
            and row.get("primitive") == "manipulation_source"
            and row.get("name") == "raw_crossed_source"
            and row.get("denominator_status") == "producer_exposed"
            for row in raw_summary.get("denominators", ())
        ),
        "source_dispositions_conserved": (
            isinstance(source_disposition, Mapping)
            and source_disposition.get("balanced") is True
        ),
        "manipulation_episodes_conserved": (
            isinstance(manipulation_conservation, Mapping)
            and manipulation_conservation.get("balanced") is True
        ),
        "selected_primary_to_episode_conserved": (
            isinstance(selected_primary_join, Mapping)
            and selected_primary_join.get("balanced") is True
        ),
        "group5_contract_exposed": (
            group5.get("denominator_status") == "producer_exposed"
        ),
        "group5_identity_funnels_conserved": (
            isinstance(group5_manipulation_funnel, list)
            and isinstance(group5_zone_funnel, list)
            and all(
                isinstance(row, Mapping)
                and isinstance(row.get("conservation"), Mapping)
                and row["conservation"].get("terminal_balanced") is True
                and row["conservation"].get("funnel_monotone") is True
                for rows in (group5_manipulation_funnel, group5_zone_funnel)
                for row in rows
            )
        ),
        "group5_exact_identities_conserved": (
            isinstance(group5_identity, Mapping)
            and group5_identity.get("exact_identity_conserved") is True
            and group5_identity.get("exact_identity_violation_count") == 0
        ),
        "group5_path_order_conserved": group5.get("path_order_errors") == 0,
        "favr_identity_contract_exposed": (
            isinstance(favr_chain, Mapping)
            and favr_chain.get("status")
            not in {
                None,
                "authoritative_join_not_exposed",
                "authoritative_identity_join_exposed_with_violations",
            }
            and not favr_chain.get("identity_violations")
            and not favr_chain.get("authoritative_join_not_exposed")
        ),
        "all_lifecycle_latest_states_classified": bool(funnel_rows)
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("conservation"), Mapping)
            and row["conservation"].get("balanced") is True
            for row in funnel_rows
        ),
        "registered_case_selection_method_used": (
            case_selection.get("method") == EXPECTED_CASE_SELECTION_METHOD
        ),
        "registered_case_categories_used": (
            tuple(case_selection.get("frozen_strata") or ())
            == EXPECTED_CASE_CATEGORIES
        ),
        "selected_case_categories_registered": all(
            isinstance(case, Mapping)
            and case.get("stratum") in EXPECTED_CASE_CATEGORIES
            for case in case_index
        ),
        "selected_case_count_consistent": (
            type(case_selection.get("selected")) is int
            and int(case_selection["selected"]) == len(case_index)
            and 20 <= len(case_index) <= 40
        ),
        "future_and_pnl_absent_from_case_selection": (
            case_selection.get("future_or_pnl_used") is False
        ),
    }


def _augment_summary(
    raw: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    scan_identity: str,
    warmup_start: pd.Timestamp,
    source_rows: int,
    emitted_bars: int,
    in_window_bars: int,
    scan_completed: bool,
    evidence_integrity_checks: Mapping[str, bool],
    max_bars: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = deepcopy(dict(raw))
    cases = list(summary.pop("case_index", ()))
    profile = payload["profile"]
    summary["registered_window"] = {
        **profile["windows"][0],
        "warmup_start": warmup_start.isoformat(),
        "warmup_calendar_days": profile["warmup_calendar_days"],
        "timezone": profile["timezone"],
    }
    integrity_passed = bool(
        scan_completed
        and evidence_integrity_checks
        and all(evidence_integrity_checks.values())
    )
    summary["scan_status"] = {
        # ``complete`` is retained as the composite compatibility field.  The
        # two explicit fields prevent source exhaustion from masquerading as
        # valid authority evidence.
        "complete": bool(scan_completed and integrity_passed),
        "scan_completed": scan_completed,
        "evidence_integrity_passed": integrity_passed,
        "partial_reason": (
            None
            if integrity_passed
            else "evidence_integrity_failed"
            if scan_completed
            else "max_bars_smoke"
        ),
        "evidence_integrity_checks": dict(evidence_integrity_checks),
        "source_rows": source_rows,
        "emitted_completed_1m_bars": emitted_bars,
        "in_window_observations": in_window_bars,
        "max_in_window_bars": max_bars,
        "execution_reality_status": "not_evaluated",
    }
    summary["case_index"] = {
        "count": len(cases),
        "future_hidden_on_first_review": True,
        "separate_artifact": True,
    }
    summary["authority_contract"] = {
        "mature_range_target": deepcopy(profile["mature_range_target"]),
        "stopping_rules": list(profile["stopping_rules"]),
        "blind_case_sampling": deepcopy(profile["blind_case_sampling"]),
    }
    summary["run_manifest"] = {
        "profile": payload["profile_name"],
        "scan_identity": scan_identity,
        "runtime_switches": RUNTIME_SWITCHES,
        "identity": dict(run_identity),
    }
    case_index = {
        "schema_version": 1,
        "profile": payload["profile_name"],
        "scan_identity": scan_identity,
        "registered_window": summary["registered_window"],
        "selection": summary.get("case_selection", {}),
        "future_hidden_on_first_review": True,
        "cases": cases,
    }
    return summary, case_index


def run_scan(
    *,
    output: Path,
    profile: str = CANONICAL_PROFILE,
    max_bars: int | None = None,
    stop_after_bars: int | None = None,
    force: bool = False,
) -> dict[str, Any] | None:
    """Run or resume the registered scan; return ``None`` on simulated stop."""

    if max_bars is not None and (type(max_bars) is not int or max_bars < 1):
        raise ValueError("max-bars must be a positive integer")
    if stop_after_bars is not None and (
        type(stop_after_bars) is not int or stop_after_bars < 1
    ):
        raise ValueError("stop-after-bars must be a positive integer")
    payload = _registered_payload(profile=profile)
    _validate_registered_payload(payload)
    profile_payload = payload["profile"]
    window = profile_payload["windows"][0]
    start = _aware(window["start"], name="registered window start")
    end = _aware(window["end_exclusive"], name="registered window end")
    warmup_start = _calendar_warmup_start(
        start,
        days=int(profile_payload["warmup_calendar_days"]),
        timezone=str(profile_payload["timezone"]),
    )
    validation = load_validation_protocol(DEFAULT_CONFIG)
    role = validation.classify_ohlcv(start, end)
    warmup_role = validation.classify_ohlcv(warmup_start, end)
    if role.role != "calibration" or warmup_role.role != "calibration":
        raise ValueError("registered eye window is outside its calibration role")

    run_identity = _run_identity(
        payload,
        require_clean=max_bars is None,
    )
    scan_identity = _scan_identity(payload, run_identity)
    binding = _checkpoint_binding(
        payload=payload,
        scan_identity=scan_identity,
        max_bars=max_bars,
    )
    output = output.resolve()
    checkpoint_path = output / "checkpoint.pkl"
    summary_path = output / "summary.json"
    cases_path = output / "case_index.json"
    if force:
        for path in (
            checkpoint_path,
            summary_path,
            cases_path,
        ):
            path.unlink(missing_ok=True)
    permanent_summary = ROOT / str(profile_payload["permanent_result_path"])
    permanent_cases = ROOT / str(
        profile_payload["permanent_case_index_path"]
    )
    if max_bars is None and not force and (
        permanent_summary.exists() or permanent_cases.exists()
    ):
        raise FileExistsError(
            "permanent eye evidence already exists; use --force to replace it"
        )
    output.mkdir(parents=True, exist_ok=True)

    source_path = ROOT / str(payload["source"]["path"])
    loaded = load_ohlcv(
        source_path,
        start=warmup_start,
        end=end,
    )
    _validate_loaded_source(loaded, expected_path=source_path)
    bars = iter_completed_bars(
        loaded.frame,
        allow_data_gap_reset=bool(profile_payload["allow_data_gap_reset"]),
    )
    if checkpoint_path.is_file():
        checkpoint = _load_checkpoint(
            checkpoint_path,
            binding=binding,
        )
        reader = checkpoint["reader"]
        observer = checkpoint["observer"]
        statistics = checkpoint["statistics"]
        previous_observation = checkpoint["previous_observation"]
        emitted_bars = int(checkpoint["emitted_bars"])
        in_window_bars = int(checkpoint["in_window_bars"])
        _skip_emitted(bars, emitted_bars)
        print(f"resume: skipped {emitted_bars:,} emitted bars", flush=True)
    else:
        reader, observer = _build_eye(payload)
        statistics = EyeAuthorityStatistics(
            start=start,
            end_exclusive=end,
            coverage_start=warmup_start,
            group4_protocol=Group4Protocol.from_file(
                ROOT / EXPECTED_PROTOCOLS["group4"]
            ),
        )
        previous_observation = None
        emitted_bars = 0
        in_window_bars = 0

    checkpoint_every = int(profile_payload["checkpoint_every_completed_1m"])
    invocation_bars = 0
    next_progress = 5
    if (
        previous_observation is not None
        and previous_observation.asof >= start
    ):
        completed_progress = int(
            100 * (previous_observation.asof - start) / (end - start)
        )
        next_progress = max(5, (completed_progress // 5 + 1) * 5)
    source_exhausted = True
    for bar in bars:
        update = reader.on_bar(bar)
        observation = observer.observe(update)
        emitted_bars += 1
        invocation_bars += 1
        if start <= observation.asof < end:
            in_window_bars += 1
            progress = int(
                100 * (observation.asof - start) / (end - start)
            )
            while progress >= next_progress and next_progress <= 100:
                print(f"[{window['id']}] {next_progress}%", flush=True)
                next_progress += 5
        statistics.observe(
            update,
            observation,
            previous_observation=previous_observation,
        )
        previous_observation = observation
        if emitted_bars % checkpoint_every == 0:
            _write_checkpoint(
                checkpoint_path,
                {
                    "binding": binding,
                    "emitted_bars": emitted_bars,
                    "in_window_bars": in_window_bars,
                    "reader": reader,
                    "observer": observer,
                    "statistics": statistics,
                    "previous_observation": previous_observation,
                },
            )
            print(f"checkpoint: {emitted_bars:,} emitted bars", flush=True)
        # The report-window cap is the semantic boundary of a smoke run.  It
        # must win over a simulated interruption on the same bar; otherwise a
        # resume would consume one observation beyond the registered cap.
        if max_bars is not None and in_window_bars >= max_bars:
            source_exhausted = False
            break
        if (
            stop_after_bars is not None
            and invocation_bars >= stop_after_bars
        ):
            _write_checkpoint(
                checkpoint_path,
                {
                    "binding": binding,
                    "emitted_bars": emitted_bars,
                    "in_window_bars": in_window_bars,
                    "reader": reader,
                    "observer": observer,
                    "statistics": statistics,
                    "previous_observation": previous_observation,
                },
            )
            print(f"simulated stop: {emitted_bars:,} emitted bars", flush=True)
            return None

    scan_completed = bool(source_exhausted and max_bars is None)
    raw_summary = statistics.finalize()
    integrity_checks: dict[str, bool] = {}
    if scan_completed:
        expected_tail = _expected_last_observation_asof(
            start=start,
            end_exclusive=end,
            timezone=str(profile_payload["timezone"]),
        )
        integrity_checks = _evidence_integrity_checks(
            raw_summary,
            in_window_bars=in_window_bars,
            last_observation_asof=(
                None
                if previous_observation is None
                else previous_observation.asof
            ),
            expected_last_observation_asof=expected_tail,
        )
    summary, case_index = _augment_summary(
        raw_summary,
        payload=payload,
        run_identity=run_identity,
        scan_identity=scan_identity,
        warmup_start=warmup_start,
        source_rows=len(loaded.frame),
        emitted_bars=emitted_bars,
        in_window_bars=in_window_bars,
        scan_completed=scan_completed,
        evidence_integrity_checks=integrity_checks,
        max_bars=max_bars,
    )
    integrity_passed = bool(
        summary["scan_status"]["evidence_integrity_passed"]
    )
    if scan_completed and integrity_passed:
        _write_json(permanent_summary, summary)
        _write_json(permanent_cases, case_index)
        summary_path.unlink(missing_ok=True)
        cases_path.unlink(missing_ok=True)
    else:
        _write_json(summary_path, summary)
        _write_json(cases_path, case_index)
    checkpoint_path.unlink(missing_ok=True)
    if scan_completed and not integrity_passed:
        failed = sorted(
            name for name, passed in integrity_checks.items() if not passed
        )
        raise RuntimeError(
            "full-year eye scan completed but evidence integrity failed: "
            + ", ".join(failed)
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=CANONICAL_PROFILE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-bars", type=int)
    parser.add_argument("--stop-after-bars", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output or (DEFAULT_OUTPUT / args.profile)
    if not output.is_absolute():
        output = ROOT / output
    result = run_scan(
        output=output,
        profile=args.profile,
        max_bars=args.max_bars,
        stop_after_bars=args.stop_after_bars,
        force=args.force,
    )
    if result is None:
        print(f"checkpoint retained under {output}", flush=True)
    else:
        status = result["scan_status"]
        print(
            json.dumps(
                {
                    "complete": status["complete"],
                    "emitted_completed_1m_bars": status[
                        "emitted_completed_1m_bars"
                    ],
                    "in_window_observations": status[
                        "in_window_observations"
                    ],
                    "output": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
