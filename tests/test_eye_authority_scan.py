from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pandas as pd
import pytest

from scripts import run_eye_authority_scan as scan
from smc_trader.io import LoadedOHLCV


def _small_loaded_source() -> LoadedOHLCV:
    index = pd.date_range(
        "2023-01-03 09:30",
        periods=18,
        freq="1min",
        tz="America/New_York",
    )
    rows = []
    for offset, _ in enumerate(index):
        open_price = 12_000.0 + 0.25 * (offset % 5)
        close = open_price + (0.25 if offset % 2 == 0 else -0.25)
        rows.append(
            {
                "open": open_price,
                "high": max(open_price, close) + 0.25,
                "low": min(open_price, close) - 0.25,
                "close": close,
                "volume": 100 + offset,
                "symbol": "NQH3",
                "instrument_id": 1,
            }
        )
    return LoadedOHLCV(
        frame=pd.DataFrame(rows, index=index),
        source=(
            scan.ROOT
            / "data/processed/"
            "nq_1m_previous_session_front_v2_3_2017_2026.parquet"
        ),
        source_role="processed_continuous_front",
        contract_selection_causal=True,
        warnings=(),
    )


def _small_loaded_source_with_warmup() -> LoadedOHLCV:
    loaded = _small_loaded_source()
    warmup = loaded.frame.iloc[:4].copy()
    warmup.index = pd.date_range(
        "2022-12-30 16:56",
        periods=4,
        freq="1min",
        tz="America/New_York",
    )
    frame = pd.concat((warmup, loaded.frame)).sort_index(kind="stable")
    return LoadedOHLCV(
        frame=frame,
        source=loaded.source,
        source_role=loaded.source_role,
        contract_selection_causal=loaded.contract_selection_causal,
        warnings=loaded.warnings,
    )


def _fake_identity() -> dict[str, object]:
    return {
        "git": {"commit": "0" * 40, "clean": True},
        "source": {"path": "registered", "sha256": "1" * 64},
        "model": {"path": "configs/model.json", "sha256": "2" * 64},
        "protocols": {},
        "code": {},
    }


def test_registered_profile_is_exact_and_fail_closed() -> None:
    payload = scan._registered_payload()

    scan._validate_registered_payload(payload)
    profile = payload["profile"]
    assert payload["profile_name"] == scan.CANONICAL_PROFILE
    assert profile["windows"] == [scan.EXPECTED_WINDOW]
    assert profile["warmup_calendar_days"] == 7
    assert scan._PROGRESS_EVERY_COMPLETED_1M == 5000
    assert scan._PROGRESS_EVERY_SECONDS == 60.0
    assert profile["protocols"] == scan.EXPECTED_PROTOCOLS
    assert profile["project_scene_graph"] is False
    assert profile["materialize_event_view"] is False
    assert all(
        profile[name] is False
        for name in scan.RUNTIME_SWITCHES
        if name in profile
    )

    with pytest.raises(ValueError, match="only the registered 2023 profile"):
        scan._registered_payload(profile="arbitrary-window")

    for field, value in (
        ("brain_used", True),
        ("project_scene_graph", True),
        ("warmup_calendar_days", 3),
    ):
        altered = deepcopy(payload)
        altered["profile"][field] = value
        with pytest.raises(ValueError, match="differs from the canonical"):
            scan._validate_registered_payload(altered)

    altered = deepcopy(payload)
    altered["profile"]["windows"] = [
        {
            "id": "arbitrary",
            "start": "2023-06-01T00:00:00-04:00",
            "end_exclusive": "2023-07-01T00:00:00-04:00",
        }
    ]
    with pytest.raises(ValueError, match="differs from the canonical"):
        scan._validate_registered_payload(altered)

    altered = deepcopy(payload)
    altered["profile"]["protocols"]["group5"] = "configs/old-entry.json"
    with pytest.raises(ValueError, match="differs from the canonical"):
        scan._validate_registered_payload(altered)

    for mutate in (
        lambda value: value["profile"]["blind_case_sampling"].update(
            selection_method="random_after_results"
        ),
        lambda value: value["profile"]["blind_case_sampling"][
            "categories"
        ].append("unregistered_case"),
        lambda value: value["profile"]["mature_range_target"].update(
            semantic_name="GenericDealingRange"
        ),
        lambda value: value["profile"]["stopping_rules"].append(
            "keep tuning until profitable"
        ),
    ):
        altered = deepcopy(payload)
        mutate(altered)
        with pytest.raises(ValueError, match="differs from the canonical"):
            scan._validate_registered_payload(altered)


def test_eye_builder_enables_only_lightweight_typed_observer() -> None:
    payload = scan._registered_payload()
    scan._validate_registered_payload(payload)

    _, observer = scan._build_eye(payload)

    assert observer.config.eye_authority_mode is True
    assert observer.config.project_scene_graph is False
    assert observer.config.materialize_event_view is False
    assert observer.config.group4_projection_only is False
    assert observer._displacement_eye is not None
    assert observer._group3_tracker is not None
    assert observer._group4_tracker is not None
    assert observer._group5_reducer is not None


def test_macos_authority_scan_rejects_rosetta_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scan.sys, "platform", "darwin")
    monkeypatch.setattr(scan.platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError, match="native arm64 Python"):
        scan._runtime_environment()

    monkeypatch.setattr(scan.platform, "machine", lambda: "arm64")
    runtime = scan._runtime_environment()
    assert runtime["machine"] == "arm64"
    assert runtime["native_arm64_required"] is True


def test_checkpoint_resume_matches_uninterrupted_partial_smoke(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _small_loaded_source()
    monkeypatch.setattr(scan, "load_ohlcv", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(
        scan,
        "_run_identity",
        lambda payload, **kwargs: _fake_identity(),
    )
    resumed_output = tmp_path / "resumed"
    baseline_output = tmp_path / "baseline"

    interrupted = scan.run_scan(
        output=resumed_output,
        max_bars=12,
        stop_after_bars=5,
        force=True,
    )

    assert interrupted is None
    assert (resumed_output / "checkpoint.pkl").is_file()
    interrupted_progress = json.loads(
        (resumed_output / "progress.json").read_text(encoding="utf-8")
    )
    assert interrupted_progress["emitted_bars"] == 5
    assert interrupted_progress["in_window_bars"] == 5
    assert interrupted_progress["checkpoint_bars"] == 5
    assert interrupted_progress["complete"] is False
    assert not (resumed_output / ".progress.json.tmp").exists()
    assert not (resumed_output / "checkpoint.json").exists()
    encoded = (resumed_output / "checkpoint.pkl").read_bytes()
    assert encoded.startswith(scan._CHECKPOINT_MAGIC)
    checksum, separator, payload = encoded[
        len(scan._CHECKPOINT_MAGIC) :
    ].partition(b"\n")
    assert separator == b"\n"
    assert checksum.decode("ascii") == hashlib.sha256(payload).hexdigest()
    assert not (resumed_output / "summary.json").exists()
    assert not (resumed_output / "case_index.json").exists()

    with pytest.raises(RuntimeError, match="identity or switches disagree"):
        scan.run_scan(output=resumed_output, max_bars=13)

    checkpoint_pickle = resumed_output / "checkpoint.pkl"
    original_checkpoint = checkpoint_pickle.read_bytes()
    checkpoint_pickle.write_bytes(
        original_checkpoint[:-1]
        + bytes((original_checkpoint[-1] ^ 0x01,))
    )
    with pytest.raises(RuntimeError, match="pickle integrity failed"):
        scan.run_scan(output=resumed_output, max_bars=12)
    checkpoint_pickle.write_bytes(original_checkpoint)

    resumed = scan.run_scan(output=resumed_output, max_bars=12)
    baseline = scan.run_scan(
        output=baseline_output,
        max_bars=12,
        force=True,
    )

    assert resumed == baseline
    assert resumed is not None
    assert resumed["scan_status"]["complete"] is False
    assert resumed["scan_status"]["scan_completed"] is False
    assert resumed["scan_status"]["evidence_integrity_passed"] is False
    assert resumed["scan_status"]["partial_reason"] == "max_bars_smoke"
    assert resumed["scan_status"]["emitted_completed_1m_bars"] == 12
    assert resumed["run_manifest"]["runtime_switches"] == scan.RUNTIME_SWITCHES
    assert not (resumed_output / "checkpoint.pkl").exists()
    assert not (resumed_output / "checkpoint.json").exists()
    assert (resumed_output / "summary.json").is_file()
    assert (resumed_output / "case_index.json").is_file()
    resumed_progress = json.loads(
        (resumed_output / "progress.json").read_text(encoding="utf-8")
    )
    assert resumed_progress["emitted_bars"] == 12
    assert resumed_progress["in_window_bars"] == 12
    assert resumed_progress["checkpoint_bars"] == 5
    assert resumed_progress["complete"] is False
    assert not tuple(resumed_output.glob("*.tmp"))


def test_max_bars_limits_only_in_window_observations(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _small_loaded_source_with_warmup()
    monkeypatch.setattr(scan, "load_ohlcv", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(
        scan,
        "_run_identity",
        lambda payload, **kwargs: _fake_identity(),
    )

    summary = scan.run_scan(
        output=tmp_path / "window-limited",
        max_bars=5,
        force=True,
    )

    assert summary is not None
    assert summary["scan_status"]["emitted_completed_1m_bars"] == 9
    assert summary["scan_status"]["in_window_observations"] == 5
    assert summary["scan_status"]["max_in_window_bars"] == 5
    assert summary["window"]["observations"] == 5
    assert summary["authority_contract"]["mature_range_target"] == (
        scan.EXPECTED_MATURE_RANGE_TARGET
    )


def test_max_bars_wins_over_same_bar_simulated_stop(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _small_loaded_source_with_warmup()
    monkeypatch.setattr(scan, "load_ohlcv", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(
        scan,
        "_run_identity",
        lambda payload, **kwargs: _fake_identity(),
    )
    output = tmp_path / "same-bar-cap"

    summary = scan.run_scan(
        output=output,
        max_bars=5,
        stop_after_bars=9,
        force=True,
    )

    assert summary is not None
    assert summary["scan_status"]["in_window_observations"] == 5
    assert summary["scan_status"]["emitted_completed_1m_bars"] == 9
    assert not (output / "checkpoint.pkl").exists()


def test_formal_identity_requires_clean_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scan, "_git_commit", lambda: "0" * 40)
    monkeypatch.setattr(
        scan,
        "_git_worktree_status",
        lambda: " M smc_trader/group4.py",
    )

    with pytest.raises(RuntimeError, match="requires a clean Git worktree"):
        scan._git_identity(require_clean=True)
    assert scan._git_identity(require_clean=False) == {
        "commit": "0" * 40,
        "clean": False,
        "dirty_runtime_allowed": True,
    }


def _integrity_summary(*, observations: int = 5) -> dict[str, object]:
    return {
        "window": {"observations": observations},
        "group3": {
            "order_block_admission": {
                "denominator_status": "producer_exposed",
                "conservation": {"balanced": True},
            }
        },
        "group4": {
            "range_formation_funnel": {
                "denominator_status": "producer_exposed",
                "conservation": {"balanced": True},
            },
            "source_disposition_conservation": {"balanced": True},
            "manipulation_conservation": {
                "conservation": {"balanced": True}
            },
            "selected_primary_to_episode_join": {"balanced": True},
        },
        "group5": {
            "denominator_status": "producer_exposed",
            "manipulation_to_path_identity_funnel": [],
            "qualified_zone_to_terminal_identity_funnel": [],
            "exact_identity_conservation": {
                "exact_identity_conserved": True,
                "exact_identity_violation_count": 0,
            },
            "path_order_errors": 0,
            "favr_observation_chain": {
                "status": "authoritative_identity_join_exposed_no_root_cases",
                "identity_violations": {},
                "authoritative_join_not_exposed": {},
            },
        },
        "denominators": [
            {
                "group": "group4",
                "primitive": "manipulation_source",
                "name": "raw_crossed_source",
                "denominator_status": "producer_exposed",
            }
        ],
        "funnels": [{"conservation": {"balanced": True}}],
        "case_selection": {
            "method": scan.EXPECTED_CASE_SELECTION_METHOD,
            "future_or_pnl_used": False,
            "frozen_strata": list(scan.EXPECTED_CASE_CATEGORIES),
            "selected": 20,
        },
        "case_index": [
            {
                "case_id": f"case-{index}",
                "stratum": scan.EXPECTED_CASE_CATEGORIES[
                    index % len(scan.EXPECTED_CASE_CATEGORIES)
                ],
            }
            for index in range(20)
        ],
    }


def test_registered_tail_and_integrity_checks_fail_closed() -> None:
    start = pd.Timestamp("2023-01-03 09:30", tz="America/New_York")
    end = pd.Timestamp("2023-01-03 09:33", tz="America/New_York")
    expected_tail = scan._expected_last_observation_asof(
        start=start,
        end_exclusive=end,
        timezone="America/New_York",
    )
    assert expected_tail == end

    checks = scan._evidence_integrity_checks(
        _integrity_summary(),
        in_window_bars=5,
        last_observation_asof=expected_tail,
        expected_last_observation_asof=expected_tail,
    )
    assert checks and all(checks.values())

    broken = _integrity_summary()
    broken["group4"]["range_formation_funnel"]["conservation"][
        "balanced"
    ] = False
    checks = scan._evidence_integrity_checks(
        broken,
        in_window_bars=5,
        last_observation_asof=expected_tail,
        expected_last_observation_asof=expected_tail,
    )
    assert checks["range_conserved"] is False

    insufficient = _integrity_summary()
    insufficient["case_index"] = insufficient["case_index"][:19]
    insufficient["case_selection"]["selected"] = 19
    checks = scan._evidence_integrity_checks(
        insufficient,
        in_window_bars=5,
        last_observation_asof=expected_tail,
        expected_last_observation_asof=expected_tail,
    )
    assert checks["selected_case_count_consistent"] is False


def test_full_scan_integrity_failure_is_not_permanent_evidence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _small_loaded_source()
    monkeypatch.setattr(scan, "load_ohlcv", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(
        scan,
        "_run_identity",
        lambda payload, **kwargs: _fake_identity(),
    )
    monkeypatch.setattr(
        scan,
        "_evidence_integrity_checks",
        lambda *args, **kwargs: {"forced_failure": False},
    )
    output = tmp_path / "failed-full"

    with pytest.raises(RuntimeError, match="evidence integrity failed"):
        scan.run_scan(output=output, force=True)

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["scan_status"]["scan_completed"] is True
    assert summary["scan_status"]["evidence_integrity_passed"] is False
    assert summary["scan_status"]["complete"] is False
    progress = json.loads(
        (output / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["complete"] is False
