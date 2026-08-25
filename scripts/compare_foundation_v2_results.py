#!/usr/bin/env python3
"""Create the fixed open-data legacy-versus-Foundation-v2 comparison receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping

sys.dont_write_bytecode = True

class ComparisonError(RuntimeError):
    pass

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)

def find_root() -> Path:
    script = Path(__file__).resolve()
    candidates = [script.parents[1], Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for candidate in dict.fromkeys(candidates):
        if all(
            (candidate / item).exists()
            for item in ("pyproject.toml", "smc_trader", "experiments")
        ):
            return candidate
    raise ComparisonError("cannot resolve repository root")

ROOT = find_root()
SCRIPT = Path(__file__).resolve()
FINAL_SCRIPT = "scripts/compare_foundation_v2_results.py"
FINAL_OUTPUT = (
    "experiments/results/foundation_v2_2024_06_vs_historical_comparison_v1.json"
)
FOUNDATION_VERSION = "smc_semantic_foundation_v2.0"
FOUNDATION_IDENTITY = (
    "ac04636919931d774309a0c306764fdf8eb53aee41df0f31d4d94e5b9125732b"
)
MAX_OBJECT_BYTES = 2 * 1024 * 1024
MAX_JSONL_BYTES = 8 * 1024 * 1024
MAX_JSONL_ROWS = 10_000
P6_SAME_SOURCE_BINDINGS = (
    "mbo_feature_artifact",
    "mbo_feature_manifest",
    "ohlcv_artifact",
    "ohlcv_manifest",
    "raw_mbo_partition_manifest",
    "split_registry",
)

P45_STAGES = (
    "E1_level_touch",
    "E2_sweep",
    "E3_sweep_displacement",
    "E4_sweep_displacement_mss",
    "E5_plus_fvg",
    "E6_plus_parent_alignment",
)
P45_CONTROLS = (
    "quiet_zero_event",
    "same_session_non_sweep_touch",
    "pseudo_level_touch",
    "forward_time_shift",
)
P6_FAMILY = (
    "sweep_rejection",
    "acceptance_continuation",
    "displacement_impact",
    "mss_flow_shift",
    "fvg_retest_response",
)
WINDOWS = {
    "W1": {
        "id": "2024-06-week-1",
        "start": "2024-06-02T22:00:00Z",
        "end_exclusive": "2024-06-07T21:01:00Z",
        "role": "development_comparison",
        "registered": 6900,
        "real": 6899,
        "synthetic": 1,
    },
    "W2": {
        "id": "2024-06-week-2",
        "start": "2024-06-09T22:00:00Z",
        "end_exclusive": "2024-06-14T21:01:00Z",
        "role": "historical_validation_comparison",
        "registered": 6900,
        "real": 6899,
        "synthetic": 1,
    },
}

# name: result, result SHA (old only), manifest, manifest SHA (old only), kind, window
OLD = {
    "phase45_january": (
        "experiments/results/smc_semantics_v1_2_2024_01_phase5_diagnostic_v3_r2.json",
        "378379df57b9efebc1ed134bfb9382aee20161088d7eefee997cf5048551ac64",
        "experiments/manifests/smc_semantics_v1_2_2024_01_phase5_diagnostic_v3_r2.yaml",
        "830eed086ccc0cfd2d16245437583d80523f3dcbb2155f72aa0fe1ac8719052c",
        "p45",
        None,
    ),
    "phase6_w1": (
        "experiments/results/smc_semantics_v1_2_2024_06_phase6_mbo_week1_v5.json",
        "30829d4c39c7d32b04ccf5aa575407433691df269a1526db59af3f3ab580f3f7",
        "experiments/manifests/smc_semantics_v1_2_2024_06_phase6_mbo_week1_v5.yaml",
        "77e5c43a157fc9aa5ac27a1f209daa869e07d649284630a186b3a50b71c689c4",
        "p6",
        "W1",
    ),
    "phase6_w2_pooled": (
        "experiments/results/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.json",
        "98f0f334cbae050a093bebca4cfbb85fcc877e96ba729fe04ea477e2761ddf99",
        "experiments/manifests/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.yaml",
        "99901b1893cdea71615239fd9c536a318ec3c8088f0c491e48cb710b9d0eeafc",
        "p6",
        "W2",
    ),
}
NEW = {
    "phase45_w1": (
        "experiments/results/foundation_v2_2024_06_phase45_w1_development_comparison_v1.json",
        "experiments/manifests/foundation_v2_2024_06_phase45_w1_development_comparison_v1.yaml",
        "p45",
        "W1",
    ),
    "phase45_w2": (
        "experiments/results/foundation_v2_2024_06_phase45_w2_historical_validation_comparison_v1.json",
        "experiments/manifests/foundation_v2_2024_06_phase45_w2_historical_validation_comparison_v1.yaml",
        "p45",
        "W2",
    ),
    "phase6_w1": (
        "experiments/results/foundation_v2_2024_06_phase6_mbo_w1_development_comparison_v1.json",
        "experiments/manifests/foundation_v2_2024_06_phase6_mbo_w1_development_comparison_v1.yaml",
        "p6",
        "W1",
    ),
    "phase6_w2": (
        "experiments/results/foundation_v2_2024_06_phase6_mbo_w2_historical_validation_comparison_v1.json",
        "experiments/manifests/foundation_v2_2024_06_phase6_mbo_w2_historical_validation_comparison_v1.yaml",
        "p6",
        "W2",
    ),
}
NEW_MANIFEST_SHA256 = {
    "phase45_w1": "4d03649ceaea9a337fb8a95ed586c80e8b735f763c57cba8c635c73860d5bbc6",
    "phase45_w2": "945f12fa366c982d1430ddc596556e990371a505142dcbaf10618e1efd7317f4",
    "phase6_w1": "1b832e72084684735bfe95b827c83282d0285ecdc1a5bdd83643036019b68b98",
    "phase6_w2": "4c6595c19bab5ed1d547edc4306e8c0647413242894ce7ffbf497df44be22584",
}
W2_PAIRS = (
    "experiments/results/"
    "smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.matched_pairs.jsonl"
)
W2_PAIRS_SHA = "f312e21f33df2bf1f72eac6d06ae33468ff07e32e7376a97ded257905969e73a"
EVALUATOR = "smc_trader/mbo_mechanism_research.py"
P45_IDENTITY_EVALUATOR = "smc_trader/signal_research.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from smc_trader.mbo_mechanism_research import (  # noqa: E402
    PHASE6_FIXED_FAMILY,
    canonical_identity as canonical_p6_identity,
    evaluate_fixed_mechanism_family,
)
from smc_trader.signal_research import (  # noqa: E402
    canonical_result_identity as canonical_p45_identity,
)

def repo_file(relative: str, label: str) -> Path:
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe {label} path")
    unresolved = ROOT / rel
    require(not unresolved.is_symlink(), f"{label} must not be a symlink")
    path = unresolved.resolve(strict=False)
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ComparisonError(f"{label} escapes repository") from error
    require(path.is_file(), f"missing {label}: {relative}")
    return path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

def snapshot(path: Path, label: str, maximum: int) -> tuple[bytes, str]:
    """Read one bounded regular-file snapshot used for both hash and parsing."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            require(stat.S_ISREG(before.st_mode), f"{label} must be regular")
            require(before.st_size <= maximum, f"{label} exceeds byte limit")
            raw = handle.read(maximum + 1)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise ComparisonError(f"cannot read {label}: {path}") from error
    require(len(raw) <= maximum, f"{label} exceeds byte limit")
    require(len(raw) == before.st_size, f"{label} size changed while reading")
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"{label} changed while reading",
    )
    return raw, hashlib.sha256(raw).hexdigest()

def no_constant(value: str) -> None:
    raise ComparisonError(f"non-finite JSON constant: {value}")

def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = child
    return value

def parse_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=no_duplicate_keys,
            parse_constant=no_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ComparisonError(f"cannot parse {label}") from error
    require(isinstance(value, dict), f"{label} must be an object")
    return value

def snapshot_object(path: Path, label: str) -> tuple[dict[str, Any], str]:
    raw, observed_sha = snapshot(path, label, MAX_OBJECT_BYTES)
    return parse_object(raw, label), observed_sha

def parse_jsonl(raw: bytes, label: str, expected_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = raw.decode("utf-8").splitlines()
        require(len(lines) <= MAX_JSONL_ROWS, f"{label} exceeds row limit")
        require(len(lines) == expected_rows, f"{label} row drift")
        for line_no, line in enumerate(lines, 1):
            require(bool(line.strip()), f"blank {label} row {line_no}")
            value = json.loads(line, object_pairs_hook=no_duplicate_keys, parse_constant=no_constant)
            require(isinstance(value, dict), f"invalid {label} row {line_no}")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ComparisonError(f"cannot parse {label}") from error
    return rows

def snapshot_jsonl(path: Path, label: str, expected_rows: int) -> tuple[list[dict[str, Any]], str]:
    raw, observed_sha = snapshot(path, label, MAX_JSONL_BYTES)
    return parse_jsonl(raw, label, expected_rows), observed_sha

def utc(value: Any) -> Any:
    return value.replace("+00:00", "Z") if isinstance(value, str) else value

def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)

def p6_source_bindings(name: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    identities = manifest.get("identity_bindings", {})
    output = {}
    for key in P6_SAME_SOURCE_BINDINGS:
        binding = identities.get(key, {})
        require(isinstance(binding.get("path"), str) and is_sha256(binding.get("sha256")), f"{name} {key} binding drift")
        output[key] = {"path": binding["path"], "sha256": binding["sha256"]}
    return output

def verify_binding(
    name: str,
    result_path: str,
    manifest_path: str,
    kind: str,
    result_sha: str | None = None,
    manifest_sha: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    result_file = repo_file(result_path, f"{name} result")
    manifest_file = repo_file(manifest_path, f"{name} manifest")
    result, observed_result_sha = snapshot_object(result_file, f"{name} result")
    manifest, observed_manifest_sha = snapshot_object(manifest_file, f"{name} manifest")
    if result_sha is not None:
        require(observed_result_sha == result_sha, f"{name} result SHA drift")
    if manifest_sha is not None:
        require(observed_manifest_sha == manifest_sha, f"{name} manifest SHA drift")
    require(result.get("manifest_path") == manifest_path, f"{name} manifest path drift")
    require(
        result.get("manifest_sha256") == observed_manifest_sha,
        f"{name} manifest binding drift",
    )
    identity = canonical_p45_identity(result) if kind == "p45" else canonical_p6_identity(result)
    require(result.get("result_identity") == identity, f"{name} result identity drift")
    require(result.get("semantic_version") == "smc_semantics_v1.2", f"{name} semantic version drift")
    return result, manifest, observed_result_sha, observed_manifest_sha

def verify_p45(
    name: str,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    window_key: str | None,
) -> None:
    require(result.get("schema_version") == 3, f"{name} schema drift")
    require(result.get("research_protocol_version") == 3, f"{name} protocol drift")
    require(tuple(result.get("nested_chain", {})) == P45_STAGES, f"{name} stage family drift")
    require(set(result.get("control_comparisons", {})) == set(P45_CONTROLS), f"{name} controls drift")
    coverage = result.get("coverage", {})
    if window_key is None:
        require(result.get("validation_state") == "development_diagnostic_unvalidated", f"{name} state drift")
        require(result.get("split_authority", {}).get("role") == "brain_calibration_trial", f"{name} role drift")
        require(
            (coverage.get("diagnostic_completed_bars"), coverage.get("diagnostic_real_rows"), coverage.get("diagnostic_synthetic_bars"))
            == (30479, 30477, 2),
            f"{name} January census drift",
        )
        return
    window = WINDOWS[window_key]
    require(result.get("status") == manifest.get("status"), f"{name} status drift")
    require(result.get("authority") == manifest.get("authority"), f"{name} authority drift")
    require(result.get("validation_state") == "fixed_historical_comparison_unvalidated", f"{name} state drift")
    split = result.get("split_authority", {})
    require((split.get("window_id"), split.get("comparison_role")) == (window["id"], window["role"]), f"{name} window drift")
    classification = result.get("artifact_classification", {})
    require(
        (classification.get("complete_registered_window"), classification.get("max_bars_smoke_limit"),
         classification.get("truncated_by_max_bars"), classification.get("inference_allowed"),
         classification.get("parameter_change_allowed"), classification.get("model_action_allowed"),
         classification.get("max_bars_artifact_is_permanently_incomplete"))
        == (True, None, False, False, False, False, False), f"{name} classification drift")
    expected = manifest.get("input_census", {})
    for field in (
        "emitted_bars_including_warmup", "diagnostic_completed_bars", "diagnostic_real_rows",
        "diagnostic_ready_real_rows", "diagnostic_synthetic_bars", "warmup_data_gap_resets",
        "diagnostic_data_gap_resets", "contract_changes",
    ):
        require(coverage.get(field) == expected.get(f"expected_{field}"), f"{name} {field} drift")
    for field in ("first_diagnostic_asof", "last_processed_asof", "last_diagnostic_asof"):
        require(utc(coverage.get(field)) == utc(expected.get(f"expected_{field}")), f"{name} {field} drift")
    require(tuple(map(utc, coverage.get("synthetic_clocks", ()))) == tuple(map(utc, expected.get("expected_synthetic_clocks", ()))), f"{name} synthetic clocks drift")
    require(coverage.get("contracts") == expected.get("expected_contracts"), f"{name} contracts drift")
    require((coverage.get("diagnostic_completed_bars"), coverage.get("diagnostic_real_rows"), coverage.get("diagnostic_synthetic_bars")) == (window["registered"], window["real"], window["synthetic"]), f"{name} June census drift")
    period = manifest.get("diagnostic_period", {})
    require((period.get("start"), period.get("end_exclusive")) == (window["start"], window["end_exclusive"]), f"{name} dates drift")
    require((manifest.get("foundation_version"), manifest.get("foundation_registry_identity")) == (FOUNDATION_VERSION, FOUNDATION_IDENTITY), f"{name} Foundation drift")

def verify_p6(
    name: str,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    window_key: str,
    comparison: bool,
) -> None:
    window = WINDOWS[window_key]
    coverage = result.get("coverage", {})
    census = coverage.get("reader_active_census", {})
    require(result.get("engineering_status") == "pass", f"{name} engineering failed")
    require(result.get("raw_partition_hashes_verified") is True, f"{name} raw hashes unverified")
    source_bindings = p6_source_bindings(name, manifest)
    feature_binding = source_bindings["mbo_feature_artifact"]
    require((result.get("feature_artifact"), result.get("feature_artifact_sha256")) == (feature_binding["path"], feature_binding["sha256"]), f"{name} feature binding drift")
    require(
        (coverage.get("feature_rows"), coverage.get("ohlcv_real_source_rows"), census.get("completed"), census.get("real"), census.get("synthetic"))
        == (6900, 6899, 6900, 6899, 1),
        f"{name} census drift",
    )
    require(tuple(result.get("holm", {}).get("family_order", ())) == P6_FAMILY, f"{name} family drift")
    require(set(result.get("mechanisms", {})) == set(P6_FAMILY), f"{name} mechanisms drift")
    matching = coverage.get("primary_matching", {})
    require(set(matching) == set(P6_FAMILY), f"{name} matching family drift")
    current_primary = sum(int(matching[item]["packed"]) for item in P6_FAMILY)
    total_primary = coverage.get("primary_fixed_holm_pair_rows")
    require(total_primary == coverage.get("matched_pair_rows") - coverage.get("pseudo_zone_descriptive_pair_rows"), f"{name} pair census drift")
    require(current_primary <= total_primary, f"{name} current pairs exceed total")
    require(coverage.get("episode_rows") == coverage.get("prior_week1_episode_rows") + coverage.get("current_week_episode_rows"), f"{name} episode decomposition drift")
    require(coverage.get("matched_pair_rows") == coverage.get("prior_week1_matched_pair_rows") + coverage.get("current_week_matched_pair_rows"), f"{name} matched-pair decomposition drift")
    if coverage.get("prior_week1_matched_pair_rows") == 0:
        require(current_primary == total_primary, f"{name} unpooled primary census drift")
    active = result.get("active_window", {})
    require(
        (active.get("id"), utc(active.get("start")), utc(active.get("end_exclusive")))
        == (window["id"], window["start"], window["end_exclusive"]),
        f"{name} active window drift",
    )
    if not comparison:
        return
    require(result.get("status") == "phase6_foundation_v2_comparison_complete_no_admission", f"{name} status drift")
    require(result.get("comparison_validation_only") is True, f"{name} not validation-only")
    require(result.get("phase7_evidence_allowlist") == [], f"{name} admitted evidence")
    require(set(result.get("phase7_excluded_mechanisms", ())) == set(P6_FAMILY), f"{name} exclusions drift")
    supported = [item for item in P6_FAMILY if result["mechanisms"][item]["status"] == "supported"]
    require(result.get("comparison_supported_mechanisms_not_admitted") == supported, f"{name} non-admission receipt drift")
    require((coverage.get("prior_week1_episode_rows"), coverage.get("prior_week1_matched_pair_rows")) == (0, 0), f"{name} pooled prior W1")
    require(current_primary == total_primary, f"{name} primary pairs are not current-only")
    require(result.get("comparison_contract", {}).get("comparison_role") == window["role"], f"{name} role drift")
    require((manifest.get("foundation_version"), manifest.get("foundation_registry_identity")) == (FOUNDATION_VERSION, FOUNDATION_IDENTITY), f"{name} Foundation drift")

def load_old() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, (result_path, result_sha, manifest_path, manifest_sha, kind, window) in OLD.items():
        result, manifest, observed_result_sha, observed_manifest_sha = verify_binding(name, result_path, manifest_path, kind, result_sha, manifest_sha)
        if kind == "p45":
            verify_p45(name, result, manifest, window)
        else:
            verify_p6(name, result, manifest, str(window), False)
        values[name] = {"result": result, "manifest": manifest, "result_sha256": observed_result_sha, "manifest_sha256": observed_manifest_sha}
    values["w2_reconstruction"] = reconstruct_w2(
        values["phase6_w2_pooled"]["result"],
        values["phase6_w2_pooled"]["manifest"],
    )
    verify_w2_decomposition(values["phase6_w1"]["result"], values["phase6_w2_pooled"]["result"], values["w2_reconstruction"])
    return values

def load_new() -> dict[str, Any]:
    missing = [spec[0] for spec in NEW.values() if not (ROOT / spec[0]).is_file()]
    require(not missing, "required new result(s) missing: " + ", ".join(missing))
    missing_manifests = [spec[1] for spec in NEW.values() if not (ROOT / spec[1]).is_file()]
    require(not missing_manifests, "required new manifest(s) missing: " + ", ".join(missing_manifests))
    unresolved = [name for name, value in NEW_MANIFEST_SHA256.items() if not is_sha256(value)]
    require(not unresolved, "pin NEW manifest SHA after source commit: " + ", ".join(unresolved))
    values: dict[str, Any] = {}
    for name, (result_path, manifest_path, kind, window) in NEW.items():
        result, manifest, observed_result_sha, observed_manifest_sha = verify_binding(
            name, result_path, manifest_path, kind, manifest_sha=NEW_MANIFEST_SHA256[name])
        if kind == "p45":
            verify_p45(name, result, manifest, window)
        else:
            verify_p6(name, result, manifest, window, True)
        values[name] = {"result": result, "manifest": manifest, "result_sha256": observed_result_sha, "manifest_sha256": observed_manifest_sha}
    return values

def reconstruct_w2(old: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    require(tuple(PHASE6_FIXED_FAMILY) == P6_FAMILY, "current evaluator family drift")
    path = repo_file(W2_PAIRS, "old W2 matched-pairs ledger")
    meta = old.get("ledgers", {}).get("matched_pairs", {})
    require((meta.get("path"), meta.get("sha256")) == (W2_PAIRS, W2_PAIRS_SHA), "old W2 ledger binding drift")
    require(isinstance(meta.get("rows"), int) and meta["rows"] == old["coverage"]["matched_pair_rows"], "old W2 ledger rows missing")
    ledger, observed_sha = snapshot_jsonl(path, "old W2 matched-pairs ledger", meta["rows"])
    require(observed_sha == W2_PAIRS_SHA, "old W2 matched-pairs SHA drift")
    for row in ledger:
        require(row.get("treatment_study_week") == row.get("control_study_week") in {"week_1", "week_2"}, "old W2 ledger week drift")
    rows = {name: [] for name in P6_FAMILY}
    for row in ledger:
        if row.get("comparison") != "primary_fixed_holm":
            continue
        hypothesis = row.get("hypothesis")
        require(hypothesis in rows, "old W2 ledger hypothesis drift")
        rows[str(hypothesis)].append(row)
    kwargs = {
        "minimum_matched": manifest["minimum_matched_episodes"],
        "alpha": manifest["inference"]["alpha"],
        "bootstrap_replicates": manifest["inference"]["bootstrap_replicates"],
        "bootstrap_seed": manifest["inference"]["bootstrap_seed"],
        "stability_stratum_minimum_n": manifest["support_rule"]["stability_stratum_minimum_n"],
    }
    pooled = evaluate_fixed_mechanism_family(rows, **kwargs)
    for key in ("mechanisms", "holm", "phase7_evidence_allowlist", "phase7_excluded_mechanisms", "extension_required", "causal_claim"):
        require(pooled[key] == old[key], f"historical pooled evaluator drift: {key}")
    w2_rows = {
        name: [row for row in values if row.get("treatment_study_week") == "week_2" and row.get("control_study_week") == "week_2"]
        for name, values in rows.items()
    }
    rebuilt = evaluate_fixed_mechanism_family(w2_rows, **kwargs)
    pair_counts = {name: len(values) for name, values in w2_rows.items()}
    require(all(rebuilt["mechanisms"][name]["primary_matched_n"] == pair_counts[name] for name in P6_FAMILY), "W2-only pair census drift")
    current_matching = {name: int(old["coverage"]["primary_matching"][name]["packed"]) for name in P6_FAMILY}
    require(pair_counts == current_matching, "W2-only reconstruction disagrees with coverage matching")
    w2_all = [row for row in ledger if row.get("treatment_study_week") == "week_2"]
    require(len(w2_all) == old["coverage"]["current_week_matched_pair_rows"], "W2-only total pair census drift")
    pair_ids = {
        (str(row["hypothesis"]), str(row["treatment_episode_id"]), str(row["control_episode_id"]))
        for values in w2_rows.values() for row in values
    }
    require(len(pair_ids) == sum(pair_counts.values()), "W2-only duplicate pair identity")
    return {
        "pooled_reproduction_verified": True,
        "filter": {"comparison": "primary_fixed_holm", "treatment_study_week": "week_2", "control_study_week": "week_2"},
        "inference_parameters": kwargs,
        "pair_counts": pair_counts,
        "w2_all_pair_rows": len(w2_all),
        "w2_pseudo_pair_rows": len(w2_all) - sum(pair_counts.values()),
        "_pair_identities": frozenset(pair_ids),
        "mechanisms": rebuilt["mechanisms"],
        "holm": rebuilt["holm"],
    }

def verify_w2_decomposition(w1: Mapping[str, Any], pooled: Mapping[str, Any], rebuilt: Mapping[str, Any]) -> None:
    w1_coverage, pooled_coverage = w1["coverage"], pooled["coverage"]
    require(pooled_coverage["prior_week1_episode_rows"] == w1_coverage["episode_rows"], "pooled prior W1 episode census drift")
    require(pooled_coverage["prior_week1_matched_pair_rows"] == w1_coverage["matched_pair_rows"], "pooled prior W1 pair census drift")
    for name in P6_FAMILY:
        require(pooled["mechanisms"][name]["primary_matched_n"] == w1["mechanisms"][name]["primary_matched_n"] + rebuilt["pair_counts"][name], f"pooled {name} decomposition drift")
    require(pooled_coverage["primary_fixed_holm_pair_rows"] == w1_coverage["primary_fixed_holm_pair_rows"] + sum(rebuilt["pair_counts"].values()), "pooled primary decomposition drift")
    require(pooled_coverage["pseudo_zone_descriptive_pair_rows"] == w1_coverage["pseudo_zone_descriptive_pair_rows"] + rebuilt["w2_pseudo_pair_rows"], "pooled pseudo decomposition drift")

def primary_pair_identities(result: Mapping[str, Any], label: str, week: str) -> frozenset[tuple[str, str, str]]:
    meta = result.get("ledgers", {}).get("matched_pairs", {})
    require(isinstance(meta.get("path"), str) and is_sha256(meta.get("sha256")) and isinstance(meta.get("rows"), int) and meta["rows"] == result["coverage"]["matched_pair_rows"], f"{label} pair ledger metadata drift")
    rows, observed_sha = snapshot_jsonl(repo_file(meta["path"], f"{label} pair ledger"), f"{label} pair ledger", meta["rows"])
    require(observed_sha == meta["sha256"], f"{label} pair ledger SHA drift")
    require(all(row.get("treatment_study_week") == row.get("control_study_week") == week for row in rows), f"{label} pair ledger week drift")
    identities = frozenset(
        (str(row["hypothesis"]), str(row["treatment_episode_id"]), str(row["control_episode_id"]))
        for row in rows if row.get("comparison") == "primary_fixed_holm"
    )
    expected = sum(int(result["mechanisms"][name]["primary_matched_n"]) for name in P6_FAMILY)
    require(len(identities) == expected, f"{label} primary pair identity drift")
    return identities

def pair_identity_audit(old: frozenset[Any], new: frozenset[Any]) -> dict[str, Any]:
    shared = len(old & new)
    return {
        "legacy_pair_ids": len(old), "foundation_v2_pair_ids": len(new), "shared_pair_ids": shared,
        "exact_pair_identity_sets_equal": old == new,
        "comparability": "not_same_pair_identity_set" if old != new else "same_ids_but_no_registered_pairwise_delta_estimand",
        "pairwise_delta_inference_authorized": False,
    }

def p45_extract(result: Mapping[str, Any]) -> dict[str, Any]:
    coverage = result["coverage"]
    real = int(coverage["diagnostic_real_rows"])
    stages = {}
    for name, value in result["nested_chain"].items():
        stages[name] = {
            key: value[key]
            for key in (
                "signals", "resolved_n", "successes", "raw_success_rate",
                "laplace_success_rate", "minimum_sample_met", "censored_n",
                "ambiguous_n", "full_horizon_outcome_n",
                "median_time_to_target_completed_bars",
                "median_time_to_invalidation_completed_bars",
            )
        }
        stages[name]["signals_per_1000_real_rows"] = 1000.0 * value["signals"] / real
    controls = {}
    for name, value in result["control_comparisons"].items():
        controls[name] = {
            "requested": value["requested"],
            "matched": value["matched"],
            "match_coverage": 0.0 if value["requested"] == 0 else value["matched"] / value["requested"],
            "paired_n": result["inference"]["exact_mcnemar"][name]["paired_n"],
            "descriptive_exact_p": result["inference"]["exact_mcnemar"][name]["descriptive_exact_p_value"],
            "holm_adjusted_p": result["inference"]["holm_fixed_family"]["adjusted_p_values"][name],
            "rejected": result["inference"]["holm_fixed_family"]["rejected"][name],
        }
    return {
        "coverage": {key: coverage[key] for key in (
            "emitted_bars_including_warmup", "diagnostic_completed_bars",
            "diagnostic_real_rows", "diagnostic_ready_real_rows", "diagnostic_synthetic_bars",
            "warmup_data_gap_resets", "diagnostic_data_gap_resets", "contract_changes",
            "first_diagnostic_asof", "last_processed_asof", "last_diagnostic_asof",
            "synthetic_clocks", "contracts", "atomic_events", "audit_events", "audit_fingerprint",
        ) if key in coverage},
        "stages": stages,
        "controls": controls,
    }

def mechanism_extract(values: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for name, value in values.items():
        effect = value["primary_effect"]
        output[name] = {
            "status": value["status"],
            "primary_matched_n": value["primary_matched_n"],
            "primary_metric": value["primary_metric"],
            "mean_effect": effect["mean_effect"],
            "ci_low": effect["ci_low"],
            "ci_high": effect["ci_high"],
            "positive": effect["positive"],
            "negative": effect["negative"],
            "ties": effect["ties"],
            "exact_sign_p": effect["exact_sign_p_value"],
            "holm_adjusted_p": value["holm_adjusted_p_value"],
            "holm_rejected": value["holm_rejected"],
            "stability_complete": value["stability_complete"],
            "systematic_sign_reversal": value["systematic_sign_reversal"],
        }
    return output

def p6_extract(
    result: Mapping[str, Any],
    mechanisms: Mapping[str, Any] | None = None,
    holm: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    coverage = result["coverage"]
    primary = sum(int(coverage["primary_matching"][name]["packed"]) for name in P6_FAMILY)
    matched = int(coverage["current_week_matched_pair_rows"])
    return {
        "window": result["active_window"],
        "coverage": {
            "episode_rows": coverage["current_week_episode_rows"],
            "matched_pair_rows": matched,
            "primary_fixed_holm_pair_rows": primary,
            "pseudo_zone_descriptive_pair_rows": matched - primary,
            "primary_matching": coverage["primary_matching"],
        },
        "mechanisms": mechanism_extract(result["mechanisms"] if mechanisms is None else mechanisms),
        "holm": result["holm"] if holm is None else holm,
    }

def deltas(old: Mapping[str, Any], new: Mapping[str, Any], pair_audit: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for name in P6_FAMILY:
        before, after = old[name], new[name]
        require(before["primary_metric"] == after["primary_metric"], f"{name} metric drift")
        output[name] = {
            "primary_metric": before["primary_metric"],
            "aggregate_matched_n_arithmetic_difference": after["primary_matched_n"] - before["primary_matched_n"],
            "aggregate_mean_effect_arithmetic_difference": None if before["mean_effect"] is None or after["mean_effect"] is None else after["mean_effect"] - before["mean_effect"],
            "status_transition": {"legacy": before["status"], "foundation_v2": after["status"], "changed": before["status"] != after["status"]},
            "holm_rejected_changed": before["holm_rejected"] != after["holm_rejected"],
            "stability_changed": before["stability_complete"] != after["stability_complete"],
        }
    return {
        "authority": {
            "estimand": "foundation_v2_aggregate_minus_legacy_aggregate",
            "independently_matched_aggregates": True,
            "pairwise_effect_delta": False,
            "delta_inference": False,
        },
        "pair_identity_audit": dict(pair_audit),
        "mechanisms": output,
    }

def source_ref(spec: tuple[Any, ...], value: Mapping[str, Any]) -> dict[str, Any]:
    result_path, manifest_path = spec[0], spec[2] if len(spec) == 6 else spec[1]
    result, manifest = value["result"], value["manifest"]
    return {
        "path": result_path,
        "sha256": value["result_sha256"],
        "result_identity": result["result_identity"],
        "manifest_path": manifest_path,
        "manifest_sha256": value["manifest_sha256"],
        "semantic_version": result["semantic_version"],
        "semantic_registry_identity": result.get("semantic_registry_identity"),
        "foundation_version": manifest.get("foundation_version"),
        "foundation_registry_identity": manifest.get("foundation_registry_identity"),
        "split_authority": result.get("split_authority"),
        "active_window": result.get("active_window"),
    }

def comparison_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def build_payload() -> dict[str, Any]:
    current = load_new()  # Formal mode always rejects missing new results first.
    require(SCRIPT == (ROOT / FINAL_SCRIPT).resolve(strict=False), f"install script at {FINAL_SCRIPT}")
    old = load_old()
    same_source_bindings, pair_audits = {}, {}
    for key in ("W1", "W2"):
        legacy = old["phase6_w1" if key == "W1" else "phase6_w2_pooled"]["result"]
        foundation = current["phase6_w1" if key == "W1" else "phase6_w2"]["result"]
        require(legacy["active_window"] == foundation["active_window"], f"Phase6 {key} window differs")
        require(legacy["feature_artifact_sha256"] == foundation["feature_artifact_sha256"], f"Phase6 {key} feature source differs")
        legacy_manifest = old["phase6_w1" if key == "W1" else "phase6_w2_pooled"]["manifest"]
        foundation_manifest = current["phase6_w1" if key == "W1" else "phase6_w2"]["manifest"]
        legacy_bindings = p6_source_bindings(f"legacy Phase6 {key}", legacy_manifest)
        require(legacy_bindings == p6_source_bindings(f"Foundation Phase6 {key}", foundation_manifest), f"Phase6 {key} exact source bindings differ")
        same_source_bindings[key] = legacy_bindings
        old_ids = primary_pair_identities(legacy, f"legacy Phase6 {key}", f"week_{1 if key == 'W1' else 2}") if key == "W1" else old["w2_reconstruction"]["_pair_identities"]
        new_ids = primary_pair_identities(foundation, f"Foundation Phase6 {key}", f"week_{1 if key == 'W1' else 2}")
        pair_audits[key] = pair_identity_audit(old_ids, new_ids)
    old_w1 = p6_extract(old["phase6_w1"]["result"])
    old_w2 = p6_extract(
        old["phase6_w2_pooled"]["result"],
        old["w2_reconstruction"]["mechanisms"],
        old["w2_reconstruction"]["holm"],
    )
    new_w1 = p6_extract(current["phase6_w1"]["result"])
    new_w2 = p6_extract(current["phase6_w2"]["result"])
    sources = {f"legacy_{name}": source_ref(spec, old[name]) for name, spec in OLD.items()}
    sources.update({f"foundation_{name}": source_ref(spec, current[name]) for name, spec in NEW.items()})
    sources["legacy_phase6_w2_matched_pairs"] = {
        "path": W2_PAIRS,
        "rows": old["phase6_w2_pooled"]["result"]["ledgers"]["matched_pairs"]["rows"],
        "sha256": W2_PAIRS_SHA,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "comparison_id": "foundation_v2_2024_06_vs_historical_comparison_v1",
        "status": "complete_open_development_comparison_no_admission",
        "authority": {
            "comparison_only": True,
            "development_data_only": True,
            "w2_historical_validation_is_oof": False,
            "w2_historical_validation_is_oos": False,
            "sealed_source_opened": False,
            "artifact_fit_allowed": False,
            "model_admission_allowed": False,
            "trading_authority": False,
        },
        "generator": {
            "path": FINAL_SCRIPT,
            "sha256": sha256(SCRIPT),
            "phase45_identity_evaluator": {
                "path": P45_IDENTITY_EVALUATOR,
                "sha256": sha256(repo_file(P45_IDENTITY_EVALUATOR, "Phase45 identity evaluator")),
            },
        },
        "evaluator": {"path": EVALUATOR, "sha256": sha256(repo_file(EVALUATOR, "evaluator"))},
        "source_artifacts": sources,
        "cohorts": {
            "phase45_january_context": {
                "start": old["phase45_january"]["manifest"]["diagnostic_period"]["start"],
                "end_exclusive": old["phase45_january"]["manifest"]["diagnostic_period"]["end_exclusive"],
                "market_timezone": "America/New_York",
                "diagnostic_completed_clocks": 30479,
                "diagnostic_real_clocks": 30477,
                "diagnostic_synthetic_clocks": 2,
                "comparison_role": "context_only_nonidentical_window",
                "direct_comparison_authorized": False,
            },
            "W1": {**WINDOWS["W1"], "market_timezone": "America/New_York"},
            "W2": {**WINDOWS["W2"], "market_timezone": "America/New_York"},
        },
        "comparison_policy": {
            "phase45_january_vs_june": "context_only_nonidentical_window_no_delta_or_improvement_claim",
            "phase6_w1": "exact_same_source_window_descriptive_comparison",
            "phase6_w2": "exact_same_source_window_using_legacy_w2_only_reconstruction",
            "pool_w1_w2": False,
            "legacy_w2_top_level_used_as_w2_baseline": False,
            "legacy_w2_pooled_reproduction_verified": True,
            "causal_language_allowed": False,
            "support_does_not_equal_admission": True,
            "exact_same_source_bindings": same_source_bindings,
            "legacy_w2_reconstruction": {key: old["w2_reconstruction"][key] for key in ("filter", "inference_parameters", "pair_counts", "w2_all_pair_rows", "w2_pseudo_pair_rows")},
        },
        "phase45": {
            "comparison_mode": "context_only_nonidentical_windows",
            "direct_comparison_authorized": False,
            "legacy_january_context": p45_extract(old["phase45_january"]["result"]),
            "foundation_w1": p45_extract(current["phase45_w1"]["result"]),
            "foundation_w2": p45_extract(current["phase45_w2"]["result"]),
        },
        "phase6": {
            "W1_same_window": {"same_source_window_verified": True, "legacy": old_w1, "foundation_v2": new_w1, "deltas": deltas(old_w1["mechanisms"], new_w1["mechanisms"], pair_audits["W1"])},
            "W2_same_window": {"same_source_window_verified": True, "legacy_w2_only_reconstructed": old_w2, "foundation_v2": new_w2, "deltas": deltas(old_w2["mechanisms"], new_w2["mechanisms"], pair_audits["W2"])},
        },
        "admission_boundary": {
            "comparison_supported_mechanisms_not_admitted": {key: current[f"phase6_{key.lower()}"]["result"]["comparison_supported_mechanisms_not_admitted"] for key in ("W1", "W2")},
            "phase7_evidence_allowlist": {key: current[f"phase6_{key.lower()}"]["result"]["phase7_evidence_allowlist"] for key in ("W1", "W2")},
            "comparison_admitted_artifacts": [],
            "comparison_authorized_trade_intents": [],
            "production_state": "unchanged_neutral_fail_closed",
        },
        "limitations": [
            "January Phase4/5 and June Foundation Phase4/5 are contextual, not a direct comparison.",
            "Phase6 same-window estimates are development associations, not causal effects.",
            "Phase6 differences are arithmetic differences of independently matched aggregates; no pairwise delta estimand or delta inference is authorized.",
            "W1 and W2 are neither rolling OOF nor sealed OOS.",
            "No result may fit/admit a model, authorize an intent, or grant trading authority.",
        ],
    }
    payload["comparison_identity"] = comparison_digest(payload)
    return payload

def pretty(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")

def registered_output_path() -> Path:
    relative = Path(FINAL_OUTPUT)
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe output path")
    logical = ROOT
    for part in relative.parts:
        logical /= part
        try:
            mode = logical.lstat().st_mode
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(mode), f"output component is a symlink: {logical}")
    try:
        logical.resolve(strict=False).relative_to(ROOT)
    except ValueError as error:
        raise ComparisonError("output escapes repository") from error
    require(logical.parent.is_dir(), "output directory missing")
    return logical

def atomic_no_clobber(destination: Path, data: bytes) -> None:
    require(destination == registered_output_path(), "unregistered output")
    require(not destination.exists() and not destination.is_symlink(), "output exists")
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        require(destination == registered_output_path(), "output path changed")
        try:
            os.link(temp, destination)
        except FileExistsError as error:
            raise ComparisonError("output appeared concurrently") from error
        # The exclusive hard-link is the publication linearization point.
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass

def baseline_receipt() -> dict[str, Any]:
    old = load_old()
    historical = {}
    for spec in OLD.values():
        historical[spec[0]], historical[spec[2]] = spec[1], spec[3]
    historical[W2_PAIRS] = W2_PAIRS_SHA
    rebuilt = old["w2_reconstruction"]
    return {
        "status": "historical_baselines_valid",
        "mode": "validate_baselines_only",
        "repository_root": str(ROOT),
        "candidate_path": str(SCRIPT),
        "candidate_sha256": sha256(SCRIPT),
        "phase45_identity_evaluator": {
            "path": P45_IDENTITY_EVALUATOR,
            "sha256": sha256(repo_file(P45_IDENTITY_EVALUATOR, "Phase45 identity evaluator")),
        },
        "evaluator": {"path": EVALUATOR, "sha256": sha256(repo_file(EVALUATOR, "evaluator"))},
        "historical_artifacts": historical,
        "legacy_w2": {
            "pooled_reproduction_verified": rebuilt["pooled_reproduction_verified"],
            "w2_only_pair_counts": rebuilt["pair_counts"],
            "w2_only_status": {name: value["status"] for name, value in rebuilt["mechanisms"].items()},
        },
        "new_results_opened": False,
        "sealed_source_opened": False,
        "artifacts_written": [],
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-baselines-only", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    require(not (args.validate_baselines_only and args.validate_existing), "modes conflict")
    if args.validate_baselines_only:
        print(json.dumps(baseline_receipt(), sort_keys=True, indent=2, allow_nan=False))
        return
    payload = build_payload()
    data = pretty(payload)
    destination = registered_output_path()
    if args.validate_existing:
        require(destination.is_file() and not destination.is_symlink(), "output missing")
        existing, existing_sha = snapshot(destination, "comparison output", MAX_OBJECT_BYTES)
        require(existing == data, "output differs from exact regeneration")
        print(json.dumps({"status": "valid", "output": FINAL_OUTPUT, "sha256": existing_sha, "comparison_identity": payload["comparison_identity"]}, sort_keys=True))
        return
    atomic_no_clobber(destination, data)
    print(json.dumps({"status": "written", "output": FINAL_OUTPUT, "sha256": hashlib.sha256(data).hexdigest(), "comparison_identity": payload["comparison_identity"]}, sort_keys=True))

if __name__ == "__main__":
    main()
