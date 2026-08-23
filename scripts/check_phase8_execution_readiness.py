#!/usr/bin/env python3
"""Audit the blocked Phase-8 execution-study template without opening data.

The repository has an execution evaluator and order FSM, but it has no
admitted non-zero TradeIntent ledger or frozen wait/cancel/stop/target study.
This checker deliberately cannot run a study or write an empirical artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import smc_trader.execution_research as execution_research_module  # noqa: E402
from smc_trader.execution_research import (  # noqa: E402
    EXECUTION_RESEARCH_SCHEMA_VERSION,
    ExecutionMethod,
    load_execution_research_config,
)


DEFAULT_TEMPLATE = (
    ROOT / "experiments/manifests/execution_research_phase8_v1_template.yaml"
)
TEMPLATE_STATUS = "template_incomplete_not_authorized_to_run"
CHECKER_BINDINGS = {
    "protocol_config",
    "runtime_execution_research",
    "readiness_checker",
}
INPUT_BINDINGS = {
    "trade_intent_ledger",
    "minute_execution_input_ledger",
    "ohlcv_artifact",
    "phase6_mbo_feature_artifact",
    "phase6_mbo_feature_manifest",
    "instrument_mapping_registry",
}
EXPECTED_BLOCKERS = (
    "not_frozen",
    "missing_experiment_identity",
    "missing_trade_intent_ledger",
    "missing_minute_execution_input_ledger",
    "missing_ohlcv_artifact_binding",
    "missing_phase6_mbo_feature_artifact_binding",
    "missing_phase6_mbo_feature_manifest_binding",
    "missing_instrument_mapping_registry",
    "limit_method_price_rules_not_frozen",
    "wait_cancel_stop_target_variants_not_frozen",
    "formal_runner_not_implemented",
    "outputs_not_registered",
)
ROOT_FIELDS = {
    "schema_version",
    "research_protocol",
    "status",
    "authority",
    "frozen_before_run",
    "experiment_id",
    "frozen_at",
    "identity_bindings",
    "reader_contract",
    "preregistration",
    "formal_input_schema",
    "method_price_rules",
    "variants",
    "readiness_blockers",
    "outputs",
}


class Phase8ReadinessError(ValueError):
    """Raised when the blocked template claims authority or drifts identity."""


def _duplicate_guard(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Phase8ReadinessError(f"JSON contains duplicate key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Phase8ReadinessError("Phase-8 template must be a regular file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_guard,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase8ReadinessError("Phase-8 template is not valid JSON") from error
    if not isinstance(value, dict):
        raise Phase8ReadinessError("Phase-8 template root must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise Phase8ReadinessError("Phase-8 binding path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise Phase8ReadinessError("Phase-8 binding must be repository-relative")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise Phase8ReadinessError("Phase-8 binding escapes the repository")
    return path


def _require_binding(
    root: Path,
    value: Any,
    *,
    expected_path: Path,
) -> str:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise Phase8ReadinessError("Phase-8 checker binding is incomplete")
    path = _resolve(root, value["path"])
    expected_sha = value["sha256"]
    if (
        path != expected_path.resolve()
        or path.is_symlink()
        or not path.is_file()
        or not isinstance(expected_sha, str)
        or _sha256(path) != expected_sha
    ):
        raise Phase8ReadinessError("Phase-8 checker binding identity differs")
    return expected_sha


def audit_execution_readiness(
    path: str | Path = DEFAULT_TEMPLATE,
    *,
    project_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Return a machine-readable blocked result; never open input ledgers."""

    root = Path(project_root).resolve()
    template_path = Path(path).resolve()
    payload = _load_json(template_path)
    if set(payload) != ROOT_FIELDS:
        raise Phase8ReadinessError("Phase-8 template fields changed")
    if (
        payload["schema_version"] != "phase8_execution_research_manifest_v1"
        or payload["research_protocol"] != EXECUTION_RESEARCH_SCHEMA_VERSION
        or payload["status"] != TEMPLATE_STATUS
        or payload["frozen_before_run"] is not False
        or payload["experiment_id"] is not None
        or payload["frozen_at"] is not None
    ):
        raise Phase8ReadinessError("Phase-8 template identity is not blocked")
    if payload["authority"] != {
        "research_only": True,
        "order_submission": False,
        "brain_decision_integration": False,
        "execution_fsm_integration": False,
        "queue_truth_claimed": False,
    }:
        raise Phase8ReadinessError("Phase-8 authority boundary changed")

    bindings = payload["identity_bindings"]
    if not isinstance(bindings, Mapping) or set(bindings) != (
        CHECKER_BINDINGS | INPUT_BINDINGS
    ):
        raise Phase8ReadinessError("Phase-8 binding set changed")
    protocol_path = root / "configs/execution_research_v1.json"
    protocol_sha = _require_binding(
        root,
        bindings["protocol_config"],
        expected_path=protocol_path,
    )
    _require_binding(
        root,
        bindings["runtime_execution_research"],
        expected_path=Path(execution_research_module.__file__).resolve(),
    )
    _require_binding(
        root,
        bindings["readiness_checker"],
        expected_path=Path(__file__).resolve(),
    )
    for name in INPUT_BINDINGS:
        if bindings[name] != {"path": None, "sha256": None}:
            raise Phase8ReadinessError(
                f"blocked Phase-8 input binding unexpectedly populated: {name}"
            )

    config = load_execution_research_config(
        protocol_path,
        expected_sha256=protocol_sha,
    )
    if payload["preregistration"] != {
        "canonical_config_id": config.config_id,
        "method_prices_frozen_before_outcomes": True,
        "same_intent_pairing_only": True,
        "market_comparator": "market",
        "fixed_contrasts": 6,
        "primary_metric": "implementation_shortfall_points",
        "inference": "two_sided_exact_sign_test_with_Holm",
        "minimum_complete_pairs": 30,
        "cross_intent_independence_claimed": False,
        "confirmatory_claim": False,
        "tuning_allowed": False,
        "causal_claim": False,
        "queue_level_claim": False,
        "realized_expectancy_status": "not_implemented_no_empirical_claim",
    }:
        raise Phase8ReadinessError("Phase-8 preregistration changed")
    if payload["formal_input_schema"] != {
        "case_ledger_schema_version": "phase8_execution_case_ledger_v1",
        "minute_ledger_schema_version": "phase8_execution_minute_ledger_v1",
        "case_grain": (
            "one_nonzero_source_trade_intent_with_preoutcome_price_provenance"
        ),
        "minute_grain": "one_source_trade_intent_x_one_exact_completed_minute",
        "nonzero_source_trade_intent_required": True,
        "fixture_rows_authorized_as_evidence": False,
        "bare_method_prices_allowed": False,
        "missing_variant_values_allowed": False,
    }:
        raise Phase8ReadinessError("Phase-8 formal input boundary changed")
    if payload["reader_contract"] != {
        "grain": "one_exact_frozen_research_intent_x_one_completed_minute",
        "clock": "causal_completed_minute_decision_time",
        "logical_instrument_identity": (
            "exact_string_or_integer_no_coercion_with_hash_bound_vendor_mapping"
        ),
        "fill_quantity": "whole_contract_floor_only",
        "same_intent_methods": [method.value for method in ExecutionMethod],
        "invalid_source_disposition": "censor_never_unfilled",
        "synthetic_source_eligible": False,
        "reset_source_eligible": False,
    }:
        raise Phase8ReadinessError("Phase-8 reader contract changed")

    method_rules = payload["method_price_rules"]
    if not isinstance(method_rules, Mapping) or set(method_rules) != {
        method.value for method in ExecutionMethod
    }:
        raise Phase8ReadinessError("Phase-8 method-price family changed")
    if method_rules["market"] != {
        "ledger_row_required": False,
        "source_semantic_type": "causal_arrival_bbo",
        "derivation_id": "first_causal_valid_best_quote_v1",
    } or any(
        method_rules[method.value] is not None
        for method in ExecutionMethod
        if method.value != "market"
    ):
        raise Phase8ReadinessError("Phase-8 method prices are not blocked")
    if payload["variants"] != {
        "cross_product": True,
        "inference_unit": "source_trade_intent_id_within_exact_variant",
        "wait_time": [],
        "cancel_rule": [],
        "stop": [],
        "target": [],
    }:
        raise Phase8ReadinessError("Phase-8 variants must remain unfrozen")
    if tuple(payload["readiness_blockers"]) != EXPECTED_BLOCKERS:
        raise Phase8ReadinessError("Phase-8 readiness blockers changed")
    if payload["outputs"] != {
        "per_method_outcomes_jsonl": None,
        "paired_contrasts_json": None,
        "summary_json": None,
    }:
        raise Phase8ReadinessError("blocked Phase-8 template cannot name outputs")
    return {
        "schema_version": "phase8_execution_readiness_v1",
        "status": TEMPLATE_STATUS,
        "ready": False,
        "readiness_blockers": list(EXPECTED_BLOCKERS),
        "opened_input_ledgers": False,
        "artifacts_written": [],
        "formal_runner_implemented": False,
        "claim_scope": "readiness_only_no_execution_or_empirical_claim",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    args = parser.parse_args(argv)
    try:
        result = audit_execution_readiness(args.template)
    except Phase8ReadinessError as error:
        print(json.dumps({"ready": False, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
