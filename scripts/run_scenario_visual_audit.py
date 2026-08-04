#!/usr/bin/env python3
"""Causal scenario audit and two-pass AI-primitive review workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_continuous_replay import (  # noqa: E402
    _deadline,
    _load_mbo_execution,
    _optional_protocol_sha256,
    _sha256_file,
)
from smc_trader.ai_review import (  # noqa: E402
    AIReviewAdapter,
    CausalPrimitiveRegistry,
    ai_review_identity,
)
from smc_trader.action_equivalence import (  # noqa: E402
    action_equivalence_code_fingerprint,
)
from smc_trader.calibration import model_code_fingerprint  # noqa: E402
from smc_trader.engine import ContinuousSMCEngine  # noqa: E402
from smc_trader.decision_trace import (  # noqa: E402
    build_decision_trace,
    build_frozen_decision_packet,
    decision_packet_payload_sha256,
    decision_packet_sha256,
    read_verified_decision_packet,
    sealed_path_audit_context,
    write_frozen_decision_packet,
)
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.managed_net_value import (  # noqa: E402
    build_v2_2_policy_base_engine,
)
from smc_trader.model import AccountState, Timeframe, to_primitive  # noqa: E402
from smc_trader.observation import ExecutionRealityInput  # noqa: E402
from smc_trader.path_evidence import (  # noqa: E402
    PrimitivePathEvidenceQuery,
    PrimitivePathEvidenceRecorder,
)
from smc_trader.simulation import SequentialReplay  # noqa: E402
from smc_trader.validation import (  # noqa: E402
    FrozenPathTestRecorder,
    PathTestResult,
    evaluate_primitive_implementation_case,
    finalize_primitive_evaluation,
    load_validation_protocol,
    records_frame,
)
from smc_trader.visual_audit import (  # noqa: E402
    ScenarioVisualAuditSampler,
    visual_audit_code_fingerprint,
)
from smc_trader.visualization import (  # noqa: E402
    DecisionVisualizer,
    SealedVisualAudit,
)


def _aware(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("America/New_York")
        if timestamp.tzinfo is None
        else timestamp
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default="configs/model_v3_development.json",
    )
    parser.add_argument(
        "--validation-protocol",
        default="configs/validation_protocol_v2.json",
    )
    parser.add_argument("--mbo-execution")
    parser.add_argument("--warmup-days", type=int, default=45)
    parser.add_argument("--quota-per-scenario", type=int, default=3)
    parser.add_argument("--ai-review-directory")
    parser.add_argument(
        "--primitive-evaluation-directory",
        help=(
            "optional directory of verified frozen decision packets from a "
            "different registered window for proposal implementation checks; "
            "this does not approve semantic path validity"
        ),
    )
    parser.add_argument(
        "--candidate-only",
        type=int,
        default=0,
        help=(
            "seal exactly one complete-sequence decision packet and stop "
            "at its decision clock; repeat in an isolated output for each "
            "additional blind case"
        ),
    )
    return parser.parse_args()


def _review_template(
    snapshot: Any,
    hypothesis_key: str | None,
    *,
    decision_packet_hash: str,
    decision_packet_sha256: str,
) -> dict[str, Any]:
    return {
        **ai_review_identity(snapshot, hypothesis_key),
        "decision_packet_hash": decision_packet_hash,
        "decision_packet_sha256": decision_packet_sha256,
        "reviewer_id": "pending_ai_reviewer",
        "issues": [],
    }


def _review_filename(identity: dict[str, str | None]) -> str:
    suffix = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{identity['decision_hash']}--{suffix}.json"


def _audit_model_code_hash() -> str:
    digest = hashlib.sha256()
    for value in (
        model_code_fingerprint(),
        action_equivalence_code_fingerprint(),
        visual_audit_code_fingerprint(),
    ):
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _proposal_entity_ids(proposal: Any) -> tuple[str, ...]:
    """Collect explicit typed identities already frozen in the proposal."""

    identities = {
        value
        for value in (
            proposal.setup_id,
            proposal.entry_location_id,
            proposal.entry_path_id,
        )
        if isinstance(value, str) and value
    }
    values = proposal.origin_value.values
    if isinstance(values, Mapping):
        for name, value in values.items():
            if name.endswith("_id") and isinstance(value, str) and value:
                identities.add(value)
            elif name.endswith("_ids") and isinstance(value, (tuple, list)):
                identities.update(
                    item
                    for item in value
                    if isinstance(item, str) and item
                )
    return tuple(sorted(identities))


def _path_boundary_reason(anomalies: Any) -> str | None:
    values = set(anomalies)
    if "contract_change_history_reset" in values:
        return "contract_change_reset"
    if "data_gap_history_reset" in values:
        return "data_gap_reset"
    if "tick_size_mismatch" in values:
        return "tick_size_mismatch"
    if "data_anomaly" in values:
        return "data_anomaly"
    return None


def _development_output_directory(requested: str | Path) -> Path:
    """Return a writable run directory without deleting prior development work."""

    root = Path(requested)
    if root.exists() and not root.is_dir():
        raise NotADirectoryError("visual audit output exists and is not a directory")
    if not root.exists() or not any(root.iterdir()):
        return root
    reruns = root / "development-runs"
    index = 1
    while (reruns / f"run-{index:04d}").exists():
        index += 1
    return reruns / f"run-{index:04d}"


def main() -> None:
    args = parse_args()
    if args.warmup_days < 1 or args.quota_per_scenario < 1:
        raise ValueError("warmup and scenario quota must be positive")
    if args.candidate_only not in {0, 1}:
        raise ValueError(
            "candidate-only must be 0 or 1 to prevent cross-case future leakage"
        )
    start = _aware(args.start)
    end = _aware(args.end)
    if end <= start:
        raise ValueError("visual audit interval must be positive")
    validation = load_validation_protocol(args.validation_protocol)
    window = validation.classify_ohlcv(start, end)
    if window.role == "sealed_holdout":
        raise RuntimeError("scenario audit refuses the sealed OHLCV holdout")
    source_hash = _sha256_file(args.source)
    if source_hash != validation.causal_front_sha256:
        raise RuntimeError("scenario audit source differs from frozen front")
    loaded = load_ohlcv(
        args.source,
        start=start - pd.Timedelta(days=args.warmup_days),
        end=end,
    )
    if not loaded.contract_selection_causal:
        raise RuntimeError("scenario audit requires causal contract selection")
    execution_manifest: dict[str, Any] = {}
    if args.mbo_execution:
        execution_store, execution_manifest = _load_mbo_execution(
            args.mbo_execution,
            validation=validation,
            start=start,
            end=end,
            reveal_sealed_holdout=False,
        )
        if execution_manifest.get("validation_window_role") == "sealed_holdout":
            raise RuntimeError("scenario audit refuses the sealed MBO holdout")
    else:
        execution_store = None
    requested_destination = Path(args.output)
    destination = _development_output_directory(requested_destination)
    destination.mkdir(parents=True, exist_ok=True)

    config = Path(args.config)
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    version = str(config_payload.get("version", ""))
    if version.startswith("2.2."):
        if config_payload.get("managed_net_value_artifact") is not None:
            raise ValueError(
                "fitted v2.2 visual audit engine is not available until its "
                "temporal gates pass"
            )
        engine = build_v2_2_policy_base_engine(config)
    elif version.startswith(("3.", "4.")):
        engine = ContinuousSMCEngine.from_config(config)
    else:
        raise ValueError("scenario audit requires a v2.2, v3, or v4 config")
    sequential = SequentialReplay(engine=engine)
    visualizer = DecisionVisualizer()
    sampler = ScenarioVisualAuditSampler(
        destination / "scenarios",
        quota_per_scenario=args.quota_per_scenario,
    )
    ai_adapter = AIReviewAdapter()
    ai_registry = CausalPrimitiveRegistry()
    review_root = (
        None
        if args.ai_review_directory is None
        else Path(args.ai_review_directory)
    )
    if review_root is not None and not review_root.is_dir():
        raise NotADirectoryError("AI review directory does not exist")
    path_tests = FrozenPathTestRecorder(
        config_hash=_sha256_file(config),
        code_hash=_audit_model_code_hash(),
    )
    visual_audits: dict[str, SealedVisualAudit] = {}
    path_evidence_recorders: dict[
        str,
        tuple[Any, PrimitivePathEvidenceRecorder, Path],
    ] = {}
    path_evidence_records: list[dict[str, Any]] = []
    audit_records: list[str] = []
    review_files_used: list[str] = []
    proposals_by_identity: dict[
        tuple[str | None, ...],
        tuple[
            bool,
            str | None,
            str | None,
            tuple[Any, ...],
        ],
    ] = {}
    candidates: list[dict[str, Any]] = []
    last_asof = start

    def proposals_for(
        snapshot: Any,
        hypothesis_key: str | None,
        histories: Any,
        previous_snapshot: Any,
        source_bar: Any,
        account_state: Any,
        belief_position_input: Any,
    ) -> tuple[bool, str | None, str | None, tuple[Any, ...]]:
        identity = ai_review_identity(snapshot, hypothesis_key)
        proposals: tuple[Any, ...] = ()
        review_present = False
        packet_hash: str | None = None
        packet_sha256: str | None = None
        if review_root is not None:
            source = review_root / _review_filename(identity)
            if source.exists():
                review_present = True
                setup_id = identity["setup_id"]
                if setup_id is None:
                    raise ValueError(
                        "reviewed decision lacks a frozen setup identity"
                    )
                packet = build_frozen_decision_packet(
                    snapshot,
                    histories,
                    previous_snapshot,
                    hypothesis_key=hypothesis_key,
                    audit_context=sealed_path_audit_context(setup_id),
                    source_bar=source_bar,
                    account_state=account_state,
                    belief_position_input=belief_position_input,
                )
                packet_hash = packet["packet_hash"]
                packet_sha256 = decision_packet_payload_sha256(packet)
                cache_key = (
                    *tuple(identity.values()),
                    packet_hash,
                    packet_sha256,
                )
                cached = proposals_by_identity.get(cache_key)
                if cached is not None:
                    return cached
                review = json.loads(source.read_text(encoding="utf-8"))
                proposals = ai_adapter.convert(
                    review,
                    snapshot,
                    hypothesis_key=hypothesis_key,
                    decision_packet_hash=packet_hash,
                    decision_packet_sha256=packet_sha256,
                    decision_packet=packet,
                )
                ai_registry.register(proposals)
                review_files_used.append(str(source))
                output = (
                    review_present,
                    packet_hash,
                    packet_sha256,
                    proposals,
                )
                proposals_by_identity[cache_key] = output
                return output
        output = (review_present, packet_hash, packet_sha256, proposals)
        return output

    def reveal_new(results: Any) -> None:
        for result in results:
            audit = visual_audits.pop(result.setup_id, None)
            if audit is None:
                continue
            _, record = audit.reveal(result)
            audit_records.append(str(record))

    def write_finalized_path_evidence() -> None:
        for proposal_id, (
            proposal,
            recorder,
            output,
        ) in tuple(path_evidence_recorders.items()):
            if not recorder.finalized:
                continue
            recorder.write(output)
            evidence = recorder.evidence
            if (
                evidence.identity.query.query_id
                != proposal.proposal_id
                or evidence.identity.query.issue
                != proposal.issue.value
                or evidence.identity.query.primitive_name
                != proposal.primitive_name
                or evidence.identity.query.formula_version
                != proposal.formula_version
                or evidence.identity.query.definition_hash
                != proposal.definition_hash
            ):
                raise ValueError(
                    "typed future evidence differs from its AI proposal"
                )
            path_evidence_records.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "issue": proposal.issue.value,
                    "hypothesis_key": evidence.identity.hypothesis_key,
                    "setup_id": evidence.identity.setup_id,
                    "entry_location_id": (
                        evidence.identity.entry_location_id
                    ),
                    "entry_path_id": evidence.identity.entry_path_id,
                    "decision_hash": evidence.identity.decision_hash,
                    "decision_packet_hash": (
                        evidence.identity.decision_packet_hash
                    ),
                    "decision_packet_sha256": (
                        evidence.identity.decision_packet_sha256
                    ),
                    "evidence_hash": evidence.evidence_hash,
                    "finalized_at": evidence.finalized_at,
                    "finalization_reason": (
                        evidence.finalization_reason
                    ),
                    "points": len(evidence.points),
                    "path": str(output),
                    "semantic_path_property_assessed": False,
                    "used_action_or_pnl_label": False,
                }
            )
            path_evidence_recorders.pop(proposal_id)

    stopped_for_candidates = False
    for bar in iter_completed_bars(loaded.frame):
        if bar.end >= end:
            break
        if bar.end < start:
            execution = ExecutionRealityInput(
                spread_points=None,
                expected_slippage_points=0.0,
                commission_per_contract_per_side=0.0,
                deadline=_deadline(bar.end),
                source="ohlcv_only_execution_unavailable",
            )
        elif execution_store is not None:
            execution = execution_store.for_bar(
                bar,
                deadline=_deadline(bar.end),
            )
        else:
            execution = ExecutionRealityInput(
                spread_points=None,
                expected_slippage_points=0.0,
                commission_per_contract_per_side=0.0,
                deadline=_deadline(bar.end),
                source="ohlcv_only_execution_unavailable",
            )
        previous_snapshot = engine.last_snapshot
        step = sequential.on_bar(bar, execution=execution)
        snapshot = step.snapshot
        prior_result_count = len(path_tests.results)
        path_tests.on_bar(bar)
        newly_resolved_paths = path_tests.results[
            prior_result_count:
        ]
        for audit in visual_audits.values():
            audit.on_bar(bar)
        if path_evidence_recorders:
            trace = build_decision_trace(
                snapshot,
                previous_snapshot,
                source_bar=bar,
                account_state=step.account_state,
                belief_position_input=step.belief_position_input,
            )
            boundary_reason = _path_boundary_reason(
                snapshot.observation.anomalies
            )
            for _, recorder, _ in tuple(
                path_evidence_recorders.values()
            ):
                recorder.observe(
                    bar,
                    events_added=trace["events_added"],
                    events_ended=trace["events_ended"],
                    events_invalidated=trace[
                        "events_invalidated"
                    ],
                    typed_state_transitions=trace[
                        "typed_state_transitions"
                    ],
                    hard_boundary_reason=boundary_reason,
                )
            write_finalized_path_evidence()
        reveal_new(newly_resolved_paths)
        if snapshot.observation.asof < start:
            continue
        last_asof = snapshot.observation.asof
        histories = engine.histories(80)
        open_before = {item.setup_id for item in path_tests.open_tests}
        path_tests.observe(snapshot)
        newly_frozen = [
            item
            for item in path_tests.open_tests
            if item.setup_id not in open_before
        ]
        if args.candidate_only:
            for frozen in newly_frozen:
                if len(candidates) >= args.candidate_only:
                    break
                candidate_root = (
                    destination / "ai_candidates" / frozen.setup_id
                )
                audit_context = sealed_path_audit_context(
                    frozen.setup_id
                )
                packet_path = write_frozen_decision_packet(
                    snapshot,
                    histories,
                    candidate_root / "decision_packet.json",
                    previous_snapshot,
                    hypothesis_key=frozen.hypothesis_key,
                    audit_context=audit_context,
                    source_bar=bar,
                    account_state=step.account_state,
                    belief_position_input=step.belief_position_input,
                )
                packet = read_verified_decision_packet(packet_path)
                packet_hash = packet["packet_hash"]
                packet_sha256 = decision_packet_sha256(packet_path)
                artifact = visualizer.render_decision(
                    snapshot,
                    histories,
                    candidate_root / "decision.png",
                    audit_hypothesis_key=frozen.hypothesis_key,
                    audit_context=audit_context,
                )
                template_path = (
                    destination
                    / "review_templates"
                    / _review_filename(
                        ai_review_identity(
                            snapshot,
                            frozen.hypothesis_key,
                        )
                    )
                )
                template_path.parent.mkdir(parents=True, exist_ok=True)
                template_path.write_text(
                    json.dumps(
                        _review_template(
                            snapshot,
                            frozen.hypothesis_key,
                            decision_packet_hash=packet_hash,
                            decision_packet_sha256=packet_sha256,
                        ),
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                candidates.append(
                    {
                        "setup_id": frozen.setup_id,
                        "hypothesis_key": frozen.hypothesis_key,
                        "entry_location_id": frozen.entry_location_id,
                        "entry_path_id": frozen.entry_path_id,
                        "decision_hash": snapshot.snapshot_hash,
                        "decision_asof": snapshot.observation.asof,
                        "decision_image": str(artifact.path),
                        "decision_image_sha256": artifact.sha256,
                        "review_template": str(template_path),
                        "maximum_market_time": artifact.maximum_market_time,
                        "decision_packet": str(packet_path),
                        "decision_packet_hash": packet_hash,
                        "decision_packet_sha256": packet_sha256,
                        "future_after_this_candidate_exposed": False,
                        "candidate_loop_stopped_at_decision_clock": True,
                    }
                )
            if len(candidates) >= args.candidate_only:
                stopped_for_candidates = True
                break
            continue

        sampler.observe(
            snapshot,
            histories,
            visualizer,
            closed_trades=step.closed_trades,
            ai_proposals=(),
            previous_snapshot=previous_snapshot,
            source_bar=bar,
            account_state=step.account_state,
            belief_position_input=step.belief_position_input,
        )
        for frozen in newly_frozen:
            (
                review_present,
                reviewed_packet_hash,
                reviewed_packet_sha256,
                setup_proposals,
            ) = proposals_for(
                snapshot,
                frozen.hypothesis_key,
                histories,
                previous_snapshot,
                bar,
                step.account_state,
                step.belief_position_input,
            )
            if review_present:
                audit = SealedVisualAudit.seal(
                    visualizer,
                    snapshot,
                    histories,
                    destination / "ai_path_audits" / frozen.setup_id,
                    ai_proposals=setup_proposals,
                    hypothesis_key=frozen.hypothesis_key,
                    previous_snapshot=previous_snapshot,
                    source_bar=bar,
                    account_state=step.account_state,
                    belief_position_input=step.belief_position_input,
                    expected_decision_packet_hash=reviewed_packet_hash,
                    expected_decision_packet_sha256=(
                        reviewed_packet_sha256
                    ),
                )
                visual_audits[frozen.setup_id] = audit
                for proposal in setup_proposals:
                    if proposal.proposal_id in path_evidence_recorders:
                        raise ValueError(
                            "one primitive proposal cannot open two path recorders"
                        )
                    query = PrimitivePathEvidenceQuery.from_proposal(
                        proposal,
                        relevant_entity_ids=_proposal_entity_ids(
                            proposal
                        ),
                    )
                    recorder = (
                        PrimitivePathEvidenceRecorder.from_decision_packet(
                            audit.decision_packet_path,
                            hypothesis_key=frozen.hypothesis_key,
                            query=query,
                        )
                    )
                    path_evidence_recorders[proposal.proposal_id] = (
                        proposal,
                        recorder,
                        destination
                        / "primitive_path_evidence"
                        / f"{proposal.proposal_id}.json",
                    )

    bindings = {
        "source": str(loaded.source),
        "source_sha256": source_hash,
        "config": str(config),
        "config_sha256": _sha256_file(config),
        "model_code_hash": _audit_model_code_hash(),
        "structure_protocol_sha256": _optional_protocol_sha256(
            getattr(engine.observer.config, "structure_protocol", None)
        ),
        "liquidity_protocol_sha256": _optional_protocol_sha256(
            getattr(engine.observer.config, "liquidity_protocol", None)
        ),
        "displacement_protocol_sha256": _optional_protocol_sha256(
            getattr(engine.observer.config, "displacement_protocol", None)
        ),
        "group3_protocol_sha256": _optional_protocol_sha256(
            getattr(engine.observer.config, "group3_protocol", None)
        ),
        "group4_protocol_sha256": _optional_protocol_sha256(
            getattr(engine.observer.config, "group4_protocol", None)
        ),
        "group5_protocol_sha256": _optional_protocol_sha256(
            getattr(engine.observer.config, "group5_protocol", None)
        ),
        "validation_protocol_hash": validation.fingerprint,
        "window_role": window.role,
        "start": start,
        "end": end,
        "mbo_execution": args.mbo_execution,
        "primitive_evaluation_directory": (
            args.primitive_evaluation_directory
        ),
        "mbo_manifest_hash": (
            hashlib.sha256(
                json.dumps(
                    execution_manifest,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if execution_manifest
            else None
        ),
        "visual_audit_code_hash": visual_audit_code_fingerprint(),
        "requested_output": str(requested_destination),
        "actual_output": str(destination),
    }
    if args.candidate_only:
        if candidates and (
            len(candidates) != 1
            or last_asof
            != pd.Timestamp(candidates[0]["decision_asof"])
            or pd.Timestamp(candidates[0]["maximum_market_time"])
            > pd.Timestamp(candidates[0]["decision_asof"])
        ):
            raise AssertionError(
                "blind candidate artifacts include data beyond the decision"
            )
        payload = {
            "format_version": 1,
            "artifact": "pre_reveal_ai_audit_candidates",
            "status": (
                "candidate_quota_sealed"
                if stopped_for_candidates
                else "interval_exhausted_before_candidate_quota"
            ),
            "candidate_quota": args.candidate_only,
            "candidates": candidates,
            "last_market_time_processed": last_asof,
            "source_frame_loaded_in_process": True,
            "future_after_last_candidate_exposed": False,
            "future_path_revealed": False,
            "bindings": bindings,
        }
        (destination / "CANDIDATES_SEALED.json").write_text(
            json.dumps(
                to_primitive(payload),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "candidates": len(candidates),
                    "output": str(destination),
                },
                sort_keys=True,
            )
        )
        return

    result_count = len(path_tests.results)
    path_tests.close_unresolved(last_asof)
    reveal_new(path_tests.results[result_count:])
    for _, recorder, _ in path_evidence_recorders.values():
        if not recorder.finalized:
            recorder.close_right_boundary(last_asof)
    write_finalized_path_evidence()
    scenario_manifest = sampler.write_manifest(
        source_bindings=bindings,
    )
    records_frame(
        path_tests.results,
        record_type=PathTestResult,
    ).to_parquet(
        destination / "path_tests.parquet",
        index=False,
    )
    (destination / "ai_primitive_proposals.json").write_text(
        json.dumps(
            [to_primitive(item) for item in ai_registry.pending()],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (
        destination / "ai_primitive_typed_path_evidence.json"
    ).write_text(
        json.dumps(
            to_primitive(path_evidence_records),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    primitive_cases = []
    primitive_evaluations = []
    if args.primitive_evaluation_directory:
        evaluation_root = Path(args.primitive_evaluation_directory)
        if not evaluation_root.is_dir():
            raise NotADirectoryError(
                "primitive evaluation directory does not exist"
            )
        evaluation_packets = []
        for source in sorted(evaluation_root.rglob("*.json")):
            try:
                packet = read_verified_decision_packet(source)
            except (ValueError, json.JSONDecodeError):
                continue
            if packet.get("artifact") == "frozen_causal_decision_packet":
                evaluation_packets.append(packet)
        for proposal in ai_registry.pending():
            origin_clock = pd.Timestamp(proposal.origin_value.decision_asof)
            origin_window = validation.classify_ohlcv(
                origin_clock,
                origin_clock + pd.Timedelta(minutes=1),
            )
            proposal_cases = []
            for packet in evaluation_packets:
                hypothesis_key = packet.get("audit_hypothesis_key")
                if not isinstance(hypothesis_key, str):
                    continue
                evaluation_clock = pd.Timestamp(packet["decision_asof"])
                evaluation_window = validation.classify_ohlcv(
                    evaluation_clock,
                    evaluation_clock + pd.Timedelta(minutes=1),
                )
                case = evaluate_primitive_implementation_case(
                    proposal,
                    packet,
                    evaluation_hypothesis_key=hypothesis_key,
                    origin_window=origin_window,
                    evaluation_window=evaluation_window,
                )
                proposal_cases.append(case)
                primitive_cases.append(case)
            primitive_evaluations.append(
                finalize_primitive_evaluation(
                    proposal,
                    proposal_cases,
                )
            )
    (
        destination / "ai_primitive_implementation_checks.json"
    ).write_text(
        json.dumps(
            [to_primitive(item) for item in primitive_cases],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (destination / "ai_primitive_evaluations.json").write_text(
        json.dumps(
            [to_primitive(item) for item in primitive_evaluations],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary = {
        "format_version": 1,
        "artifact": "scenario_visual_and_ai_primitive_audit",
        "bindings": bindings,
        "scenario_manifest": str(scenario_manifest),
        "scenario_manifest_sha256": _sha256_file(scenario_manifest),
        "AI_review_files_used": review_files_used,
        "AI_primitive_proposals": len(ai_registry.pending()),
        "AI_primitive_implementation_checks": len(primitive_cases),
        "AI_primitive_path_acceptance_available": False,
        "AI_primitive_typed_future_evidence": len(
            path_evidence_records
        ),
        "AI_primitive_typed_future_evidence_index": str(
            destination / "ai_primitive_typed_path_evidence.json"
        ),
        "AI_primitive_typed_future_evidence_index_sha256": (
            _sha256_file(
                destination
                / "ai_primitive_typed_path_evidence.json"
            )
        ),
        "AI_primitive_semantic_path_assessment_pending": bool(
            path_evidence_records
        ),
        "AI_primitive_evaluations": [
            to_primitive(item) for item in primitive_evaluations
        ],
        "AI_has_model_action_authority": False,
        "AI_path_audit_records": audit_records,
        "path_tests": len(path_tests.results),
        "blind_pre_reveal_images_exclude_future": True,
        "post_outcome_views_are_explicit": True,
    }
    (destination / "summary.json").write_text(
        json.dumps(
            to_primitive(summary),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "scenario_captures": len(sampler.captures),
                "AI_proposals": len(ai_registry.pending()),
                "output": str(destination),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
