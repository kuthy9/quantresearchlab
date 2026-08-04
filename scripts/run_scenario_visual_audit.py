#!/usr/bin/env python3
"""Sample 20–40 causal decision traces, then reveal their future separately.

This is a development diagnostic, not a release-governance runner.  It keeps
the useful boundaries—completed bars, frozen model state, stratified actions,
separate future views and outcome-free AI primitives—without create-once
custody trees or parallel policy engines.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
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
    _sha256_file,
)
from smc_trader.ai_review import (  # noqa: E402
    AIReviewAdapter,
    CausalPrimitiveRegistry,
    ai_review_identity,
    path_evidence_query_from_proposal,
)
from smc_trader.decision_trace import (  # noqa: E402
    PrimitivePathEvidenceRecorder,
    build_decision_trace,
    build_frozen_decision_packet,
    decision_packet_payload_sha256,
)
from smc_trader.engine import ContinuousSMCEngine  # noqa: E402
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.model import Action, Bar, Candle, Timeframe, to_primitive  # noqa: E402
from smc_trader.observation import ExecutionRealityInput  # noqa: E402
from smc_trader.simulation import SequentialReplay  # noqa: E402
from smc_trader.validation import load_validation_protocol  # noqa: E402
from smc_trader.visualization import (  # noqa: E402
    DecisionVisualizer,
    RevealPermit,
    VisualArtifact,
)


AUDIT_ACTIONS = frozenset({Action.ENTER, Action.WAIT, Action.ABSTAIN})


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
    parser.add_argument("--config", default="configs/model.json")
    parser.add_argument(
        "--validation-protocol",
        default="configs/data_splits.json",
    )
    parser.add_argument("--mbo-execution")
    parser.add_argument("--warmup-days", type=int, default=45)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--future-minutes", type=int, default=120)
    parser.add_argument("--ai-review-directory")
    return parser.parse_args()


def _review_filename(identity: Mapping[str, str | None]) -> str:
    suffix = hashlib.sha256(
        json.dumps(dict(identity), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{identity['decision_hash']}--{suffix}.json"


def _bar_candle(bar: Bar) -> Candle:
    return Candle(
        timeframe=Timeframe.M1,
        start=bar.start,
        end=bar.end,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        symbol=bar.symbol,
        instrument_id=bar.instrument_id,
        observed_minutes=1,
        expected_minutes=1,
        complete=True,
        synthetic_minutes=1 if bar.synthetic_no_trade else 0,
    )


def _boundary_reason(anomalies: Any) -> str | None:
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


def _sample_stratum(snapshot: Any) -> tuple[str, ...]:
    key = snapshot.decision.best_hypothesis_key
    hypothesis = (
        None if key is None else snapshot.belief.hypotheses.get(key)
    )
    h4_direction = float(
        snapshot.observation.frame(Timeframe.H4).metrics.get(
            "structure_direction",
            0.0,
        )
    )
    regime = (
        "h4_up"
        if h4_direction > 0
        else "h4_down"
        if h4_direction < 0
        else "h4_flat"
    )
    hour = snapshot.observation.asof.tz_convert("America/New_York").hour
    session = "overnight" if hour < 8 else "morning" if hour < 12 else "afternoon"
    return (
        snapshot.risk.final_action.value,
        "none" if hypothesis is None else hypothesis.playbook.value,
        "none" if hypothesis is None else hypothesis.direction.value,
        regime,
        session,
    )


@dataclass
class _OpenAudit:
    sample_id: str
    snapshot: Any
    hypothesis_key: str | None
    permit: RevealPermit
    decision_artifact: VisualArtifact
    directory: Path
    reveal_at: pd.Timestamp
    proposals: tuple[Any, ...]
    recorders: list[tuple[Any, PrimitivePathEvidenceRecorder, Path]]
    stratum: tuple[str, ...]
    future_1m: list[Candle] = field(default_factory=list)


class _StratifiedAuditBatch:
    def __init__(
        self,
        root: Path,
        *,
        sample_count: int,
        future_minutes: int,
        visualizer: DecisionVisualizer,
        review_root: Path | None,
    ) -> None:
        self.root = root
        self.sample_count = sample_count
        self.future_minutes = future_minutes
        self.visualizer = visualizer
        self.review_root = review_root
        self.ai_adapter = AIReviewAdapter()
        self.ai_registry = CausalPrimitiveRegistry()
        self.open: list[_OpenAudit] = []
        self.rows: list[dict[str, Any]] = []
        self._stratum_counts: dict[tuple[str, ...], int] = {}
        self._action_counts: dict[str, int] = {}
        self._last_action_clock: dict[str, pd.Timestamp] = {}

    def _accepts(self, snapshot: Any) -> bool:
        action = snapshot.risk.final_action
        if action not in AUDIT_ACTIONS or len(self.rows) + len(self.open) >= self.sample_count:
            return False
        action_name = action.value
        action_cap = max(1, int(math.ceil(self.sample_count * 0.60)))
        if self._action_counts.get(action_name, 0) >= action_cap:
            return False
        previous = self._last_action_clock.get(action_name)
        if previous is not None and snapshot.observation.asof - previous < pd.Timedelta(minutes=5):
            return False
        stratum = _sample_stratum(snapshot)
        return self._stratum_counts.get(stratum, 0) < 3

    def _review_proposals(
        self,
        snapshot: Any,
        histories: Mapping[Timeframe, Any],
        previous_snapshot: Any,
        source_bar: Bar,
        account_state: Any,
        belief_position_input: Any,
        sample_root: Path,
    ) -> tuple[Any, ...]:
        key = snapshot.decision.best_hypothesis_key
        identity = ai_review_identity(snapshot, key)
        packet = build_frozen_decision_packet(
            snapshot,
            histories,
            previous_snapshot,
            hypothesis_key=key,
            source_bar=source_bar,
            account_state=account_state,
            belief_position_input=belief_position_input,
        )
        packet_sha = decision_packet_payload_sha256(packet)
        template = {
            **identity,
            "decision_packet_hash": packet["packet_hash"],
            "decision_packet_sha256": packet_sha,
            "reviewer_id": "pending_ai_reviewer",
            "issues": [],
        }
        template_path = sample_root / "ai_review_template.json"
        template_path.write_text(
            json.dumps(to_primitive(template), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if self.review_root is None:
            return ()
        source = self.review_root / _review_filename(identity)
        if not source.exists():
            return ()
        proposals = self.ai_adapter.convert(
            json.loads(source.read_text(encoding="utf-8")),
            snapshot,
            hypothesis_key=key,
            decision_packet_hash=packet["packet_hash"],
            decision_packet_sha256=packet_sha,
            decision_packet=packet,
        )
        self.ai_registry.register(proposals)
        return proposals

    def maybe_open(
        self,
        snapshot: Any,
        histories: Mapping[Timeframe, Any],
        previous_snapshot: Any,
        source_bar: Bar,
        account_state: Any,
        belief_position_input: Any,
    ) -> None:
        if not self._accepts(snapshot):
            return
        action = snapshot.risk.final_action.value
        stratum = _sample_stratum(snapshot)
        index = len(self.rows) + len(self.open) + 1
        sample_id = (
            f"{index:02d}-{snapshot.observation.asof:%Y%m%d-%H%M}-"
            f"{action}"
        )
        sample_root = self.root / "samples" / sample_id
        sample_root.mkdir(parents=True, exist_ok=True)
        key = snapshot.decision.best_hypothesis_key
        trace = build_decision_trace(
            snapshot,
            previous_snapshot,
            hypothesis_key=key,
            source_bar=source_bar,
            account_state=account_state,
            belief_position_input=belief_position_input,
        )
        (sample_root / "decision_trace.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        proposals = self._review_proposals(
            snapshot,
            histories,
            previous_snapshot,
            source_bar,
            account_state,
            belief_position_input,
            sample_root,
        )
        artifact = self.visualizer.render_decision(
            snapshot,
            histories,
            sample_root / "decision.png",
            ai_proposals=proposals,
            audit_hypothesis_key=key,
            audit_context={
                "scenario": "stratified_decision_trace",
                "future_present": False,
                "stratum": stratum,
            },
        )
        reveal_at = snapshot.observation.asof + pd.Timedelta(
            minutes=self.future_minutes
        )
        recorders: list[tuple[Any, PrimitivePathEvidenceRecorder, Path]] = []
        for proposal in proposals:
            try:
                recorder = PrimitivePathEvidenceRecorder.from_snapshot(
                    snapshot,
                    hypothesis_key=str(key),
                    query=path_evidence_query_from_proposal(proposal),
                )
            except ValueError:
                continue
            recorders.append(
                (
                    proposal,
                    recorder,
                    sample_root / f"future-{proposal.proposal_id}.json",
                )
            )
            reveal_at = min(reveal_at, recorder.identity.deadline)
        self.open.append(
            _OpenAudit(
                sample_id=sample_id,
                snapshot=snapshot,
                hypothesis_key=key,
                permit=self.visualizer.seal_reveal(snapshot, key),
                decision_artifact=artifact,
                directory=sample_root,
                reveal_at=reveal_at,
                proposals=proposals,
                recorders=recorders,
                stratum=stratum,
            )
        )
        self._stratum_counts[stratum] = self._stratum_counts.get(stratum, 0) + 1
        self._action_counts[action] = self._action_counts.get(action, 0) + 1
        self._last_action_clock[action] = snapshot.observation.asof

    def _finalize(self, audit: _OpenAudit, at: pd.Timestamp) -> None:
        evidence_rows = []
        for proposal, recorder, output in audit.recorders:
            if not recorder.finalized:
                recorder.close_right_boundary(
                    min(at, recorder.identity.deadline)
                )
            recorder.write(output)
            evidence_rows.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "path": str(output),
                    "points": len(recorder.evidence.points),
                    "finalized_at": recorder.evidence.finalized_at,
                    "finalization_reason": recorder.evidence.finalization_reason,
                }
            )
        reveal_path = None
        reveal_maximum = audit.snapshot.observation.asof
        if audit.future_1m:
            reveal = self.visualizer.render_reveal(
                audit.snapshot,
                audit.permit,
                audit.future_1m,
                audit.directory / "future_reveal.png",
                revealed_at=max(at, audit.future_1m[-1].end),
                ai_proposals=audit.proposals,
                audit_hypothesis_key=audit.hypothesis_key,
            )
            reveal_path = str(reveal.path)
            reveal_maximum = reveal.maximum_market_time
        row = {
            "sample_id": audit.sample_id,
            "decision_hash": audit.snapshot.snapshot_hash,
            "decision_asof": audit.snapshot.observation.asof,
            "action": audit.snapshot.risk.final_action.value,
            "stratum": audit.stratum,
            "decision_trace": str(audit.directory / "decision_trace.json"),
            "decision_image": str(audit.decision_artifact.path),
            "future_reveal": reveal_path,
            "future_maximum_market_time": reveal_maximum,
            "future_physically_separate": True,
            "ai_proposals": len(audit.proposals),
            "primitive_future_paths": evidence_rows,
        }
        (audit.directory / "audit.json").write_text(
            json.dumps(to_primitive(row), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.rows.append(row)
        self.open.remove(audit)

    def advance(
        self,
        bar: Bar,
        snapshot: Any,
        previous_snapshot: Any,
        account_state: Any,
        belief_position_input: Any,
    ) -> None:
        if not self.open:
            return
        trace = build_decision_trace(
            snapshot,
            previous_snapshot,
            source_bar=bar,
            account_state=account_state,
            belief_position_input=belief_position_input,
        )
        boundary = _boundary_reason(snapshot.observation.anomalies)
        for audit in tuple(self.open):
            if bar.start < audit.snapshot.observation.asof:
                continue
            same_contract = (bar.symbol, bar.instrument_id) == (
                audit.snapshot.observation.symbol,
                audit.snapshot.observation.instrument_id,
            )
            if same_contract and bar.start < audit.reveal_at:
                audit.future_1m.append(_bar_candle(bar))
            for _, recorder, _ in audit.recorders:
                if not recorder.finalized:
                    recorder.observe(
                        bar,
                        events_added=trace["events_added"],
                        events_ended=trace["events_ended"],
                        events_invalidated=trace["events_invalidated"],
                        typed_state_transitions=trace[
                            "typed_state_transitions"
                        ],
                        hard_boundary_reason=boundary,
                    )
            if not same_contract or boundary is not None or bar.end >= audit.reveal_at:
                self._finalize(audit, min(bar.end, audit.reveal_at))

    def close(self, at: pd.Timestamp) -> None:
        for audit in tuple(self.open):
            self._finalize(audit, min(at, audit.reveal_at))

    def write_manifest(self, bindings: Mapping[str, Any]) -> Path:
        manifest = {
            "schema_version": 1,
            "artifact": "stratified_decision_trace_audit",
            "requested_samples": self.sample_count,
            "captured_samples": len(self.rows),
            "action_counts": self._action_counts,
            "sampling": (
                "causal first-observed strata; ENTER/WAIT/ABSTAIN; maximum "
                "three per playbook-direction-regime-session stratum"
            ),
            "future_reveal_is_separate": True,
            "ai_has_action_authority": False,
            "rows": self.rows,
            "ai_primitive_proposals": [
                to_primitive(item) for item in self.ai_registry.pending()
            ],
            "bindings": dict(bindings),
        }
        output = self.root / "manifest.json"
        output.write_text(
            json.dumps(
                to_primitive(manifest),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return output


def main() -> None:
    args = parse_args()
    if args.warmup_days < 1 or not 20 <= args.sample_count <= 40:
        raise ValueError("warmup must be positive and sample-count must be 20–40")
    if args.future_minutes < 1:
        raise ValueError("future-minutes must be positive")
    start, end = _aware(args.start), _aware(args.end)
    if end <= start:
        raise ValueError("visual audit interval must be positive")
    validation = load_validation_protocol(args.validation_protocol)
    window = validation.classify_ohlcv(start, end)
    if window.role == "sealed_holdout":
        raise RuntimeError("scenario audit refuses the sealed OHLCV holdout")
    source_hash = _sha256_file(args.source)
    if source_hash != validation.causal_source.sha256:
        raise RuntimeError("scenario audit source differs from the causal front")
    loaded = load_ohlcv(
        args.source,
        start=start - pd.Timedelta(days=args.warmup_days),
        end=end,
    )
    if not loaded.contract_selection_causal:
        raise RuntimeError("scenario audit requires causal contract selection")
    if args.mbo_execution:
        execution_store, execution_manifest = _load_mbo_execution(
            args.mbo_execution,
            validation=validation,
            start=start,
            end=end,
            reveal_sealed_holdout=False,
        )
    else:
        execution_store, execution_manifest = None, {}
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    review_root = (
        None
        if args.ai_review_directory is None
        else Path(args.ai_review_directory)
    )
    if review_root is not None and not review_root.is_dir():
        raise NotADirectoryError("AI review directory does not exist")

    engine = ContinuousSMCEngine.from_config(args.config)
    sequential = SequentialReplay(engine=engine)
    batch = _StratifiedAuditBatch(
        destination,
        sample_count=args.sample_count,
        future_minutes=args.future_minutes,
        visualizer=DecisionVisualizer(),
        review_root=review_root,
    )
    last_asof = start
    for bar in iter_completed_bars(loaded.frame, allow_data_gap_reset=True):
        if bar.end >= end:
            break
        execution = (
            execution_store.for_bar(bar, deadline=_deadline(bar.end))
            if execution_store is not None
            else ExecutionRealityInput(
                spread_points=None,
                expected_slippage_points=0.0,
                commission_per_contract_per_side=0.0,
                deadline=_deadline(bar.end),
                source="ohlcv_only_execution_unavailable",
            )
        )
        previous_snapshot = engine.last_snapshot
        step = sequential.on_bar(bar, execution=execution)
        snapshot = step.snapshot
        batch.advance(
            bar,
            snapshot,
            previous_snapshot,
            step.account_state,
            step.belief_position_input,
        )
        if snapshot.observation.asof < start:
            continue
        last_asof = snapshot.observation.asof
        batch.maybe_open(
            snapshot,
            engine.histories(80),
            previous_snapshot,
            bar,
            step.account_state,
            step.belief_position_input,
        )
    batch.close(last_asof)
    manifest = batch.write_manifest(
        {
            "source": str(loaded.source),
            "source_sha256": source_hash,
            "config": str(args.config),
            "config_sha256": _sha256_file(args.config),
            "window_role": window.role,
            "start": start,
            "end": end,
            "mbo_execution": args.mbo_execution,
            "mbo_manifest": execution_manifest,
        }
    )
    print(
        json.dumps(
            {
                "samples": len(batch.rows),
                "manifest": str(manifest),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
