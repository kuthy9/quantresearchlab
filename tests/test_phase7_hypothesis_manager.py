from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pandas as pd
import pytest

import smc_trader.path_belief as path_belief_module

from smc_trader.path_belief import (
    HypothesisManager,
    PathKind,
    PathStatus,
    create_path_competition_set,
    load_path_belief_protocol,
)


T0 = pd.Timestamp("2026-08-21 09:30:00", tz="America/New_York")
EXPIRY = pd.Timestamp("2026-08-21 16:00:00", tz="America/New_York")
PHASE6_MANIFEST_SHA256 = (
    "99901b1893cdea71615239fd9c536a318ec3c8088f0c491e48cb710b9d0eeafc"
)
PHASE6_RESULT_SHA256 = (
    "98f0f334cbae050a093bebca4cfbb85fcc877e96ba729fe04ea477e2761ddf99"
)
PHASE6_RESULT_IDENTITY = (
    "8e69ca54f4a6c8c1ae11878a9a0552c037910c54bfa3478aef443b1a27a6bc77"
)


def _production_protocol():
    return load_path_belief_protocol("configs/path_hypotheses.json")


def _admitted_bayesian_protocol(
    tmp_path: Path,
    *,
    continuation_decay: float = 0.0,
):
    payload = json.loads(
        Path("configs/path_hypotheses.json").read_text(encoding="utf-8")
    )
    payload["model_version"] = "phase7-hand-calculated-diagnostic-v1"
    payload["model_admission_status"] = "diagnostic_likelihood_admitted"
    payload["likelihood_artifact_status"] = "diagnostic_admitted"
    payload["real_completed_bar_decay"]["continuation"] = continuation_decay
    payload["evidence_rules"]["acceptance_continuation"][
        "conditional_likelihood"
    ] = {
        "continuation": 0.6,
        "deeper_retracement": 0.2,
        "reversal": 0.1,
        "balance": 0.1,
        "failed_breakout": 0.1,
        "residual_unknown": 0.3,
    }
    payload["evidence_rules"]["displacement_impact"][
        "conditional_likelihood"
    ] = {path.value: 0.5 for path in PathKind}
    for rule in payload["evidence_rules"].values():
        rule.pop("log_likelihood_increment")
    destination = tmp_path / "admitted_path_model.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return load_path_belief_protocol(destination)


def _synthetic_three_family_protocol(tmp_path: Path):
    """Build a test-only admitted model with one rule per impulse stage."""

    payload = json.loads(
        Path("configs/path_hypotheses.json").read_text(encoding="utf-8")
    )
    payload["model_version"] = "three-family-dependence-regression-v1"
    payload["model_admission_status"] = "diagnostic_likelihood_admitted"
    payload["likelihood_artifact_status"] = "diagnostic_admitted"
    payload["phase6_evidence_allowlist"] = [
        "sweep_rejection",
        "displacement_impact",
        "mss_flow_shift",
    ]
    likelihoods = {path.value: 0.5 for path in PathKind}
    payload["evidence_rules"] = {
        rule_id: {
            "description": f"test-only {rule_id} conditional contribution",
            "evidence_family": family,
            "conditional_likelihood": likelihoods,
        }
        for rule_id, family in (
            ("sweep_rejection", "sweep"),
            ("displacement_impact", "displacement"),
            ("mss_flow_shift", "mss"),
        )
    }
    destination = tmp_path / "three_family_path_model.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return load_path_belief_protocol(destination)


def _pristine(protocol):
    return create_path_competition_set(
        protocol,
        instrument_id="NQ:front",
        market_epoch_id="epoch:nq:2026-08-21",
        authority_structure_id="h1-structure:123",
        horizon_id="ny-session:2026-08-21",
        formed_at=T0,
        common_expires_at=EXPIRY,
    )


def _initialized_manager(protocol):
    manager = HypothesisManager(protocol)
    state, _ = manager.initialize_state(
        _pristine(protocol),
        real_completed_bar=False,
    )
    return manager, state


def test_phase6_identity_admits_evidence_family_but_not_likelihood_model() -> None:
    protocol = _production_protocol()

    assert protocol.phase6_manifest_sha256 == PHASE6_MANIFEST_SHA256
    assert protocol.phase6_result_sha256 == PHASE6_RESULT_SHA256
    assert protocol.phase6_result_identity == PHASE6_RESULT_IDENTITY
    assert protocol.evidence_allowlist == (
        "acceptance_continuation",
        "displacement_impact",
    )
    assert tuple(rule.rule_id for rule in protocol.evidence_rules) == (
        "acceptance_continuation",
        "displacement_impact",
    )
    assert protocol.likelihood_artifact_status == "missing"
    assert not protocol.can_apply_bayesian_update
    assert not protocol.can_authorize_action


def test_missing_likelihood_artifact_rejects_unfitted_path_decay(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        Path("configs/path_hypotheses.json").read_text(encoding="utf-8")
    )
    payload["real_completed_bar_decay"]["continuation"] = 0.01
    destination = tmp_path / "unfitted_decay.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="separately admitted temporal artifact"):
        load_path_belief_protocol(destination)


def test_likelihood_artifact_cannot_smuggle_unadmitted_temporal_decay(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="separately admitted temporal artifact"):
        _admitted_bayesian_protocol(tmp_path, continuation_decay=0.01)


def test_hand_calculated_bayes_normalizes_all_paths_including_residual(
    tmp_path: Path,
) -> None:
    protocol = _admitted_bayesian_protocol(tmp_path)
    manager = HypothesisManager(protocol)
    pristine = _pristine(protocol)
    evidence = protocol.make_contribution(
        competition_set_id=pristine.competition_set_id,
        rule_id="acceptance_continuation",
        source_event_ids=("event:acceptance:1",),
        known_at=T0,
        correlation_key="acceptance_continuation:generation-1",
        require_admitted=True,
    )

    state, record = manager.initialize_state(
        pristine,
        contributions=(evidence,),
        real_completed_bar=False,
    )

    likelihoods = {
        PathKind.CONTINUATION: 0.6,
        PathKind.DEEPER_RETRACEMENT: 0.2,
        PathKind.REVERSAL: 0.1,
        PathKind.BALANCE: 0.1,
        PathKind.FAILED_BREAKOUT: 0.1,
        PathKind.RESIDUAL_UNKNOWN: 0.3,
    }
    denominator = sum(likelihoods.values())
    for path, likelihood in likelihoods.items():
        assert state.member(path).probability == pytest.approx(
            likelihood / denominator
        )
        assert state.member(path).log_weight == pytest.approx(
            math.log(likelihood)
        )
    assert sum(item.probability for item in state.members) == pytest.approx(1.0)
    assert record.bayesian_update_applied
    assert not record.evidence_admission_only
    assert manager.updater.update_mode == "bayesian_log_likelihood"
    assert not manager.action_authority


def test_missing_likelihood_model_journals_neutral_evidence_and_has_no_authority() -> None:
    protocol = _production_protocol()
    manager = HypothesisManager(protocol)
    pristine = _pristine(protocol)
    evidence = protocol.make_contribution(
        competition_set_id=pristine.competition_set_id,
        rule_id="acceptance_continuation",
        source_event_ids=("event:acceptance:neutral",),
        known_at=T0,
        correlation_key="acceptance_continuation:neutral-generation",
        require_admitted=True,
    )

    state, record = manager.initialize_state(
        pristine,
        contributions=(evidence,),
        real_completed_bar=True,
    )

    assert all(item.probability == pytest.approx(1.0 / 6.0) for item in state.members)
    assert state.evidence_ledger == (evidence,)
    assert record.evidence_admission_only
    assert not record.bayesian_update_applied
    assert manager.updater.update_mode == "evidence_admission_only"
    assert not manager.action_authority


def test_correlated_family_is_applied_once_independent_of_input_order(
    tmp_path: Path,
) -> None:
    protocol = _admitted_bayesian_protocol(tmp_path)
    pristine = _pristine(protocol)
    values = tuple(
        protocol.make_contribution(
            competition_set_id=pristine.competition_set_id,
            rule_id="acceptance_continuation",
            source_event_ids=(f"event:acceptance:{suffix}",),
            known_at=T0,
            correlation_key="acceptance_continuation:same-generation",
            require_admitted=True,
        )
        for suffix in ("a", "b")
    )

    first = HypothesisManager(protocol)
    first_state, first_record = first.initialize_state(
        pristine,
        contributions=values,
        real_completed_bar=False,
    )
    second = HypothesisManager(protocol)
    second_state, second_record = second.initialize_state(
        pristine,
        contributions=tuple(reversed(values)),
        real_completed_bar=False,
    )

    assert first_state == second_state
    assert first_record == second_record
    assert len(first_state.evidence_ledger) == 1
    assert len(first_record.applied_contributions) == 1
    assert len(first_record.correlated_duplicate_contribution_ids) == 1


def test_same_dependency_cluster_cannot_be_multiplied_across_evidence_families(
    tmp_path: Path,
) -> None:
    protocol = _admitted_bayesian_protocol(tmp_path)
    pristine = _pristine(protocol)
    values = tuple(
        protocol.make_contribution(
            competition_set_id=pristine.competition_set_id,
            rule_id=rule_id,
            source_event_ids=(source_id,),
            known_at=T0,
            correlation_key="shared-upstream-instance",
            require_admitted=True,
        )
        for rule_id, source_id in (
            ("acceptance_continuation", "event:acceptance:shared"),
            ("displacement_impact", "event:displacement:shared"),
        )
    )

    manager = HypothesisManager(protocol)
    with pytest.raises(
        ValueError,
        match="one registered joint or history-conditioned contribution",
    ):
        manager.initialize_state(
            pristine,
            contributions=values,
            real_completed_bar=False,
        )


def test_one_source_event_cannot_rename_its_dependency_cluster(
    tmp_path: Path,
) -> None:
    protocol = _admitted_bayesian_protocol(tmp_path)
    pristine = _pristine(protocol)
    values = tuple(
        protocol.make_contribution(
            competition_set_id=pristine.competition_set_id,
            rule_id=rule_id,
            source_event_ids=("event:one-upstream-fact",),
            known_at=T0,
            correlation_key=cluster,
            require_admitted=True,
        )
        for rule_id, cluster in (
            ("acceptance_continuation", "cluster:claimed-a"),
            ("displacement_impact", "cluster:claimed-b"),
        )
    )

    with pytest.raises(
        ValueError,
        match="one source event cannot declare multiple dependency clusters",
    ):
        HypothesisManager(protocol).initialize_state(
            pristine,
            contributions=values,
            real_completed_bar=False,
        )


def test_sweep_displacement_and_mss_from_one_impulse_are_not_three_multipliers(
    tmp_path: Path,
) -> None:
    protocol = _synthetic_three_family_protocol(tmp_path)
    pristine = _pristine(protocol)
    values = tuple(
        protocol.make_contribution(
            competition_set_id=pristine.competition_set_id,
            rule_id=rule_id,
            source_event_ids=(source_id,),
            known_at=T0,
            correlation_key="impulse:nq:2026-08-21:09:30:1",
            require_admitted=True,
        )
        for rule_id, source_id in (
            ("sweep_rejection", "event:sweep:one-impulse"),
            ("displacement_impact", "event:displacement:one-impulse"),
            ("mss_flow_shift", "event:mss:one-impulse"),
        )
    )

    with pytest.raises(
        ValueError,
        match="one registered joint or history-conditioned contribution",
    ):
        HypothesisManager(protocol).initialize_state(
            pristine,
            contributions=values,
            real_completed_bar=False,
        )


def test_neutral_cross_family_cluster_is_ledgered_without_bayesian_multiplier() -> None:
    protocol = _production_protocol()
    pristine = _pristine(protocol)
    values = tuple(
        protocol.make_contribution(
            competition_set_id=pristine.competition_set_id,
            rule_id=rule_id,
            source_event_ids=(source_id,),
            known_at=T0,
            correlation_key="impulse:shared-neutral-source",
            require_admitted=True,
        )
        for rule_id, source_id in (
            ("acceptance_continuation", "event:acceptance:shared"),
            ("displacement_impact", "event:displacement:shared"),
        )
    )

    state, record = HypothesisManager(protocol).initialize_state(
        pristine,
        contributions=values,
        real_completed_bar=False,
    )

    assert len(state.evidence_ledger) == 2
    assert len(record.applied_contributions) == 2
    assert record.evidence_admission_only
    assert not record.bayesian_update_applied
    assert all(member.log_weight == 0.0 for member in state.members)
    assert all(
        member.probability == pytest.approx(1.0 / len(PathKind))
        for member in state.members
    )


def test_equal_logit_subtraction_is_probability_invariant() -> None:
    state = _pristine(_production_protocol())
    logits = {
        path: value
        for path, value in zip(
            PathKind,
            (2.5, 0.75, -0.5, 1.25, -1.0, 0.0),
            strict=True,
        )
    }
    original, _ = path_belief_module._normalized_members(
        tuple(
            replace(member, log_weight=logits[member.path])
            for member in state.members
        )
    )
    shifted, _ = path_belief_module._normalized_members(
        tuple(
            replace(member, log_weight=logits[member.path] - 7.0)
            for member in state.members
        )
    )

    assert tuple(item.probability for item in shifted) == pytest.approx(
        tuple(item.probability for item in original)
    )


def test_order_replay_checkpoint_resume_and_reset_are_deterministic(
    tmp_path: Path,
) -> None:
    protocol = _admitted_bayesian_protocol(tmp_path)
    manager, state = _initialized_manager(protocol)
    first = protocol.make_contribution(
        competition_set_id=state.competition_set_id,
        rule_id="acceptance_continuation",
        source_event_ids=("event:acceptance:checkpoint",),
        known_at=T0 + pd.Timedelta(minutes=1),
        correlation_key="acceptance_continuation:checkpoint",
        require_admitted=True,
    )
    second = protocol.make_contribution(
        competition_set_id=state.competition_set_id,
        rule_id="displacement_impact",
        source_event_ids=("event:displacement:checkpoint",),
        known_at=T0 + pd.Timedelta(minutes=1),
        correlation_key="displacement_impact:checkpoint",
        require_admitted=True,
    )
    expected, expected_record = manager.advance(
        asof=T0 + pd.Timedelta(minutes=1),
        contributions=(second, first),
        real_completed_bar=False,
    )

    restored = HypothesisManager.from_checkpoint(
        manager.checkpoint_payload(),
        protocol=protocol,
    )
    replayed = restored.replay_evidence_posterior()

    assert restored.checkpoint_payload() == manager.checkpoint_payload()
    assert restored.state == expected
    assert replayed.members == expected.members
    assert replayed.evidence_ledger == expected.evidence_ledger
    assert expected_record.applied_contributions == tuple(
        sorted((first, second), key=lambda item: item.contribution_id)
    )

    next_evidence = protocol.make_contribution(
        competition_set_id=expected.competition_set_id,
        rule_id="displacement_impact",
        source_event_ids=("event:displacement:resume",),
        known_at=T0 + pd.Timedelta(minutes=2),
        correlation_key="displacement_impact:resume",
        require_admitted=True,
    )
    uninterrupted, uninterrupted_record = manager.advance(
        asof=T0 + pd.Timedelta(minutes=2),
        contributions=(next_evidence,),
        real_completed_bar=False,
    )
    resumed, resumed_record = restored.advance(
        asof=T0 + pd.Timedelta(minutes=2),
        contributions=(next_evidence,),
        real_completed_bar=False,
    )
    assert resumed == uninterrupted
    assert resumed_record == uninterrupted_record

    restored.reset()
    assert restored.state is None
    assert restored.update_ledger == ()


def test_checkpoint_rejects_ledger_payload_that_differs_from_admitted_rule() -> None:
    protocol = _production_protocol()
    manager, state = _initialized_manager(protocol)
    evidence = protocol.make_contribution(
        competition_set_id=state.competition_set_id,
        rule_id="acceptance_continuation",
        source_event_ids=("event:acceptance:tamper",),
        known_at=T0 + pd.Timedelta(minutes=1),
        correlation_key="acceptance_continuation:tamper",
        require_admitted=True,
    )
    manager.advance(
        asof=T0 + pd.Timedelta(minutes=1),
        contributions=(evidence,),
        real_completed_bar=False,
    )
    payload = json.loads(json.dumps(manager.checkpoint_payload()))
    payload["state"]["evidence_ledger"][0]["log_likelihood_increments"][
        "continuation"
    ] = 0.25

    with pytest.raises(ValueError, match="registered rule"):
        HypothesisManager.from_checkpoint(payload, protocol=protocol)


def test_checkpoint_rejects_posterior_that_differs_from_causal_ledger() -> None:
    protocol = _production_protocol()
    manager, _state = _initialized_manager(protocol)
    payload = manager.checkpoint_payload()
    payload["state"]["members"][0]["probability"] = 0.25
    remaining = 0.75 / 5.0
    for member in payload["state"]["members"][1:]:
        member["probability"] = remaining

    with pytest.raises(ValueError, match="causal ledger"):
        HypothesisManager.from_checkpoint(payload, protocol=protocol)


def test_checkpoint_cannot_terminalize_path_without_exact_terminal_event() -> None:
    protocol = _production_protocol()
    manager, _state = _initialized_manager(protocol)
    payload = manager.checkpoint_payload()
    forged = payload["state"]["members"][0]
    forged.update(
        {
            "status": "invalidated",
            "log_weight": None,
            "probability": 0.0,
            "terminal_at": T0.isoformat(),
            "terminal_reason": "forged_without_terminal_event",
            "terminal_source_event_ids": ["event:forged-terminal"],
        }
    )
    for member in payload["state"]["members"][1:]:
        member["probability"] = 0.2

    with pytest.raises(ValueError, match="competition set"):
        HypothesisManager.from_checkpoint(payload, protocol=protocol)


def test_new_evidence_cannot_be_delayed_from_an_already_reduced_clock() -> None:
    protocol = _production_protocol()
    manager, state = _initialized_manager(protocol)
    delayed = protocol.make_contribution(
        competition_set_id=state.competition_set_id,
        rule_id="acceptance_continuation",
        source_event_ids=("event:acceptance:late",),
        known_at=T0,
        correlation_key="acceptance_continuation:late",
        require_admitted=True,
    )

    with pytest.raises(ValueError, match="follow the reducer state clock"):
        manager.advance(
            asof=T0 + pd.Timedelta(minutes=1),
            contributions=(delayed,),
            real_completed_bar=False,
        )


def test_path_terminal_cannot_be_delayed_from_an_already_reduced_clock() -> None:
    protocol = _production_protocol()
    manager, state = _initialized_manager(protocol)
    delayed = protocol.make_terminal_event(
        competition_set_id=state.competition_set_id,
        path=PathKind.BALANCE,
        rule_id="registered_path_invalidation",
        reason="delayed_terminal",
        source_event_ids=("event:path-terminal:late",),
        known_at=T0,
    )

    with pytest.raises(ValueError, match="follow the reducer state clock"):
        manager.advance(
            asof=T0 + pd.Timedelta(minutes=1),
            terminal_events=(delayed,),
            real_completed_bar=False,
        )


def test_explicit_outcome_realizes_one_winner_and_terminalizes_competitors() -> None:
    protocol = _production_protocol()
    manager, state = _initialized_manager(protocol)
    outcome = protocol.make_outcome_event(
        competition_set_id=state.competition_set_id,
        winner_path=PathKind.REVERSAL,
        reason="registered_reversal_path_realized",
        source_event_ids=("event:path-outcome:1",),
        known_at=T0 + pd.Timedelta(minutes=1),
    )

    realized, record = manager.advance(
        asof=T0 + pd.Timedelta(minutes=1),
        outcome_events=(outcome,),
        real_completed_bar=False,
    )

    assert realized.status is PathStatus.REALIZED
    assert realized.winner_path is PathKind.REVERSAL
    assert realized.outcome_event_id == outcome.outcome_event_id
    assert realized.member(PathKind.REVERSAL).status is PathStatus.REALIZED
    assert realized.member(PathKind.REVERSAL).probability == 1.0
    assert all(
        item.status is PathStatus.INVALIDATED and item.probability == 0.0
        for item in realized.members
        if item.path is not PathKind.REVERSAL
    )
    assert record.applied_outcome_event == outcome
    restored = HypothesisManager.from_checkpoint(
        manager.checkpoint_payload(),
        protocol=protocol,
    )
    assert restored.state == realized


def test_outcome_preserves_a_competitor_that_expired_before_realization() -> None:
    protocol = _production_protocol()
    manager, state = _initialized_manager(protocol)
    expiry_clock = T0 + pd.Timedelta(minutes=1)
    terminal = protocol.make_terminal_event(
        competition_set_id=state.competition_set_id,
        path=PathKind.BALANCE,
        rule_id="registered_path_expiry",
        reason="registered_balance_path_expired",
        source_event_ids=("event:path-expiry:balance",),
        known_at=expiry_clock,
    )
    state, _ = manager.advance(
        asof=expiry_clock,
        terminal_events=(terminal,),
        real_completed_bar=False,
    )
    outcome_clock = T0 + pd.Timedelta(minutes=2)
    outcome = protocol.make_outcome_event(
        competition_set_id=state.competition_set_id,
        winner_path=PathKind.REVERSAL,
        reason="registered_reversal_path_realized",
        source_event_ids=("event:path-outcome:after-expiry",),
        known_at=outcome_clock,
    )

    realized, _ = manager.advance(
        asof=outcome_clock,
        outcome_events=(outcome,),
        real_completed_bar=False,
    )

    assert realized.status is PathStatus.REALIZED
    assert realized.member(PathKind.BALANCE).status is PathStatus.EXPIRED
    assert realized.member(PathKind.REVERSAL).status is PathStatus.REALIZED
    restored = HypothesisManager.from_checkpoint(
        manager.checkpoint_payload(),
        protocol=protocol,
    )
    assert restored.state == realized


def test_active_set_cannot_contain_a_realized_member() -> None:
    protocol = _production_protocol()
    _manager, state = _initialized_manager(protocol)
    members = tuple(
        replace(
            member,
            status=PathStatus.REALIZED,
            log_weight=None,
            probability=1.0,
            terminal_at=T0,
            terminal_reason="forged_realization",
            terminal_source_event_ids=("event:forged",),
        )
        if member.path is PathKind.CONTINUATION
        else replace(member, probability=0.2)
        for member in state.members
    )

    with pytest.raises(ValueError, match="competition set"):
        replace(state, members=members)


def test_realized_checkpoint_rejects_a_forged_outcome_identity() -> None:
    protocol = _production_protocol()
    manager, state = _initialized_manager(protocol)
    outcome = protocol.make_outcome_event(
        competition_set_id=state.competition_set_id,
        winner_path=PathKind.REVERSAL,
        reason="registered_reversal_path_realized",
        source_event_ids=("event:path-outcome:checkpoint",),
        known_at=T0 + pd.Timedelta(minutes=1),
    )
    manager.advance(
        asof=T0 + pd.Timedelta(minutes=1),
        outcome_events=(outcome,),
        real_completed_bar=False,
    )
    payload = manager.checkpoint_payload()
    payload["state"]["outcome_event_id"] = "path-outcome:" + "0" * 32

    with pytest.raises(ValueError, match="competition set"):
        HypothesisManager.from_checkpoint(payload, protocol=protocol)
