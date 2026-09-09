from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from brain.core.market_belief import (
    PATH_KINDS,
    PathBeliefProtocolError,
    PathKind,
    PathStatus,
    create_path_competition_set,
    initialize_path_competition_set,
    load_path_belief_protocol,
    reduce_path_competition_set,
    restore_path_competition_set,
)


T0 = pd.Timestamp("2026-08-21 09:30:00", tz="America/New_York")
EXPIRY = pd.Timestamp("2026-08-21 16:00:00", tz="America/New_York")


def _protocol():
    return load_path_belief_protocol("brain/configs/path_hypotheses.json")


def _state():
    return create_path_competition_set(
        _protocol(),
        instrument_id="NQ:front",
        market_epoch_id="epoch:nq:2026-08-21",
        authority_structure_id="h1-structure:123",
        horizon_id="ny-session:2026-08-21",
        formed_at=T0,
        common_expires_at=EXPIRY,
    )


def _member(state, path: PathKind):
    return state.member(path)


def test_protocol_is_explicitly_unvalidated_shadow_only() -> None:
    protocol = _protocol()

    assert protocol.status == "development_unvalidated"
    assert protocol.authority == "shadow_only"
    assert protocol.path_kinds == PATH_KINDS
    assert "not_calibrated" in protocol.probability_interpretation
    assert "conditional_likelihood" in protocol.weight_interpretation
    assert protocol.likelihood_artifact_status == "missing"
    assert protocol.evidence_allowlist == (
        "acceptance_continuation",
        "displacement_impact",
    )
    assert not protocol.can_apply_bayesian_update
    assert not protocol.can_authorize_action
    assert protocol.fingerprint


def test_protocol_rejects_any_claim_of_trading_authority(tmp_path: Path) -> None:
    source = Path("brain/configs/path_hypotheses.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["authority"] = "trading_authority"
    destination = tmp_path / "path_hypotheses.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PathBeliefProtocolError, match="shadow_only"):
        load_path_belief_protocol(destination)


def test_competition_set_is_mutually_exclusive_exhaustive_and_normalized() -> None:
    first = _state()
    second = _state()

    assert first == second
    assert first.status is PathStatus.ACTIVE
    assert tuple(member.path for member in first.members) == PATH_KINDS
    assert len({member.hypothesis_id for member in first.members}) == len(PATH_KINDS)
    assert all(member.common_expires_at == EXPIRY for member in first.members)
    assert sum(member.probability for member in first.members) == pytest.approx(1.0)
    assert all(member.probability == pytest.approx(1.0 / 6.0) for member in first.members)


def test_contribution_has_exact_causal_identity_without_unfitted_update() -> None:
    protocol = _protocol()
    state = _state()
    contribution = protocol.make_contribution(
        competition_set_id=state.competition_set_id,
        rule_id="acceptance_continuation",
        source_event_ids=("event:h1-protected", "event:h1-bos"),
        known_at=T0 + pd.Timedelta(minutes=1),
    )

    updated, record = reduce_path_competition_set(
        protocol,
        state,
        asof=T0 + pd.Timedelta(minutes=1),
        contributions=(contribution,),
        real_completed_bar=False,
    )

    assert contribution.model_version == protocol.model_version
    assert contribution.protocol_fingerprint == protocol.fingerprint
    assert contribution.rule_id == "acceptance_continuation"
    assert contribution.source_event_ids == (
        "event:h1-bos",
        "event:h1-protected",
    )
    assert contribution.known_at == T0 + pd.Timedelta(minutes=1)
    assert record.applied_contributions == (contribution,)
    assert record.log_normalizer is not None
    assert record.evidence_admission_only
    assert not record.bayesian_update_applied
    assert sum(member.probability for member in updated.members) == pytest.approx(1.0)
    assert _member(updated, PathKind.CONTINUATION).probability == pytest.approx(
        _member(updated, PathKind.REVERSAL).probability
    )


def test_initial_evidence_is_assimilated_at_the_exact_formation_clock() -> None:
    protocol = _protocol()
    state = _state()
    contribution = protocol.make_contribution(
        competition_set_id=state.competition_set_id,
        rule_id="acceptance_continuation",
        source_event_ids=("event:h1-authority",),
        known_at=T0,
    )

    initialized, record = initialize_path_competition_set(
        protocol,
        state,
        initial_contributions=(contribution,),
        initial_real_completed_bar=True,
    )

    assert initialized.formed_at == T0
    assert initialized.asof == T0
    assert initialized.last_real_completed_at == T0
    assert initialized.real_completed_bar_count == 1
    assert initialized.applied_contribution_ids == (
        contribution.contribution_id,
    )
    assert record.initialization
    assert record.from_asof == record.asof == T0
    assert record.applied_contributions == (contribution,)
    assert record.evidence_admission_only
    assert not record.bayesian_update_applied
    assert _member(initialized, PathKind.CONTINUATION).probability == pytest.approx(
        _member(initialized, PathKind.REVERSAL).probability
    )

    with pytest.raises(ValueError, match="pristine"):
        initialize_path_competition_set(
            protocol,
            initialized,
            initial_contributions=(contribution,),
            initial_real_completed_bar=False,
        )


def test_same_event_is_assimilated_once_even_when_replayed() -> None:
    protocol = _protocol()
    state = _state()
    contribution = protocol.make_contribution(
        competition_set_id=state.competition_set_id,
        rule_id="acceptance_continuation",
        source_event_ids=("event:balance",),
        known_at=T0 + pd.Timedelta(minutes=1),
    )
    once, _ = reduce_path_competition_set(
        protocol,
        state,
        asof=T0 + pd.Timedelta(minutes=1),
        contributions=(contribution, contribution),
        real_completed_bar=False,
    )
    twice, record = reduce_path_competition_set(
        protocol,
        once,
        asof=T0 + pd.Timedelta(minutes=2),
        contributions=(contribution,),
        real_completed_bar=False,
    )

    assert twice.members == once.members
    assert record.applied_contributions == ()
    assert record.duplicate_contribution_ids == (contribution.contribution_id,)
    assert once.applied_contribution_ids == (contribution.contribution_id,)

    much_later, late_record = reduce_path_competition_set(
        protocol,
        twice,
        asof=T0 + pd.Timedelta(minutes=10),
        contributions=(contribution,),
        real_completed_bar=False,
    )
    assert much_later.members == twice.members
    assert late_record.duplicate_contribution_ids == (
        contribution.contribution_id,
    )


@pytest.mark.parametrize(
    ("rule_id", "terminal_status"),
    [
        ("registered_path_invalidation", PathStatus.INVALIDATED),
        ("registered_path_expiry", PathStatus.EXPIRED),
    ],
)
def test_terminal_path_has_zero_probability_and_survivors_renormalize(
    rule_id: str,
    terminal_status: PathStatus,
) -> None:
    protocol = _protocol()
    state = _state()
    terminal = protocol.make_terminal_event(
        competition_set_id=state.competition_set_id,
        path=PathKind.CONTINUATION,
        rule_id=rule_id,
        reason=f"test_{terminal_status.value}",
        source_event_ids=("event:terminal",),
        known_at=T0 + pd.Timedelta(minutes=1),
    )

    updated, record = reduce_path_competition_set(
        protocol,
        state,
        asof=T0 + pd.Timedelta(minutes=1),
        terminal_events=(terminal,),
        real_completed_bar=False,
    )

    ended = _member(updated, PathKind.CONTINUATION)
    assert ended.status is terminal_status
    assert ended.log_weight is None
    assert ended.probability == 0.0
    assert ended.terminal_source_event_ids == ("event:terminal",)
    assert sum(member.probability for member in updated.members) == pytest.approx(1.0)
    assert record.applied_terminal_events == (terminal,)


def test_residual_unknown_cannot_be_removed_before_common_horizon() -> None:
    protocol = _protocol()
    state = _state()
    terminal = protocol.make_terminal_event(
        competition_set_id=state.competition_set_id,
        path=PathKind.RESIDUAL_UNKNOWN,
        rule_id="registered_path_invalidation",
        reason="attempted_residual_removal",
        source_event_ids=("event:bad",),
        known_at=T0 + pd.Timedelta(minutes=1),
    )

    with pytest.raises(ValueError, match="residual_unknown"):
        reduce_path_competition_set(
            protocol,
            state,
            asof=T0 + pd.Timedelta(minutes=1),
            terminal_events=(terminal,),
            real_completed_bar=False,
        )


def test_same_clock_terminal_precedes_descriptive_evidence() -> None:
    protocol = _protocol()
    state = _state()
    clock = T0 + pd.Timedelta(minutes=1)
    support = protocol.make_contribution(
        competition_set_id=state.competition_set_id,
        rule_id="acceptance_continuation",
        source_event_ids=("event:support",),
        known_at=clock,
    )
    terminal = protocol.make_terminal_event(
        competition_set_id=state.competition_set_id,
        path=PathKind.CONTINUATION,
        rule_id="registered_path_invalidation",
        reason="same_clock_invalidation",
        source_event_ids=("event:invalidation",),
        known_at=clock,
    )

    updated, record = reduce_path_competition_set(
        protocol,
        state,
        asof=clock,
        contributions=(support,),
        terminal_events=(terminal,),
        real_completed_bar=False,
    )

    continuation = _member(updated, PathKind.CONTINUATION)
    assert continuation.status is PathStatus.INVALIDATED
    assert continuation.probability == 0.0
    assert continuation.log_weight is None
    assert record.applied_terminal_events == (terminal,)
    assert record.applied_contributions == (support,)


def test_missing_likelihood_artifact_disables_unfitted_time_decay() -> None:
    protocol = _protocol()
    state = _state()
    synthetic, synthetic_record = reduce_path_competition_set(
        protocol,
        state,
        asof=T0 + pd.Timedelta(minutes=1),
        real_completed_bar=False,
    )
    real, real_record = reduce_path_competition_set(
        protocol,
        synthetic,
        asof=T0 + pd.Timedelta(minutes=2),
        real_completed_bar=True,
    )

    assert synthetic.members == state.members
    assert synthetic.real_completed_bar_count == 0
    assert synthetic.last_real_completed_at is None
    assert not synthetic_record.decay_applied
    assert real.real_completed_bar_count == 1
    assert real.last_real_completed_at == T0 + pd.Timedelta(minutes=2)
    assert not real_record.decay_applied
    assert real.members == synthetic.members


def test_common_horizon_expires_every_member_without_fake_distribution() -> None:
    protocol = _protocol()
    state = _state()

    expired, record = reduce_path_competition_set(
        protocol,
        state,
        asof=EXPIRY,
        real_completed_bar=True,
    )

    assert expired.status is PathStatus.EXPIRED
    assert all(member.status is PathStatus.EXPIRED for member in expired.members)
    assert all(member.log_weight is None for member in expired.members)
    assert all(member.probability == 0.0 for member in expired.members)
    assert all(
        member.terminal_source_event_ids == ()
        for member in expired.members
    )
    assert record.log_normalizer is None
    assert record.common_horizon_expired


def test_update_order_checkpoint_and_replay_are_deterministic() -> None:
    protocol = _protocol()
    state = _state()
    first = protocol.make_contribution(
        competition_set_id=state.competition_set_id,
        rule_id="acceptance_continuation",
        source_event_ids=("event:dfp",),
        known_at=T0 + pd.Timedelta(minutes=1),
    )
    second = protocol.make_contribution(
        competition_set_id=state.competition_set_id,
        rule_id="displacement_impact",
        source_event_ids=("event:obstruction",),
        known_at=T0 + pd.Timedelta(minutes=1),
    )

    forward, forward_record = reduce_path_competition_set(
        protocol,
        state,
        asof=T0 + pd.Timedelta(minutes=1),
        contributions=(first, second),
        real_completed_bar=True,
    )
    reverse, reverse_record = reduce_path_competition_set(
        protocol,
        state,
        asof=T0 + pd.Timedelta(minutes=1),
        contributions=(second, first),
        real_completed_bar=True,
    )

    assert forward == reverse
    assert forward_record == reverse_record
    restored = restore_path_competition_set(
        forward.state_dict(),
        protocol=protocol,
    )
    assert restored == forward
    assert pickle.loads(pickle.dumps(forward)) == forward

    next_contribution = protocol.make_contribution(
        competition_set_id=forward.competition_set_id,
        rule_id="acceptance_continuation",
        source_event_ids=("event:parent-acceptance",),
        known_at=T0 + pd.Timedelta(minutes=2),
    )
    expected, expected_record = reduce_path_competition_set(
        protocol,
        forward,
        asof=T0 + pd.Timedelta(minutes=2),
        contributions=(next_contribution,),
        real_completed_bar=True,
    )
    actual, actual_record = reduce_path_competition_set(
        protocol,
        restored,
        asof=T0 + pd.Timedelta(minutes=2),
        contributions=(next_contribution,),
        real_completed_bar=True,
    )
    assert actual == expected
    assert actual_record == expected_record


def test_future_or_wrong_model_evidence_fails_closed() -> None:
    protocol = _protocol()
    state = _state()
    future = protocol.make_contribution(
        competition_set_id=state.competition_set_id,
        rule_id="acceptance_continuation",
        source_event_ids=("event:future",),
        known_at=T0 + pd.Timedelta(minutes=2),
    )

    with pytest.raises(ValueError, match="future"):
        reduce_path_competition_set(
            protocol,
            state,
            asof=T0 + pd.Timedelta(minutes=1),
            contributions=(future,),
            real_completed_bar=False,
        )

    wrong_model = pickle.loads(pickle.dumps(future))
    object.__setattr__(wrong_model, "model_version", "wrong-model")
    with pytest.raises(ValueError, match="model or protocol"):
        reduce_path_competition_set(
            protocol,
            state,
            asof=T0 + pd.Timedelta(minutes=2),
            contributions=(wrong_model,),
            real_completed_bar=False,
        )
