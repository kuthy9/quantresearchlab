from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.core.eye_view import EvidenceRule, build_eye_context
from brain.core.llm_client import LLMReply, LLMTimeout, ScriptedClient
from brain.core.main_brain import MainBrain, MainBrainConfig
from brain.core.object_registry import ObjectRegistry
from brain.core.position_ledger import InMemoryPositionLedger
from contract.brain.llm import LLM_UPDATE_EXAMPLE
from shares.core.eye_factory import build_eye
from shares.tests.helpers import session_bars

ROOT = Path(__file__).resolve().parents[2]
CONFIG = MainBrainConfig.from_json(ROOT / "brain" / "configs" / "main_brain.json")
RULE = EvidenceRule(frozenset({"5m", "15m", "1H", "4H"}), frozenset({"bar_completed", "market_epoch_reset"}), "_state")
EMPTY_TAPE = {"bars": 0, "counts": {}, "recent": []}


@pytest.fixture(scope="module")
def context():
    reader, observer = build_eye(ROOT / "configs" / "model.json", root=ROOT, audit_journal_dir=None)
    registry = ObjectRegistry()
    last = None
    for bar in session_bars(1):
        obs = observer.observe(reader.on_bar(bar))
        if obs.market_snapshot is not None:
            last = build_eye_context(obs, registry, rule=RULE)
    return last, registry


def good_reply(ctx, **over) -> LLMReply:
    payload = json.loads(json.dumps(LLM_UPDATE_EXAMPLE))
    payload["evidence_verdicts"] = [
        {"evidence_id": e.evidence_id, "verdict": "NEUTRAL", "note": "", "resolves_evidence_id": None, "resolution": None}
        for e in ctx.events
    ]
    payload["market_understanding"] = "synthetic tape"
    payload["watch_next"] = []
    payload["destination_candidates"] = []
    payload.update(over)
    return LLMReply(json.dumps(payload), "thinking…", {"prompt_tokens": 1, "completion_tokens": 1}, 5, "fake")


def test_config_and_prompt_load() -> None:
    assert CONFIG.model == "deepseek-flash" and "json" in CONFIG.system_prompt.lower()
    assert len(CONFIG.prompt_sha256) == 64 and len(CONFIG.sha256) == 64
    assert "step_14" in CONFIG.system_prompt and "{EXAMPLE}" not in CONFIG.system_prompt
    assert CONFIG.retry_policy.max_retries == 3


def test_input_carries_prior_state_and_every_alias(context) -> None:
    ctx, registry = context
    brain = MainBrain(client=ScriptedClient([]), config=CONFIG, ledger=InMemoryPositionLedger())
    llm_input = brain.build_input(episode_id="EP_1", context=ctx, trigger_kind="WAKE", reasons=["ev_x"], tape=EMPTY_TAPE, prior=None)
    payload = llm_input.to_dict()
    assert payload["prior_state"] is None and payload["trigger"] == {"kind": "WAKE", "reasons": ["ev_x"]}
    assert {rel["object_id"] for rel in payload["price_relations"]} == ctx.visible_aliases()
    assert "close" in payload["bar"] and payload["known_at"] == llm_input.known_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert payload["tape_since_last_update"] == EMPTY_TAPE


def test_wake_step_reduces_a_good_reply(context) -> None:
    ctx, registry = context
    brain = MainBrain(client=ScriptedClient([good_reply(ctx)]), config=CONFIG, ledger=InMemoryPositionLedger())
    step = brain.step(episode_id="EP_1", context=ctx, trigger_kind="WAKE", reasons=[], tape=EMPTY_TAPE, prior=None, registry=registry, tick=0.25)
    assert step.result.state.revision == 0 and step.result.state.market_understanding == "synthetic tape"
    assert step.outcome.reply.reasoning_content == "thinking…" and step.result.incident is None
    assert set(step.result.state.object_registry) >= ctx.visible_aliases()


def test_wake_step_with_exhausted_timeouts_opens_an_empty_state(context) -> None:
    ctx, registry = context
    brain = MainBrain(client=ScriptedClient([LLMTimeout("t")] * 4), config=CONFIG, ledger=InMemoryPositionLedger(), sleep=lambda s: None)
    step = brain.step(episode_id="EP_1", context=ctx, trigger_kind="WAKE", reasons=[], tape=EMPTY_TAPE, prior=None, registry=registry, tick=0.25)
    assert step.result.incident == "LLMTimeout" and step.result.state.revision == 0
    assert step.result.state.continue_active is True and step.outcome.attempts == 4


def test_update_step_carries_state_forward_on_incident(context) -> None:
    ctx, registry = context
    brain = MainBrain(client=ScriptedClient([good_reply(ctx)] + [LLMTimeout("t")] * 4), config=CONFIG, ledger=InMemoryPositionLedger(), sleep=lambda s: None)
    first = brain.step(episode_id="EP_1", context=ctx, trigger_kind="WAKE", reasons=[], tape=EMPTY_TAPE, prior=None, registry=registry, tick=0.25)
    second = brain.step(episode_id="EP_1", context=ctx, trigger_kind="UPDATE", reasons=["ev_1"], tape={"bars": 1, "counts": {}, "recent": []}, prior=first.result.state, registry=registry, tick=0.25)
    assert second.result.incident == "LLMTimeout" and second.result.state.revision == 1
    assert second.result.state.market_understanding == first.result.state.market_understanding
    assert second.llm_input.to_dict()["prior_state"]["revision"] == 0


def test_opportunity_naming_a_visible_object_survives_reduce(context) -> None:
    ctx, registry = context
    zones = [v for v in ctx.objects if v.kind in {"fvg", "ob"}]
    pools_above = [v for v in ctx.objects if v.kind == "bsl" and v.lower > ctx.close]
    pools_below = [v for v in ctx.objects if v.kind == "ssl" and v.upper < ctx.close]
    if not (zones and pools_above and pools_below):
        pytest.skip("synthetic tape has no zone/pool triple on this bar")
    zone = min(zones, key=lambda v: abs(v.anchor - ctx.close))
    opportunity = {
        "state": "DEVELOPING", "direction": "LONG", "entry_object_id": zone.alias,
        "invalidation_object_id": min(pools_below, key=lambda v: v.upper).alias,
        "target_object_id": max(pools_above, key=lambda v: v.lower).alias,
    }
    brain = MainBrain(client=ScriptedClient([good_reply(ctx, opportunity=opportunity)]), config=CONFIG, ledger=InMemoryPositionLedger())
    step = brain.step(episode_id="EP_1", context=ctx, trigger_kind="WAKE", reasons=[], tape=EMPTY_TAPE, prior=None, registry=registry, tick=0.25)
    assert step.result.state.opportunity.entry_object_id == zone.alias and step.result.rejections == ()
