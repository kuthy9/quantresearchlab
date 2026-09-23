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
    listed = {rel["object_id"] for rel in payload["price_relations"]}
    assert listed <= ctx.visible_aliases() and listed
    kinds = {alias: view.kind for alias, view in ctx.object_map().items()}
    assert all(rel["offset_atr"] is None or kinds[rel["object_id"]] in ("bsl", "ssl") or abs(rel["offset_atr"]) <= CONFIG.relation_atr_limit for rel in payload["price_relations"])
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


def test_prior_state_evidence_is_bounded_with_counts(context) -> None:
    import pandas as pd
    from contract.brain.state import EvidenceItem, EvidenceLedger, Verdict
    from brain.tests.test_brain_state import make_state

    ctx, registry = context
    many = tuple(EvidenceItem(f"ev_{i}", pd.Timestamp("2022-01-04T14:41:00Z"), "k", "5m", None, Verdict.SUPPORT) for i in range(50))
    prior = make_state(evidence=EvidenceLedger(supporting=many), watch_next=(), destination_candidates=())
    brain = MainBrain(client=ScriptedClient([]), config=CONFIG, ledger=InMemoryPositionLedger())
    payload = brain.build_input(episode_id="EP_1", context=ctx, trigger_kind="UPDATE", reasons=[], tape=EMPTY_TAPE, prior=prior).to_dict()
    evidence = payload["prior_state"]["evidence"]
    assert len(evidence["supporting"]) == CONFIG.prior_evidence_limit == 12
    assert evidence["supporting"][-1]["evidence_id"] == "ev_49" and evidence["counts"]["supporting"] == 50
    assert "object_registry" not in payload["prior_state"]


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
        "thesis_id": "T1", "grade": "BASE", "invalidation_mode": "TOUCH",
    }
    brain = MainBrain(client=ScriptedClient([good_reply(ctx, opportunity=opportunity, bias={"direction": "LONG", "scale": "15m", "basis": "scripted"})]), config=CONFIG, ledger=InMemoryPositionLedger())
    step = brain.step(episode_id="EP_1", context=ctx, trigger_kind="WAKE", reasons=[], tape=EMPTY_TAPE, prior=None, registry=registry, tick=0.25)
    assert step.result.state.opportunity.entry_object_id == zone.alias and step.result.rejections == ()


def test_pending_evidence_is_reoffered_and_neutral_history_is_bounded(context) -> None:
    import pandas as pd
    from contract.brain.state import EvidenceItem, EvidenceLedger, Verdict
    from brain.tests.test_brain_state import make_state

    ctx, registry = context
    at = pd.Timestamp("2022-01-04T14:41:00Z")
    pending = EvidenceItem("ev_pending", at, "sweep_confirmed", "5m", None)
    neutral = tuple(EvidenceItem(f"ev_n{i}", at, "k", "5m", None, Verdict.NEUTRAL) for i in range(30))
    prior = make_state(evidence=EvidenceLedger(unresolved=neutral + (pending,)), watch_next=(), destination_candidates=())
    brain = MainBrain(client=ScriptedClient([]), config=CONFIG, ledger=InMemoryPositionLedger())
    llm_input = brain.build_input(episode_id="EP_1", context=ctx, trigger_kind="UPDATE", reasons=[], tape=EMPTY_TAPE, prior=prior)
    payload = llm_input.to_dict()
    ids = [e["evidence_id"] for e in payload["new_evidence"]]
    assert ids == [e.evidence_id for e in ctx.events] + ["ev_pending"]
    assert payload["new_evidence"][-1]["pending_since"] == "2022-01-04T14:41:00Z"
    evidence = payload["prior_state"]["evidence"]
    assert all(e["evidence_id"] != "ev_pending" for e in evidence["unresolved"])
    assert len(evidence["unresolved"]) == CONFIG.prior_evidence_limit
    assert evidence["counts"] == {"supporting": 0, "contradicting": 0, "unresolved": 30, "pending": 1}
    assert set(evidence["unresolved"][0]) == {"evidence_id", "kind", "timeframe", "object_id", "verdict", "note"}


def test_update_step_files_a_pending_item_by_the_verdict_it_finally_gets(context) -> None:
    import pandas as pd
    from contract.brain.state import EvidenceItem, EvidenceLedger
    from brain.tests.test_brain_state import make_state

    ctx, registry = context
    pending = EvidenceItem("ev_pending", pd.Timestamp("2022-01-04T14:41:00Z"), "sweep_confirmed", "5m", None)
    prior = make_state(evidence=EvidenceLedger(unresolved=(pending,)), watch_next=(), destination_candidates=())
    verdicts = [
        {"evidence_id": e.evidence_id, "verdict": "NEUTRAL", "note": "", "resolves_evidence_id": None, "resolution": None}
        for e in ctx.events
    ] + [{"evidence_id": "ev_pending", "verdict": "SUPPORT", "note": "late", "resolves_evidence_id": None, "resolution": None}]
    client = ScriptedClient([good_reply(ctx, evidence_verdicts=verdicts, continue_active=False)])
    brain = MainBrain(client=client, config=CONFIG, ledger=InMemoryPositionLedger())
    step = brain.step(
        episode_id="EP_1", context=ctx, trigger_kind="UPDATE", reasons=[], tape=EMPTY_TAPE, prior=prior, registry=registry, tick=0.25,
    )
    assert step.result.incident is None
    assert [e.evidence_id for e in step.result.state.evidence.supporting] == ["ev_pending"]
    assert not any(e.verdict is None for e in step.result.state.evidence.unresolved)
    assert "sleep_refused:unresolved_evidence" not in step.result.rejections


def test_prior_notes_are_truncated_and_relations_are_bounded_by_atr_distance(context) -> None:
    import dataclasses
    import pandas as pd
    from contract.brain.state import EvidenceItem, EvidenceLedger, Verdict, WatchItem
    from brain.tests.test_brain_state import make_state

    ctx, registry = context
    far = max(ctx.price_relations, key=lambda r: abs(r["offset_atr"] or 0.0))
    if not far["offset_atr"] or abs(far["offset_atr"]) < 0.5:
        pytest.skip("synthetic tape has no object beyond 0.5 ATR")
    prior = make_state(
        evidence=EvidenceLedger(supporting=(EvidenceItem("ev_1", pd.Timestamp("2022-01-04T14:41:00Z"), "k", "5m", None, Verdict.SUPPORT, "x" * 500),)),
        watch_next=(WatchItem(far["object_id"], "far but watched"),), destination_candidates=(),
        object_registry={far["object_id"]: registry.get(far["object_id"])},
    )
    config = dataclasses.replace(CONFIG, relation_atr_limit=0.25, note_limit=40)
    brain = MainBrain(client=ScriptedClient([]), config=config, ledger=InMemoryPositionLedger())
    payload = brain.build_input(episode_id="EP_1", context=ctx, trigger_kind="UPDATE", reasons=[], tape=EMPTY_TAPE, prior=prior).to_dict()
    assert len(payload["prior_state"]["evidence"]["supporting"][0]["note"]) == 40
    kept = {r["object_id"]: r for r in payload["price_relations"]}
    assert far["object_id"] in kept, "a watched object is kept whatever its distance"
    for alias, rel in kept.items():
        assert alias == far["object_id"] or ctx.object_map()[alias].kind in ("bsl", "ssl") or rel["offset_atr"] is None or abs(rel["offset_atr"]) <= 0.25  # pools are placed at any distance (2026-09-20)
    assert len(kept) < len(ctx.price_relations)


def test_deferred_bookkeeping_evidence_rides_the_next_call(context) -> None:
    import pandas as pd
    from contract.brain.state import EvidenceItem

    ctx, registry = context
    deferred = EvidenceItem("ev_deferred", pd.Timestamp("2022-01-04T14:41:00Z"), "fvg_created", "5m", None)
    brain = MainBrain(client=ScriptedClient([]), config=CONFIG, ledger=InMemoryPositionLedger())
    payload = brain.build_input(episode_id="EP_1", context=ctx, trigger_kind="UPDATE", reasons=[], tape=EMPTY_TAPE, prior=None, deferred=(deferred,)).to_dict()
    ids = [e["evidence_id"] for e in payload["new_evidence"]]
    assert ids == [e.evidence_id for e in ctx.events] + ["ev_deferred"]
    assert payload["new_evidence"][-1]["pending_since"] is None


def test_prior_state_carries_the_execution_view_and_a_wake_call_does_not(context) -> None:
    from brain.core.position_ledger import IDLE_VIEW

    ctx, registry = context

    class Engaged(InMemoryPositionLedger):
        def execution_view(self):
            return {**IDLE_VIEW, "status": "WORKING", "order": {"direction": "LONG", "entry_object_id": "FVG_5m_1", "bars_working": 2}}

    brain = MainBrain(client=ScriptedClient([good_reply(ctx)]), config=CONFIG, ledger=Engaged())
    first = brain.step(episode_id="EP_1", context=ctx, trigger_kind="WAKE", reasons=[], tape=EMPTY_TAPE, prior=None, registry=registry, tick=0.25)
    assert first.llm_input.to_dict()["prior_state"] is None
    payload = brain.build_input(episode_id="EP_1", context=ctx, trigger_kind="UPDATE", reasons=[], tape=EMPTY_TAPE, prior=first.result.state).to_dict()
    idle_view = json.loads(json.dumps(dict(IDLE_VIEW)))  # the input is JSON: tuples become lists
    assert payload["prior_state"]["execution"] == {**idle_view, "status": "WORKING", "order": {"direction": "LONG", "entry_object_id": "FVG_5m_1", "bars_working": 2}}
    assert set(payload["prior_state"]["execution"]) == {"status", "order", "positions", "theses", "cooldown_bars_left", "daily_stop", "halted", "last_outcome", "last_veto"}
    idle = MainBrain(client=ScriptedClient([]), config=CONFIG, ledger=InMemoryPositionLedger())
    payload = idle.build_input(episode_id="EP_1", context=ctx, trigger_kind="UPDATE", reasons=[], tape=EMPTY_TAPE, prior=first.result.state).to_dict()
    assert payload["prior_state"]["execution"] == idle_view


def test_the_prompt_explains_the_execution_view() -> None:
    assert "prior_state.execution" in CONFIG.system_prompt and "last_veto" in CONFIG.system_prompt
    for word in ("thesis_id", "governing_timeframe", "CLOSE_BEYOND", "A_PLUS", "cooldown_bars_left", "daily_stop", "halted", "theses"):
        assert word in CONFIG.system_prompt, word
    assert "name a nearer invalidation" not in CONFIG.system_prompt and "never move the invalidation" in CONFIG.system_prompt


def test_the_prompt_defines_position_and_offset() -> None:
    assert "`position`" in CONFIG.system_prompt and "`offset_atr`" in CONFIG.system_prompt
    assert "above_price" in CONFIG.system_prompt and "below_price" in CONFIG.system_prompt and "contains_price" in CONFIG.system_prompt
    assert "distance_atr" not in CONFIG.system_prompt


def test_timings_split_input_call_and_reduce(context) -> None:
    from shares.core.timing import Timings

    ctx, registry = context
    timings = Timings()
    brain = MainBrain(client=ScriptedClient([good_reply(ctx)]), config=CONFIG, ledger=InMemoryPositionLedger(), timings=timings)
    brain.step(episode_id="EP_1", context=ctx, trigger_kind="WAKE", reasons=[], tape=EMPTY_TAPE, prior=None, registry=registry, tick=0.25)
    summary = timings.summary()
    assert {"input", "llm", "reduce"} <= set(summary) and all(summary[k]["count"] == 1 for k in ("input", "llm", "reduce"))


def test_the_prompt_defines_the_bias_and_how_the_facts_set_it() -> None:
    text = CONFIG.system_prompt
    for word in ("## Bias", "`bias`", "active_leg_direction", "forming_leg_atr", "displacement_age_bars", "`reset`", "drift_atr", "contains_price", "live delivery"):
        assert word in text, word
    assert "The thesis is judged on its governing scale" not in text
    assert "judged on the bias scale" in text and "judged on the thesis scale" in text
    assert "stays live" in text and "`bias.scale` ∈ 4H | 1H | 15m;" in text  # run B's amendment: hysteresis, no 5m bias


def test_config_carries_the_pending_evidence_bound() -> None:
    assert CONFIG.max_pending_evidence == 32


def test_the_prior_view_shows_the_last_rejections_and_the_input_schema_is_3() -> None:
    from brain.tests.test_brain_state import T1, make_state
    from contract.brain.llm import LLM_INPUT_SCHEMA_VERSION
    from contract.brain.state import LastUpdate

    brain = MainBrain(client=ScriptedClient([]), config=CONFIG, ledger=InMemoryPositionLedger())
    state = make_state(last_update=LastUpdate(T1, True, {}, None, rejections=("opportunity_incoherent:LONG entry 105 lies above price 103",)))
    assert brain._prior_view(state)["last_update"]["rejections"] == ["opportunity_incoherent:LONG entry 105 lies above price 103"]
    assert LLM_INPUT_SCHEMA_VERSION == 3


def test_the_prompt_separates_the_bias_from_the_entry() -> None:
    text = CONFIG.system_prompt
    for word in ("## Expression", "the pullback picks the price", "Nearest first", "Follow the leg", "last_update.rejections", "ttl_bars", "midpoint", "not an entry"):
        assert word in text, word
    assert "the order fills now" not in text and "entry.range" not in text


def test_every_pool_is_placed_whatever_its_distance_and_zones_stay_near(context) -> None:
    # 2026-09-20: pools are where trades go, so the LLM must see which side of price each one lies on;
    # zones and swings are where trades are entered and stay bounded by relation_atr_limit.
    ctx, registry = context
    from dataclasses import replace

    config = replace(CONFIG, relation_atr_limit=0.25)
    brain = MainBrain(client=ScriptedClient([]), config=config, ledger=InMemoryPositionLedger())
    rows = {row["object_id"]: row for row in brain._relations_view(ctx, None)}
    kinds = {alias: view.kind for alias, view in ctx.object_map().items()}
    offsets = {row["object_id"]: row["offset_atr"] for row in ctx.price_relations}
    far_pools = [a for a, k in kinds.items() if k in ("bsl", "ssl") and offsets.get(a) is not None and abs(offsets[a]) > 0.25]
    far_zones = [a for a, k in kinds.items() if k in ("fvg", "ob", "swing_high", "swing_low", "range") and offsets.get(a) is not None and abs(offsets[a]) > 0.25]
    assert far_pools and far_zones, "the synthetic tape should have pools and zones beyond a quarter ATR"
    assert all(a in rows for a in far_pools)
    assert not any(a in rows for a in far_zones)


def test_the_prompt_names_the_stop_floor_and_the_governing_scale_target() -> None:
    # 2026-09-21: the stop never sits nearer the entry than one bar of the governing scale, so the target is named on that scale
    text = CONFIG.system_prompt
    for word in ("never nearer the entry than one bar", "governing scale", "two governing bars", "stop.floor.governing_bar"):
        assert word in text, word


def test_the_prompt_names_the_thesis_scale_the_bias_decay_and_the_structural_exit() -> None:
    # 2026-09-22: the thesis scale is code's (one below the bias scale, never below the 15m), the target lies on it,
    # a bias decays on the scales below it and comes back on structure only, and a position leaves on its own scale's reversal
    text = CONFIG.system_prompt
    for word in (
        "thesis scale", "one scale below the bias scale", "never below the 15m", "opportunity_target_scale", "bias_decayed",
        "bias_reassert_refused", "structure_reversed", "`since`", "`decayed`", "`decayed_at`", "No trade in a balance", "survives sleep", "`account_risk`", "all three are `null`",
    ):
        assert word in text, word
    for gone in ("bias_reversed", "`governing_timeframe` — `4H`, `1H`, `15m` or `5m`", "up to three", "opportunity_scale_above_bias"):
        assert gone not in text, gone
    assert "governing_timeframe" not in json.dumps(LLM_UPDATE_EXAMPLE), "the reply no longer names the thesis scale"
