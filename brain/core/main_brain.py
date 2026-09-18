"""The Main Brain: one reasoning step.

Build the ``LLMInput`` from the ``EyeContext`` and the prior state, prove it
causal, call the LLM under the retry policy, parse the reply against the
contract, and hand everything to the reducer.  A failed call is an incident,
not an exception: the reducer carries the state forward and the runtime stays
ACTIVE.  The fourteen-step framework lives in the system prompt, whose sha256
is part of every run's identity."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from brain.core.eye_view import EyeContext, assert_causal
from brain.core.llm_client import CallOutcome, LLMClient, RetryPolicy, call_with_policy
from brain.core.object_registry import ObjectRegistry
from brain.core.opportunity_geometry import coherence_error
from brain.core.position_ledger import PositionLedger, engaged
from brain.core.reducer import ReduceContext, ReduceResult, apply, pending_evidence
from contract.brain.llm import LLM_UPDATE_EXAMPLE, LLMInput, parse_update
from contract.brain.state import BrainState, EvidenceItem, Opportunity, isoformat_utc
from shares.core.timing import NO_TIMINGS, Timings, timed

MAIN_BRAIN_SCHEMA_VERSION = 1
EXAMPLE_TOKEN = "{EXAMPLE}"


@dataclass(frozen=True)
class MainBrainConfig:
    model: str
    timeout_s: float
    max_tokens: int
    max_retries: int
    backoff_base_s: float
    backoff_cap_s: float
    system_prompt: str
    sha256: str
    prompt_sha256: str
    # How many of the most recent supporting / contradicting evidence items the
    # LLM sees in ``prior_state``.  The state keeps the whole ledger; the input
    # carries a bounded tail plus the counts, or a long episode's input grows
    # without limit (measured: 12k → 190k characters over one Globex session).
    prior_evidence_limit: int = 20
    # Evidence notes are cut to this many characters in ``prior_state``.
    note_limit: int = 160
    # Only objects within this many 1m ATRs of the close are listed in
    # ``price_relations`` — plus every object the prior state names.
    relation_atr_limit: float | None = 4.0
    # DeepSeek's reasoning budget (low / high / max); None sends nothing.
    reasoning_effort: str | None = None

    @classmethod
    def from_json(cls, path: Path, *, root: Path | None = None) -> "MainBrainConfig":
        path = Path(path)
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
        if payload.get("schema_version") != MAIN_BRAIN_SCHEMA_VERSION:
            raise ValueError("unsupported main_brain schema_version")
        base = Path(root) if root is not None else path.resolve().parents[2]
        prompt_path = base / payload["system_prompt"]
        template = prompt_path.read_text(encoding="utf-8")
        if EXAMPLE_TOKEN not in template:
            raise ValueError(f"system prompt {prompt_path} lacks the {EXAMPLE_TOKEN} token")
        example = json.dumps(LLM_UPDATE_EXAMPLE, indent=2, ensure_ascii=False)
        prompt = template.replace(EXAMPLE_TOKEN, example)
        return cls(
            model=str(payload["model"]),
            timeout_s=float(payload["timeout_s"]),
            max_tokens=int(payload["max_tokens"]),
            max_retries=int(payload["max_retries"]),
            backoff_base_s=float(payload["backoff_base_s"]),
            backoff_cap_s=float(payload["backoff_cap_s"]),
            system_prompt=prompt,
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            prior_evidence_limit=int(payload["prior_evidence_limit"]),
            note_limit=int(payload["note_limit"]),
            relation_atr_limit=None if payload["relation_atr_limit"] is None else float(payload["relation_atr_limit"]),
            reasoning_effort=None if payload.get("reasoning_effort") is None else str(payload["reasoning_effort"]),
        )

    @property
    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(self.max_retries, self.backoff_base_s, self.backoff_cap_s)


@dataclass(frozen=True)
class BrainStep:
    result: ReduceResult
    llm_input: LLMInput
    outcome: CallOutcome


class MainBrain:
    def __init__(
        self,
        *,
        client: LLMClient,
        config: MainBrainConfig,
        ledger: PositionLedger,
        sleep: Callable[[float], None] = time.sleep,
        timings: Timings = NO_TIMINGS,
    ) -> None:
        self._client = client
        self._config = config
        self._ledger = ledger
        self._sleep = sleep
        self._timings = timings

    @property
    def config(self) -> MainBrainConfig:
        return self._config

    def build_input(
        self,
        *,
        episode_id: str,
        context: EyeContext,
        trigger_kind: str,
        reasons: Sequence[str],
        tape: Mapping[str, Any],
        prior: BrainState | None,
        deferred: Sequence[EvidenceItem] = (),
    ) -> LLMInput:
        evidence = self.evidence_for(context, prior, deferred)
        fresh = len(context.events) + len(deferred)
        llm_input = LLMInput(
            episode_id=episode_id,
            known_at=context.known_at,
            trigger={"kind": trigger_kind, "reasons": list(reasons)},
            bar=context.bar,
            session=context.session,
            scales=context.scales,
            interaction=context.interaction,
            new_evidence=tuple(
                {
                    "evidence_id": item.evidence_id,
                    "kind": item.kind,
                    "timeframe": item.timeframe,
                    "direction": item.direction,
                    "side": item.side,
                    "object_id": item.object_id,
                    "known_at": isoformat_utc(item.known_at),
                    # Set on an item an incident bar left unjudged: it is
                    # re-offered on every call until the LLM verdicts it.
                    "pending_since": None if index < fresh else isoformat_utc(item.known_at),
                }
                for index, item in enumerate(evidence)
            ),
            tape_since_last_update=tape,
            price_relations=self._relations_view(context, prior),
            prior_state=None if prior is None else self._prior_view(prior),
        )
        assert_causal(llm_input.to_dict(), context.known_at)
        return llm_input

    @staticmethod
    def evidence_for(
        context: EyeContext, prior: BrainState | None, deferred: Sequence[EvidenceItem] = ()
    ) -> tuple[EvidenceItem, ...]:
        """This bar's evidence, then the bookkeeping evidence of the TICK bars
        since the last call, then the prior state's pending items — what the
        LLM must verdict, and what the reducer files."""
        pending = () if prior is None else pending_evidence(prior.evidence)
        return tuple(context.events) + tuple(deferred) + pending

    def _relations_view(self, context: EyeContext, prior: BrainState | None) -> tuple[Mapping[str, Any], ...]:
        """``price_relations`` bounded to the objects near price, plus every
        object the prior state names (watched, destination, opportunity)."""
        limit = self._config.relation_atr_limit
        if limit is None:
            return context.price_relations
        named: set[str] = set()
        if prior is not None:
            named.update(item.object_id for item in prior.watch_next)
            named.update(prior.destination_candidates)
            named.update(prior.opportunity.object_ids)
        return tuple(
            row for row in context.price_relations
            if row["object_id"] in named or row["offset_atr"] is None or abs(row["offset_atr"]) <= limit
        )

    def _prior_view(self, prior: BrainState) -> dict[str, Any]:
        """The prior state as the LLM sees it: the state without its registry
        (the LLM names aliases; entity ids are the journal's business), the
        evidence ledger bounded to its most recent items in a compact shape,
        plus the counts it dropped, and ``execution`` — what the executor did
        with the last opportunity (the ledger's view: working order, position,
        last outcome, last veto).  Pending items are not listed here — they
        travel in ``new_evidence``."""
        payload = prior.to_dict()
        payload.pop("object_registry")
        payload["execution"] = dict(self._ledger.execution_view())
        limit = self._config.prior_evidence_limit
        note_limit = self._config.note_limit

        def compact(item: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "evidence_id": item["evidence_id"],
                "kind": item["kind"],
                "timeframe": item["timeframe"],
                "object_id": item["object_id"],
                "verdict": item["verdict"],
                "note": item["note"][:note_limit],
            }

        ledger = payload["evidence"]
        judged = [item for item in ledger["unresolved"] if item["verdict"] is not None]
        bounded = {
            "supporting": [compact(item) for item in ledger["supporting"][-limit:]],
            "contradicting": [compact(item) for item in ledger["contradicting"][-limit:]],
            "unresolved": [compact(item) for item in judged[-limit:]],
            "counts": {
                "supporting": len(ledger["supporting"]),
                "contradicting": len(ledger["contradicting"]),
                "unresolved": len(judged),
                "pending": len(ledger["unresolved"]) - len(judged),
            },
        }
        payload["evidence"] = bounded
        return payload

    def step(
        self,
        *,
        episode_id: str,
        context: EyeContext,
        trigger_kind: str,
        reasons: Sequence[str],
        tape: Mapping[str, Any],
        prior: BrainState | None,
        registry: ObjectRegistry,
        tick: float,
        deferred: Sequence[EvidenceItem] = (),
        idle_updates: int = 0,
        idle_archive_after: int | None = None,
    ) -> BrainStep:
        with timed(self._timings, "input"):
            llm_input = self.build_input(
                episode_id=episode_id, context=context, trigger_kind=trigger_kind,
                reasons=reasons, tape=tape, prior=prior, deferred=deferred,
            )
        evidence = self.evidence_for(context, prior, deferred)
        evidence_ids = {item.evidence_id for item in evidence}
        object_ids = set(context.visible_aliases())
        if prior is not None:
            object_ids |= set(prior.object_registry)

        def parse(text: str):
            return parse_update(text, evidence_ids=evidence_ids, object_ids=object_ids)

        with timed(self._timings, "llm"):
            outcome = call_with_policy(
                self._client,
                system=self._config.system_prompt,
                user=llm_input.to_json(),
                parse=parse,
                policy=self._config.retry_policy,
                sleep=self._sleep,
            )
        geometries = context.geometries()

        def coherence(opportunity: Opportunity) -> str | None:
            return coherence_error(opportunity, geometries, close=context.close, tick=tick, atr_1m=context.atr_1m)

        def timeframe_of(alias: str) -> str | None:
            view = geometries.get(alias)
            return None if view is None else view.timeframe

        ctx = ReduceContext(
            known_at=context.known_at,
            has_open_position=engaged(self._ledger),
            open_interaction=context.open_interaction,
            visible_aliases=context.visible_aliases(),
            registry=registry.snapshot(),
            coherence=coherence,
            idle_updates=idle_updates,
            idle_archive_after=idle_archive_after,
            timeframe_of=timeframe_of,
        )
        with timed(self._timings, "reduce"):
            result = apply(
                prior,
                episode_id=episode_id,
                evidence=evidence,
                update=outcome.update,
                ctx=ctx,
                incident=outcome.incident,
            )
        return BrainStep(result=result, llm_input=llm_input, outcome=outcome)


__all__ = ["EXAMPLE_TOKEN", "MAIN_BRAIN_SCHEMA_VERSION", "BrainStep", "MainBrain", "MainBrainConfig"]
