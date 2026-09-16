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
from brain.core.position_ledger import PositionLedger
from brain.core.reducer import ReduceContext, ReduceResult, apply
from contract.brain.llm import LLM_UPDATE_EXAMPLE, LLMInput, parse_update
from contract.brain.state import BrainState, Opportunity

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
    ) -> None:
        self._client = client
        self._config = config
        self._ledger = ledger
        self._sleep = sleep

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
    ) -> LLMInput:
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
                    "known_at": item.to_dict()["known_at"],
                }
                for item in context.events
            ),
            tape_since_last_update=tape,
            price_relations=context.price_relations,
            prior_state=None if prior is None else prior.to_dict(),
        )
        assert_causal(llm_input.to_dict(), context.known_at)
        return llm_input

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
    ) -> BrainStep:
        llm_input = self.build_input(
            episode_id=episode_id, context=context, trigger_kind=trigger_kind,
            reasons=reasons, tape=tape, prior=prior,
        )
        evidence_ids = {item.evidence_id for item in context.events}
        object_ids = set(context.visible_aliases())
        if prior is not None:
            object_ids |= set(prior.object_registry)

        def parse(text: str):
            return parse_update(text, evidence_ids=evidence_ids, object_ids=object_ids)

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
            return coherence_error(opportunity, geometries, close=context.close, tick=tick)

        ctx = ReduceContext(
            known_at=context.known_at,
            has_open_position=self._ledger.has_open_position(),
            open_interaction=context.open_interaction,
            visible_aliases=context.visible_aliases(),
            registry=registry.snapshot(),
            coherence=coherence,
        )
        result = apply(
            prior,
            episode_id=episode_id,
            evidence=context.events,
            update=outcome.update,
            ctx=ctx,
            incident=outcome.incident,
        )
        return BrainStep(result=result, llm_input=llm_input, outcome=outcome)


__all__ = ["EXAMPLE_TOKEN", "MAIN_BRAIN_SCHEMA_VERSION", "BrainStep", "MainBrain", "MainBrainConfig"]
