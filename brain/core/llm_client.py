"""The LLM behind the Main Brain, and the policy that calls it.

``DeepSeekClient`` speaks DeepSeek's chat-completions API over ``urllib``
(no dependency), in JSON mode, reading its key from ``DEEPSEEK_API_KEY`` only.
``ScriptedClient`` and ``EchoClient`` serve tests and key-less smoke runs;
``RecordedClient`` answers a replay from the journal.  ``call_with_policy``
owns retries: transport errors back off and retry, a malformed reply earns
one repair attempt, and anything past that becomes an incident the runtime
records instead of a crash."""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
import socket
import ssl
import time
from typing import Any, Protocol
import urllib.error
import urllib.request

import certifi

from contract.brain.llm import LLM_UPDATE_EXAMPLE, MalformedReply

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
API_KEY_ENV = "DEEPSEEK_API_KEY"
BASE_URL_ENV = "DEEPSEEK_BASE_URL"


class LLMClientError(RuntimeError):
    """A call could not be completed."""


class LLMTimeout(LLMClientError):
    """The request exceeded its timeout."""


class LLMRateLimited(LLMClientError):
    """HTTP 429; ``retry_after`` in seconds."""

    def __init__(self, message: str, *, retry_after: float = 5.0) -> None:
        super().__init__(message)
        self.retry_after = float(retry_after)


class LLMServerError(LLMClientError):
    """HTTP 5xx or a transport failure that is not a timeout."""


class LLMRequestRejected(LLMClientError):
    """HTTP 4xx other than 429 — retrying will not help."""


@dataclass(frozen=True)
class LLMReply:
    content: str
    reasoning_content: str | None
    usage: Mapping[str, int]
    latency_ms: int
    model: str
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "usage": dict(self.usage),
            "latency_ms": int(self.latency_ms),
            "model": self.model,
            "finish_reason": self.finish_reason,
        }


REPAIR_SUFFIX = (
    "\n\nYour previous reply was rejected: {error}. "
    "Reply again with one JSON object that follows the contract exactly."
)


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str) -> LLMReply: ...


class DeepSeekClient:
    def __init__(
        self,
        *,
        model: str,
        timeout_s: float,
        max_tokens: int,
        api_key: str | None = None,
        base_url: str | None = None,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if not key:
            raise LLMClientError(f"{API_KEY_ENV} is not set; the DeepSeek client reads its key from the environment only")
        self._key = key
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        self.base_url = (base_url or os.environ.get(BASE_URL_ENV) or DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")
        self._opener = opener
        # python.org's macOS builds ship without a CA bundle, so a bare
        # urlopen fails TLS verification; certifi's bundle is the authority.
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    def request_body(self, *, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "stream": False,
        }

    def complete(self, *, system: str, user: str) -> LLMReply:
        body = json.dumps(self.request_body(system=system, user=user)).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with self._opener(request, timeout=self.timeout_s, context=self._ssl_context) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            status = int(error.code)
            if status == 429:
                retry_after = error.headers.get("Retry-After", "5") if error.headers else "5"
                try:
                    seconds = float(retry_after)
                except ValueError:
                    seconds = 5.0
                raise LLMRateLimited(f"DeepSeek rate limited (HTTP 429)", retry_after=seconds) from None
            if status >= 500:
                raise LLMServerError(f"DeepSeek server error (HTTP {status})") from None
            raise LLMRequestRejected(f"DeepSeek rejected the request (HTTP {status})") from None
        except (socket.timeout, TimeoutError) as error:
            raise LLMTimeout(f"DeepSeek request timed out after {self.timeout_s}s") from None
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", None)
            if isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in str(reason).lower():
                raise LLMTimeout(f"DeepSeek request timed out after {self.timeout_s}s") from None
            if isinstance(reason, ssl.SSLError):
                # A certificate or protocol failure does not fix itself on retry.
                raise LLMRequestRejected(f"DeepSeek TLS failure: {reason}") from None
            raise LLMServerError(f"DeepSeek transport failure: {reason}") from None
        except ssl.SSLError as error:
            raise LLMRequestRejected(f"DeepSeek TLS failure: {error}") from None
        latency_ms = int((time.monotonic() - started) * 1000)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise MalformedReply("DeepSeek response body is not JSON") from None
        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise MalformedReply("DeepSeek response has no choices[0].message") from None
        finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise MalformedReply("DeepSeek reply content is empty")
        if finish_reason == "length":
            # Reasoning tokens count against max_tokens; a cut reply is never valid JSON.
            raise MalformedReply(
                f"DeepSeek reply truncated at max_tokens={self.max_tokens} (finish_reason=length)"
            )
        reasoning = message.get("reasoning_content")
        usage_raw = payload.get("usage") or {}
        usage = {
            str(key): int(value)
            for key, value in usage_raw.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        return LLMReply(
            content=content,
            reasoning_content=reasoning if isinstance(reasoning, str) else None,
            usage=usage,
            latency_ms=latency_ms,
            model=str(payload.get("model", self.model)),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
        )


class ScriptedClient:
    """Replies (or raises) in order; records every call."""

    def __init__(self, replies: Sequence[LLMReply | Exception]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str) -> LLMReply:
        self.calls.append((system, user))
        if not self._replies:
            raise LLMClientError("scripted client has no reply left")
        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class EchoClient:
    """A contract-valid reply for any input, with no reasoning behind it.

    Verdicts every piece of evidence NEUTRAL until ``sleep_after`` calls, then
    SUPPORT with ``continue_active`` false so the episode can close.  For
    key-less smoke runs and tests only."""

    def __init__(self, *, sleep_after: int | None = None, fail_calls: Collection[int] = ()) -> None:
        self.calls = 0
        self.sleep_after = sleep_after
        self.fail_calls = frozenset(int(call) for call in fail_calls)

    def complete(self, *, system: str, user: str) -> LLMReply:
        self.calls += 1
        if self.calls in self.fail_calls:
            raise LLMTimeout("echo client scripted timeout")
        request = json.loads(user)
        payload = json.loads(json.dumps(LLM_UPDATE_EXAMPLE))
        closing = self.sleep_after is not None and self.calls >= self.sleep_after
        payload["evidence_verdicts"] = [
            {
                "evidence_id": item["evidence_id"],
                "verdict": "SUPPORT" if closing else "NEUTRAL",
                "note": "echo",
                "resolves_evidence_id": None,
                "resolution": None,
            }
            for item in request["new_evidence"]
        ]
        payload["market_understanding"] = f"echo of {request['known_at']}"
        payload["watch_next"] = []
        payload["destination_candidates"] = []
        payload["continue_active"] = not closing
        if closing:
            # Spend this call's verdicts resolving what earlier calls left open,
            # one pending item per new evidence, so the exit conditions can hold.
            prior = request.get("prior_state") or {}
            pending = [item["evidence_id"] for item in (prior.get("evidence") or {}).get("unresolved", [])]
            for target, verdict in zip(pending, payload["evidence_verdicts"]):
                verdict.update(verdict="RESOLVE", resolves_evidence_id=target, resolution="SUPPORT")
        return LLMReply(json.dumps(payload), None, {}, 1, "echo")


_INCIDENT_CLASSES: Mapping[str, type[LLMClientError]] = {
    "LLMTimeout": LLMTimeout,
    "LLMRateLimited": LLMRateLimited,
    "LLMServerError": LLMServerError,
    "LLMRequestRejected": LLMRequestRejected,
}


class RecordedClient:
    """Answers every input with what a journal recorded for its sha: the reply,
    or the transport incident the original call ended in.  A repair prompt is
    keyed by the input it repairs, so a twice-malformed reply replays as such."""

    def __init__(
        self,
        replies_by_input_sha: Mapping[str, LLMReply],
        incidents_by_input_sha: Mapping[str, str] | None = None,
    ) -> None:
        self._replies = dict(replies_by_input_sha)
        self._incidents = dict(incidents_by_input_sha or {})

    @staticmethod
    def input_sha(user: str) -> str:
        marker = REPAIR_SUFFIX.split("{error}")[0]
        cut = user.find(marker)
        original = user if cut < 0 else user[:cut]
        return hashlib.sha256(original.encode("utf-8")).hexdigest()

    def complete(self, *, system: str, user: str) -> LLMReply:
        sha = self.input_sha(user)
        incident = self._incidents.get(sha)
        if incident is not None and incident in _INCIDENT_CLASSES:
            if incident == "LLMRateLimited":
                raise LLMRateLimited("recorded incident", retry_after=0.0)
            raise _INCIDENT_CLASSES[incident]("recorded incident")
        reply = self._replies.get(sha)
        if reply is None:
            raise LLMClientError(f"no recorded reply for input {sha[:16]}")
        return reply


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    backoff_base_s: float = 1.0
    backoff_cap_s: float = 30.0


@dataclass(frozen=True)
class CallOutcome:
    update: Any | None
    reply: LLMReply | None
    incident: str | None
    attempts: int
    repaired: bool
    incident_message: str | None = None
    # Why the first reply was refused, when a repair attempt followed; and the
    # refused reply itself, so the journal keeps what the model actually said.
    repair_reason: str | None = None
    rejected_reply: LLMReply | None = None


def call_with_policy(
    client: LLMClient,
    *,
    system: str,
    user: str,
    parse: Callable[[str], Any],
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> CallOutcome:
    attempts = 0
    transport_failures = 0
    repaired = False
    repair_reason: str | None = None
    rejected: LLMReply | None = None
    prompt = user
    while True:
        attempts += 1
        try:
            reply = client.complete(system=system, user=prompt)
        except LLMRequestRejected as error:
            return CallOutcome(None, None, type(error).__name__, attempts, repaired, str(error), repair_reason, rejected)
        except (LLMTimeout, LLMRateLimited, LLMServerError) as error:
            transport_failures += 1
            if transport_failures > policy.max_retries:
                return CallOutcome(None, None, type(error).__name__, attempts, repaired, str(error), repair_reason, rejected)
            if isinstance(error, LLMRateLimited):
                delay = error.retry_after
            else:
                delay = min(policy.backoff_cap_s, policy.backoff_base_s * (2 ** (transport_failures - 1)))
            sleep(delay)
            continue
        except MalformedReply as error:
            if repaired:
                return CallOutcome(None, None, "MalformedReply", attempts, repaired, str(error), repair_reason, rejected)
            repaired = True
            repair_reason = str(error)
            prompt = user + REPAIR_SUFFIX.format(error=error)
            continue
        try:
            update = parse(reply.content)
        except MalformedReply as error:
            if repaired:
                return CallOutcome(None, reply, "MalformedReply", attempts, repaired, str(error), repair_reason, rejected)
            repaired = True
            repair_reason = str(error)
            rejected = reply
            prompt = user + REPAIR_SUFFIX.format(error=error)
            continue
        return CallOutcome(update, reply, None, attempts, repaired, None, repair_reason, rejected)


__all__ = [
    "API_KEY_ENV",
    "BASE_URL_ENV",
    "DEFAULT_DEEPSEEK_BASE_URL",
    "CallOutcome",
    "DeepSeekClient",
    "EchoClient",
    "LLMClient",
    "LLMClientError",
    "LLMRateLimited",
    "LLMReply",
    "LLMRequestRejected",
    "LLMServerError",
    "LLMTimeout",
    "REPAIR_SUFFIX",
    "RecordedClient",
    "RetryPolicy",
    "ScriptedClient",
    "call_with_policy",
]
