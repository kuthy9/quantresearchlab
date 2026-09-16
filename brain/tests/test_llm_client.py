from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

import pytest

from brain.core.llm_client import (
    DeepSeekClient,
    LLMClientError,
    LLMRateLimited,
    LLMReply,
    LLMRequestRejected,
    LLMServerError,
    LLMTimeout,
    RecordedClient,
    RetryPolicy,
    ScriptedClient,
    call_with_policy,
)
from contract.brain.llm import MalformedReply


def reply(text: str) -> LLMReply:
    return LLMReply(content=text, reasoning_content=None, usage={"prompt_tokens": 1, "completion_tokens": 1}, latency_ms=1, model="fake")


def parse_ok(text: str):
    if text != "GOOD":
        raise MalformedReply("bad")
    return "UPDATE"


def run(steps, policy=RetryPolicy(max_retries=2, backoff_base_s=0.0)):
    slept: list[float] = []
    client = ScriptedClient(steps)
    out = call_with_policy(client, system="s", user="u", parse=parse_ok, policy=policy, sleep=slept.append)
    return out, client, slept


def test_timeout_is_retried_then_becomes_an_incident() -> None:
    out, client, _ = run([LLMTimeout("t"), LLMTimeout("t"), LLMTimeout("t")])
    assert out.update is None and out.incident == "LLMTimeout" and out.attempts == 3 and len(client.calls) == 3


def test_rate_limit_honours_retry_after_then_succeeds() -> None:
    out, _, slept = run([LLMRateLimited("429", retry_after=7.5), reply("GOOD")])
    assert out.update == "UPDATE" and slept == [7.5] and out.attempts == 2


def test_backoff_grows_and_is_capped() -> None:
    out, _, slept = run([LLMServerError("500")] * 3 + [reply("GOOD")], policy=RetryPolicy(max_retries=3, backoff_base_s=1.0, backoff_cap_s=3.0))
    assert out.update == "UPDATE" and slept == [1.0, 2.0, 3.0]


def test_server_error_retried_and_bad_request_not() -> None:
    out, _, _ = run([LLMServerError("500"), reply("GOOD")])
    assert out.update == "UPDATE"
    out, client, _ = run([LLMRequestRejected("400")])
    assert out.incident == "LLMRequestRejected" and len(client.calls) == 1


def test_malformed_gets_exactly_one_repair_attempt() -> None:
    out, client, _ = run([reply("BAD"), reply("GOOD")])
    assert out.update == "UPDATE" and out.repaired is True and "bad" in client.calls[1][1]
    assert client.calls[0][1] == "u"
    out, client, _ = run([reply("BAD"), reply("BAD")])
    assert out.incident == "MalformedReply" and len(client.calls) == 2 and out.reply is not None


def test_transport_malformed_reply_counts_as_malformed() -> None:
    out, client, _ = run([MalformedReply("empty content"), reply("GOOD")])
    assert out.update == "UPDATE" and out.repaired


class _Handler(BaseHTTPRequestHandler):
    scenario: list[tuple[int, dict, dict]] = []
    bodies: list[dict] = []
    headers_seen: list[dict] = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        _Handler.bodies.append(json.loads(self.rfile.read(length)))
        _Handler.headers_seen.append(dict(self.headers))
        status, headers, body = _Handler.scenario.pop(0)
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *args):  # silence the test server
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_deepseek_client_request_shape_and_reply(server, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    _Handler.scenario = [(200, {}, {
        "model": "deepseek-flash", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "choices": [{"message": {"content": "{\"x\":1}", "reasoning_content": "thought"}}],
    })]
    client = DeepSeekClient(model="deepseek-flash", timeout_s=5, max_tokens=100, base_url=server)
    out = client.complete(system="SYS", user="USER")
    assert out.content == '{"x":1}' and out.reasoning_content == "thought" and out.usage["prompt_tokens"] == 10
    assert out.model == "deepseek-flash" and out.latency_ms >= 0
    body = _Handler.bodies[-1]
    assert body["model"] == "deepseek-flash" and body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 100 and body["stream"] is False
    assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert _Handler.headers_seen[-1]["Authorization"] == "Bearer test-key"


@pytest.mark.parametrize("status,headers,expected", [
    (429, {"Retry-After": "3"}, LLMRateLimited),
    (503, {}, LLMServerError),
    (400, {}, LLMRequestRejected),
    (401, {}, LLMRequestRejected),
])
def test_deepseek_client_maps_http_errors(server, monkeypatch, status, headers, expected) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    _Handler.scenario = [(status, headers, {"error": "x"})]
    client = DeepSeekClient(model="deepseek-flash", timeout_s=5, max_tokens=100, base_url=server)
    with pytest.raises(expected) as info:
        client.complete(system="s", user="u")
    if expected is LLMRateLimited:
        assert info.value.retry_after == 3.0
    assert "test-key" not in str(info.value)


def test_deepseek_empty_content_is_malformed(server, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    _Handler.scenario = [(200, {}, {"choices": [{"message": {"content": ""}}], "usage": {}})]
    client = DeepSeekClient(model="deepseek-flash", timeout_s=5, max_tokens=100, base_url=server)
    with pytest.raises(MalformedReply):
        client.complete(system="s", user="u")


def test_deepseek_timeout_maps(monkeypatch) -> None:
    import socket

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def slow(request, timeout, context=None):
        raise socket.timeout("timed out")

    client = DeepSeekClient(model="deepseek-flash", timeout_s=0.01, max_tokens=10, base_url="http://127.0.0.1:9", opener=slow)
    with pytest.raises(LLMTimeout):
        client.complete(system="s", user="u")


def test_deepseek_tls_failure_is_not_retried(monkeypatch) -> None:
    import ssl
    import urllib.error

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    seen: dict[str, object] = {}

    def refuse(request, timeout, context=None):
        seen["context"] = context
        raise urllib.error.URLError(ssl.SSLCertVerificationError(1, "certificate verify failed"))

    client = DeepSeekClient(model="deepseek-flash", timeout_s=1, max_tokens=10, base_url="http://127.0.0.1:9", opener=refuse)
    with pytest.raises(LLMRequestRejected, match="TLS"):
        client.complete(system="s", user="u")
    assert isinstance(seen["context"], ssl.SSLContext) and seen["context"].verify_mode == ssl.CERT_REQUIRED
    out = call_with_policy(client, system="s", user="u", parse=parse_ok, policy=RetryPolicy(max_retries=3, backoff_base_s=0.0), sleep=lambda s: None)
    assert out.incident == "LLMRequestRejected" and out.attempts == 1 and "certificate verify failed" in out.incident_message


def test_incident_message_is_kept() -> None:
    out, _, _ = run([LLMTimeout("slow upstream")] * 3)
    assert out.incident_message == "slow upstream"


def test_missing_key_refuses_construction(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(LLMClientError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient(model="deepseek-flash", timeout_s=5, max_tokens=100)


def test_base_url_env_is_honoured(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://proxy.local/v1/")
    assert DeepSeekClient(model="m", timeout_s=1, max_tokens=1).base_url == "http://proxy.local/v1"


def test_recorded_client_answers_by_input_sha() -> None:
    client = RecordedClient({hashlib.sha256(b"u").hexdigest(): reply("GOOD")})
    assert client.complete(system="s", user="u").content == "GOOD"
    with pytest.raises(LLMClientError):
        client.complete(system="s", user="other")
