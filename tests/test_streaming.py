"""Phase 5 Group 1 — SSE streaming passthrough + accounting.

Adapter-level tests use httpx.MockTransport to fake an upstream SSE response; the sse re-emitter is
unit-tested directly; one endpoint test drives the full streaming branch over an in-process ASGI
transport with a fake streaming adapter + fakeredis (no real provider/Redis).
"""

import asyncio
import json

import fakeredis.aioredis
import httpx
import pytest

from gateway import main
from gateway.providers.base import UpstreamError
from gateway.providers.gemini import GeminiAdapter
from gateway.providers.openai_compat import OpenAICompatAdapter
from gateway.ratelimit.limiter import RateLimiter
from gateway.resilience.circuit_breaker import BreakerRegistry, CircuitState
from gateway.resilience.retry import AllTargetsFailed, ResilientExecutor
from gateway.routing.policies import Target
from gateway.routing.router import Router
from gateway.schemas import ChatCompletionRequest, Message
from gateway.streaming.sse import stream_sse
from gateway.tenancy.auth import Tenant, TenantRegistry


def _sse_bytes(*objs, done=True):
    parts = [f"data: {json.dumps(o)}\n\n" for o in objs]
    if done:
        parts.append("data: [DONE]\n\n")
    return "".join(parts).encode()


def _req():
    return ChatCompletionRequest(
        model="m", messages=[Message(role="user", content="hi")], stream=True
    )


# --- adapter streaming -------------------------------------------------------------------
def test_openai_compat_stream_parses_and_requests_usage():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_sse_bytes(
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"total_tokens": 3}},
            ),
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatAdapter("groq", "http://up/v1", "key", client)
            return [c async for c in adapter.stream_chat_completion(_req())]

    chunks = asyncio.run(run())
    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"]["include_usage"] is True  # requested upstream
    assert [c["choices"][0]["delta"].get("content") for c in chunks[:2]] == ["Hel", "lo"]
    assert chunks[-1]["usage"]["total_tokens"] == 3
    assert len(chunks) == 3  # the [DONE] terminator is consumed, not yielded


def test_stream_raises_before_first_chunk_on_non_2xx():
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "rate"}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatAdapter("groq", "http://up/v1", "key", client)
            with pytest.raises(UpstreamError) as exc:
                async for _ in adapter.stream_chat_completion(_req()):
                    pass
            return exc.value

    err = asyncio.run(run())
    assert err.status_code == 429
    assert err.body["error"]["message"] == "rate"


def test_gemini_keeps_usage_only_on_final_chunk():
    # Gemini emits usage in *every* chunk; the adapter must strip all but the final one.
    def handler(request):
        return httpx.Response(
            200,
            content=_sse_bytes(
                {"choices": [{"delta": {"content": "a"}}], "usage": {"total_tokens": 1}},
                {"choices": [{"delta": {"content": "b"}}], "usage": {"total_tokens": 2}},
                {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"total_tokens": 3}},
            ),
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = GeminiAdapter("gemini", "http://up/v1", "key", client)
            return [c async for c in adapter.stream_chat_completion(_req())]

    chunks = asyncio.run(run())
    assert chunks[0].get("usage") is None
    assert chunks[1].get("usage") is None
    assert chunks[-1]["usage"]["total_tokens"] == 3  # only the final chunk carries usage


# --- sse re-emitter ----------------------------------------------------------------------
def test_stream_sse_emits_done_and_captures_usage():
    async def chunks():
        yield {"choices": [{"delta": {"content": "hi"}}]}
        yield {"choices": [{"delta": {}}], "usage": {"total_tokens": 5}}

    seen = {}

    async def on_finish(usage, completed):
        seen["usage"], seen["completed"] = usage, completed

    async def run():
        gen = chunks()
        first = await gen.__anext__()
        return b"".join([b async for b in stream_sse(first, gen, on_finish)])

    out = asyncio.run(run())
    assert b"data: [DONE]\n\n" in out
    assert seen["completed"] is True
    assert seen["usage"]["total_tokens"] == 5


def test_stream_sse_refunds_on_midstream_error():
    async def chunks():
        yield {"choices": [{"delta": {"content": "partial"}}]}
        raise RuntimeError("boom")

    seen = {}

    async def on_finish(usage, completed):
        seen["usage"], seen["completed"] = usage, completed

    async def run():
        gen = chunks()
        first = await gen.__anext__()
        out = []
        with pytest.raises(RuntimeError):
            async for b in stream_sse(first, gen, on_finish):
                out.append(b)
        return out

    out = asyncio.run(run())
    assert any(b"partial" in b for b in out)  # partial data reached the client
    assert seen["completed"] is False and seen["usage"] is None  # -> caller refunds


# --- endpoint + executor fallback --------------------------------------------------------
class _StreamAdapter:
    def __init__(self, chunks=None, error=None):
        self._chunks = chunks or []
        self._error = error
        self.opened = 0

    async def stream_chat_completion(self, request):
        self.opened += 1
        if self._error is not None:
            raise self._error  # raised on the first __anext__ (before any chunk)
        for chunk in self._chunks:
            yield chunk


class _Registry:
    def __init__(self, adapters):
        self._adapters = adapters

    def get(self, name):
        return self._adapters.get(name)


async def _no_sleep(_):
    return None


def _executor(**kwargs):
    return ResilientExecutor(
        BreakerRegistry(min_calls=1, failure_threshold=0.5),
        base_delay_s=0.0,
        sleep=_no_sleep,
        **kwargs,
    )


async def _drain(first, rest):
    out = [] if first is None else [first]
    out += [c async for c in rest]
    return out


def test_execute_stream_falls_back_before_first_byte():
    reg = _Registry(
        {
            "groq": _StreamAdapter(error=UpstreamError(503, {"error": "down"})),
            "ollama": _StreamAdapter([{"choices": [{"delta": {"content": "hi"}}]}]),
        }
    )
    targets = [Target("groq", "a"), Target("ollama", "b")]

    async def run():
        ex = _executor()
        first, rest, target = await ex.execute_stream(targets, reg, _req())
        return target, await _drain(first, rest), reg, ex

    target, out, reg, ex = asyncio.run(run())
    assert target.provider == "ollama"  # fell back past the 503 primary
    assert out[0]["choices"][0]["delta"]["content"] == "hi"
    assert reg.get("groq").opened == 1  # primary was attempted
    assert ex.breakers.get("groq").state == CircuitState.OPEN  # its open-failure was recorded


def test_execute_stream_transport_error_falls_back():
    reg = _Registry(
        {
            "groq": _StreamAdapter(error=httpx.ConnectError("refused")),
            "ollama": _StreamAdapter([{"choices": [{"delta": {"content": "ok"}}]}]),
        }
    )
    targets = [Target("groq", "a"), Target("ollama", "b")]

    async def run():
        first, rest, target = await _executor().execute_stream(targets, reg, _req())
        return target, await _drain(first, rest)

    target, out = asyncio.run(run())
    assert target.provider == "ollama"
    assert out[0]["choices"][0]["delta"]["content"] == "ok"


def test_execute_stream_non_retryable_surfaces_without_fallback():
    secondary = _StreamAdapter([{"choices": [{"delta": {"content": "x"}}]}])
    reg = _Registry(
        {"groq": _StreamAdapter(error=UpstreamError(400, {"error": "bad"})), "ollama": secondary}
    )
    targets = [Target("groq", "a"), Target("ollama", "b")]

    async def run():
        with pytest.raises(UpstreamError) as exc:
            await _executor().execute_stream(targets, reg, _req())
        return exc.value

    err = asyncio.run(run())
    assert err.status_code == 400
    assert secondary.opened == 0  # a bad request is not retried against the next provider


def test_execute_stream_all_targets_failed():
    reg = _Registry(
        {
            "groq": _StreamAdapter(error=UpstreamError(503, {"error": "a"})),
            "ollama": _StreamAdapter(error=UpstreamError(500, {"error": "b"})),
        }
    )
    targets = [Target("groq", "a"), Target("ollama", "b")]

    async def run():
        with pytest.raises(AllTargetsFailed) as exc:
            await _executor().execute_stream(targets, reg, _req())
        return exc.value

    err = asyncio.run(run())
    assert isinstance(err.last_error, UpstreamError)


def test_execute_stream_skips_open_circuit():
    groq = _StreamAdapter([{"choices": [{"delta": {"content": "groq"}}]}])
    reg = _Registry(
        {"groq": groq, "ollama": _StreamAdapter([{"choices": [{"delta": {"content": "ollama"}}]}])}
    )
    targets = [Target("groq", "a"), Target("ollama", "b")]

    async def run():
        ex = _executor()
        ex.breakers.get("groq").record_failure()  # trip groq (min_calls=1)
        first, rest, target = await ex.execute_stream(targets, reg, _req())
        return target, await _drain(first, rest), groq

    target, out, groq = asyncio.run(run())
    assert target.provider == "ollama"
    assert groq.opened == 0  # open circuit -> never even attempted


def test_streaming_endpoint_forwards_chunks_and_reconciles_tokens():
    async def run():
        redis_client = fakeredis.aioredis.FakeRedis()
        main.app.state.tenants = TenantRegistry({"k1": Tenant("t1", 100, 1000)})
        main.app.state.limiter = RateLimiter(redis_client, window_s=60, now=lambda: 1000.0)
        main.app.state.router = Router({"fast": ("priority", [Target("groq", "m")])})
        main.app.state.registry = _Registry(
            {
                "groq": _StreamAdapter(
                    [
                        {"choices": [{"delta": {"content": "Hel"}}]},
                        {"choices": [{"delta": {"content": "lo"}}]},
                        {
                            "choices": [{"delta": {}, "finish_reason": "stop"}],
                            "usage": {"total_tokens": 5},
                        },
                    ]
                )
            }
        )
        main.app.state.breakers = BreakerRegistry()
        main.app.state.executor = ResilientExecutor(main.app.state.breakers)
        main.app.state.cache = None
        main.app.state.cache_per_tenant = False

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "fast",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Authorization": "Bearer k1"},
            )
        tok = float(await redis_client.hget("rl:t1:tok", "tokens"))
        return resp, tok

    resp, tok = asyncio.run(run())
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert resp.headers["x-switchyard-cache"] == "bypass"
    body = resp.text
    assert "Hel" in body and "lo" in body and "data: [DONE]" in body
    assert tok == 1000 - 5  # admission charged the estimate; reconciled to actual total_tokens=5
