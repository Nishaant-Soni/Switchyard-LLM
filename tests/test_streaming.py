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
from gateway.resilience.circuit_breaker import BreakerRegistry
from gateway.resilience.retry import ResilientExecutor
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


# --- endpoint (full streaming branch) ----------------------------------------------------
class _StreamAdapter:
    def __init__(self, chunks):
        self._chunks = chunks

    async def stream_chat_completion(self, request):
        for chunk in self._chunks:
            yield chunk


class _Registry:
    def __init__(self, adapters):
        self._adapters = adapters

    def get(self, name):
        return self._adapters.get(name)


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
