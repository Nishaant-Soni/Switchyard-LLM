"""Endpoint-level pipeline tests for gateway.main.chat_completions.

Covers the rate-limit accounting rules wired in Phase 3 iteration 7:
  1. A bad alias is validated before admission, so it charges nothing (no tokens, no request slot).
  2. Token refund when an upstream attempt produces nothing (non-retryable upstream error or all
     targets failed) — actual usage is 0, so tokens are refunded while the request slot stays
     charged (a real attempt was made).
  3. Auth runs before the streaming 501, and a rejected streaming request is never charged.

Everything runs hermetically over an in-process ASGI transport with fakeredis + fake provider
adapters (no real Redis/providers). Each test runs on a single event loop via asyncio.run, so the
fakeredis client the endpoint uses and the post-request assertions share that loop.
"""

import asyncio
import hashlib

import fakeredis.aioredis
import httpx
import numpy as np

from gateway import main
from gateway.cache.semantic_cache import SemanticCache
from gateway.providers.base import UpstreamError
from gateway.ratelimit.limiter import RateLimiter
from gateway.resilience.circuit_breaker import BreakerRegistry
from gateway.resilience.retry import ResilientExecutor
from gateway.routing.policies import Target
from gateway.routing.router import Router
from gateway.schemas import ChatCompletionResponse, Choice, Message, Usage
from gateway.tenancy.auth import Tenant, TenantRegistry

TOKENS_PER_MIN = 1000
REQUESTS_PER_MIN = 100
KEY = "k1"
_AUTH = {"Authorization": f"Bearer {KEY}"}
_REQ = {"model": "fast", "messages": [{"role": "user", "content": "hello there"}]}


def _ok_response(total_tokens=2):
    return ChatCompletionResponse(
        id="x",
        object="chat.completion",
        created=0,
        model="m",
        choices=[Choice(index=0, message=Message(role="assistant", content="hi"))],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=total_tokens),
    )


class _FakeAdapter:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = 0

    async def chat_completion(self, request):
        self.calls += 1
        if isinstance(self.behavior, Exception):
            raise self.behavior
        return self.behavior


class _FakeRegistry:
    def __init__(self, adapters):
        self._adapters = adapters

    def get(self, name):
        return self._adapters.get(name)


async def _no_sleep(_):
    return None


def _install_state(behavior, cache=None, cache_per_tenant=False):
    """Wire main.app.state with hermetic fakes; returns the fakeredis client so a test can read
    bucket levels on the same event loop afterward."""
    client = fakeredis.aioredis.FakeRedis()
    main.app.state.tenants = TenantRegistry({KEY: Tenant("t1", REQUESTS_PER_MIN, TOKENS_PER_MIN)})
    main.app.state.limiter = RateLimiter(client, window_s=60, now=lambda: 1000.0)
    main.app.state.router = Router({"fast": ("priority", [Target("groq", "a")])})
    main.app.state.registry = _FakeRegistry({"groq": _FakeAdapter(behavior)})
    main.app.state.breakers = BreakerRegistry(min_calls=100)  # never trips for a single request
    main.app.state.executor = ResilientExecutor(
        main.app.state.breakers, base_delay_s=0.0, sleep=_no_sleep
    )
    main.app.state.cache = cache
    main.app.state.cache_per_tenant = cache_per_tenant
    return client


async def _post(json, headers):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        return await ac.post("/v1/chat/completions", json=json, headers=headers)


async def _tokens(client):
    return float(await client.hget("rl:t1:tok", "tokens"))


async def _requests(client):
    return float(await client.hget("rl:t1:req", "tokens"))


def test_success_reconciles_against_actual_usage():
    async def run():
        client = _install_state(_ok_response(total_tokens=2))
        resp = await _post(_REQ, _AUTH)
        return resp.status_code, await _tokens(client)

    status, tokens = asyncio.run(run())
    assert status == 200
    assert tokens == TOKENS_PER_MIN - 2  # charged the estimate, reconciled down to actual (2)


def test_unknown_alias_charges_nothing():
    async def run():
        client = _install_state(_ok_response())
        resp = await _post({**_REQ, "model": "nope"}, _AUTH)
        tok = await client.hget("rl:t1:tok", "tokens")
        req = await client.hget("rl:t1:req", "tokens")
        return resp.status_code, tok, req

    status, tok, req = asyncio.run(run())
    assert status == 400
    # A bad alias is validated before rate-limit admission, so nothing is charged at all:
    # the buckets are never even created.
    assert tok is None and req is None


def test_all_targets_failed_refunds_tokens():
    async def run():
        client = _install_state(UpstreamError(500, {"error": "boom"}))
        resp = await _post(_REQ, _AUTH)
        return resp.status_code, await _tokens(client)

    status, tokens = asyncio.run(run())
    assert status == 500  # last upstream status forwarded
    assert tokens == TOKENS_PER_MIN


def test_non_retryable_upstream_error_refunds_tokens():
    async def run():
        client = _install_state(UpstreamError(400, {"error": "bad"}))
        resp = await _post(_REQ, _AUTH)
        return resp.status_code, await _tokens(client)

    status, tokens = asyncio.run(run())
    assert status == 400
    assert tokens == TOKENS_PER_MIN


def test_streaming_requires_auth_before_rejection():
    async def run():
        _install_state(_ok_response())
        resp = await _post({**_REQ, "stream": True}, {"Authorization": "Bearer wrong"})
        return resp.status_code

    assert asyncio.run(run()) == 401  # auth runs before the streaming 501


def test_streaming_rejected_without_charging_tokens():
    async def run():
        client = _install_state(_ok_response())
        resp = await _post({**_REQ, "stream": True}, _AUTH)
        tok = await client.hget("rl:t1:tok", "tokens")
        req = await client.hget("rl:t1:req", "tokens")
        return resp.status_code, tok, req

    status, tok, req = asyncio.run(run())
    assert status == 501
    assert tok is None and req is None  # admission never ran => no buckets created


class _HashEmbedder:
    """Deterministic, real-embedder-free: identical text -> identical (centered) vector -> cosine
    1.0 (hit); different text -> ~orthogonal -> low cosine (miss). Keeps cache pipeline tests
    hermetic (no torch/MiniLM)."""

    def embed(self, texts):
        rows = []
        for t in texts:
            digest = hashlib.sha256(t.encode()).digest()[:16]
            rows.append(np.frombuffer(digest, dtype="uint8").astype("float32") - 128.0)
        arr = np.array(rows, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def test_cache_hit_serves_from_cache_and_refunds_tokens():
    async def run():
        cache = SemanticCache(_HashEmbedder(), threshold=0.9)
        client = _install_state(_ok_response(total_tokens=2), cache=cache)
        r1 = await _post(_REQ, _AUTH)  # miss -> upstream -> write
        r2 = await _post(_REQ, _AUTH)  # identical -> hit -> served from cache
        adapter = main.app.state.registry.get("groq")
        return r1, r2, adapter.calls, await _tokens(client), await _requests(client)

    r1, r2, calls, tokens, requests = asyncio.run(run())
    assert r1.status_code == 200 and r1.headers["x-switchyard-cache"] == "miss"
    assert r2.status_code == 200 and r2.headers["x-switchyard-cache"] == "hit"
    assert calls == 1  # second request never hit the provider
    assert tokens == TOKENS_PER_MIN - 2  # only the miss's actual usage; the hit was refunded
    assert requests == REQUESTS_PER_MIN - 2  # both requests still cost a request slot


def test_cache_miss_marks_header_and_populates():
    async def run():
        cache = SemanticCache(_HashEmbedder(), threshold=0.9)
        _install_state(_ok_response(), cache=cache)
        resp = await _post(_REQ, _AUTH)
        # a *different* prompt in the same scope must not hit the first entry
        other = await _post(
            {**_REQ, "messages": [{"role": "user", "content": "totally different"}]}, _AUTH
        )
        return resp, other

    resp, other = asyncio.run(run())
    assert resp.headers["x-switchyard-cache"] == "miss"
    assert other.headers["x-switchyard-cache"] == "miss"  # distinct prompt => not a false hit
