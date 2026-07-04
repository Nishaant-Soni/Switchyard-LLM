"""Phase 6 Group 1 — observability instrumentation (metrics + counterfactual cost).

Drives the full pipeline over an in-process ASGI transport with hermetic fakes (fakeredis + fake
adapters) and asserts that the Prometheus series move as expected. The metrics registry is a
process-global cumulative object, so each test snapshots the relevant samples before/after and
asserts the *delta* — no reset needed, and tests don't interfere.

Verifies the Group-1 contract: instrumentation is additive (status codes / token accounting are
unchanged — those are covered in test_pipeline / test_streaming; here we only assert the metrics).
"""

import asyncio
import hashlib

import fakeredis.aioredis
import httpx
import numpy as np

from gateway import main
from gateway.cache.semantic_cache import SemanticCache
from gateway.observability import metrics
from gateway.observability.cost import PriceBook
from gateway.resilience.circuit_breaker import BreakerRegistry
from gateway.resilience.retry import ResilientExecutor
from gateway.routing.policies import Target
from gateway.routing.router import Router
from gateway.schemas import ChatCompletionResponse, Choice, Message, Usage
from gateway.tenancy.auth import Tenant, TenantRegistry

KEY = "k1"
_AUTH = {"Authorization": f"Bearer {KEY}"}
_REQ = {"model": "fast", "messages": [{"role": "user", "content": "hello there"}]}
# cost = (prompt*input + completion*output) / 1e6
_PRICES = {"gm": {"input": 1.0, "output": 2.0}}


def _response(prompt=10, completion=20):
    return ChatCompletionResponse(
        id="x",
        object="chat.completion",
        created=0,
        model="gm",
        choices=[Choice(index=0, message=Message(role="assistant", content="hi"))],
        usage=Usage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        ),
    )


class _Adapter:
    def __init__(self, behavior):
        self.behavior = behavior

    async def chat_completion(self, request):
        if isinstance(self.behavior, Exception):
            raise self.behavior
        return self.behavior


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


async def _no_sleep(_):
    return None


def _install(adapter, *, cache=None, requests_per_min=100, tokens_per_min=100_000):
    client = fakeredis.aioredis.FakeRedis()
    main.app.state.tenants = TenantRegistry({KEY: Tenant("t1", requests_per_min, tokens_per_min)})
    from gateway.ratelimit.limiter import RateLimiter

    main.app.state.limiter = RateLimiter(client, window_s=60, now=lambda: 1000.0)
    main.app.state.router = Router({"fast": ("priority", [Target("groq", "gm")])})
    main.app.state.registry = _Registry({"groq": adapter})
    main.app.state.breakers = BreakerRegistry(min_calls=100)
    main.app.state.executor = ResilientExecutor(
        main.app.state.breakers, base_delay_s=0.0, sleep=_no_sleep
    )
    main.app.state.cache = cache
    main.app.state.cache_per_tenant = False
    main.app.state.pricebook = PriceBook(_PRICES)
    return client


async def _post(json, headers=_AUTH):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        return await ac.post("/v1/chat/completions", json=json, headers=headers)


def _val(name, labels):
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


class _HashEmbedder:
    """Deterministic embedder (identical text -> cosine 1.0); keeps cache tests hermetic."""

    def embed(self, texts):
        rows = []
        for t in texts:
            digest = hashlib.sha256(t.encode()).digest()[:16]
            rows.append(np.frombuffer(digest, dtype="uint8").astype("float32") - 128.0)
        arr = np.array(rows, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def test_success_records_request_latency_tokens_and_cost():
    req_labels = {"alias": "fast", "provider": "groq", "outcome": "success", "stream": "false"}
    lat_labels = {"provider": "groq", "stream": "false"}
    prompt_labels = {"provider": "groq", "direction": "prompt"}
    completion_labels = {"provider": "groq", "direction": "completion"}
    cost_labels = {"tenant": "t1", "provider": "groq"}

    async def run():
        _install(_Adapter(_response(prompt=10, completion=20)))
        before = (
            _val("switchyard_requests_total", req_labels),
            _val("switchyard_request_latency_seconds_count", lat_labels),
            _val("switchyard_tokens_total", prompt_labels),
            _val("switchyard_tokens_total", completion_labels),
            _val("switchyard_cost_usd_total", cost_labels),
        )
        resp = await _post(_REQ)
        after = (
            _val("switchyard_requests_total", req_labels),
            _val("switchyard_request_latency_seconds_count", lat_labels),
            _val("switchyard_tokens_total", prompt_labels),
            _val("switchyard_tokens_total", completion_labels),
            _val("switchyard_cost_usd_total", cost_labels),
        )
        return resp.status_code, before, after

    status, before, after = asyncio.run(run())
    assert status == 200
    assert after[0] - before[0] == 1  # one success request
    assert after[1] - before[1] == 1  # one latency observation
    assert after[2] - before[2] == 10  # prompt tokens
    assert after[3] - before[3] == 20  # completion tokens
    # cost = (10*1 + 20*2) / 1e6
    assert abs((after[4] - before[4]) - (10 * 1.0 + 20 * 2.0) / 1_000_000) < 1e-12


def test_cache_hit_and_miss_recorded():
    hit_labels = {"event": "hit"}
    miss_labels = {"event": "miss"}
    cache_req = {"alias": "fast", "provider": "cache", "outcome": "success", "stream": "false"}

    async def run():
        cache = SemanticCache(_HashEmbedder(), threshold=0.9)
        _install(_Adapter(_response()), cache=cache)
        before = (
            _val("switchyard_cache_events_total", miss_labels),
            _val("switchyard_cache_events_total", hit_labels),
            _val("switchyard_requests_total", cache_req),
        )
        await _post(_REQ)  # miss -> upstream -> write
        await _post(_REQ)  # identical -> hit
        after = (
            _val("switchyard_cache_events_total", miss_labels),
            _val("switchyard_cache_events_total", hit_labels),
            _val("switchyard_requests_total", cache_req),
        )
        return before, after

    before, after = asyncio.run(run())
    assert after[0] - before[0] == 1  # one miss
    assert after[1] - before[1] == 1  # one hit
    assert after[2] - before[2] == 1  # the hit was served from cache


def test_unknown_alias_records_error():
    labels = {"type": "invalid_request_error"}

    async def run():
        _install(_Adapter(_response()))
        before = _val("switchyard_errors_total", labels)
        resp = await _post({**_REQ, "model": "nope"})
        return resp.status_code, before, _val("switchyard_errors_total", labels)

    status, before, after = asyncio.run(run())
    assert status == 400
    assert after - before == 1


def test_rate_limited_records_error_and_throttled_request():
    err_labels = {"type": "rate_limit_exceeded"}
    throttled = {"alias": "fast", "provider": "none", "outcome": "throttled", "stream": "false"}

    async def run():
        _install(_Adapter(_response()), requests_per_min=0)  # zero-capacity -> always throttled
        before = (
            _val("switchyard_errors_total", err_labels),
            _val("switchyard_requests_total", throttled),
        )
        resp = await _post(_REQ)
        after = (
            _val("switchyard_errors_total", err_labels),
            _val("switchyard_requests_total", throttled),
        )
        return resp.status_code, before, after

    status, before, after = asyncio.run(run())
    assert status == 429
    assert after[0] - before[0] == 1
    assert after[1] - before[1] == 1


def test_streaming_records_bypass_success_and_usage():
    bypass = {"event": "bypass"}
    req_labels = {"alias": "fast", "provider": "groq", "outcome": "success", "stream": "true"}
    cost_labels = {"tenant": "t1", "provider": "groq"}

    async def run():
        adapter = _StreamAdapter(
            [
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
                },
            ]
        )
        _install(adapter)
        before = (
            _val("switchyard_cache_events_total", bypass),
            _val("switchyard_requests_total", req_labels),
            _val("switchyard_cost_usd_total", cost_labels),
        )
        resp = await _post({**_REQ, "stream": True})
        after = (
            _val("switchyard_cache_events_total", bypass),
            _val("switchyard_requests_total", req_labels),
            _val("switchyard_cost_usd_total", cost_labels),
        )
        return resp.status_code, before, after

    status, before, after = asyncio.run(run())
    assert status == 200
    assert after[0] - before[0] == 1  # cache bypass recorded
    assert after[1] - before[1] == 1  # streamed success recorded
    # cost from the final-chunk usage: (3*1 + 7*2) / 1e6
    assert abs((after[2] - before[2]) - (3 * 1.0 + 7 * 2.0) / 1_000_000) < 1e-12


def test_metrics_endpoint_exposition():
    async def run():
        _install(_Adapter(_response()))
        await _post(_REQ)  # generate at least one series
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            return await ac.get("/metrics")

    resp = asyncio.run(run())
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "version=0.0.4" in resp.headers["content-type"]
    assert "switchyard_requests_total" in resp.text
