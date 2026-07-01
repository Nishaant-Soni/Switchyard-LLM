import asyncio

import fakeredis
import fakeredis.aioredis

from gateway.ratelimit.estimate import estimate_tokens
from gateway.ratelimit.limiter import RateLimiter
from gateway.schemas import ChatCompletionRequest, Message
from gateway.tenancy.auth import Tenant, TenantRegistry, extract_bearer


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _client(server=None):
    return fakeredis.aioredis.FakeRedis(server=server or fakeredis.FakeServer())


# --- token bucket ------------------------------------------------------------------------
def test_burst_blocks_after_request_capacity():
    async def run():
        limiter = RateLimiter(_client(), window_s=60, now=lambda: 1000.0)  # frozen: no refill
        tenant = Tenant("t1", requests_per_min=3, tokens_per_min=10_000)
        return [await limiter.check(tenant, estimated_tokens=1) for _ in range(4)]

    results = asyncio.run(run())
    assert [r.allowed for r in results] == [True, True, True, False]
    assert results[-1].blocked_dimension == "requests"
    assert results[-1].retry_after_s > 0


def test_token_limit_binds_before_request_limit():
    async def run():
        limiter = RateLimiter(_client(), window_s=60, now=lambda: 1000.0)
        # plenty of request budget, tiny token budget
        tenant = Tenant("t2", requests_per_min=100, tokens_per_min=100)
        first = await limiter.check(tenant, estimated_tokens=60)
        second = await limiter.check(tenant, estimated_tokens=60)  # 40 left < 60
        return first, second

    first, second = asyncio.run(run())
    assert first.allowed is True
    assert second.allowed is False
    assert second.blocked_dimension == "tokens"  # tokens bind before requests


def test_bucket_refills_over_time():
    async def run():
        clock = FakeClock(1000.0)
        limiter = RateLimiter(_client(), window_s=60, now=clock)
        tenant = Tenant("t3", requests_per_min=2, tokens_per_min=10_000)
        drained = [(await limiter.check(tenant, 1)).allowed for _ in range(3)]  # 2 ok, 1 blocked
        clock.advance(30)  # refill 30 * (2/60) = 1 request
        after = (await limiter.check(tenant, 1)).allowed
        return drained, after

    drained, after = asyncio.run(run())
    assert drained == [True, True, False]
    assert after is True


def test_counters_persist_across_limiter_instances():
    async def run():
        server = fakeredis.FakeServer()  # shared "Redis"
        tenant = Tenant("t4", requests_per_min=2, tokens_per_min=10_000)
        limiter_a = RateLimiter(_client(server), window_s=60, now=lambda: 1000.0)
        await limiter_a.check(tenant, 1)
        await limiter_a.check(tenant, 1)  # bucket now empty
        # New limiter + new client, same backing server = "gateway restart".
        limiter_b = RateLimiter(_client(server), window_s=60, now=lambda: 1000.0)
        return await limiter_b.check(tenant, 1)

    result = asyncio.run(run())
    assert result.allowed is False  # state survived the "restart"


# --- tenancy -----------------------------------------------------------------------------
def test_tenant_resolution():
    reg = TenantRegistry({"k1": Tenant("free", 10, 100)})
    assert reg.enabled is True
    assert reg.resolve("k1").id == "free"
    assert reg.resolve("unknown") is None
    assert reg.resolve(None) is None


def test_registry_disabled_when_empty():
    assert TenantRegistry({}).enabled is False


def test_from_config_missing_file_is_disabled():
    assert TenantRegistry.from_config("/nonexistent/tenants.yaml").enabled is False


def test_from_config_reads_tenants(tmp_path):
    p = tmp_path / "tenants.yaml"
    p.write_text(
        "tenants:\n  k1:\n    tenant: free\n    requests_per_min: 5\n    tokens_per_min: 50\n"
    )
    reg = TenantRegistry.from_config(str(p))
    tenant = reg.resolve("k1")
    assert tenant.requests_per_min == 5
    assert tenant.tokens_per_min == 50


def test_extract_bearer():
    assert extract_bearer("Bearer abc") == "abc"
    assert extract_bearer("abc") == "abc"  # tolerate a raw key
    assert extract_bearer(None) is None
    assert extract_bearer("") is None


# --- reconciliation ----------------------------------------------------------------------
async def _read_tokens(client, tenant_id):
    return float(await client.hget(f"rl:{tenant_id}:tok", "tokens"))


def test_reconcile_refunds_when_over_estimated():
    async def run():
        client = _client()
        limiter = RateLimiter(client, window_s=60, now=lambda: 1000.0)
        tenant = Tenant("r1", requests_per_min=100, tokens_per_min=1000)
        await limiter.check(tenant, estimated_tokens=500)  # 1000 -> 500
        await limiter.reconcile(tenant, actual_tokens=100, estimated_tokens=500)  # +400
        return await _read_tokens(client, "r1")

    assert asyncio.run(run()) == 900  # 500 + (500 - 100)


def test_reconcile_charges_when_under_estimated():
    async def run():
        client = _client()
        limiter = RateLimiter(client, window_s=60, now=lambda: 1000.0)
        tenant = Tenant("r2", requests_per_min=100, tokens_per_min=1000)
        await limiter.check(tenant, estimated_tokens=100)  # 1000 -> 900
        await limiter.reconcile(tenant, actual_tokens=600, estimated_tokens=100)  # -500
        return await _read_tokens(client, "r2")

    assert asyncio.run(run()) == 400  # 900 - (600 - 100)


def test_reconcile_can_drive_bucket_negative_and_throttle():
    async def run():
        client = _client()
        limiter = RateLimiter(client, window_s=60, now=lambda: 1000.0)
        tenant = Tenant("r3", requests_per_min=100, tokens_per_min=100)
        await limiter.check(tenant, estimated_tokens=50)  # 100 -> 50
        await limiter.reconcile(tenant, actual_tokens=300, estimated_tokens=50)  # -250 -> -200
        level = await _read_tokens(client, "r3")
        blocked = await limiter.check(tenant, estimated_tokens=1)  # debt throttles
        return level, blocked.allowed

    level, allowed = asyncio.run(run())
    assert level == -200
    assert allowed is False


def test_reconcile_noop_when_exact():
    async def run():
        client = _client()
        limiter = RateLimiter(client, window_s=60, now=lambda: 1000.0)
        tenant = Tenant("r4", requests_per_min=100, tokens_per_min=1000)
        await limiter.check(tenant, estimated_tokens=200)
        return await limiter.reconcile(tenant, actual_tokens=200, estimated_tokens=200)

    assert asyncio.run(run()) is None


# --- estimator ---------------------------------------------------------------------------
def test_estimate_tokens_heuristic_fallback():
    req = ChatCompletionRequest(
        model="fast",
        messages=[Message(role="user", content="a" * 40)],
        max_tokens=100,
    )
    # encoder=None forces the heuristic: overhead(3) + chars/4+1(11) + output(100)
    assert estimate_tokens(req, encoder=None) == 3 + (40 // 4 + 1) + 100


def test_estimate_tokens_default_output_when_no_max():
    req = ChatCompletionRequest(model="fast", messages=[Message(role="user", content="hi")])
    assert estimate_tokens(req, encoder=None) == 3 + (2 // 4 + 1) + 256


def test_estimate_tokens_with_injected_encoder():
    class WordEncoder:
        def encode(self, text):
            return text.split()

    req = ChatCompletionRequest(
        model="fast",
        messages=[Message(role="user", content="one two three")],
        max_tokens=10,
    )
    # overhead(3) + 3 words + output(10)
    assert estimate_tokens(req, encoder=WordEncoder()) == 3 + 3 + 10


def test_estimate_tokens_default_encoder_is_positive():
    req = ChatCompletionRequest(model="fast", messages=[Message(role="user", content="hello")])
    assert estimate_tokens(req) > 0
