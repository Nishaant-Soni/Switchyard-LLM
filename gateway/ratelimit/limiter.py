"""Two-dimensional token-bucket rate limiter backed by Redis.

Each tenant has two buckets — request rate and token rate — because LLM cost is
token-denominated and the token limit often binds before the request limit. A single Lua script
refills and checks *both* buckets and consumes from them only if both have capacity, so the two
dimensions stay consistent under concurrency (a plain read-then-write would race).

State lives in Redis, so counters survive gateway restarts and are shared across instances.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from gateway.tenancy.auth import Tenant

# KEYS[1]=request bucket, KEYS[2]=token bucket
# ARGV: now, req_cap, req_rate, req_cost, tok_cap, tok_rate, tok_cost, ttl
# Returns: {allowed(0/1), retry_after_seconds(string), blocked_dimension(string)}
_LUA = """
local now = tonumber(ARGV[1])
local rcap, rrate, rcost = tonumber(ARGV[2]), tonumber(ARGV[3]), tonumber(ARGV[4])
local tcap, trate, tcost = tonumber(ARGV[5]), tonumber(ARGV[6]), tonumber(ARGV[7])
local ttl = tonumber(ARGV[8])

local function level(key, cap, rate)
  local d = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens, ts = tonumber(d[1]), tonumber(d[2])
  if tokens == nil then tokens, ts = cap, now end
  local elapsed = now - ts
  if elapsed < 0 then elapsed = 0 end
  return math.min(cap, tokens + elapsed * rate)
end

local rtok = level(KEYS[1], rcap, rrate)
local ttok = level(KEYS[2], tcap, trate)

local allowed = 1
if rtok < rcost or ttok < tcost then allowed = 0 end
if allowed == 1 then
  rtok = rtok - rcost
  ttok = ttok - tcost
end

redis.call('HSET', KEYS[1], 'tokens', rtok, 'ts', now)
redis.call('HSET', KEYS[2], 'tokens', ttok, 'ts', now)
if ttl > 0 then
  redis.call('EXPIRE', KEYS[1], ttl)
  redis.call('EXPIRE', KEYS[2], ttl)
end

local retry, blocked = 0, 'none'
if allowed == 0 then
  local rneed = 0
  if ttok < tcost then
    blocked = 'tokens'
    if trate > 0 then retry = (tcost - ttok) / trate end
  else
    blocked = 'requests'
    if rrate > 0 then retry = (rcost - rtok) / rrate end
  end
end
return {allowed, tostring(retry), blocked}
"""


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after_s: float
    blocked_dimension: str  # 'none' | 'requests' | 'tokens'


class RateLimiter:
    def __init__(self, client, *, window_s: int = 60, now: Callable[[], float] = time.time):
        self.client = client
        self.window_s = window_s
        self.now = now
        self._script = client.register_script(_LUA)

    async def check(self, tenant: Tenant, estimated_tokens: int) -> RateLimitResult:
        """Atomically admit (and charge) one request costing `estimated_tokens` for a tenant."""
        req_cap = tenant.requests_per_min
        tok_cap = tenant.tokens_per_min
        keys = [f"rl:{tenant.id}:req", f"rl:{tenant.id}:tok"]
        args = [
            self.now(),
            req_cap,
            req_cap / self.window_s,
            1,
            tok_cap,
            tok_cap / self.window_s,
            estimated_tokens,
            self.window_s * 2,
        ]
        allowed, retry, blocked = await self._script(keys=keys, args=args)
        if isinstance(blocked, bytes):
            blocked = blocked.decode()
        if isinstance(retry, bytes):
            retry = retry.decode()
        return RateLimitResult(bool(allowed), float(retry), blocked)
