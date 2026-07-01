"""Resilient execution over the router's ordered targets: circuit-breaker guarding + retry
with jittered backoff + cross-provider fallback, bounded by a per-request deadline.

Retries advance to the *next* target (a different provider), never the same struggling one —
retrying a failing backend just amplifies its load.
"""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable

import httpx

from gateway.providers.base import ProviderAdapter, UpstreamError
from gateway.providers.registry import ProviderRegistry
from gateway.resilience.circuit_breaker import BreakerRegistry
from gateway.routing.policies import Target
from gateway.schemas import ChatCompletionRequest, ChatCompletionResponse

logger = logging.getLogger("gateway.resilience")


def _is_retryable_status(status_code: int) -> bool:
    """429 (rate limit) and 5xx are transient/overload -> try the next provider."""
    return status_code == 429 or status_code >= 500


class CircuitOpenError(Exception):
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"circuit open for provider {provider!r}")


class AllTargetsFailed(Exception):
    """Every target was exhausted (failed, unavailable, or circuit-open)."""

    def __init__(self, last_error: Exception | None):
        self.last_error = last_error
        super().__init__(str(last_error) if last_error else "no targets available")


class ResilientExecutor:
    def __init__(
        self,
        breakers: BreakerRegistry,
        *,
        base_delay_s: float = 0.2,
        max_delay_s: float = 2.0,
        deadline_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ):
        self.breakers = breakers
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.deadline_s = deadline_s
        self.clock = clock
        self.sleep = sleep
        self.rng = rng or random.Random()

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter. attempt is 1-based (delay before attempt N)."""
        ceiling = min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
        return self.rng.uniform(0, ceiling)

    async def execute(
        self,
        targets: list[Target],
        registry: ProviderRegistry,
        request: ChatCompletionRequest,
    ) -> tuple[ChatCompletionResponse, Target]:
        start = self.clock()
        last_error: Exception | None = None
        attempts = 0

        for target in targets:
            if self.clock() - start > self.deadline_s:
                logger.warning("request deadline exceeded before provider=%s", target.provider)
                break

            adapter: ProviderAdapter | None = registry.get(target.provider)
            if adapter is None:
                last_error = UpstreamError(
                    503,
                    {
                        "error": {
                            "message": f"provider {target.provider!r} not configured",
                            "type": "provider_unavailable",
                        }
                    },
                )
                continue

            breaker = self.breakers.get(target.provider)
            if not breaker.allow():
                logger.info("skip provider=%s: circuit %s", target.provider, breaker.state.value)
                last_error = CircuitOpenError(target.provider)
                continue

            if attempts > 0:
                delay = self._backoff(attempts)
                if delay:
                    await self.sleep(delay)
            attempts += 1

            upstream = request.model_copy(update={"model": target.model})
            try:
                resp = await adapter.chat_completion(upstream)
            except UpstreamError as exc:
                if _is_retryable_status(exc.status_code):
                    breaker.record_failure()
                    last_error = exc
                    logger.info(
                        "provider=%s status=%s -> fallback", target.provider, exc.status_code
                    )
                    continue
                # Non-retryable client error (e.g. 400/401/404): provider is healthy, the request
                # is the problem. Don't trip the breaker; surface it to the client immediately.
                breaker.record_success()
                raise
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                breaker.record_failure()
                last_error = exc
                logger.info("provider=%s transport error -> fallback: %s", target.provider, exc)
                continue
            else:
                breaker.record_success()
                logger.info("served by provider=%s model=%s", target.provider, target.model)
                return resp, target

        raise AllTargetsFailed(last_error)
