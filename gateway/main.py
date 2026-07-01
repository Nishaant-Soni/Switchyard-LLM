"""Switchyard LLM Gateway — Phase 3 (Group 1): per-tenant rate limiting.

Request pipeline: auth (API key -> tenant) -> rate-limit admission (Redis token-bucket on request
rate AND token rate) -> routing -> resilient execution. Rate limiting is active only when tenants
are configured in config/tenants.yaml; otherwise the gateway runs open (no Redis needed).
"""

import logging
import math
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Header, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from gateway.config import get_settings
from gateway.providers.base import UpstreamError
from gateway.providers.registry import ProviderRegistry
from gateway.ratelimit.estimate import estimate_tokens
from gateway.ratelimit.limiter import RateLimiter
from gateway.resilience.circuit_breaker import BreakerRegistry
from gateway.resilience.retry import AllTargetsFailed, ResilientExecutor
from gateway.routing.router import Router
from gateway.schemas import ChatCompletionRequest, ChatCompletionResponse
from gateway.tenancy.auth import TenantRegistry, extract_bearer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = httpx.AsyncClient(timeout=settings.request_timeout_s)
    app.state.client = client
    app.state.registry = ProviderRegistry.from_config(settings.providers_config, client)
    app.state.router = Router.from_config(settings.models_config)
    app.state.breakers = BreakerRegistry()
    app.state.executor = ResilientExecutor(app.state.breakers)

    app.state.tenants = TenantRegistry.from_config(settings.tenants_config)
    app.state.redis = None
    app.state.limiter = None
    if app.state.tenants.enabled:
        app.state.redis = redis.from_url(settings.redis_url)
        app.state.limiter = RateLimiter(app.state.redis, window_s=settings.rate_limit_window_s)
        logger.info("rate limiting ENABLED (redis=%s)", settings.redis_url)
    else:
        logger.info("rate limiting disabled (no tenants configured)")

    logger.info(
        "gateway ready: providers=%s aliases=%s",
        app.state.registry.available(),
        app.state.router.aliases(),
    )
    yield
    await client.aclose()
    if app.state.redis is not None:
        await app.state.redis.aclose()


app = FastAPI(title="Switchyard LLM Gateway", version="0.3.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "circuits": app.state.breakers.snapshot()}


@app.get("/metrics")
async def metrics():
    # Populated in Phase 6 (Prometheus). Empty exposition for now.
    return PlainTextResponse("", media_type="text/plain; version=0.0.4")


def _error(status_code: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    response: Response,
    authorization: str | None = Header(default=None),
):
    if request.stream:
        return _error(
            501, "Streaming is not supported yet (arrives in Phase 5).", "not_implemented"
        )

    tenants: TenantRegistry = app.state.tenants
    tenant = None
    estimated_tokens = 0
    if tenants.enabled:
        tenant = tenants.resolve(extract_bearer(authorization))
        if tenant is None:
            return _error(401, "Invalid or missing API key.", "invalid_api_key")
        estimated_tokens = estimate_tokens(request)
        rl = await app.state.limiter.check(tenant, estimated_tokens)
        if not rl.allowed:
            resp = _error(
                429,
                f"Rate limit exceeded for tenant {tenant.id!r} ({rl.blocked_dimension}).",
                "rate_limit_exceeded",
            )
            resp.headers["Retry-After"] = str(max(1, math.ceil(rl.retry_after_s)))
            return resp

    router: Router = app.state.router
    registry: ProviderRegistry = app.state.registry
    executor: ResilientExecutor = app.state.executor

    try:
        targets = router.resolve(request.model)
    except KeyError:
        return _error(
            400,
            f"Unknown model alias {request.model!r}. Known aliases: {router.aliases()}.",
            "invalid_request_error",
        )

    try:
        result, target = await executor.execute(targets, registry, request)
    except UpstreamError as exc:
        # Non-retryable client error (e.g. 400/401) forwarded verbatim.
        return JSONResponse(status_code=exc.status_code, content=exc.body)
    except AllTargetsFailed as exc:
        last = exc.last_error
        if isinstance(last, UpstreamError):
            # e.g. every target rate-limited -> forward the last upstream status/body.
            return JSONResponse(status_code=last.status_code, content=last.body)
        return _error(
            502,
            f"All targets failed for alias {request.model!r}: {last}",
            "all_targets_failed",
        )

    # Reconcile the token bucket against actual usage (Group 2): admission charged the estimate.
    if tenant is not None and result.usage is not None:
        await app.state.limiter.reconcile(tenant, result.usage.total_tokens, estimated_tokens)

    response.headers["x-switchyard-provider"] = target.provider
    response.headers["x-switchyard-model"] = target.model
    return result
