"""Switchyard LLM Gateway.

Request pipeline (non-streaming): auth (API key -> tenant) -> alias validation -> rate-limit
admission (Redis token-bucket on request rate AND token rate) -> semantic-cache read -> resilient
execution -> cache write. Rate limiting is active only when tenants are configured; the semantic
cache is active when cache_enabled (both degrade gracefully when off).
"""

import logging
import math
import time
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Header, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from gateway.cache.embedder import SentenceTransformerEmbedder
from gateway.cache.semantic_cache import SemanticCache, build_query_text, build_scope_key
from gateway.config import get_settings
from gateway.observability import metrics
from gateway.observability.cost import PriceBook
from gateway.providers.base import UpstreamError
from gateway.providers.registry import ProviderRegistry
from gateway.ratelimit.estimate import estimate_tokens
from gateway.ratelimit.limiter import RateLimiter
from gateway.resilience.circuit_breaker import BreakerRegistry
from gateway.resilience.retry import AllTargetsFailed, ResilientExecutor
from gateway.routing.policies import Target
from gateway.routing.router import Router
from gateway.routing.signals import RoutingSignals
from gateway.schemas import ChatCompletionRequest, ChatCompletionResponse
from gateway.streaming.sse import stream_sse
from gateway.tenancy.auth import Tenant, TenantRegistry, extract_bearer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = httpx.AsyncClient(timeout=settings.request_timeout_s)
    app.state.client = client
    app.state.registry = ProviderRegistry.from_config(settings.providers_config, client)
    app.state.pricebook = PriceBook.from_config(settings.pricing_config)
    # Live in-process signals (per-provider EWMA latency + list-price cost) shared by the executor
    # (writer) and the router's latency-/cost-aware policies (reader).
    app.state.signals = RoutingSignals(app.state.pricebook)
    app.state.router = Router.from_config(settings.models_config, signals=app.state.signals)
    app.state.breakers = BreakerRegistry()
    app.state.executor = ResilientExecutor(app.state.breakers, signals=app.state.signals)

    app.state.tenants = TenantRegistry.from_config(settings.tenants_config)
    app.state.redis = None
    app.state.limiter = None
    if app.state.tenants.enabled:
        app.state.redis = redis.from_url(settings.redis_url)
        app.state.limiter = RateLimiter(app.state.redis, window_s=settings.rate_limit_window_s)
        logger.info("rate limiting ENABLED (redis=%s)", settings.redis_url)
    else:
        logger.info("rate limiting disabled (no tenants configured)")

    app.state.cache = None
    app.state.cache_per_tenant = settings.cache_per_tenant
    if settings.cache_enabled:
        app.state.cache = SemanticCache(
            SentenceTransformerEmbedder(),
            threshold=settings.cache_similarity_threshold,
            ttl_s=settings.cache_ttl_s,
            max_entries=settings.cache_max_entries,
        )
        logger.info(
            "semantic cache ENABLED (threshold=%s, ttl_s=%s, per_tenant=%s)",
            settings.cache_similarity_threshold,
            settings.cache_ttl_s,
            settings.cache_per_tenant,
        )
    else:
        logger.info("semantic cache disabled")

    logger.info(
        "gateway ready: providers=%s aliases=%s",
        app.state.registry.available(),
        app.state.router.aliases(),
    )
    yield
    await client.aclose()
    if app.state.redis is not None:
        await app.state.redis.aclose()


app = FastAPI(title="Switchyard LLM Gateway", version="0.7.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "circuits": app.state.breakers.snapshot()}


@app.get("/metrics")
async def metrics_endpoint():
    data, content_type = metrics.render()
    return PlainTextResponse(data, media_type=content_type)


def _error(status_code: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
    )


async def _refund_tokens(tenant: Tenant | None, estimated_tokens: int) -> None:
    """Refund tokens charged at admission for a request whose upstream attempt produced nothing
    (a non-retryable provider error, or every target failing): actual usage was 0, so reconcile
    against zero. The request slot stays charged — a real upstream attempt was made, so request-rate
    limiting still throttles a client retrying a dead route. (An unknown alias is validated before
    admission, so it is never charged and never reaches here.)"""
    if tenant is not None:
        await app.state.limiter.reconcile(tenant, 0, estimated_tokens)


def _cache_scope(request: ChatCompletionRequest, tenant: Tenant | None) -> str:
    """Cache scope = model alias + output-affecting params. Shared across tenants by default;
    prepend the tenant id when per-tenant isolation is configured."""
    scope = build_scope_key(request)
    if app.state.cache_per_tenant and tenant is not None:
        scope = f"{tenant.id}\x1f{scope}"
    return scope


async def _streaming_completion(
    request: ChatCompletionRequest,
    targets: list[Target],
    registry: ProviderRegistry,
    tenant: Tenant | None,
    estimated_tokens: int,
) -> Response:
    """Stream a completion with breaker + cross-provider fallback up to first byte. The executor
    peeks the first chunk, so a pre-first-byte failure returns a proper error status (not a broken
    200); token reconciliation runs when the stream finishes (see `on_finish`)."""
    executor: ResilientExecutor = app.state.executor
    metrics.record_cache("bypass")  # streaming never touches the semantic cache
    started = time.perf_counter()
    try:
        first, chunks, target = await executor.execute_stream(targets, registry, request)
    except UpstreamError as exc:
        # Non-retryable client error before first byte, forwarded verbatim.
        await _refund_tokens(tenant, estimated_tokens)
        metrics.record_error(f"upstream_{exc.status_code}")
        metrics.record_request(request.model, "none", "error", True)
        return JSONResponse(status_code=exc.status_code, content=exc.body)
    except AllTargetsFailed as exc:
        await _refund_tokens(tenant, estimated_tokens)
        metrics.record_error("all_targets_failed")
        metrics.record_request(request.model, "none", "error", True)
        last = exc.last_error
        if isinstance(last, UpstreamError):
            return JSONResponse(status_code=last.status_code, content=last.body)
        return _error(
            502,
            f"All targets failed for alias {request.model!r}: {last}",
            "all_targets_failed",
        )

    # Committed (first byte returned): record the served request + time-to-first-byte. Tokens/cost
    # are only known when the stream finishes, so they land in on_finish.
    metrics.observe_latency(target.provider, True, time.perf_counter() - started)
    metrics.record_request(request.model, target.provider, "success", True)

    async def on_finish(usage: dict | None, completed: bool) -> None:
        if tenant is not None:
            total = usage.get("total_tokens") if usage else None
            if completed and total is not None:
                await app.state.limiter.reconcile(tenant, total, estimated_tokens)
            elif not completed:
                # Errored / disconnected before a final usage -> refund (produced nothing complete).
                await _refund_tokens(tenant, estimated_tokens)
            # completed but no usage reported -> leave the estimate (don't hand out free tokens).
        if completed and usage:
            prompt = int(usage.get("prompt_tokens") or 0)
            completion = int(usage.get("completion_tokens") or 0)
            cost = app.state.pricebook.cost_usd(target.model, prompt, completion)
            metrics.record_usage(
                tenant.id if tenant else "anonymous", target.provider, prompt, completion, cost
            )

    logger.info(
        "stream alias=%s provider=%s model=%s", request.model, target.provider, target.model
    )
    return StreamingResponse(
        stream_sse(first, chunks, on_finish),
        media_type="text/event-stream",
        headers={
            "x-switchyard-provider": target.provider,
            "x-switchyard-model": target.model,
            "x-switchyard-cache": "bypass",
        },
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    response: Response,
    authorization: str | None = Header(default=None),
):
    tenants: TenantRegistry = app.state.tenants
    tenant = None
    estimated_tokens = 0

    # Auth first — even a request we may not serve must present a valid key.
    if tenants.enabled:
        tenant = tenants.resolve(extract_bearer(authorization))
        if tenant is None:
            metrics.record_error("invalid_api_key")
            return _error(401, "Invalid or missing API key.", "invalid_api_key")

    router: Router = app.state.router
    registry: ProviderRegistry = app.state.registry
    executor: ResilientExecutor = app.state.executor

    # Validate the alias BEFORE rate-limit admission: an unknown alias is a client-side validation
    # error (400) that does no billable work and never reaches a provider, so it must not consume
    # any quota (no charge, not a charge-then-refund).
    try:
        targets = router.resolve(request.model)
    except KeyError:
        metrics.record_error("invalid_request_error")
        return _error(
            400,
            f"Unknown model alias {request.model!r}. Known aliases: {router.aliases()}.",
            "invalid_request_error",
        )

    # Rate-limit admission: charge the pre-call token estimate against both buckets.
    if tenant is not None:
        estimated_tokens = estimate_tokens(request)
        rl = await app.state.limiter.check(tenant, estimated_tokens)
        if not rl.allowed:
            metrics.record_error("rate_limit_exceeded")
            metrics.record_request(request.model, "none", "throttled", request.stream)
            resp = _error(
                429,
                f"Rate limit exceeded for tenant {tenant.id!r} ({rl.blocked_dimension}).",
                "rate_limit_exceeded",
            )
            resp.headers["Retry-After"] = str(max(1, math.ceil(rl.retry_after_s)))
            return resp

    # Streaming bypasses the cache and streams chunks straight through. Group 1 = single target;
    # breaker + cross-provider fallback up to first byte lands in Group 2.
    if request.stream:
        return await _streaming_completion(request, targets, registry, tenant, estimated_tokens)

    # Semantic cache read (after admission, before the upstream call). A hit serves with no
    # upstream call. Embedding is the cache's one expensive/blocking step, so it runs off the event
    # loop *once*; the vector is reused for the write on a miss (no double-embed).
    cache: SemanticCache | None = app.state.cache
    scope = query_vec = None
    if cache is not None:
        scope = _cache_scope(request, tenant)
        query_vec = await run_in_threadpool(cache.embed_query, build_query_text(request))
        cached = cache.get(scope, query_vec)
        if cached is not None:
            # Hit: zero upstream tokens -> refund the estimate (request slot stays charged).
            await _refund_tokens(tenant, estimated_tokens)
            metrics.record_cache("hit")
            metrics.record_request(request.model, "cache", "success", False)
            logger.info("cache HIT alias=%s", request.model)
            response.headers["x-switchyard-cache"] = "hit"
            return cached
        metrics.record_cache("miss")

    started = time.perf_counter()
    try:
        result, target = await executor.execute(targets, registry, request)
    except UpstreamError as exc:
        # An upstream attempt was made but returned a non-retryable client error (forwarded
        # verbatim). Keep the request slot charged (an attempt happened), refund the tokens.
        await _refund_tokens(tenant, estimated_tokens)
        metrics.record_error(f"upstream_{exc.status_code}")
        metrics.record_request(request.model, "none", "error", False)
        return JSONResponse(status_code=exc.status_code, content=exc.body)
    except AllTargetsFailed as exc:
        await _refund_tokens(tenant, estimated_tokens)
        metrics.record_error("all_targets_failed")
        metrics.record_request(request.model, "none", "error", False)
        last = exc.last_error
        if isinstance(last, UpstreamError):
            # e.g. every target rate-limited -> forward the last upstream status/body.
            return JSONResponse(status_code=last.status_code, content=last.body)
        return _error(
            502,
            f"All targets failed for alias {request.model!r}: {last}",
            "all_targets_failed",
        )

    metrics.observe_latency(target.provider, False, time.perf_counter() - started)
    metrics.record_request(request.model, target.provider, "success", False)

    # Reconcile the token bucket against actual usage: admission charged the estimate.
    if tenant is not None and result.usage is not None:
        await app.state.limiter.reconcile(tenant, result.usage.total_tokens, estimated_tokens)

    # Token throughput + counterfactual cost attribution.
    if result.usage is not None:
        cost = app.state.pricebook.cost_usd(
            target.model, result.usage.prompt_tokens, result.usage.completion_tokens
        )
        metrics.record_usage(
            tenant.id if tenant else "anonymous",
            target.provider,
            result.usage.prompt_tokens,
            result.usage.completion_tokens,
            cost,
        )

    # Cache write on a successful (non-streaming) miss — reuses the vector embedded for the read.
    if cache is not None:
        cache.set(scope, query_vec, result)
        response.headers["x-switchyard-cache"] = "miss"

    response.headers["x-switchyard-provider"] = target.provider
    response.headers["x-switchyard-model"] = target.model
    return result
