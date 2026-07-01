"""Switchyard LLM Gateway — Phase 2: resilient multi-provider routing.

Client sends a logical alias as `model`; the router resolves it to ordered targets; the resilient
executor runs them through per-provider circuit breakers with jittered-backoff retry and
cross-provider fallback, so a dead/throttled backend never reaches the client as an error.
"""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from gateway.config import get_settings
from gateway.providers.base import UpstreamError
from gateway.providers.registry import ProviderRegistry
from gateway.resilience.circuit_breaker import BreakerRegistry
from gateway.resilience.retry import AllTargetsFailed, ResilientExecutor
from gateway.routing.router import Router
from gateway.schemas import ChatCompletionRequest, ChatCompletionResponse

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
    logger.info(
        "gateway ready: providers=%s aliases=%s",
        app.state.registry.available(),
        app.state.router.aliases(),
    )
    yield
    await client.aclose()


app = FastAPI(title="Switchyard LLM Gateway", version="0.2.0", lifespan=lifespan)


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
async def chat_completions(request: ChatCompletionRequest, response: Response):
    if request.stream:
        return _error(
            501, "Streaming is not supported yet (arrives in Phase 5).", "not_implemented"
        )

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

    response.headers["x-switchyard-provider"] = target.provider
    response.headers["x-switchyard-model"] = target.model
    return result
