"""Switchyard LLM Gateway — Phase 1: multi-provider registry + policy routing.

Client sends a logical alias (fast/smart/cheap/balanced) as `model`; the router resolves it
to an ordered list of concrete (provider, model) targets under a policy. Phase 1 calls the
first target; cross-provider failover over the rest lands in Phase 2.
"""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

from gateway.config import get_settings
from gateway.providers.base import ProviderAdapter, UpstreamError
from gateway.providers.registry import ProviderRegistry
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
    logger.info(
        "gateway ready: providers=%s aliases=%s",
        app.state.registry.available(),
        app.state.router.aliases(),
    )
    yield
    await client.aclose()


app = FastAPI(title="Switchyard LLM Gateway", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


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
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return _error(
            501,
            "Streaming is not supported yet (arrives in Phase 5).",
            "not_implemented",
        )

    router: Router = app.state.router
    registry: ProviderRegistry = app.state.registry

    try:
        targets = router.resolve(request.model)
    except KeyError:
        return _error(
            400,
            f"Unknown model alias {request.model!r}. Known aliases: {router.aliases()}.",
            "invalid_request_error",
        )

    # Phase 1: call the first (policy-selected) target. Fallback over targets[1:] is Phase 2.
    target = targets[0]
    adapter: ProviderAdapter | None = registry.get(target.provider)
    if adapter is None:
        return _error(
            503,
            f"Provider {target.provider!r} for alias {request.model!r} is not configured "
            "(missing API key?).",
            "provider_unavailable",
        )

    logger.info(
        "route alias=%s policy_selected=%s model=%s",
        request.model,
        target.provider,
        target.model,
    )
    upstream_request = request.model_copy(update={"model": target.model})
    try:
        return await adapter.chat_completion(upstream_request)
    except UpstreamError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.body)
    except httpx.RequestError as exc:
        return _error(502, f"Upstream request failed: {exc}", "upstream_error")
