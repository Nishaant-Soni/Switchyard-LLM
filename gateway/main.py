"""Switchyard LLM Gateway — Phase 0 walking skeleton.

OpenAI SDK client -> gateway -> Groq -> response. Non-streaming only. No routing, cache,
rate limiting, or metrics yet — those arrive in later phases.
"""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

from gateway.config import get_settings
from gateway.providers.base import UpstreamError
from gateway.providers.openai_compat import OpenAICompatAdapter
from gateway.schemas import ChatCompletionRequest, ChatCompletionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = httpx.AsyncClient(timeout=settings.request_timeout_s)
    app.state.client = client
    app.state.adapter = OpenAICompatAdapter(
        name="groq",
        base_url=settings.groq_base_url,
        api_key=settings.groq_api_key,
        client=client,
    )
    yield
    await client.aclose()


app = FastAPI(title="Switchyard LLM Gateway", version="0.0.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    # Populated in Phase 6 (Prometheus). Empty exposition for now.
    return PlainTextResponse("", media_type="text/plain; version=0.0.4")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": "Streaming is not supported yet (arrives in Phase 5).",
                    "type": "not_implemented",
                }
            },
        )
    adapter: OpenAICompatAdapter = app.state.adapter
    try:
        return await adapter.chat_completion(request)
    except UpstreamError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.body)
    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Upstream request failed: {exc}",
                    "type": "upstream_error",
                }
            },
        )
