"""Minimal OpenAI-compatible mock upstream for the resilience demo (Phase 7 Group 2).

One server exposes two backends — `/a/v1/chat/completions` and `/b/v1/chat/completions` — so the
gateway can be configured with two providers pointing at one process. Backend `a` can be toggled to
fail (503) at runtime via `/control/a/{fail,ok}`, simulating a provider going down mid-run so the
demo can show the circuit breaker trip and traffic fall back to `b`.

Run: `uvicorn bench.mock_upstream:app --port 8100`.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Switchyard mock upstream")

_state = {"a_healthy": True}


def _completion(model: str, backend: str) -> dict:
    return {
        "id": f"mock-{backend}",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"mock response from {backend}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }


@app.post("/a/v1/chat/completions")
async def backend_a(body: dict):
    if not _state["a_healthy"]:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "backend A is down", "type": "server_error"}},
        )
    return _completion(body.get("model", "mock-model"), "a")


@app.post("/b/v1/chat/completions")
async def backend_b(body: dict):
    return _completion(body.get("model", "mock-model"), "b")


@app.post("/control/a/{action}")
async def control_a(action: str):
    """Toggle backend A: `fail` -> 503, `ok` -> healthy."""
    _state["a_healthy"] = action == "ok"
    return {"a_healthy": _state["a_healthy"]}


@app.get("/health")
async def health():
    return {"ok": True, "a_healthy": _state["a_healthy"]}
