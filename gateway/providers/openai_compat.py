"""Generic adapter for any OpenAI-compatible /chat/completions backend.

All four fleet providers speak this surface, so this adapter is near pass-through: swap
base_url + auth header. Per-provider quirks (e.g. Gemini streaming usage) become subclasses
in later phases.
"""

import json
from collections.abc import AsyncIterator

import httpx

from gateway.providers.base import ProviderAdapter, UpstreamError
from gateway.schemas import ChatCompletionRequest, ChatCompletionResponse

_DONE = object()  # sentinel for the SSE terminator `data: [DONE]`


def _parse_sse_line(line: str):
    """Parse one SSE line. Returns a chunk dict, the _DONE sentinel, or None (non-data / blank)."""
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if not data:
        return None
    if data == "[DONE]":
        return _DONE
    try:
        return json.loads(data)
    except ValueError:
        return None  # skip a malformed chunk rather than killing the stream


class OpenAICompatAdapter(ProviderAdapter):
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        client: httpx.AsyncClient,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = client

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = request.model_dump(exclude_none=True)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        if not resp.is_success:
            try:
                body = resp.json()
            except ValueError:
                body = {"error": {"message": resp.text, "type": "upstream_error"}}
            raise UpstreamError(resp.status_code, body)
        return ChatCompletionResponse.model_validate(resp.json())

    async def stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncIterator[dict]:
        payload = request.model_dump(exclude_none=True)
        payload["stream"] = True
        # Ask upstream to include token usage in the final chunk (for post-stream accounting).
        stream_options = dict(payload.get("stream_options") or {})
        stream_options["include_usage"] = True
        payload["stream_options"] = stream_options
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        async with self.client.stream(
            "POST", f"{self.base_url}/chat/completions", headers=headers, json=payload
        ) as resp:
            if not resp.is_success:
                # Error before the first byte -> read the body and raise (callers can fall back).
                raw = await resp.aread()
                try:
                    body = json.loads(raw)
                except (ValueError, TypeError):
                    body = {
                        "error": {"message": raw.decode(errors="replace"), "type": "upstream_error"}
                    }
                raise UpstreamError(resp.status_code, body)
            async for line in resp.aiter_lines():
                chunk = _parse_sse_line(line)
                if chunk is _DONE:
                    return
                if isinstance(chunk, dict):
                    yield chunk
