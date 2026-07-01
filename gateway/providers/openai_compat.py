"""Generic adapter for any OpenAI-compatible /chat/completions backend.

All four fleet providers speak this surface, so this adapter is near pass-through: swap
base_url + auth header. Per-provider quirks (e.g. Gemini streaming usage) become subclasses
in later phases.
"""

import httpx

from gateway.providers.base import ProviderAdapter, UpstreamError
from gateway.schemas import ChatCompletionRequest, ChatCompletionResponse


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

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
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
