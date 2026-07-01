"""Provider adapter interface (Strategy). Concrete adapters translate the canonical
Chat Completions request into an upstream call and normalize the response back."""

from abc import ABC, abstractmethod
from typing import Any

from gateway.schemas import ChatCompletionRequest, ChatCompletionResponse


class UpstreamError(Exception):
    """A provider returned a non-2xx HTTP response. Carries the upstream status + body
    so the gateway can forward it to the client unchanged (Phase 0). Resilience handling
    (breaker/retry/fallback) lands in Phase 2."""

    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self.body = body
        super().__init__(f"upstream returned {status_code}")


class ProviderAdapter(ABC):
    name: str

    @abstractmethod
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...
