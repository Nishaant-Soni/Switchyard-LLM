from collections.abc import AsyncIterator

from gateway.providers.openai_compat import OpenAICompatAdapter
from gateway.schemas import ChatCompletionRequest


class GeminiAdapter(OpenAICompatAdapter):
    """Google Gemini via its OpenAI-compatible endpoint.

    Non-streaming Chat Completions is spec-conformant. On the **streaming** path Gemini emits
    `usage` in *every* chunk (not just the final one, as the OpenAI spec intends), which would
    inflate accounting and leak a non-conformant stream to the client. This override keeps `usage`
    only on the final chunk via a one-chunk lookahead.
    """

    async def stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncIterator[dict]:
        prev: dict | None = None
        async for chunk in super().stream_chat_completion(request):
            if prev is not None:
                prev.pop("usage", None)  # not the final chunk -> drop the repeated usage
                yield prev
            prev = chunk
        if prev is not None:
            yield prev  # final chunk keeps its usage
