"""Per-provider conformance: send the same request to each configured provider and assert a
well-formed OpenAI Chat Completions response. Providers without a key are skipped (registry
won't build them); unreachable providers (e.g. Ollama not running) are skipped at call time.

These are live integration tests — they make real upstream calls for providers that have keys.
"""

import asyncio

import httpx
import pytest
from dotenv import load_dotenv

from gateway.providers.base import UpstreamError
from gateway.providers.registry import ProviderRegistry
from gateway.schemas import ChatCompletionRequest, ChatCompletionResponse, Message

load_dotenv()  # so provider keys in .env are visible to the registry during tests

PROVIDERS_CONFIG = "config/providers.yaml"

# One representative real model per provider (edit to match your account's available models).
CONFORMANCE_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-3-flash-preview",
    "openrouter": "openai/gpt-oss-120b:free",
    "ollama": "llama3.2",
}

# Reasoning models (e.g. Gemini 3 Flash) spend tokens on internal thinking before any visible
# output, so a tiny budget yields empty content + finish_reason=length. Give enough headroom.
CONFORMANCE_MAX_TOKENS = 256


async def _call(provider: str, model: str) -> ChatCompletionResponse:
    async with httpx.AsyncClient(timeout=60.0) as client:
        registry = ProviderRegistry.from_config(PROVIDERS_CONFIG, client)
        adapter = registry.get(provider)
        if adapter is None:
            pytest.skip(f"provider {provider!r} not configured (no API key)")
        request = ChatCompletionRequest(
            model=model,
            messages=[Message(role="user", content="Reply with the single word: pong")],
            max_tokens=CONFORMANCE_MAX_TOKENS,
        )
        try:
            return await adapter.chat_completion(request)
        except httpx.RequestError as exc:
            pytest.skip(f"provider {provider!r} unreachable: {exc}")
        except UpstreamError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:
                # Transient availability: 429 = free-tier quota, 5xx = upstream overloaded
                # (PRD §10). Phase 2 fallback handles these in production, and they say nothing
                # about OpenAI-conformance, so skip. Genuine issues (400/401/404) still fail.
                pytest.skip(f"provider {provider!r} unavailable ({exc.status_code})")
            raise


@pytest.mark.parametrize("provider,model", list(CONFORMANCE_MODELS.items()))
def test_provider_conformance(provider: str, model: str):
    resp = asyncio.run(_call(provider, model))
    assert isinstance(resp, ChatCompletionResponse)
    assert resp.choices, "response had no choices"
    assert resp.choices[0].message.content is not None
    assert resp.usage is not None and resp.usage.total_tokens > 0
