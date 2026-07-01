"""Pre-call token estimate for rate-limit admission.

Uses `tiktoken` (cl100k_base) to count input tokens — an approximation for non-OpenAI
tokenizers (Groq/Gemini/OpenRouter/Ollama), reconciled after the call against the provider's real
`usage`. Falls back to a chars/4 heuristic if tiktoken can't load (e.g. offline), so the gateway
never hard-depends on the vocab download.
"""

import logging

from gateway.schemas import ChatCompletionRequest

logger = logging.getLogger("gateway.estimate")

DEFAULT_OUTPUT_ESTIMATE = 256
PER_MESSAGE_OVERHEAD = 3  # chat framing per message (OpenAI counts ~3)

_DEFAULT = object()
_encoder = None
_encoder_loaded = False


def _get_encoder():
    """Lazily load tiktoken once; None means 'use the heuristic'. Loading is deferred so importing
    this module never triggers a network download (only the first estimate does)."""
    global _encoder, _encoder_loaded
    if not _encoder_loaded:
        _encoder_loaded = True
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # offline, download failure, etc.
            logger.warning("tiktoken unavailable, using heuristic token estimate: %s", exc)
            _encoder = None
    return _encoder


def _count_text(text: str, encoder) -> int:
    if not text:
        return 0
    if encoder is not None:
        return len(encoder.encode(text))
    return len(text) // 4 + 1  # heuristic fallback (~4 chars/token)


def estimate_tokens(request: ChatCompletionRequest, encoder=_DEFAULT) -> int:
    """Estimate total tokens (input + expected output) for admission.

    `encoder` is injectable for tests: pass a real/fake encoder, or None to force the heuristic.
    """
    if encoder is _DEFAULT:
        encoder = _get_encoder()

    input_tokens = 0
    for message in request.messages:
        input_tokens += PER_MESSAGE_OVERHEAD
        content = message.content
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        input_tokens += _count_text(content, encoder)

    output_estimate = request.max_tokens or DEFAULT_OUTPUT_ESTIMATE
    return input_tokens + output_estimate
