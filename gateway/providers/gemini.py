from gateway.providers.openai_compat import OpenAICompatAdapter


class GeminiAdapter(OpenAICompatAdapter):
    """Google Gemini via its OpenAI-compatible endpoint.

    Phase 1: identical to the generic adapter (Gemini's non-streaming Chat Completions is
    spec-conformant). The known streaming-usage quirk — usage repeated in every chunk
    rather than only the final one — is normalized in Phase 5 when streaming lands.
    """
