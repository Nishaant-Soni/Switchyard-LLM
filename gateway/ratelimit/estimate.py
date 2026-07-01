"""Pre-call token estimate for rate-limit admission.

Group 1 uses a cheap heuristic (~4 chars/token + expected output). Group 2 replaces the input
side with a `tiktoken` approximation and reconciles against the provider's real `usage`.
"""

from gateway.schemas import ChatCompletionRequest

DEFAULT_OUTPUT_ESTIMATE = 256


def estimate_tokens(request: ChatCompletionRequest) -> int:
    chars = 0
    for message in request.messages:
        if isinstance(message.content, str):
            chars += len(message.content)
        elif message.content is not None:
            chars += len(str(message.content))
    input_estimate = chars // 4 + 1
    output_estimate = request.max_tokens or DEFAULT_OUTPUT_ESTIMATE
    return input_estimate + output_estimate
