"""Re-emit normalized provider chunks as an SSE stream, tapping the final `usage` for post-stream
token accounting.

The first chunk is peeked by the caller (so a pre-first-byte failure returns a proper error status
instead of a broken 200); this generator forwards it plus the rest, appends the `[DONE]` terminator,
and runs `on_finish` exactly once when the stream ends — normally or via error/disconnect — so the
token bucket is reconciled (or refunded) even if the client goes away mid-stream. It also closes the
upstream chunk generator on the way out, releasing the httpx connection promptly on a client
disconnect rather than leaving it to GC finalization.
"""

import json
from collections.abc import AsyncIterator, Awaitable, Callable

# on_finish(usage_or_None, completed): reconcile against usage when completed, else refund.
OnFinish = Callable[[dict | None, bool], Awaitable[None]]


def _sse(chunk: dict) -> bytes:
    return f"data: {json.dumps(chunk)}\n\n".encode()


async def stream_sse(
    first: dict | None,
    rest: AsyncIterator[dict],
    on_finish: OnFinish,
) -> AsyncIterator[bytes]:
    usage: dict | None = None
    completed = False
    try:
        if first is not None:
            usage = first.get("usage") or usage
            yield _sse(first)
        async for chunk in rest:
            usage = chunk.get("usage") or usage
            yield _sse(chunk)
        yield b"data: [DONE]\n\n"
        completed = True
    finally:
        # Close the upstream generator (releasing its httpx stream) before accounting, so the
        # connection is freed deterministically even on a mid-stream client disconnect (which
        # aclose()s this generator). Guarded: the parameter type only promises an async iterator.
        aclose = getattr(rest, "aclose", None)
        if aclose is not None:
            await aclose()
        await on_finish(usage if completed else None, completed)
