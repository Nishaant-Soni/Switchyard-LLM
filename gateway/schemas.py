"""OpenAI-compatible Chat Completions models — the gateway's canonical internal representation.

Phase 0 covers the non-streaming request/response shape. `extra="allow"` on every model so
unknown OpenAI params/fields pass through unchanged instead of being silently dropped.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    n: int | None = None
    stop: Any | None = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    message: Message
    finish_reason: str | None = None


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None
