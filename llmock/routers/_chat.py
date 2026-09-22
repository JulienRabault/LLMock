"""Chat plumbing shared by every router that speaks the OpenAI chat format.

OpenAI, Groq, Together, Perplexity, xAI, Mistral and AI21 keep their own
request and response models -- each has real quirks worth mimicking -- but
decide *what* to say and *how to stream it* through these helpers, so a
scripted reply or a stream fault behaves the same on all of them.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from starlette.requests import Request
from starlette.responses import StreamingResponse

from llmock.completion import Completion, plan_of, resolve
from llmock.simulation import MockResponseSettings, estimate_tokens, flatten_text
from llmock.streaming import openai_chat_chunks, sse_response

__all__ = ["complete", "is_streaming", "openai_stream", "prompt_of", "raw_body"]


def raw_body(request: Request) -> dict[str, Any]:
    """The request JSON as the middleware parsed it, including undeclared fields."""
    return getattr(request.state, "llmock_body", None) or {}


def is_streaming(request: Request) -> bool:
    info = getattr(request.state, "llmock_info", None)
    if info is not None:
        return bool(info.stream)
    return raw_body(request).get("stream") is True


def prompt_of(messages: Iterable[Any]) -> tuple[str, int]:
    """Prompt text and token estimate for a list of chat messages.

    Tolerates list content (images, parts) and ``None`` content (assistant
    tool calls), which a naive ``" ".join(m.content ...)`` would crash on.
    """
    contents = [getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
                for m in messages]
    text = " ".join(part for part in (flatten_text(c) for c in contents if c is not None) if part)
    return text, estimate_tokens(*(c for c in contents if c is not None))


def complete(request: Request, *, model: str, prompt_text: str, prompt_tokens: int) -> Completion:
    settings: MockResponseSettings = request.app.state.mock_response_settings
    return resolve(
        plan=plan_of(request),
        settings=settings,
        model=model,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    )


def openai_stream(
    request: Request,
    completion: Completion,
    *,
    model: str,
    choices: int = 1,
    id_prefix: str = "chatcmpl",
) -> StreamingResponse:
    options = raw_body(request).get("stream_options") or {}
    return sse_response(
        openai_chat_chunks(
            completion,
            completion_id=f"{id_prefix}-{uuid.uuid4().hex[:24]}",
            model=model,
            include_usage=bool(options.get("include_usage")),
            choices=max(1, choices),
        )
    )
