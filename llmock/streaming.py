"""Server-sent event encoders, one per wire format.

Each encoder yields one complete SSE event per item. The middleware relies
on that: one ASGI body message is one event, which is what lets it cut,
stall or corrupt a stream at a precise point for every provider alike.

Three formats cover all ten providers:

- OpenAI ``chat.completion.chunk`` -- OpenAI, Groq, Together, Perplexity,
  xAI, Mistral and AI21 all speak it;
- Anthropic's typed events (``message_start`` ... ``message_stop``);
- Gemini's ``streamGenerateContent?alt=sse`` candidates.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Iterator
from typing import Any

from starlette.responses import StreamingResponse

from llmock.completion import Completion

__all__ = [
    "anthropic_events",
    "gemini_chunks",
    "openai_chat_chunks",
    "sse",
    "sse_response",
    "text_pieces",
]

_SSE_HEADERS = {"cache-control": "no-cache", "x-accel-buffering": "no"}

# Tool-call arguments are streamed in fragments of this many characters,
# as providers do, so clients have to reassemble them.
_ARGUMENT_FRAGMENT = 12

_PIECE_RE = re.compile(r"\s*\S+\s*")


def sse(data: Any, event: str | None = None) -> bytes:
    """Encode one SSE event. ``data`` is JSON-encoded unless it is a str."""
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n".encode()


def sse_response(events: Iterable[bytes]) -> StreamingResponse:
    return StreamingResponse(events, media_type="text/event-stream", headers=_SSE_HEADERS)


def text_pieces(text: str) -> list[str]:
    """Split text into word-sized pieces that concatenate back to ``text``."""
    pieces = _PIECE_RE.findall(text)
    return pieces if "".join(pieces) == text else [text]


def _fragments(value: str, size: int = _ARGUMENT_FRAGMENT) -> list[str]:
    return [value[i : i + size] for i in range(0, len(value), size)] or [""]


# -- OpenAI chat.completion.chunk ---------------------------------------------

_OPENAI_FINISH = {"stop": "stop", "length": "length", "tool_calls": "tool_calls",
                  "content_filter": "content_filter"}


def openai_chat_chunks(
    completion: Completion,
    *,
    completion_id: str,
    model: str,
    include_usage: bool = False,
    choices: int = 1,
    created: int | None = None,
) -> Iterator[bytes]:
    created = int(time.time()) if created is None else created

    def chunk(choice_list: list[dict[str, Any]], usage: dict[str, int] | None = None) -> bytes:
        data: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": None,
            "choices": choice_list,
        }
        if include_usage:
            data["usage"] = usage
        return sse(data)

    def choice(index: int, delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {"index": index, "delta": delta, "logprobs": None, "finish_reason": finish}

    for index in range(choices):
        if completion.tool_calls and not completion.text:
            yield chunk([choice(index, {"role": "assistant", "content": None})])
        else:
            yield chunk([choice(index, {"role": "assistant", "content": ""})])
            for piece in text_pieces(completion.text):
                yield chunk([choice(index, {"content": piece})])

        for position, call in enumerate(completion.tool_calls):
            yield chunk([choice(index, {"tool_calls": [{
                "index": position,
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": ""},
            }]})])
            for fragment in _fragments(call.arguments):
                yield chunk([choice(index, {"tool_calls": [{
                    "index": position,
                    "function": {"arguments": fragment},
                }]})])

        finish = _OPENAI_FINISH.get(completion.finish_reason, completion.finish_reason)
        yield chunk([choice(index, {}, finish)])

    if include_usage:
        yield chunk([], usage={
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "total_tokens": completion.total_tokens,
        })
    yield sse("[DONE]")


# -- Anthropic messages -------------------------------------------------------

_ANTHROPIC_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                   "content_filter": "refusal"}


def anthropic_events(completion: Completion, *, message_id: str, model: str) -> Iterator[bytes]:
    yield sse({
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": completion.prompt_tokens, "output_tokens": 1},
        },
    }, event="message_start")

    block = 0
    if completion.text or not completion.tool_calls:
        yield sse({"type": "content_block_start", "index": block,
                   "content_block": {"type": "text", "text": ""}}, event="content_block_start")
        yield sse({"type": "ping"}, event="ping")
        for piece in text_pieces(completion.text):
            yield sse({"type": "content_block_delta", "index": block,
                       "delta": {"type": "text_delta", "text": piece}}, event="content_block_delta")
        yield sse({"type": "content_block_stop", "index": block}, event="content_block_stop")
        block += 1

    for call in completion.tool_calls:
        yield sse({"type": "content_block_start", "index": block, "content_block": {
            "type": "tool_use", "id": call.id, "name": call.name, "input": {},
        }}, event="content_block_start")
        for fragment in _fragments(call.arguments):
            yield sse({"type": "content_block_delta", "index": block,
                       "delta": {"type": "input_json_delta", "partial_json": fragment}},
                      event="content_block_delta")
        yield sse({"type": "content_block_stop", "index": block}, event="content_block_stop")
        block += 1

    yield sse({
        "type": "message_delta",
        "delta": {
            "stop_reason": _ANTHROPIC_STOP.get(completion.finish_reason, completion.finish_reason),
            "stop_sequence": None,
        },
        "usage": {"output_tokens": completion.completion_tokens},
    }, event="message_delta")
    yield sse({"type": "message_stop"}, event="message_stop")


# -- Gemini streamGenerateContent ---------------------------------------------

_GEMINI_FINISH = {"stop": "STOP", "length": "MAX_TOKENS", "tool_calls": "STOP",
                  "content_filter": "SAFETY"}


def gemini_chunks(completion: Completion, *, model: str) -> Iterator[bytes]:
    def chunk(parts: list[dict[str, Any]], finish: str | None = None) -> bytes:
        candidate: dict[str, Any] = {"content": {"parts": parts, "role": "model"}, "index": 0}
        data: dict[str, Any] = {"candidates": [candidate], "modelVersion": model}
        if finish is not None:
            candidate["finishReason"] = finish
            data["usageMetadata"] = {
                "promptTokenCount": completion.prompt_tokens,
                "candidatesTokenCount": completion.completion_tokens,
                "totalTokenCount": completion.total_tokens,
            }
        return sse(data)

    pieces = text_pieces(completion.text)
    finish = _GEMINI_FINISH.get(completion.finish_reason, "STOP")
    # Gemini sends a function call whole, in a single part.
    calls = [{"functionCall": {"name": c.name, "args": c.parsed_arguments}}
             for c in completion.tool_calls]

    for piece in pieces[:-1]:
        yield chunk([{"text": piece}])
    last_parts = ([{"text": pieces[-1]}] if pieces else []) + calls
    yield chunk(last_parts or [{"text": ""}], finish)
