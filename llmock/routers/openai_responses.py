"""OpenAI Responses API: ``POST /v1/responses``.

The API the OpenAI Agents SDK uses by default. Output items, stream events
and their required fields follow the ``openai`` SDK's own types
(``openai.types.responses``), so that ``client.responses.create``, the
``responses.stream()`` helper and ``responses.parse()`` all accept them.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Request

from llmock.completion import Completion, resolve_request
from llmock.routers import registry
from llmock.routers._chat import is_streaming, raw_body
from llmock.simulation import estimate_tokens, flatten_text
from llmock.streaming import fragments, sse, sse_response, text_pieces

router = APIRouter(prefix="/v1", tags=["openai-responses"])


@router.post("/responses")
def create_response(request: Request):
    body = raw_body(request)
    model = str(body.get("model") or "gpt-4o")
    prompt_text, prompt_tokens = _prompt(body)
    completion = resolve_request(
        request,
        model=model,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
        dialect="openai-responses",
    )
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    items = _output_items(completion)
    if is_streaming(request):
        return sse_response(_events(body, response_id, model, completion, items))
    return _response_object(body, response_id, model, completion, items, status="completed")


def _prompt(body: dict[str, Any]) -> tuple[str, int]:
    """Text and token estimate of ``input`` (a string or a list of items)."""
    pieces: list[Any] = []
    if body.get("instructions"):
        pieces.append(body["instructions"])
    raw = body.get("input")
    if isinstance(raw, str):
        pieces.append(raw)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                pieces.append(item.get("content") if "content" in item else item.get("output", ""))
    text = " ".join(t for t in (flatten_text(p) for p in pieces) if t)
    return text, estimate_tokens(*pieces)


def _output_items(completion: Completion) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if completion.text or not completion.tool_calls:
        items.append({
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "status": "completed",
            "role": "assistant",
            "content": [_text_part(completion.text)],
        })
    for call in completion.tool_calls:
        items.append({
            "type": "function_call",
            "id": f"fc_{uuid.uuid4().hex[:24]}",
            "call_id": call.id,
            "name": call.name,
            "arguments": call.arguments,
            "status": "completed",
        })
    return items


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}


def _response_object(
    body: dict[str, Any],
    response_id: str,
    model: str,
    completion: Completion,
    items: list[dict[str, Any]],
    *,
    status: str,
) -> dict[str, Any]:
    done = status == "completed"
    incomplete = done and completion.finish_reason == "length"
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "incomplete" if incomplete else status,
        "model": model,
        "output": items if done else [],
        "parallel_tool_calls": bool(body.get("parallel_tool_calls", True)),
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools") or [],
        "instructions": body.get("instructions"),
        "metadata": body.get("metadata") or {},
        "temperature": body.get("temperature", 1.0),
        "top_p": body.get("top_p", 1.0),
        "text": body.get("text") or {"format": {"type": "text"}},
        "truncation": body.get("truncation", "disabled"),
        "previous_response_id": body.get("previous_response_id"),
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
        "usage": _usage(completion) if done else None,
    }


def _usage(completion: Completion) -> dict[str, Any]:
    return {
        "input_tokens": completion.prompt_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": completion.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": completion.total_tokens,
    }


def _events(
    body: dict[str, Any],
    response_id: str,
    model: str,
    completion: Completion,
    items: list[dict[str, Any]],
) -> Iterator[bytes]:
    sequence = 0

    def event(kind: str, **fields: Any) -> bytes:
        nonlocal sequence
        data = {"type": kind, "sequence_number": sequence, **fields}
        sequence += 1
        return sse(data, event=kind)

    pending = _response_object(body, response_id, model, completion, items, status="in_progress")
    yield event("response.created", response=pending)
    yield event("response.in_progress", response=pending)

    for index, item in enumerate(items):
        if item["type"] == "message":
            text = item["content"][0]["text"]
            yield event("response.output_item.added", output_index=index,
                        item={**item, "status": "in_progress", "content": []})
            yield event("response.content_part.added", item_id=item["id"], output_index=index,
                        content_index=0, part=_text_part(""))
            for piece in text_pieces(text):
                yield event("response.output_text.delta", item_id=item["id"], output_index=index,
                            content_index=0, delta=piece, logprobs=[])
            yield event("response.output_text.done", item_id=item["id"], output_index=index,
                        content_index=0, text=text, logprobs=[])
            yield event("response.content_part.done", item_id=item["id"], output_index=index,
                        content_index=0, part=_text_part(text))
        else:
            yield event("response.output_item.added", output_index=index,
                        item={**item, "arguments": "", "status": "in_progress"})
            for fragment in fragments(item["arguments"]):
                yield event("response.function_call_arguments.delta", item_id=item["id"],
                            output_index=index, delta=fragment)
            yield event("response.function_call_arguments.done", item_id=item["id"],
                        output_index=index, arguments=item["arguments"])
        yield event("response.output_item.done", output_index=index, item=item)

    final = _response_object(body, response_id, model, completion, items, status="completed")
    yield event("response.completed", response=final)


registry.register(router)
