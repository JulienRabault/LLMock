"""Anthropic (Claude) compatible endpoints for LLMock."""

# Official docs:
# - https://docs.anthropic.com/en/api/messages-examples
# - https://platform.claude.com/docs/en/about-claude/models/overview
# - https://docs.anthropic.com/en/api/handling-stop-reasons
# - https://docs.anthropic.com/en/api/creating-message-batches
# - https://docs.anthropic.com/en/api/retrieving-message-batches

# Official docs:
# - https://docs.anthropic.com/en/api/messages
# - https://platform.claude.com/docs/en/api/models/list

# Docs:
# - https://docs.anthropic.com/en/api/messages
# - https://docs.anthropic.com/en/api/errors
# - https://docs.anthropic.com/en/docs/about-claude/models

import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from llmock.routers import batch as batch_support
from llmock.completion import plan_of, resolve
from llmock.routers._chat import is_streaming
from llmock.simulation import MockResponseSettings, estimate_tokens, flatten_text
from llmock.streaming import anthropic_events, sse_response

router = APIRouter(prefix="/anthropic", tags=["anthropic"])


class MessageParam(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[Any]


class MessagesRequest(BaseModel):
    model: str
    messages: list[MessageParam]
    max_tokens: int
    # A plain string, or a list of content blocks (e.g. with cache_control).
    system: str | list[Any] | None = None
    temperature: float | None = 1.0
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class InputTokensUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class MessagesResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[TextBlock | ToolUseBlock]
    model: str
    stop_reason: str = "end_turn"
    stop_sequence: str | None = None
    usage: InputTokensUsage


class ModelInfo(BaseModel):
    type: Literal["model"] = "model"
    id: str
    display_name: str
    created_at: str


class ModelList(BaseModel):
    data: list[ModelInfo]
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None


_MOCK_MODELS: list[dict[str, str]] = [
    {"id": "claude-opus-4-6", "display_name": "Claude Opus 4.6", "created_at": "2025-08-01T00:00:00Z"},
    {"id": "claude-sonnet-4-6", "display_name": "Claude Sonnet 4.6", "created_at": "2025-08-01T00:00:00Z"},
    {"id": "claude-haiku-4-5-20251001", "display_name": "Claude Haiku 4.5", "created_at": "2025-10-01T00:00:00Z"},
    {"id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet", "created_at": "2024-10-22T00:00:00Z"},
    {"id": "claude-3-opus-20240229", "display_name": "Claude 3 Opus", "created_at": "2024-02-29T00:00:00Z"},
]


def _response_settings(request: Request) -> MockResponseSettings:
    return request.app.state.mock_response_settings


@router.get("/v1/models", response_model=ModelList)
def list_models() -> ModelList:
    models = [ModelInfo(**model) for model in _MOCK_MODELS]
    return ModelList(
        data=models,
        first_id=models[0].id if models else None,
        last_id=models[-1].id if models else None,
    )


_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                 "content_filter": "refusal"}


@router.post("/v1/messages", response_model=MessagesResponse)
def create_message(request: Request, body: MessagesRequest):
    prompt_text = " ".join(flatten_text(message.content) for message in body.messages)
    input_tokens = estimate_tokens(*(message.content for message in body.messages), body.system or "")
    completion = resolve(
        plan=plan_of(request),
        settings=_response_settings(request),
        model=body.model,
        prompt_text=prompt_text,
        prompt_tokens=input_tokens,
        tool_call_prefix="toolu",
    )
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    if is_streaming(request):
        return sse_response(anthropic_events(completion, message_id=message_id, model=body.model))

    stop_reason = _STOP_REASONS.get(completion.finish_reason, completion.finish_reason)
    if stop_reason == "end_turn" and completion.completion_tokens >= body.max_tokens:
        stop_reason = "max_tokens"

    content: list[TextBlock | ToolUseBlock] = []
    if completion.text or not completion.tool_calls:
        content.append(TextBlock(text=completion.text))
    content.extend(
        ToolUseBlock(id=call.id, name=call.name, input=call.parsed_arguments)
        for call in completion.tool_calls
    )
    return MessagesResponse(
        id=message_id,
        model=body.model,
        content=content,
        stop_reason=stop_reason,
        usage=InputTokensUsage(
            input_tokens=completion.prompt_tokens,
            output_tokens=completion.completion_tokens,
        ),
    )


@router.post("/v1/messages/batches")
def create_message_batch(body: dict) -> dict:
    requests_payload = body.get("requests")
    if not isinstance(requests_payload, list) or not requests_payload:
        raise batch_support.HTTPException(status_code=400, detail="'requests' must be a non-empty list.")
    batch_id = batch_support._make_batch_id("msgbatch")
    batch_support._batches[batch_id] = {
        "id": batch_id,
        "type": "message_batch",
        "processing_status": "in_progress",
        "request_counts": {
            "processing": len(requests_payload),
            "succeeded": 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        },
        "created_at": batch_support._now_iso(),
        "expires_at": batch_support._iso_from_ts(time.time() + 86400),
        "ended_at": None,
        "cancel_initiated_at": None,
        "archived_at": None,
        "results_url": f"/anthropic/v1/messages/batches/{batch_id}/results",
        "requests": requests_payload,
        "results_content": "",
        "provider": "anthropic",
        "kind": "anthropic",
        "ready_at": time.time() + batch_support._BATCH_DELAY,
        "done": False,
    }
    return batch_support._public_payload(batch_support._batches[batch_id], exclude={"requests", "results_content"})


@router.get("/v1/messages/batches")
def list_message_batches(limit: int = 20, after: str | None = None) -> dict:
    items = batch_support._sorted("anthropic", "anthropic")
    if after:
        ids = [item["id"] for item in items]
        if after in ids:
            items = items[ids.index(after) + 1 :]
    page = items[:limit]
    data = [batch_support._public_payload(item, exclude={"requests", "results_content"}) for item in page]
    return {
        "data": data,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
        "has_more": len(items) > limit,
    }


@router.get("/v1/messages/batches/{batch_id}")
def get_message_batch(batch_id: str) -> dict:
    batch = batch_support._get_batch(batch_id, provider="anthropic", kind="anthropic")
    return batch_support._public_payload(batch, exclude={"requests", "results_content"})


@router.post("/v1/messages/batches/{batch_id}/cancel")
def cancel_message_batch(batch_id: str) -> dict:
    batch = batch_support._get_batch(batch_id, provider="anthropic", kind="anthropic")
    if batch["processing_status"] == "ended":
        raise batch_support.HTTPException(status_code=400, detail="Message batch is already ended.")
    batch["cancel_initiated_at"] = batch_support._now_iso()
    batch["processing_status"] = "ended"
    batch["ended_at"] = batch_support._now_iso()
    batch["request_counts"]["processing"] = 0
    batch["request_counts"]["canceled"] = len(batch["requests"])
    batch["results_content"] = ""
    batch["done"] = True
    return batch_support._public_payload(batch, exclude={"requests", "results_content"})


@router.get("/v1/messages/batches/{batch_id}/results")
def get_message_batch_results(batch_id: str):
    batch = batch_support._get_batch(batch_id, provider="anthropic", kind="anthropic")
    return batch_support.PlainTextResponse(content=batch["results_content"], media_type="application/jsonl")


from llmock.routers import registry as _registry
_registry.register(router)
