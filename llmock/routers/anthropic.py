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
from llmock.simulation import MockResponseSettings, build_mock_text, estimate_tokens, flatten_text, raise_if_streaming

router = APIRouter(prefix="/anthropic", tags=["anthropic"])


class MessageParam(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[Any]


class MessagesRequest(BaseModel):
    model: str
    messages: list[MessageParam]
    max_tokens: int
    system: str | None = None
    temperature: float | None = 1.0
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class InputTokensUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class MessagesResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[TextBlock]
    model: str
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] = "end_turn"
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


@router.post("/v1/messages", response_model=MessagesResponse)
def create_message(request: Request, body: MessagesRequest) -> MessagesResponse:
    raise_if_streaming(body.stream)
    prompt_text = " ".join(flatten_text(message.content) for message in body.messages)
    input_tokens = estimate_tokens(*(message.content for message in body.messages), body.system or "")
    reply_text = build_mock_text(
        settings=_response_settings(request),
        model=body.model,
        prompt=prompt_text,
    )
    output_tokens = estimate_tokens(reply_text)

    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] = "end_turn"
    if output_tokens >= body.max_tokens:
        stop_reason = "max_tokens"

    return MessagesResponse(
        model=body.model,
        content=[TextBlock(text=reply_text)],
        stop_reason=stop_reason,
        usage=InputTokensUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
