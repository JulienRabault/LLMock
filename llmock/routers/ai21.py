"""AI21 Labs compatible endpoints for LLMock (OpenAI-compatible schema)."""

# Official docs:
# - https://docs.ai21.com/docs/jamba-foundation-models
# - https://docs.ai21.com/reference/jamba-api-response
# - https://docs.ai21.com/docs/jamba-batch-api

# Official docs:
# - https://docs.ai21.com/reference/jamba-1-6-api-ref
# - https://docs.ai21.com/reference/jamba-api-response
# - https://docs.ai21.com/docs/jamba-foundation-models

# Docs:
# - https://docs.ai21.com/reference/jamba-1-6-api-ref
# - https://docs.ai21.com/docs/troubleshooting-performance

import time
import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from llmock.routers import batch as batch_support
from llmock.simulation import MockResponseSettings, build_mock_text, estimate_tokens, raise_if_streaming

router = APIRouter(prefix="/ai21/v1", tags=["ai21"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.4
    max_tokens: int | None = 2048
    stream: bool = False
    n: int = 1
    top_p: float | None = 1.0
    stop: list[str] | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = 1677610602
    owned_by: str = "ai21"


class ModelList(BaseModel):
    object: str = "list"
    data: list[Model]


_MOCK_MODELS = [
    "jamba-1.5-large",
    "jamba-1.5-mini",
    "jamba-instruct",
]


def _response_settings(request: Request) -> MockResponseSettings:
    return request.app.state.mock_response_settings


@router.get("/models", response_model=ModelList)
def list_models() -> ModelList:
    return ModelList(data=[Model(id=model) for model in _MOCK_MODELS])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(request: Request, body: ChatCompletionRequest) -> ChatCompletionResponse:
    raise_if_streaming(body.stream)
    prompt_tokens = estimate_tokens(*(message.content for message in body.messages))
    prompt_text = " ".join(message.content for message in body.messages)
    reply_text = build_mock_text(
        settings=_response_settings(request),
        model=body.model,
        prompt=prompt_text,
    )
    completion_tokens = estimate_tokens(reply_text)

    choices = [
        ChatCompletionChoice(
            index=index,
            message=ChatMessage(role="assistant", content=reply_text),
        )
        for index in range(body.n)
    ]
    return ChatCompletionResponse(
        model=body.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@router.post("/files")
async def upload_batch_file(request: Request) -> dict:
    purpose, filename, raw = await batch_support._read_upload(request)
    file = batch_support._store_file(
        filename=filename,
        purpose=purpose,
        content=raw.decode("utf-8", errors="replace"),
        provider="ai21",
    )
    return batch_support._file_payload(file)


@router.get("/files/{file_id}")
def get_batch_file(file_id: str) -> dict:
    return batch_support._file_payload(batch_support._get_file_or_404(file_id))


@router.get("/files/{file_id}/content")
def get_batch_file_content(file_id: str):
    file = batch_support._get_file_or_404(file_id)
    return batch_support.PlainTextResponse(content=file["content"], media_type="application/jsonl")


@router.post("/batches")
def create_batch(body: dict) -> dict:
    input_file_id = str(body.get("input_file_id", ""))
    if input_file_id not in batch_support._files:
        raise batch_support.HTTPException(status_code=400, detail=f"Input file '{input_file_id}' not found.")
    now = batch_support._now()
    batch_id = batch_support._make_batch_id()
    batch_support._batches[batch_id] = {
        "id": batch_id,
        "object": "batch",
        "endpoint": str(body.get("endpoint", "/v1/chat/completions")),
        "errors": None,
        "input_file_id": input_file_id,
        "completion_window": str(body.get("completion_window", "24h")),
        "status": "validating",
        "output_file_id": None,
        "error_file_id": None,
        "created_at": now,
        "in_progress_at": None,
        "expires_at": now + 86400,
        "finalizing_at": None,
        "completed_at": None,
        "failed_at": None,
        "expired_at": None,
        "cancelling_at": None,
        "cancelled_at": None,
        "request_counts": {"total": 0, "completed": 0, "failed": 0},
        "metadata": body.get("metadata"),
        "provider": "ai21",
        "kind": "openai_like",
        "ready_at": time.time() + batch_support._BATCH_DELAY,
        "done": False,
    }
    return batch_support._public_payload(batch_support._batches[batch_id])


@router.get("/batches")
def list_batches(limit: int = 20, after: str | None = None) -> dict:
    items = batch_support._sorted("ai21", "openai_like")
    if after:
        ids = [item["id"] for item in items]
        if after in ids:
            items = items[ids.index(after) + 1 :]
    page = items[:limit]
    data = [batch_support._public_payload(item) for item in page]
    return {
        "object": "list",
        "data": data,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
        "has_more": len(items) > limit,
    }


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str) -> dict:
    return batch_support._public_payload(batch_support._get_batch(batch_id, provider="ai21", kind="openai_like"))


@router.post("/batches/{batch_id}/cancel")
def cancel_batch(batch_id: str) -> dict:
    batch = batch_support._get_batch(batch_id, provider="ai21", kind="openai_like")
    if batch["status"] in {"completed", "failed", "expired", "cancelled"}:
        raise batch_support.HTTPException(status_code=400, detail="Batch is already terminal.")
    batch["status"] = "cancelled"
    batch["cancelling_at"] = batch_support._now()
    batch["cancelled_at"] = batch_support._now()
    batch["done"] = True
    return batch_support._public_payload(batch)


from llmock.routers import registry as _registry
_registry.register(router)
