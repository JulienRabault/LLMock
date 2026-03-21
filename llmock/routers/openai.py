"""OpenAI-compatible endpoints for LLMock."""

# Official docs:
# - https://platform.openai.com/docs/api-reference/chat/create
# - https://platform.openai.com/docs/api-reference/embeddings/create
# - https://platform.openai.com/docs/api-reference/images/create
# - https://platform.openai.com/docs/api-reference/models/list
# - https://platform.openai.com/docs/guides/error-codes/api-error
# - https://platform.openai.com/docs/api-reference/batch
# - https://platform.openai.com/docs/api-reference/files/create

# Official docs:
# - https://platform.openai.com/docs/api-reference/chat/create-chat-completion
# - https://platform.openai.com/docs/api-reference/embeddings/create
# - https://platform.openai.com/docs/guides/images/image-generation
# - https://platform.openai.com/docs/api-reference/models/list

# Docs:
# - https://platform.openai.com/docs/api-reference/chat/create-chat-completion
# - https://platform.openai.com/docs/api-reference/embeddings/create
# - https://platform.openai.com/docs/api-reference/models/list
# - https://platform.openai.com/docs/api-reference/images/create

import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from llmock.routers import batch as batch_support
from llmock.simulation import (
    DEFAULT_IMAGE_SIZE,
    MockResponseSettings,
    build_fake_image_payload,
    build_mock_embedding,
    build_mock_text,
    estimate_tokens,
    flatten_text,
    raise_if_streaming,
)

router = APIRouter(prefix="/v1", tags=["openai"])


class ChatMessage(BaseModel):
    role: str
    content: str | list[Any]
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 1.0
    max_tokens: int | None = None
    stream: bool = False
    n: int = 1


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str = "float"


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = "gpt-image-1"
    n: int = 1
    size: str = DEFAULT_IMAGE_SIZE
    response_format: Literal["url", "b64_json"] = "url"


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage


class ImageData(BaseModel):
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None


class ImageGenerationResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ImageData]


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = 1677610602
    owned_by: str = "llmock"


class ModelList(BaseModel):
    object: str = "list"
    data: list[Model]


_MOCK_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gpt-image-1",
    "text-embedding-3-small",
    "text-embedding-3-large",
]


def _response_settings(request: Request) -> MockResponseSettings:
    return request.app.state.mock_response_settings


@router.get("/models", response_model=ModelList)
def list_models() -> ModelList:
    return ModelList(data=[Model(id=model) for model in _MOCK_MODELS])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(request: Request, body: ChatCompletionRequest) -> ChatCompletionResponse:
    raise_if_streaming(body.stream)
    prompt_text = " ".join(flatten_text(message.content) for message in body.messages)
    prompt_tokens = estimate_tokens(*(message.content for message in body.messages))
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


@router.post("/embeddings", response_model=EmbeddingResponse)
def embeddings(body: EmbeddingRequest) -> EmbeddingResponse:
    inputs = [body.input] if isinstance(body.input, str) else body.input
    data = [
        EmbeddingData(index=index, embedding=build_mock_embedding(text, 1536))
        for index, text in enumerate(inputs)
    ]
    prompt_tokens = estimate_tokens(*inputs)
    return EmbeddingResponse(
        data=data,
        model=body.model,
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=prompt_tokens),
    )


@router.post("/images/generations", response_model=ImageGenerationResponse)
def generate_images(body: ImageGenerationRequest) -> ImageGenerationResponse:
    payload = build_fake_image_payload(
        prompt=body.prompt,
        count=body.n,
        size=body.size,
        response_format=body.response_format,
    )
    return ImageGenerationResponse(data=[ImageData(**item) for item in payload])


@router.post("/files")
async def upload_batch_file(request: Request) -> dict[str, Any]:
    purpose, filename, raw = await batch_support._read_upload(request)
    file = batch_support._store_file(
        filename=filename,
        purpose=purpose,
        content=raw.decode("utf-8", errors="replace"),
        provider="openai",
    )
    return batch_support._file_payload(file)


@router.get("/files/{file_id}")
def get_batch_file(file_id: str) -> dict[str, Any]:
    return batch_support._file_payload(batch_support._get_file_or_404(file_id))


@router.get("/files/{file_id}/content")
def get_batch_file_content(file_id: str):
    file = batch_support._get_file_or_404(file_id)
    return batch_support.PlainTextResponse(content=file["content"], media_type="application/jsonl")


@router.delete("/files/{file_id}")
def delete_batch_file(file_id: str) -> dict[str, Any]:
    batch_support._get_file_or_404(file_id)
    del batch_support._files[file_id]
    return {"id": file_id, "object": "file.deleted", "deleted": True}


@router.post("/batches")
def create_batch(body: dict[str, Any]) -> dict[str, Any]:
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
        "provider": "openai",
        "kind": "openai_like",
        "ready_at": time.time() + batch_support._BATCH_DELAY,
        "done": False,
    }
    return batch_support._public_payload(batch_support._batches[batch_id])


@router.get("/batches")
def list_batches(limit: int = 20, after: str | None = None) -> dict[str, Any]:
    items = batch_support._sorted("openai", "openai_like")
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
def get_batch(batch_id: str) -> dict[str, Any]:
    return batch_support._public_payload(batch_support._get_batch(batch_id, provider="openai", kind="openai_like"))


@router.post("/batches/{batch_id}/cancel")
def cancel_batch(batch_id: str) -> dict[str, Any]:
    batch = batch_support._get_batch(batch_id, provider="openai", kind="openai_like")
    if batch["status"] in {"completed", "failed", "expired", "cancelled"}:
        raise batch_support.HTTPException(
            status_code=400,
            detail=f"Batch is already in terminal status '{batch['status']}'.",
        )
    batch["status"] = "cancelled"
    batch["cancelling_at"] = batch_support._now()
    batch["cancelled_at"] = batch_support._now()
    batch["done"] = True
    return batch_support._public_payload(batch)


from llmock.routers import registry as _registry
_registry.register(router)
