"""Perplexity AI compatible endpoints for LLMock (OpenAI-compatible schema)."""

# Official docs:
# - https://docs.perplexity.ai/docs/sonar/openai-compatibility
# - https://docs.perplexity.ai/docs/sonar/quickstart
# - https://docs.perplexity.ai/docs/sdk/error-handling
# - https://docs.perplexity.ai/docs/resources/changelog
# - https://docs.perplexity.ai/api-reference/async-sonar-post

# Official docs:
# - https://docs.perplexity.ai/api-reference/sonar-post
# - https://docs.perplexity.ai/docs/agent-api/models

# Docs:
# - https://docs.perplexity.ai/api-reference/chat-completions-post
# - https://docs.perplexity.ai/guides/model-cards

import time
import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from llmock.routers import batch as batch_support
from llmock.simulation import (
    MockResponseSettings,
    build_fake_image_data_uri,
    build_mock_text,
    estimate_tokens,
    raise_if_streaming,
)

router = APIRouter(prefix="/perplexity/v1", tags=["perplexity"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.2
    max_tokens: int | None = None
    stream: bool = False
    top_p: float | None = 0.9
    return_citations: bool = False
    search_domain_filter: list[str] | None = None
    return_images: bool = False
    return_related_questions: bool = False
    search_recency_filter: str | None = None


class SearchResult(BaseModel):
    title: str
    url: str
    date: str | None = None
    last_updated: str | None = None
    snippet: str = ""
    source: str = "web"


class ImageResult(BaseModel):
    image_url: str
    origin_url: str
    title: str
    width: int
    height: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"
    delta: dict = Field(default_factory=dict)


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
    citations: list[str] = Field(default_factory=list)
    search_results: list[SearchResult] = Field(default_factory=list)
    images: list[ImageResult] = Field(default_factory=list)
    related_questions: list[str] = Field(default_factory=list)


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = 1677610602
    owned_by: str = "perplexity"


class ModelList(BaseModel):
    object: str = "list"
    data: list[Model]


_MOCK_MODELS = [
    "sonar-pro",
    "sonar",
    "sonar-reasoning-pro",
    "sonar-reasoning",
]


def _response_settings(request: Request) -> MockResponseSettings:
    return request.app.state.mock_response_settings


def _build_search_results(prompt: str) -> list[SearchResult]:
    normalized = prompt.replace(" ", "-").lower()[:32] or "llmock"
    return [
        SearchResult(
            title=f"Mock result for {prompt[:40] or 'your query'}",
            url=f"https://example.com/mock-search/{normalized}",
            date="2026-03-20",
            last_updated="2026-03-20",
            snippet=f"Synthetic search snippet for {prompt[:48] or 'your query'}.",
        )
    ]


def _build_image_results(prompt: str) -> list[ImageResult]:
    normalized = prompt.replace(" ", "-").lower()[:32] or "llmock"
    return [
        ImageResult(
            image_url=build_fake_image_data_uri(prompt=prompt or "LLMock image", size="640x360"),
            origin_url=f"https://example.com/mock-image/{normalized}",
            title=f"Mock image for {prompt[:40] or 'your query'}",
            width=640,
            height=360,
        )
    ]


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

    search_results = _build_search_results(prompt_text)
    citations = [result.url for result in search_results] if body.return_citations else []
    images = _build_image_results(prompt_text or body.model) if body.return_images else []
    related_questions = (
        [
            f"What are the main takeaways about {prompt_text[:48] or body.model}?",
            f"How would you validate {prompt_text[:48] or body.model} in production?",
        ]
        if body.return_related_questions
        else []
    )

    choices = [
        ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=reply_text),
        )
    ]
    return ChatCompletionResponse(
        model=body.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        citations=citations,
        search_results=search_results,
        images=images,
        related_questions=related_questions,
    )


@router.post("/files")
async def upload_batch_file(request: Request) -> dict:
    purpose, filename, raw = await batch_support._read_upload(request)
    file = batch_support._store_file(
        filename=filename,
        purpose=purpose,
        content=raw.decode("utf-8", errors="replace"),
        provider="perplexity",
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
        "provider": "perplexity",
        "kind": "openai_like",
        "ready_at": time.time() + batch_support._BATCH_DELAY,
        "done": False,
    }
    return batch_support._public_payload(batch_support._batches[batch_id])


@router.get("/batches")
def list_batches(limit: int = 20, after: str | None = None) -> dict:
    items = batch_support._sorted("perplexity", "openai_like")
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
    return batch_support._public_payload(batch_support._get_batch(batch_id, provider="perplexity", kind="openai_like"))


@router.post("/batches/{batch_id}/cancel")
def cancel_batch(batch_id: str) -> dict:
    batch = batch_support._get_batch(batch_id, provider="perplexity", kind="openai_like")
    if batch["status"] in {"completed", "failed", "expired", "cancelled"}:
        raise batch_support.HTTPException(status_code=400, detail="Batch is already terminal.")
    batch["status"] = "cancelled"
    batch["cancelling_at"] = batch_support._now()
    batch["cancelled_at"] = batch_support._now()
    batch["done"] = True
    return batch_support._public_payload(batch)


from llmock.routers import registry as _registry
_registry.register(router)
