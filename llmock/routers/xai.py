"""xAI (Grok) compatible endpoints for LLMock (OpenAI-compatible schema)."""

# Official docs:
# - https://docs.x.ai/developers/model-capabilities/legacy/chat-completions
# - https://docs.x.ai/developers/rest-api-reference/inference/models
# - https://docs.x.ai/developers/debugging
# - https://docs.x.ai/developers/rest-api-reference/inference/batches

# Official docs:
# - https://docs.x.ai/developers/model-capabilities/legacy/chat-completions
# - https://docs.x.ai/developers/rest-api-reference/inference/models
# - https://docs.x.ai/developers/debugging

# Docs:
# - https://docs.x.ai/developers/model-capabilities/legacy/chat-completions
# - https://docs.x.ai/developers/rest-api-reference/inference

import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from llmock.routers import batch as batch_support
from llmock.simulation import MockResponseSettings, build_mock_text, estimate_tokens, flatten_text

router = APIRouter(prefix="/xai/v1", tags=["xai"])


class ChatMessage(BaseModel):
    role: str
    content: str | list[Any]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 1.0
    max_tokens: int | None = None
    stream: bool = False
    n: int = 1
    top_p: float | None = 1.0


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


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = 1677610602
    owned_by: str = "xai"


class ModelList(BaseModel):
    object: str = "list"
    data: list[Model]


_MOCK_MODELS = [
    "grok-3",
    "grok-3-mini",
    "grok-2-1212",
    "grok-2-vision-1212",
]


def _response_settings(request: Request) -> MockResponseSettings:
    return request.app.state.mock_response_settings


@router.get("/models", response_model=ModelList)
def list_models() -> ModelList:
    return ModelList(data=[Model(id=model) for model in _MOCK_MODELS])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(request: Request, body: ChatCompletionRequest) -> ChatCompletionResponse:
    prompt_tokens = estimate_tokens(*(message.content for message in body.messages))
    prompt_text = " ".join(flatten_text(message.content) for message in body.messages)
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


def _xai_batch_payload(record: dict) -> dict:
    payload = batch_support._public_payload(record, exclude={"requests", "results"})
    payload["batch_id"] = payload["id"]
    payload["status"] = {
        "PENDING": "pending",
        "COMPLETED": "completed",
        "FAILED": "failed",
        "CANCELLED": "cancelled",
    }.get(record["state"]["status"], record["state"]["status"].lower())
    return payload


@router.post("/files")
async def upload_batch_file(request: Request) -> dict:
    purpose, filename, raw = await batch_support._read_upload(request)
    file = batch_support._store_file(
        filename=filename,
        purpose=purpose,
        content=raw.decode("utf-8", errors="replace"),
        provider="xai",
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
    requests_payload = body.get("requests", [])
    if requests_payload and not isinstance(requests_payload, list):
        raise batch_support.HTTPException(status_code=400, detail="'requests' must be a list.")
    if not requests_payload:
        requests_payload = []
    if body.get("input_file_id"):
        file = batch_support._get_file_or_404(str(body["input_file_id"]))
        requests_payload.extend(batch_support._read_jsonl_rows(file["content"]))
    batch_id = batch_support._make_batch_id()
    batch_support._batches[batch_id] = {
        "id": batch_id,
        "object": "batch",
        "requests": requests_payload,
        "results": [],
        "state": {
            "status": "PENDING",
            "total_requests": len(requests_payload),
            "completed_requests": 0,
            "failed_requests": 0,
        },
        "create_time": batch_support._now_iso(),
        "update_time": batch_support._now_iso(),
        "provider": "xai",
        "kind": "xai",
        "ready_at": (time.time() + batch_support._BATCH_DELAY) if requests_payload else None,
        "done": False,
    }
    payload = _xai_batch_payload(batch_support._batches[batch_id])
    payload["batch_id"] = batch_id
    return payload


@router.get("/batches")
def list_batches(limit: int = 20, after: str | None = None) -> dict:
    items = batch_support._sorted("xai", "xai")
    if after:
        ids = [item["id"] for item in items]
        if after in ids:
            items = items[ids.index(after) + 1 :]
    page = items[:limit]
    return {"object": "list", "data": [_xai_batch_payload(item) for item in page]}


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str) -> dict:
    return _xai_batch_payload(batch_support._get_batch(batch_id, provider="xai", kind="xai"))


@router.get("/batches/{batch_id}/requests")
def get_batch_requests(batch_id: str) -> dict:
    batch = batch_support._get_batch(batch_id, provider="xai", kind="xai")
    return {"data": batch["requests"]}


@router.post("/batches/{batch_id}/requests")
def add_batch_requests(batch_id: str, body: dict) -> dict:
    batch = batch_support._get_batch(batch_id, provider="xai", kind="xai")
    if batch["state"]["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
        raise batch_support.HTTPException(status_code=400, detail="Batch is already terminal.")
    requests_payload = body.get("requests") or body.get("batch_requests")
    if not isinstance(requests_payload, list) or not requests_payload:
        raise batch_support.HTTPException(status_code=400, detail="'requests' must be a non-empty list.")
    batch["requests"].extend(requests_payload)
    batch["state"]["total_requests"] = len(batch["requests"])
    if batch["ready_at"] is None:
        batch["ready_at"] = time.time() + batch_support._BATCH_DELAY
    batch["update_time"] = batch_support._now_iso()
    batch["done"] = False
    payload = _xai_batch_payload(batch)
    payload["added_requests"] = len(requests_payload)
    payload["batch_id"] = batch_id
    return payload


@router.get("/batches/{batch_id}/results")
def get_batch_results(batch_id: str) -> dict:
    batch = batch_support._get_batch(batch_id, provider="xai", kind="xai")
    return {"results": batch["results"], "data": batch["results"]}


@router.post("/batches/{batch_id}:cancel")
def cancel_batch(batch_id: str) -> dict:
    batch = batch_support._get_batch(batch_id, provider="xai", kind="xai")
    if batch["state"]["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
        raise batch_support.HTTPException(status_code=400, detail="Batch is already terminal.")
    batch["state"]["status"] = "CANCELLED"
    batch["update_time"] = batch_support._now_iso()
    batch["done"] = True
    return _xai_batch_payload(batch)


from llmock.routers import registry as _registry
_registry.register(router)
