"""Mistral-compatible endpoints for LLMock."""

# Official docs:
# - https://docs.mistral.ai/api/
# - https://docs.mistral.ai/capabilities/completion/usage
# - https://docs.mistral.ai/api/endpoint/batch
# - https://docs.mistral.ai/capabilities/batch

# Official docs:
# - https://docs.mistral.ai/api
# - https://docs.mistral.ai/capabilities/completion/usage

# Docs:
# - https://docs.mistral.ai/api
# - https://docs.mistral.ai/getting-started/models/models_overview/

import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from llmock.routers import batch as batch_support
from llmock.simulation import MockResponseSettings, build_mock_text, estimate_tokens, flatten_text, raise_if_streaming

router = APIRouter(prefix="/mistral/v1", tags=["mistral"])


class InputChatMessage(BaseModel):
    role: str
    content: str | list[Any]
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[InputChatMessage]
    temperature: float | None = 0.7
    max_tokens: int | None = None
    stream: bool = False
    n: int = 1
    top_p: float | None = 1.0
    safe_prompt: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"
    logprobs: None = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{uuid.uuid4().hex[:16]}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: None = None
    is_blocking: bool = False


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mistralai"
    root: str | None = None
    parent: None = None
    permission: list[ModelPermission] = Field(default_factory=lambda: [ModelPermission()])


class ModelList(BaseModel):
    object: str = "list"
    data: list[Model]


_MOCK_MODELS = [
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
    "open-mistral-7b",
    "open-mixtral-8x7b",
    "open-mixtral-8x22b",
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
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _mistral_batch_payload(record: dict) -> dict:
    payload = batch_support._public_payload(record)
    if payload.get("output_file"):
        payload["output_file"] = batch_support._file_payload(batch_support._get_file_or_404(payload["output_file"]))
    if payload.get("error_file"):
        payload["error_file"] = batch_support._file_payload(batch_support._get_file_or_404(payload["error_file"]))
    return payload


@router.post("/files")
async def upload_batch_file(request: Request) -> dict:
    purpose, filename, raw = await batch_support._read_upload(request)
    file = batch_support._store_file(
        filename=filename,
        purpose=purpose,
        content=raw.decode("utf-8", errors="replace"),
        provider="mistral",
    )
    return batch_support._file_payload(file)


@router.get("/files/{file_id}")
def get_batch_file(file_id: str) -> dict:
    return batch_support._file_payload(batch_support._get_file_or_404(file_id))


@router.get("/files/{file_id}/content")
def get_batch_file_content(file_id: str):
    file = batch_support._get_file_or_404(file_id)
    return batch_support.PlainTextResponse(content=file["content"], media_type="application/jsonl")


@router.post("/batch/jobs")
def create_batch_job(body: dict) -> dict:
    input_files = [str(file_id) for file_id in body.get("input_files", [])]
    for file_id in input_files:
        batch_support._get_file_or_404(file_id)
    requests_payload = body.get("requests", [])
    if requests_payload and not isinstance(requests_payload, list):
        raise batch_support.HTTPException(status_code=400, detail="'requests' must be a list.")
    batch_id = batch_support._make_batch_id("job")
    batch_support._batches[batch_id] = {
        "id": batch_id,
        "object": "batch.job",
        "model": str(body.get("model", "mistral-large-latest")),
        "endpoint": str(body.get("endpoint", "/v1/chat/completions")),
        "input_files": input_files,
        "output_file": None,
        "error_file": None,
        "outputs": [],
        "errors": [],
        "status": "QUEUED",
        "created_at": batch_support._now_iso(),
        "started_at": None,
        "completed_at": None,
        "total_requests": 0,
        "succeeded_requests": 0,
        "failed_requests": 0,
        "completed_requests": 0,
        "metadata": body.get("metadata"),
        "requests": requests_payload,
        "provider": "mistral",
        "kind": "mistral",
        "ready_at": time.time() + batch_support._BATCH_DELAY,
        "done": False,
    }
    return _mistral_batch_payload(batch_support._batches[batch_id])


@router.get("/batch/jobs")
def list_batch_jobs(limit: int = 20, after: str | None = None) -> dict:
    items = batch_support._sorted("mistral", "mistral")
    if after:
        ids = [item["id"] for item in items]
        if after in ids:
            items = items[ids.index(after) + 1 :]
    page = items[:limit]
    return {"object": "list", "data": [_mistral_batch_payload(item) for item in page]}


@router.get("/batch/jobs/{job_id}")
def get_batch_job(job_id: str) -> dict:
    return _mistral_batch_payload(batch_support._get_batch(job_id, provider="mistral", kind="mistral"))


@router.post("/batch/jobs/{job_id}/cancel")
def cancel_batch_job(job_id: str) -> dict:
    batch = batch_support._get_batch(job_id, provider="mistral", kind="mistral")
    if batch["status"] in {"SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"}:
        raise batch_support.HTTPException(status_code=400, detail="Batch job is already terminal.")
    batch["status"] = "CANCELLED"
    batch["completed_at"] = batch_support._now_iso()
    batch["done"] = True
    return _mistral_batch_payload(batch)


from llmock.routers import registry as _registry
_registry.register(router)
