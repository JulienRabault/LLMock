"""Cohere compatible endpoints for LLMock."""

# Official docs:
# - https://docs.cohere.com/v2/reference/chat
# - https://docs.cohere.com/v2
# - https://docs.cohere.com/reference/create-batch
# - https://docs.cohere.com/reference/get-batch

# Official docs:
# - https://docs.cohere.com/reference/chat
# - https://docs.cohere.com/reference/list-models

# Docs:
# - https://docs.cohere.com/v2/reference/chat
# - https://docs.cohere.com/v2/docs/models

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from llmock.routers import batch as batch_support
from llmock.simulation import MockResponseSettings, build_mock_text, estimate_tokens, flatten_text

router = APIRouter(prefix="/cohere/v2", tags=["cohere"])
legacy_router = APIRouter(prefix="/cohere/v1", tags=["cohere"])


class Message(BaseModel):
    role: str
    content: str | list[Any]


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = 0.3
    p: float | None = None
    k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None


class TextContent(BaseModel):
    type: str = "text"
    text: str


class AssistantMessage(BaseModel):
    role: str = "assistant"
    content: list[TextContent]


class UsageTokens(BaseModel):
    input_tokens: int
    output_tokens: int


class Usage(BaseModel):
    billed_units: UsageTokens
    tokens: UsageTokens


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    finish_reason: str = "COMPLETE"
    message: AssistantMessage
    usage: Usage


class ModelInfo(BaseModel):
    name: str
    endpoints: list[str] = Field(default_factory=lambda: ["chat"])
    context_length: int = 128000


class ModelList(BaseModel):
    models: list[ModelInfo]


_MOCK_MODELS = [
    "command-r-plus-08-2024",
    "command-r-08-2024",
    "command-r-plus",
    "command-r",
    "command",
]


def _response_settings(request: Request) -> MockResponseSettings:
    return request.app.state.mock_response_settings


@router.get("/models", response_model=ModelList)
def list_models() -> ModelList:
    return ModelList(models=[ModelInfo(name=model) for model in _MOCK_MODELS])


@router.post("/chat", response_model=ChatResponse)
def chat(request: Request, body: ChatRequest) -> ChatResponse:
    prompt_text = " ".join(flatten_text(message.content) for message in body.messages)
    input_tokens = estimate_tokens(*(message.content for message in body.messages))
    reply_text = build_mock_text(
        settings=_response_settings(request),
        model=body.model,
        prompt=prompt_text,
    )
    output_tokens = estimate_tokens(reply_text)

    usage_units = UsageTokens(input_tokens=input_tokens, output_tokens=output_tokens)
    return ChatResponse(
        message=AssistantMessage(content=[TextContent(text=reply_text)]),
        usage=Usage(billed_units=usage_units, tokens=usage_units),
    )


@router.post("/datasets")
@legacy_router.post("/datasets")
def create_dataset(body: dict) -> dict:
    records = body.get("records") or body.get("rows")
    if not isinstance(records, list) or not records:
        raise batch_support.HTTPException(status_code=400, detail="'records' must be a non-empty list.")
    dataset_id = batch_support._make_dataset_id()
    batch_support._datasets[dataset_id] = {
        "id": dataset_id,
        "name": str(body.get("name", dataset_id)),
        "provider": "cohere",
        "created_at": batch_support._now_iso(),
        "rows": records,
    }
    return {
        "id": dataset_id,
        "name": batch_support._datasets[dataset_id]["name"],
        "record_count": len(records),
        "rows": records,
    }


@router.get("/datasets/{dataset_id}")
@legacy_router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    dataset = batch_support._datasets.get(dataset_id)
    if dataset is None or dataset.get("provider") != "cohere":
        raise batch_support.HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return {"id": dataset["id"], "name": dataset["name"], "record_count": len(dataset["rows"]), "rows": dataset["rows"]}


@router.get("/datasets/{dataset_id}/download")
@legacy_router.get("/datasets/{dataset_id}/download")
def download_dataset(dataset_id: str):
    dataset = batch_support._datasets.get(dataset_id)
    if dataset is None or dataset.get("provider") != "cohere":
        raise batch_support.HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    lines = "\n".join(json.dumps(row) for row in dataset["rows"])
    return batch_support.PlainTextResponse(content=lines, media_type="application/jsonl")


@router.post("/batches")
def create_batch(body: dict) -> dict:
    dataset_id = str(body.get("input_dataset_id", ""))
    dataset = batch_support._datasets.get(dataset_id)
    if dataset is None or dataset.get("provider") != "cohere":
        raise batch_support.HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    batch_id = batch_support._make_batch_id("cohere")
    batch_support._batches[batch_id] = {
        "id": batch_id,
        "name": str(body.get("name", "llmock-cohere-batch")),
        "status": "BATCH_STATUS_IN_PROGRESS",
        "model": str(body.get("model", "command-r-plus")),
        "input_dataset_id": dataset_id,
        "output_dataset_id": None,
        "created_at": batch_support._now_iso(),
        "updated_at": batch_support._now_iso(),
        "num_records": len(dataset["rows"]),
        "num_successful_records": 0,
        "num_failed_records": 0,
        "provider": "cohere",
        "kind": "cohere",
        "ready_at": time.time() + batch_support._BATCH_DELAY,
        "done": False,
    }
    return batch_support._public_payload(batch_support._batches[batch_id])


@router.get("/batches")
def list_batches(limit: int = 20, after: str | None = None) -> dict:
    items = batch_support._sorted("cohere", "cohere")
    if after:
        ids = [item["id"] for item in items]
        if after in ids:
            items = items[ids.index(after) + 1 :]
    page = items[:limit]
    return {"batches": [batch_support._public_payload(item) for item in page]}


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str) -> dict:
    return batch_support._public_payload(batch_support._get_batch(batch_id, provider="cohere", kind="cohere"))


@router.post("/batches/{batch_id}:cancel")
@router.post("/batches/{batch_id}/cancel")
def cancel_batch(batch_id: str) -> dict:
    batch = batch_support._get_batch(batch_id, provider="cohere", kind="cohere")
    if batch["status"] in {"BATCH_STATUS_COMPLETED", "BATCH_STATUS_FAILED", "BATCH_STATUS_CANCELLED"}:
        raise batch_support.HTTPException(status_code=400, detail="Batch is already terminal.")
    batch["status"] = "BATCH_STATUS_CANCELLED"
    batch["updated_at"] = batch_support._now_iso()
    batch["done"] = True
    return batch_support._public_payload(batch)


from llmock.routers import registry as _registry
_registry.register(router, legacy_router)
