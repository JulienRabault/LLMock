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
from llmock.completion import resolve_request
from llmock.routers._chat import is_streaming
from llmock.simulation import MockResponseSettings, estimate_tokens, flatten_text
from llmock.streaming import COHERE_FINISH, cohere_events, cohere_usage, sse_response

router = APIRouter(prefix="/cohere/v2", tags=["cohere"])
legacy_router = APIRouter(prefix="/cohere/v1", tags=["cohere"])


class Message(BaseModel):
    role: str
    # Assistant turns that only call tools have no content; tool results
    # come back as role="tool" with a tool_call_id.
    content: str | list[Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    tool_plan: str | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = 0.3
    p: float | None = None
    k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None


class TextContent(BaseModel):
    type: str = "text"
    text: str


class AssistantMessage(BaseModel):
    role: str = "assistant"
    content: list[TextContent] | None = None
    tool_calls: list[dict[str, Any]] | None = None


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
def chat(request: Request, body: ChatRequest):
    contents = [m.content for m in body.messages if m.content is not None]
    completion = resolve_request(
        request,
        model=body.model,
        prompt_text=" ".join(flatten_text(c) for c in contents),
        prompt_tokens=estimate_tokens(*contents),
    )
    message_id = uuid.uuid4().hex
    if is_streaming(request):
        return sse_response(cohere_events(completion, message_id=message_id))

    tool_calls = [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
        for c in completion.tool_calls
    ] or None
    content = (
        [TextContent(text=completion.text)] if completion.text or not tool_calls else None
    )
    return ChatResponse(
        id=message_id,
        finish_reason=COHERE_FINISH.get(completion.finish_reason, "COMPLETE"),
        message=AssistantMessage(content=content, tool_calls=tool_calls),
        usage=Usage(**cohere_usage(completion)),
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
