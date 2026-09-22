"""Google Gemini compatible endpoints for LLMock."""

# Official docs:
# - https://ai.google.dev/api

# Official docs:
# - https://ai.google.dev/api/generate-content
# - https://ai.google.dev/models/gemini
# - https://ai.google.dev/gemini-api/docs/batch-api

# Docs:
# - https://ai.google.dev/gemini-api/docs/text-generation
# - https://ai.google.dev/api/models

import time
import uuid
from typing import Any

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel, Field

from llmock.routers import batch as batch_support
from llmock.completion import Completion, resolve_request
from llmock.simulation import MockResponseSettings, estimate_tokens, flatten_text
from llmock.streaming import gemini_chunk_dicts, gemini_chunks, sse_response

router = APIRouter(prefix="/gemini/v1beta", tags=["gemini"])


class Part(BaseModel):
    # A part carries exactly one kind of payload. Agent loops send back the
    # model's functionCall and their own functionResponse, so text is optional.
    text: str | None = None
    functionCall: dict[str, Any] | None = None
    functionResponse: dict[str, Any] | None = None
    inlineData: dict[str, Any] | None = None
    fileData: dict[str, Any] | None = None


class Content(BaseModel):
    role: str | None = None
    parts: list[Part]


class GenerateContentRequest(BaseModel):
    contents: list[Content]
    systemInstruction: Content | None = None
    tools: list[dict[str, Any]] | None = None
    generationConfig: dict[str, Any] | None = None


class Candidate(BaseModel):
    content: Content
    finishReason: str = "STOP"
    index: int = 0


class UsageMetadata(BaseModel):
    promptTokenCount: int
    candidatesTokenCount: int
    totalTokenCount: int


class GenerateContentResponse(BaseModel):
    candidates: list[Candidate]
    usageMetadata: UsageMetadata


class ModelInfo(BaseModel):
    name: str
    version: str
    displayName: str
    description: str
    inputTokenLimit: int
    outputTokenLimit: int
    supportedGenerationMethods: list[str] = Field(
        default_factory=lambda: ["generateContent", "countTokens"]
    )


class ModelList(BaseModel):
    models: list[ModelInfo]


_MOCK_MODELS = [
    {"id": "gemini-2.5-pro", "display": "Gemini 2.5 Pro", "in": 1048576, "out": 65536},
    {"id": "gemini-2.0-flash", "display": "Gemini 2.0 Flash", "in": 1048576, "out": 8192},
    {"id": "gemini-1.5-pro", "display": "Gemini 1.5 Pro", "in": 2097152, "out": 8192},
    {"id": "gemini-1.5-flash", "display": "Gemini 1.5 Flash", "in": 1048576, "out": 8192},
]


def _response_settings(request: Request) -> MockResponseSettings:
    return request.app.state.mock_response_settings


@router.get("/models", response_model=ModelList)
def list_models() -> ModelList:
    models = [
        ModelInfo(
            name=f"models/{model['id']}",
            version=model["id"].split("-")[-1],
            displayName=model["display"],
            description=f"Mock {model['display']} model",
            inputTokenLimit=model["in"],
            outputTokenLimit=model["out"],
        )
        for model in _MOCK_MODELS
    ]
    return ModelList(models=models)


_FINISH = {"stop": "STOP", "length": "MAX_TOKENS", "tool_calls": "STOP", "content_filter": "SAFETY"}


def _completion(request: Request, model: str, body: GenerateContentRequest) -> Completion:
    prompt_values: list[Any] = []
    for content in body.contents:
        prompt_values.extend(p.model_dump(exclude_none=True) for p in content.parts)
    if body.systemInstruction:
        prompt_values.extend(p.model_dump(exclude_none=True) for p in body.systemInstruction.parts)
    return resolve_request(
        request,
        model=model,
        prompt_text=" ".join(flatten_text(part) for part in prompt_values),
        prompt_tokens=estimate_tokens(*prompt_values),
    )


# Omit unset fields: a real part carries one payload, never a list of nulls.
@router.post(
    "/models/{model}:generateContent",
    response_model=GenerateContentResponse,
    response_model_exclude_none=True,
)
def generate_content(request: Request, model: str, body: GenerateContentRequest):
    completion = _completion(request, model, body)
    parts: list[Part] = []
    if completion.text or not completion.tool_calls:
        parts.append(Part(text=completion.text))
    parts.extend(
        Part(functionCall={"name": c.name, "args": c.parsed_arguments})
        for c in completion.tool_calls
    )
    return GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(role="model", parts=parts),
                finishReason=_FINISH.get(completion.finish_reason, "STOP"),
                index=0,
            )
        ],
        usageMetadata=UsageMetadata(
            promptTokenCount=completion.prompt_tokens,
            candidatesTokenCount=completion.completion_tokens,
            totalTokenCount=completion.total_tokens,
        ),
    )


@router.post("/models/{model}:streamGenerateContent")
def stream_generate_content(request: Request, model: str, body: GenerateContentRequest):
    """SSE with ``alt=sse`` (what the SDKs use), a JSON array otherwise -- like the real API."""
    completion = _completion(request, model, body)
    if request.query_params.get("alt") == "sse":
        return sse_response(gemini_chunks(completion, model=model))
    return gemini_chunk_dicts(completion, model=model)


@router.post("/models/{model}:batchGenerateContent")
def create_batch(model: str, body: dict) -> dict:
    requests_payload = body.get("requests")
    if requests_payload is None and isinstance(body.get("src"), dict):
        requests_payload = body["src"].get("inlinedRequests") or body["src"].get("requests")
    if not isinstance(requests_payload, list) or not requests_payload:
        raise batch_support.HTTPException(status_code=400, detail="Provide inline batch requests.")
    batch_id = f"batches/{uuid.uuid4().hex[:16]}"
    batch_support._batches[batch_id] = {
        "id": batch_id,
        "name": batch_id,
        "model": f"models/{model}" if not model.startswith("models/") else model,
        "createTime": batch_support._now_iso(),
        "updateTime": batch_support._now_iso(),
        "endTime": None,
        "metadata": {"state": "JOB_STATE_PENDING"},
        "src": {"inlinedRequests": requests_payload},
        "dest": {"inlinedResponses": []},
        "requests": requests_payload,
        "provider": "gemini",
        "kind": "gemini",
        "ready_at": time.time() + batch_support._BATCH_DELAY,
        "done": False,
    }
    return batch_support._public_payload(batch_support._batches[batch_id], exclude={"requests"})


@router.get("/batches")
def list_batches(limit: int = 20, after: str | None = None) -> dict:
    items = batch_support._sorted("gemini", "gemini")
    if after:
        ids = [item["id"] for item in items]
        if after in ids:
            items = items[ids.index(after) + 1 :]
    page = items[:limit]
    return {"batches": [batch_support._public_payload(item, exclude={"requests"}) for item in page]}


@router.get("/batches/{batch_name:path}")
def get_batch(batch_name: str) -> dict:
    normalized = batch_name if batch_name.startswith("batches/") else f"batches/{batch_name}"
    return batch_support._public_payload(
        batch_support._get_batch(normalized, provider="gemini", kind="gemini"),
        exclude={"requests"},
    )


@router.post("/batches/{batch_name:path}:cancel")
def cancel_batch(batch_name: str) -> dict:
    normalized = batch_name if batch_name.startswith("batches/") else f"batches/{batch_name}"
    batch = batch_support._get_batch(normalized, provider="gemini", kind="gemini")
    if batch["metadata"]["state"] in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}:
        raise batch_support.HTTPException(status_code=400, detail="Batch is already terminal.")
    batch["metadata"]["state"] = "JOB_STATE_CANCELLED"
    batch["updateTime"] = batch_support._now_iso()
    batch["endTime"] = batch_support._now_iso()
    batch["done"] = True
    return batch_support._public_payload(batch, exclude={"requests"})


@router.delete("/batches/{batch_name:path}")
def delete_batch(batch_name: str):
    normalized = batch_name if batch_name.startswith("batches/") else f"batches/{batch_name}"
    batch_support._get_batch(normalized, provider="gemini", kind="gemini")
    del batch_support._batches[normalized]
    return Response(status_code=status.HTTP_204_NO_CONTENT)


from llmock.routers import registry as _registry
_registry.register(router)
