"""Batch API simulation for LLMock."""

# Official docs:
# - https://platform.openai.com/docs/api-reference/batch
# - https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
# - https://docs.mistral.ai/capabilities/batch
# - https://docs.cohere.com/reference/create-batch
# - https://ai.google.dev/gemini-api/docs/batch-api
# - https://console.groq.com/docs/batch
# - https://docs.x.ai/developers/rest-api-reference/inference/batches

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import PlainTextResponse
from llmock.simulation import (
    DEFAULT_IMAGE_SIZE,
    MockResponseSettings,
    build_fake_image_data_uri,
    build_fake_image_payload,
    build_mock_embedding,
    build_mock_text,
    estimate_tokens,
    flatten_text,
    provider_from_path,
)

_files: dict[str, dict[str, Any]] = {}
_batches: dict[str, dict[str, Any]] = {}
_datasets: dict[str, dict[str, Any]] = {}
_BATCH_DELAY: float = 3.0

_OPENAI_LIKE_PREFIXES = {
    "openai": "/v1",
    "groq": "/groq/openai/v1",
    "together": "/together/v1",
    "ai21": "/ai21/v1",
    "perplexity": "/perplexity/v1",
}
_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "groq": "llama-3.3-70b-versatile",
    "together": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "ai21": "jamba-1.5-large",
    "perplexity": "sonar-pro",
    "mistral": "mistral-large-latest",
    "anthropic": "claude-sonnet-4-6",
    "cohere": "command-r-plus",
    "gemini": "gemini-2.0-flash",
    "xai": "grok-3",
}


def _now() -> int:
    return int(time.time())


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _iso_from_ts(timestamp: float | int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _make_file_id() -> str:
    return f"file-{uuid.uuid4().hex[:24]}"


def _make_batch_id(prefix: str = "batch") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _make_dataset_id() -> str:
    return f"dataset-{uuid.uuid4().hex[:24]}"


def _mock_settings() -> MockResponseSettings:
    return MockResponseSettings.from_env()


def _normalize_path(url: str) -> str:
    path = (url or "/").strip()
    if "://" in path:
        parts = path.split("/", 3)
        path = f"/{parts[3]}" if len(parts) > 3 else "/"
    if not path.startswith("/"):
        path = f"/{path}"
    return path


def _normalize_provider_path(provider: str, url: str) -> str:
    path = _normalize_path(url)
    if provider == "mistral":
        prefix = "/mistral/v1"
    elif provider == "xai":
        prefix = "/xai/v1"
    else:
        prefix = _OPENAI_LIKE_PREFIXES[provider]
    if provider == "openai":
        return path
    if path.startswith(prefix):
        return path
    if path.startswith("/v1/"):
        return f"{prefix}{path[3:]}"
    return f"{prefix}{path}"


def _store_file(*, filename: str, purpose: str, content: str, provider: str) -> dict[str, Any]:
    record = {
        "id": _make_file_id(),
        "object": "file",
        "bytes": len(content.encode("utf-8")),
        "created_at": _now(),
        "filename": filename,
        "purpose": purpose,
        "status": "processed",
        "content": content,
        "provider": provider,
    }
    _files[record["id"]] = record
    return record


def _file_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in {"content", "provider"}}


def _get_file_or_404(file_id: str) -> dict[str, Any]:
    record = _files.get(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found.")
    return record


def _public_payload(record: dict[str, Any], *, exclude: set[str] | None = None) -> dict[str, Any]:
    hidden = {"provider", "kind", "ready_at", "done"}
    if exclude:
        hidden |= exclude
    return {key: value for key, value in record.items() if key not in hidden}


def _read_messages(messages: list[dict[str, Any]]) -> tuple[str, int]:
    prompt = " ".join(flatten_text(message.get("content", "")) for message in messages)
    tokens = estimate_tokens(*(message.get("content", "") for message in messages))
    return prompt, tokens


def _fake_perplexity_search(prompt: str) -> list[dict[str, Any]]:
    slug = prompt.replace(" ", "-").lower()[:32] or "llmock"
    return [{
        "title": f"Mock result for {prompt[:40] or 'your query'}",
        "url": f"https://example.com/mock-search/{slug}",
        "date": "2026-03-20",
        "last_updated": "2026-03-20",
        "snippet": f"Synthetic search snippet for {prompt[:48] or 'your query'}.",
        "source": "web",
    }]


def _fake_perplexity_images(prompt: str) -> list[dict[str, Any]]:
    slug = prompt.replace(" ", "-").lower()[:32] or "llmock"
    return [{
        "image_url": build_fake_image_data_uri(prompt=prompt or "LLMock image", size="640x360"),
        "origin_url": f"https://example.com/mock-image/{slug}",
        "title": f"Mock image for {prompt[:40] or 'your query'}",
        "width": 640,
        "height": 360,
    }]


def _build_openai_like_chat(provider: str, body: dict[str, Any]) -> dict[str, Any]:
    model = str(body.get("model", _DEFAULT_MODELS[provider]))
    prompt, prompt_tokens = _read_messages(body.get("messages", []))
    reply = build_mock_text(settings=_mock_settings(), model=model, prompt=prompt)
    completion_tokens = estimate_tokens(reply)
    n = max(1, int(body.get("n", 1) or 1))
    if provider == "perplexity":
        search = _fake_perplexity_search(prompt)
        return {
            "id": uuid.uuid4().hex,
            "object": "chat.completion",
            "created": _now(),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
                "delta": {},
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "citations": [item["url"] for item in search] if body.get("return_citations") else [],
            "search_results": search,
            "images": _fake_perplexity_images(prompt or model) if body.get("return_images") else [],
            "related_questions": (
                [
                    f"What are the main takeaways about {prompt[:48] or model}?",
                    f"How would you validate {prompt[:48] or model} in production?",
                ]
                if body.get("return_related_questions")
                else []
            ),
        }

    finish_reason = "eos" if provider == "together" else "stop"
    choices: list[dict[str, Any]] = []
    for index in range(n):
        choice = {
            "index": index,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": finish_reason,
        }
        if provider == "together":
            choice["text"] = reply
            choice["logprobs"] = None
        if provider == "groq":
            choice["logprobs"] = None
        choices.append(choice)

    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if provider == "groq":
        response["system_fingerprint"] = f"fp_{uuid.uuid4().hex[:8]}"
        response["x_groq"] = {"id": f"req_{uuid.uuid4().hex[:24]}"}
        response["usage"] |= {
            "prompt_time": 0.001,
            "completion_time": 0.001,
            "total_time": 0.002,
        }
    return response


def _build_mistral_chat(body: dict[str, Any]) -> dict[str, Any]:
    model = str(body.get("model", _DEFAULT_MODELS["mistral"]))
    prompt, prompt_tokens = _read_messages(body.get("messages", []))
    reply = build_mock_text(settings=_mock_settings(), model=model, prompt=prompt)
    completion_tokens = estimate_tokens(reply)
    n = max(1, int(body.get("n", 1) or 1))
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "choices": [{
            "index": index,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
            "logprobs": None,
        } for index in range(n)],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _build_anthropic_message(body: dict[str, Any]) -> dict[str, Any]:
    model = str(body.get("model", _DEFAULT_MODELS["anthropic"]))
    prompt, input_tokens = _read_messages(body.get("messages", []))
    input_tokens += estimate_tokens(body.get("system", ""))
    reply = build_mock_text(settings=_mock_settings(), model=model, prompt=prompt)
    output_tokens = estimate_tokens(reply)
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": reply}],
        "model": model,
        "stop_reason": "max_tokens" if output_tokens >= int(body.get("max_tokens", 4096)) else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _build_embeddings(body: dict[str, Any]) -> dict[str, Any]:
    model = str(body.get("model", "text-embedding-3-small"))
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    prompt_tokens = estimate_tokens(*inputs)
    return {
        "object": "list",
        "data": [{
            "object": "embedding",
            "index": index,
            "embedding": build_mock_embedding(text, 1536),
        } for index, text in enumerate(inputs)],
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        },
    }


def _build_images(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "created": _now(),
        "data": build_fake_image_payload(
            prompt=str(body.get("prompt", "LLMock image")),
            count=max(1, int(body.get("n", 1) or 1)),
            size=str(body.get("size", DEFAULT_IMAGE_SIZE)),
            response_format=str(body.get("response_format", "url")),
        ),
    }


def _build_gemini(model: str, body: dict[str, Any]) -> dict[str, Any]:
    prompt_values: list[Any] = []
    for content in body.get("contents", []):
        prompt_values.extend(content.get("parts", []))
    if body.get("systemInstruction"):
        prompt_values.extend(body["systemInstruction"].get("parts", []))
    prompt = " ".join(flatten_text(part) for part in prompt_values)
    prompt_tokens = estimate_tokens(*prompt_values)
    reply = build_mock_text(settings=_mock_settings(), model=model, prompt=prompt)
    output_tokens = estimate_tokens(reply)
    return {
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": reply}]},
            "finishReason": "STOP",
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": output_tokens,
            "totalTokenCount": prompt_tokens + output_tokens,
        },
    }


def _build_cohere(body: dict[str, Any]) -> dict[str, Any]:
    model = str(body.get("model", _DEFAULT_MODELS["cohere"]))
    prompt, input_tokens = _read_messages(body.get("messages", []))
    reply = build_mock_text(settings=_mock_settings(), model=model, prompt=prompt)
    output_tokens = estimate_tokens(reply)
    units = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return {
        "id": uuid.uuid4().hex,
        "finish_reason": "COMPLETE",
        "message": {"role": "assistant", "content": [{"type": "text", "text": reply}]},
        "usage": {"billed_units": units, "tokens": units},
    }


def _build_response(path: str, body: dict[str, Any]) -> dict[str, Any]:
    provider = provider_from_path(path)
    if path.startswith("/anthropic/v1/messages"):
        return _build_anthropic_message(body)
    if path.startswith("/gemini/v1beta/models/") and ":generateContent" in path:
        model = path.split("/models/", 1)[1].split(":generateContent", 1)[0]
        return _build_gemini(model, body)
    if path.startswith("/cohere/v2/chat"):
        return _build_cohere(body)
    if path.endswith("/embeddings"):
        return _build_embeddings(body)
    if path.endswith("/images/generations"):
        return _build_images(body)
    if path.endswith("/chat/completions"):
        if provider == "mistral":
            return _build_mistral_chat(body)
        return _build_openai_like_chat(provider, body)
    raise ValueError(f"Unsupported batch endpoint '{path}'.")


def _process_line(
    line: str,
    *,
    provider: str,
    default_endpoint: str,
    default_model: str | None = None,
) -> dict[str, Any]:
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        return {"id": str(uuid.uuid4()), "custom_id": None, "response": None, "error": {"code": "invalid_json", "message": "Could not parse request line."}}

    custom_id = item.get("custom_id") or str(uuid.uuid4())
    if item.get("method", "POST") != "POST":
        return {"id": str(uuid.uuid4()), "custom_id": custom_id, "response": None, "error": {"code": "unsupported_method", "message": "Only POST batch lines are supported."}}

    body = dict(item.get("body", {}))
    if default_model and "model" not in body and provider != "gemini":
        body["model"] = default_model

    try:
        response_body = _build_response(
            _normalize_provider_path(provider, item.get("url", default_endpoint)),
            body,
        )
    except ValueError as exc:
        return {"id": str(uuid.uuid4()), "custom_id": custom_id, "response": None, "error": {"code": "unsupported_endpoint", "message": str(exc)}}

    return {
        "id": str(uuid.uuid4()),
        "custom_id": custom_id,
        "response": {"status_code": 200, "request_id": uuid.uuid4().hex, "body": response_body},
        "error": None,
    }


def _refresh(batch: dict[str, Any]) -> None:
    if batch.get("ready_at") is None or time.time() < batch["ready_at"]:
        return
    if batch.get("done"):
        return
    kind = batch["kind"]
    if kind == "openai_like":
        _finish_openai_like(batch)
    elif kind == "mistral":
        _finish_mistral(batch)
    elif kind == "anthropic":
        _finish_anthropic(batch)
    elif kind == "gemini":
        _finish_gemini(batch)
    elif kind == "cohere":
        _finish_cohere(batch)
    elif kind == "xai":
        _finish_xai(batch)


def _sorted(provider: str, kind: str) -> list[dict[str, Any]]:
    items = []
    for batch in _batches.values():
        if batch.get("provider") == provider and batch.get("kind") == kind:
            _refresh(batch)
            items.append(batch)
    return sorted(items, key=lambda item: item.get("created_at", item.get("create_time", "")), reverse=True)


def _get_batch(batch_id: str, *, provider: str, kind: str) -> dict[str, Any]:
    batch = _batches.get(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
    same_provider = batch.get("provider") == provider
    same_kind = batch.get("kind") == kind
    legacy_openai = provider == "openai" and kind == "openai_like" and "provider" not in batch and "kind" not in batch
    if not ((same_provider and same_kind) or legacy_openai):
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
    _refresh(batch)
    return batch


def _finish_openai_like(batch: dict[str, Any]) -> None:
    if batch.get("status") in {"completed", "failed", "cancelled"}:
        batch["done"] = True
        return
    batch["status"] = "in_progress"
    batch["in_progress_at"] = batch.get("in_progress_at") or _now()
    input_file = _files.get(batch["input_file_id"])
    if input_file is None:
        batch["status"] = "failed"
        batch["failed_at"] = _now()
        batch["errors"] = {"data": [{"code": "missing_file", "message": "Input file not found."}]}
        batch["done"] = True
        return
    ok_lines: list[str] = []
    err_lines: list[str] = []
    total = completed = failed = 0
    for raw_line in input_file["content"].splitlines():
        line = raw_line.strip()
        if not line:
            continue
        total += 1
        result = _process_line(line, provider=batch["provider"], default_endpoint=batch["endpoint"])
        if result["error"] is None:
            ok_lines.append(json.dumps(result))
            completed += 1
        else:
            err_lines.append(json.dumps(result))
            failed += 1
    if ok_lines:
        batch["output_file_id"] = _store_file(
            filename=f"{batch['provider']}_batch_output_{batch['id']}.jsonl",
            purpose="batch_output",
            content="\n".join(ok_lines),
            provider=batch["provider"],
        )["id"]
    if err_lines:
        batch["error_file_id"] = _store_file(
            filename=f"{batch['provider']}_batch_error_{batch['id']}.jsonl",
            purpose="batch_error",
            content="\n".join(err_lines),
            provider=batch["provider"],
        )["id"]
        batch["errors"] = {"data": [{"code": "batch_line_error", "message": f"{failed} request(s) failed."}]}
    batch["status"] = "failed" if failed and not completed else "completed"
    batch["request_counts"] = {"total": total, "completed": completed, "failed": failed}
    batch["finalizing_at"] = _now()
    batch["completed_at"] = _now() if batch["status"] == "completed" else None
    batch["failed_at"] = _now() if batch["status"] == "failed" else None
    batch["done"] = True


def _finish_mistral(batch: dict[str, Any]) -> None:
    if batch.get("status") == "CANCELLED":
        batch["done"] = True
        return
    lines: list[str] = []
    for file_id in batch.get("input_files", []):
        file = _files.get(file_id)
        if file:
            lines.extend(line for line in file["content"].splitlines() if line.strip())
    for request_item in batch.get("requests", []):
        lines.append(json.dumps(request_item))
    ok_lines: list[str] = []
    err_lines: list[str] = []
    total = completed = failed = 0
    batch["status"] = "RUNNING"
    batch["started_at"] = batch.get("started_at") or _now_iso()
    for line in lines:
        total += 1
        result = _process_line(
            line,
            provider="mistral",
            default_endpoint=batch["endpoint"],
            default_model=batch.get("model"),
        )
        if result["error"] is None:
            ok_lines.append(json.dumps(result))
            completed += 1
        else:
            err_lines.append(json.dumps(result))
            failed += 1
    if ok_lines:
        batch["output_file"] = _store_file(
            filename=f"mistral_batch_output_{batch['id']}.jsonl",
            purpose="batch_output",
            content="\n".join(ok_lines),
            provider="mistral",
        )["id"]
    if err_lines:
        batch["error_file"] = _store_file(
            filename=f"mistral_batch_error_{batch['id']}.jsonl",
            purpose="batch_error",
            content="\n".join(err_lines),
            provider="mistral",
        )["id"]
    batch["status"] = "FAILED" if failed and not completed else "SUCCESS"
    batch["completed_at"] = _now_iso()
    batch["total_requests"] = total
    batch["succeeded_requests"] = completed
    batch["failed_requests"] = failed
    batch["completed_requests"] = completed + failed
    batch["done"] = True


def _finish_anthropic(batch: dict[str, Any]) -> None:
    if batch.get("processing_status") == "ended":
        batch["done"] = True
        return
    lines: list[str] = []
    succeeded = errored = 0
    for request_item in batch.get("requests", []):
        custom_id = request_item.get("custom_id", str(uuid.uuid4()))
        result = {"custom_id": custom_id, "result": {"type": "succeeded", "message": _build_anthropic_message(request_item.get("params", {}))}}
        lines.append(json.dumps(result))
        succeeded += 1
    batch["processing_status"] = "ended"
    batch["ended_at"] = _now_iso()
    batch["results_content"] = "\n".join(lines)
    batch["request_counts"] = {"processing": 0, "succeeded": succeeded, "errored": errored, "canceled": 0, "expired": 0}
    batch["done"] = True


def _finish_gemini(batch: dict[str, Any]) -> None:
    if batch["metadata"]["state"] == "JOB_STATE_CANCELLED":
        batch["done"] = True
        return
    responses = []
    for request_item in batch.get("requests", []):
        request_body = request_item.get("request") or request_item
        responses.append({
            "custom_id": request_item.get("custom_id", str(uuid.uuid4())),
            "response": _build_gemini(batch["model"], request_body),
        })
    batch["metadata"]["state"] = "JOB_STATE_SUCCEEDED"
    batch["metadata"]["completedRequestCount"] = len(responses)
    batch["dest"] = {"inlinedResponses": responses}
    batch["updateTime"] = _now_iso()
    batch["done"] = True


def _finish_cohere(batch: dict[str, Any]) -> None:
    if batch.get("status") == "BATCH_STATUS_CANCELLED":
        batch["done"] = True
        return
    dataset = _datasets.get(batch["input_dataset_id"])
    if dataset is None:
        batch["status"] = "BATCH_STATUS_FAILED"
        batch["status_reason"] = "Input dataset not found."
        batch["done"] = True
        return
    rows = []
    for row in dataset["rows"]:
        body = dict(row.get("body", {})) or {"messages": row.get("messages", [])}
        body.setdefault("model", batch["model"])
        rows.append({
            "custom_id": row.get("custom_id", str(uuid.uuid4())),
            "status": "succeeded",
            "response": _build_cohere(body),
        })
    output_id = _make_dataset_id()
    _datasets[output_id] = {"id": output_id, "name": f"{dataset['name']}-output", "provider": "cohere", "created_at": _now_iso(), "rows": rows}
    batch["status"] = "BATCH_STATUS_COMPLETED"
    batch["output_dataset_id"] = output_id
    batch["updated_at"] = _now_iso()
    batch["num_successful_records"] = len(rows)
    batch["num_failed_records"] = 0
    batch["done"] = True


def _finish_xai(batch: dict[str, Any]) -> None:
    if batch["state"]["status"] == "CANCELLED":
        batch["done"] = True
        return
    results = []
    completed = failed = 0
    for request_item in batch.get("requests", []):
        result = _process_line(
            json.dumps(request_item),
            provider="xai",
            default_endpoint="/v1/chat/completions",
            default_model=_DEFAULT_MODELS["xai"],
        )
        results.append({
            "request_id": result["id"],
            "custom_id": result["custom_id"],
            "state": "failed" if result["error"] is not None else "succeeded",
            "response": None if result["response"] is None else result["response"]["body"],
            "error": result["error"],
        })
        if result["error"] is None:
            completed += 1
        else:
            failed += 1
    batch["results"] = results
    batch["state"] = {"status": "FAILED" if failed and not completed else "COMPLETED", "total_requests": len(batch["requests"]), "completed_requests": completed, "failed_requests": failed}
    batch["update_time"] = _now_iso()
    batch["done"] = True


async def _read_upload(request: Request) -> tuple[str, str, bytes]:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        file_field = form.get("file")
        if file_field is None:
            raise HTTPException(status_code=400, detail="Missing 'file' field in form data.")
        raw = await file_field.read()  # type: ignore[union-attr]
        return str(form.get("purpose", "batch")), getattr(file_field, "filename", "upload.jsonl") or "upload.jsonl", raw
    raw = await request.body()
    return "batch", "upload.jsonl", raw


def _read_jsonl_rows(content: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in content.splitlines() if line.strip()]
