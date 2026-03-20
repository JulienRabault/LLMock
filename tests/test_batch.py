"""Tests for the Batch API simulation."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

import llmock.routers.batch as batch_mod
from llmock.main import create_app
from llmock.routers.batch import _batches, _files


@pytest.fixture(autouse=True)
def clear_state():
    """Reset in-memory stores before each test."""
    for name in (
        "_files",
        "_batches",
        "_datasets",
    ):
        getattr(batch_mod, name).clear()
    yield
    for name in (
        "_files",
        "_batches",
        "_datasets",
    ):
        getattr(batch_mod, name).clear()


@pytest.fixture
def client():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _upload_jsonl(client: TestClient, content: str) -> str:
    resp = client.post("/v1/files", content=content.encode())
    assert resp.status_code == 200
    return resp.json()["id"]


def _create_batch(client: TestClient, file_id: str, endpoint: str = "/v1/chat/completions") -> dict:
    resp = client.post(
        "/v1/batches",
        json={"input_file_id": file_id, "endpoint": endpoint, "completion_window": "24h"},
    )
    assert resp.status_code == 200
    return resp.json()


def _upload_jsonl_to(client: TestClient, path: str, content: str) -> str:
    resp = client.post(path, content=content.encode())
    assert resp.status_code == 200
    return resp.json()["id"]


def _get_nested(payload: dict, path: str):
    value = payload
    for part in path.split("."):
        value = value[part]
    return value


def _wait_for_terminal_status(
    client: TestClient,
    path: str,
    *,
    field: str = "status",
    terminal_statuses: set[str],
    timeout_s: float = 2.0,
) -> dict:
    deadline = time.time() + timeout_s
    payload = client.get(path).json()
    current = _get_nested(payload, field)
    while current not in terminal_statuses and time.time() < deadline:
        time.sleep(0.1)
        payload = client.get(path).json()
        current = _get_nested(payload, field)
    return payload


SAMPLE_LINE = json.dumps({
    "custom_id": "req-1",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
})


# ---------------------------------------------------------------------------
# File upload tests
# ---------------------------------------------------------------------------

def test_upload_file_raw_body(client):
    resp = client.post("/v1/files", content=SAMPLE_LINE.encode())
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "file"
    assert data["purpose"] == "batch"
    assert data["id"] in _files


def test_upload_file_multipart(client):
    resp = client.post(
        "/v1/files",
        files={"file": ("data.jsonl", SAMPLE_LINE.encode(), "application/jsonl")},
        data={"purpose": "batch"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "data.jsonl"


def test_get_file_metadata(client):
    resp = client.post("/v1/files", content=SAMPLE_LINE.encode())
    file_id = resp.json()["id"]

    resp2 = client.get(f"/v1/files/{file_id}")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == file_id


def test_get_file_not_found(client):
    resp = client.get("/v1/files/file-nonexistent")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Batch lifecycle tests
# ---------------------------------------------------------------------------

def test_create_batch_unknown_file(client):
    resp = client.post(
        "/v1/batches",
        json={"input_file_id": "file-doesnotexist", "endpoint": "/v1/chat/completions", "completion_window": "24h"},
    )
    assert resp.status_code == 400


def test_create_batch_returns_validating_or_in_progress(client):
    file_id = _upload_jsonl(client, SAMPLE_LINE)
    data = _create_batch(client, file_id)

    assert data["object"] == "batch"
    assert data["status"] in ("validating", "in_progress")
    assert data["input_file_id"] == file_id
    assert data["endpoint"] == "/v1/chat/completions"


def test_batch_completes_after_delay(client, monkeypatch):
    import llmock.routers.batch as batch_mod
    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)

    file_id = _upload_jsonl(client, SAMPLE_LINE)
    batch = _create_batch(client, file_id)
    batch_id = batch["id"]

    # Poll until completed (max ~2 s)
    deadline = time.time() + 2.0
    status = batch["status"]
    while status not in ("completed", "failed", "cancelled") and time.time() < deadline:
        time.sleep(0.1)
        resp = client.get(f"/v1/batches/{batch_id}")
        status = resp.json()["status"]

    assert status == "completed"
    data = client.get(f"/v1/batches/{batch_id}").json()
    assert data["output_file_id"] is not None
    assert data["request_counts"]["total"] == 1
    assert data["request_counts"]["completed"] == 1


def test_batch_output_content(client, monkeypatch):
    import llmock.routers.batch as batch_mod
    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)

    file_id = _upload_jsonl(client, SAMPLE_LINE)
    batch = _create_batch(client, file_id)
    batch_id = batch["id"]

    deadline = time.time() + 2.0
    status = batch["status"]
    while status not in ("completed", "failed", "cancelled") and time.time() < deadline:
        time.sleep(0.1)
        status = client.get(f"/v1/batches/{batch_id}").json()["status"]

    assert status == "completed"
    output_file_id = client.get(f"/v1/batches/{batch_id}").json()["output_file_id"]

    content_resp = client.get(f"/v1/files/{output_file_id}/content")
    assert content_resp.status_code == 200
    result = json.loads(content_resp.text.strip())
    assert result["custom_id"] == "req-1"
    assert result["response"]["status_code"] == 200
    assert result["error"] is None


def test_batch_embedding_output(client, monkeypatch):
    import llmock.routers.batch as batch_mod
    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)

    line = json.dumps({
        "custom_id": "emb-1",
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {"model": "text-embedding-3-small", "input": "hello world"},
    })
    file_id = _upload_jsonl(client, line)
    batch = _create_batch(client, file_id, endpoint="/v1/embeddings")
    batch_id = batch["id"]

    deadline = time.time() + 2.0
    status = batch["status"]
    while status not in ("completed", "failed", "cancelled") and time.time() < deadline:
        time.sleep(0.1)
        status = client.get(f"/v1/batches/{batch_id}").json()["status"]

    assert status == "completed"
    output_file_id = client.get(f"/v1/batches/{batch_id}").json()["output_file_id"]
    content = client.get(f"/v1/files/{output_file_id}/content").text.strip()
    result = json.loads(content)
    assert result["custom_id"] == "emb-1"
    body = result["response"]["body"]
    assert body["object"] == "list"
    assert len(body["data"]) == 1


def test_batch_image_output(client, monkeypatch):
    import llmock.routers.batch as batch_mod

    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)

    line = json.dumps(
        {
            "custom_id": "img-1",
            "method": "POST",
            "url": "/v1/images/generations",
            "body": {"model": "gpt-image-1", "prompt": "A mock image", "response_format": "url"},
        }
    )
    file_id = _upload_jsonl(client, line)
    batch = _create_batch(client, file_id, endpoint="/v1/images/generations")
    batch_id = batch["id"]

    deadline = time.time() + 2.0
    status = batch["status"]
    while status not in ("completed", "failed", "cancelled") and time.time() < deadline:
        time.sleep(0.1)
        status = client.get(f"/v1/batches/{batch_id}").json()["status"]

    assert status == "completed"
    output_file_id = client.get(f"/v1/batches/{batch_id}").json()["output_file_id"]
    content = client.get(f"/v1/files/{output_file_id}/content").text.strip()
    result = json.loads(content)
    assert result["custom_id"] == "img-1"
    assert result["response"]["body"]["data"][0]["url"].startswith("data:image/svg+xml;base64,")


def test_cancel_batch(client):
    # Inject a batch directly in "in_progress" state to bypass background task execution.
    import time, uuid
    batch_id = f"batch_{uuid.uuid4().hex[:24]}"
    file_id = _upload_jsonl(client, SAMPLE_LINE)
    _batches[batch_id] = {
        "id": batch_id,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "errors": None,
        "input_file_id": file_id,
        "completion_window": "24h",
        "status": "in_progress",
        "output_file_id": None,
        "error_file_id": None,
        "created_at": int(time.time()),
        "in_progress_at": int(time.time()),
        "expires_at": int(time.time()) + 86400,
        "finalizing_at": None,
        "completed_at": None,
        "failed_at": None,
        "expired_at": None,
        "cancelling_at": None,
        "cancelled_at": None,
        "request_counts": {"total": 0, "completed": 0, "failed": 0},
        "metadata": None,
    }

    cancel_resp = client.post(f"/v1/batches/{batch_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"


def test_cancel_already_cancelled(client):
    # Re-use the same direct-inject approach so background tasks don't interfere.
    import time, uuid
    batch_id = f"batch_{uuid.uuid4().hex[:24]}"
    file_id = _upload_jsonl(client, SAMPLE_LINE)
    _batches[batch_id] = {
        "id": batch_id,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "errors": None,
        "input_file_id": file_id,
        "completion_window": "24h",
        "status": "in_progress",
        "output_file_id": None,
        "error_file_id": None,
        "created_at": int(time.time()),
        "in_progress_at": int(time.time()),
        "expires_at": int(time.time()) + 86400,
        "finalizing_at": None,
        "completed_at": None,
        "failed_at": None,
        "expired_at": None,
        "cancelling_at": None,
        "cancelled_at": None,
        "request_counts": {"total": 0, "completed": 0, "failed": 0},
        "metadata": None,
    }

    client.post(f"/v1/batches/{batch_id}/cancel")
    resp2 = client.post(f"/v1/batches/{batch_id}/cancel")
    assert resp2.status_code == 400


def test_get_batch_not_found(client):
    resp = client.get("/v1/batches/batch_nonexistent")
    assert resp.status_code == 404


def test_list_batches(client):
    file_id = _upload_jsonl(client, SAMPLE_LINE)
    for _ in range(3):
        _create_batch(client, file_id)

    resp = client.get("/v1/batches")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 3


def test_list_batches_limit(client):
    file_id = _upload_jsonl(client, SAMPLE_LINE)
    for _ in range(5):
        _create_batch(client, file_id)

    resp = client.get("/v1/batches?limit=2")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 2
    assert data["has_more"] is True


def test_multi_line_batch(client, monkeypatch):
    import llmock.routers.batch as batch_mod
    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)

    lines = "\n".join([
        json.dumps({"custom_id": f"r{i}", "method": "POST", "url": "/v1/chat/completions",
                    "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": f"msg {i}"}]}})
        for i in range(4)
    ])
    file_id = _upload_jsonl(client, lines)
    batch = _create_batch(client, file_id)
    batch_id = batch["id"]

    deadline = time.time() + 2.0
    status = batch["status"]
    while status not in ("completed", "failed", "cancelled") and time.time() < deadline:
        time.sleep(0.1)
        status = client.get(f"/v1/batches/{batch_id}").json()["status"]

    assert status == "completed"
    data = client.get(f"/v1/batches/{batch_id}").json()
    assert data["request_counts"]["total"] == 4
    assert data["request_counts"]["completed"] == 4


def test_groq_batch_output(client, monkeypatch):
    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)

    file_id = _upload_jsonl_to(client, "/groq/openai/v1/files", SAMPLE_LINE)
    resp = client.post(
        "/groq/openai/v1/batches",
        json={"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
    )
    assert resp.status_code == 200
    batch_id = resp.json()["id"]

    batch = _wait_for_terminal_status(
        client,
        f"/groq/openai/v1/batches/{batch_id}",
        terminal_statuses={"completed", "failed", "cancelled"},
    )
    assert batch["status"] == "completed"
    content = client.get(f"/groq/openai/v1/files/{batch['output_file_id']}/content").text.strip()
    body = json.loads(content)["response"]["body"]
    assert body["x_groq"]["id"].startswith("req_")


def test_together_batch_output(client, monkeypatch):
    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)

    file_id = _upload_jsonl_to(client, "/together/v1/files", SAMPLE_LINE)
    resp = client.post(
        "/together/v1/batches",
        json={"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
    )
    assert resp.status_code == 200
    batch_id = resp.json()["id"]

    batch = _wait_for_terminal_status(
        client,
        f"/together/v1/batches/{batch_id}",
        terminal_statuses={"completed", "failed", "cancelled"},
    )
    assert batch["status"] == "completed"
    content = client.get(f"/together/v1/files/{batch['output_file_id']}/content").text.strip()
    choice = json.loads(content)["response"]["body"]["choices"][0]
    assert choice["text"]
    assert choice["finish_reason"] == "eos"


def test_mistral_batch_inline_requests(client, monkeypatch):
    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)

    resp = client.post(
        "/mistral/v1/batch/jobs",
        json={
            "model": "mistral-large-latest",
            "endpoint": "/v1/chat/completions",
            "requests": [
                {
                    "custom_id": "mistral-1",
                    "body": {"messages": [{"role": "user", "content": "hello mistral"}]},
                }
            ],
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["id"]

    job = _wait_for_terminal_status(
        client,
        f"/mistral/v1/batch/jobs/{job_id}",
        terminal_statuses={"SUCCESS", "FAILED", "CANCELLED"},
    )
    assert job["status"] == "SUCCESS"
    output_file_id = job["output_file"]["id"]
    result = json.loads(client.get(f"/mistral/v1/files/{output_file_id}/content").text.strip())
    assert result["custom_id"] == "mistral-1"
    assert result["response"]["body"]["model"] == "mistral-large-latest"


def test_anthropic_message_batch_results(client, monkeypatch):
    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)

    resp = client.post(
        "/anthropic/v1/messages/batches",
        json={
            "requests": [
                {
                    "custom_id": "anthropic-1",
                    "params": {
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 128,
                        "messages": [{"role": "user", "content": "hello claude"}],
                    },
                }
            ]
        },
    )
    assert resp.status_code == 200
    batch_id = resp.json()["id"]

    batch = _wait_for_terminal_status(
        client,
        f"/anthropic/v1/messages/batches/{batch_id}",
        field="processing_status",
        terminal_statuses={"ended"},
    )
    assert batch["request_counts"]["succeeded"] == 1
    results = client.get(batch["results_url"])
    assert results.status_code == 200
    line = json.loads(results.text.strip())
    assert line["custom_id"] == "anthropic-1"
    assert line["result"]["type"] == "succeeded"


def test_cohere_batch_with_dataset(client, monkeypatch):
    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)

    dataset_resp = client.post(
        "/cohere/v2/datasets",
        json={
            "name": "cohere-input",
            "records": [
                {
                    "custom_id": "cohere-1",
                    "body": {
                        "model": "command-r-plus",
                        "messages": [{"role": "user", "content": "hello cohere"}],
                    },
                }
            ],
        },
    )
    assert dataset_resp.status_code == 200
    dataset_id = dataset_resp.json()["id"]

    resp = client.post(
        "/cohere/v2/batches",
        json={
            "name": "cohere-batch",
            "input_dataset_id": dataset_id,
            "model": "command-r-plus",
            "endpoint": "/v2/chat",
        },
    )
    assert resp.status_code == 200
    batch_id = resp.json()["id"]

    batch = _wait_for_terminal_status(
        client,
        f"/cohere/v2/batches/{batch_id}",
        terminal_statuses={"BATCH_STATUS_COMPLETED", "BATCH_STATUS_FAILED", "BATCH_STATUS_CANCELLED"},
    )
    assert batch["status"] == "BATCH_STATUS_COMPLETED"
    output = client.get(f"/cohere/v2/datasets/{batch['output_dataset_id']}/download")
    assert output.status_code == 200
    line = json.loads(output.text.strip())
    assert line["custom_id"] == "cohere-1"
    assert line["status"] == "succeeded"


def test_gemini_batch_generate_content(client, monkeypatch):
    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)

    resp = client.post(
        "/gemini/v1beta/models/gemini-2.0-flash:batchGenerateContent",
        json={
            "requests": [
                {
                    "custom_id": "gemini-1",
                    "body": {
                        "contents": [{"role": "user", "parts": [{"text": "hello gemini"}]}],
                    },
                }
            ]
        },
    )
    assert resp.status_code == 200
    batch_name = resp.json()["name"]

    batch = _wait_for_terminal_status(
        client,
        f"/gemini/v1beta/{batch_name}",
        field="metadata.state",
        terminal_statuses={"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"},
    )
    assert batch["metadata"]["state"] == "JOB_STATE_SUCCEEDED"
    response = batch["dest"]["inlinedResponses"][0]["response"]
    assert response["candidates"][0]["content"]["parts"][0]["text"]


def test_xai_batch_add_requests_and_results(client, monkeypatch):
    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)

    create_resp = client.post("/xai/v1/batches", json={})
    assert create_resp.status_code == 200
    batch_id = create_resp.json()["id"]

    add_resp = client.post(
        f"/xai/v1/batches/{batch_id}/requests",
        json={
            "requests": [
                {
                    "custom_id": "xai-1",
                    "url": "/v1/images/generations",
                    "body": {"model": "grok-2-image", "prompt": "synthetic image"},
                }
            ]
        },
    )
    assert add_resp.status_code == 200

    batch = _wait_for_terminal_status(
        client,
        f"/xai/v1/batches/{batch_id}",
        terminal_statuses={"completed", "failed", "cancelled"},
    )
    assert batch["status"] == "completed"
    results = client.get(f"/xai/v1/batches/{batch_id}/results")
    assert results.status_code == 200
    payload = results.json()["data"][0]
    assert payload["state"] == "succeeded"
    assert payload["response"]["data"][0]["url"].startswith("data:image/svg+xml;base64,")
