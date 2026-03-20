"""Provider-specific batch endpoint tests."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from llmock.main import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def _wait(client: TestClient, path: str, is_done, timeout: float = 2.0) -> dict:
    deadline = time.time() + timeout
    data = client.get(path).json()
    while time.time() < deadline and not is_done(data):
        time.sleep(0.05)
        data = client.get(path).json()
    return data


@pytest.mark.parametrize(
    ("prefix", "model", "check_key"),
    [
        ("/groq/openai/v1", "llama-3.3-70b-versatile", "x_groq"),
        ("/together/v1", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", None),
        ("/ai21/v1", "jamba-1.5-large", None),
        ("/perplexity/v1", "sonar-pro", "search_results"),
    ],
)
def test_openai_like_provider_batches(monkeypatch, client, prefix, model, check_key):
    import llmock.routers.batch as batch_mod

    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)
    line = json.dumps(
        {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": "Hello"}]},
        }
    )
    file_id = client.post(f"{prefix}/files", content=line.encode()).json()["id"]
    batch = client.post(
        f"{prefix}/batches",
        json={"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
    ).json()
    data = _wait(
        client,
        f"{prefix}/batches/{batch['id']}",
        lambda item: item["status"] in {"completed", "failed", "cancelled", "COMPLETED", "FAILED", "CANCELLED"},
    )

    assert data["status"] in {"completed", "COMPLETED"}
    output = client.get(f"{prefix}/files/{data['output_file_id']}/content").text.strip()
    result = json.loads(output)
    body = result["response"]["body"]
    assert result["custom_id"] == "req-1"
    assert body["choices"][0]["message"]["role"] == "assistant"
    if check_key:
        assert check_key in body


def test_mistral_batch_jobs_support_inline_requests(monkeypatch, client):
    import llmock.routers.batch as batch_mod

    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)
    job = client.post(
        "/mistral/v1/batch/jobs",
        json={
            "model": "mistral-large-latest",
            "endpoint": "/v1/chat/completions",
            "requests": [{"custom_id": "m-1", "body": {"messages": [{"role": "user", "content": "Salut"}]}}],
        },
    ).json()
    data = _wait(client, f"/mistral/v1/batch/jobs/{job['id']}", lambda item: item["status"] in {"SUCCESS", "FAILED", "CANCELLED"})

    assert data["status"] == "SUCCESS"
    output = client.get(f"/mistral/v1/files/{data['output_file']['id']}/content").text.strip()
    assert json.loads(output)["custom_id"] == "m-1"


def test_anthropic_message_batch_results(monkeypatch, client):
    import llmock.routers.batch as batch_mod

    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)
    batch = client.post(
        "/anthropic/v1/messages/batches",
        json={
            "requests": [
                {
                    "custom_id": "claude-1",
                    "params": {
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 64,
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                }
            ]
        },
    ).json()
    data = _wait(client, f"/anthropic/v1/messages/batches/{batch['id']}", lambda item: item["processing_status"] == "ended")

    assert data["request_counts"]["succeeded"] == 1
    results = client.get(f"/anthropic/v1/messages/batches/{batch['id']}/results").text.strip()
    line = json.loads(results)
    assert line["custom_id"] == "claude-1"
    assert line["result"]["type"] == "succeeded"


def test_gemini_batch_generate_content(monkeypatch, client):
    import llmock.routers.batch as batch_mod

    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)
    batch = client.post(
        "/gemini/v1beta/models/gemini-2.0-flash:batchGenerateContent",
        json={
            "requests": [
                {
                    "custom_id": "g-1",
                    "request": {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
                }
            ]
        },
    ).json()
    batch_id = batch["name"].split("/", 1)[1]
    data = _wait(client, f"/gemini/v1beta/batches/{batch_id}", lambda item: item["metadata"]["state"] in {"JOB_STATE_SUCCEEDED", "JOB_STATE_CANCELLED"})

    assert data["metadata"]["state"] == "JOB_STATE_SUCCEEDED"
    assert data["dest"]["inlinedResponses"][0]["response"]["candidates"][0]["content"]["role"] == "model"


def test_cohere_dataset_backed_batch(monkeypatch, client):
    import llmock.routers.batch as batch_mod

    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)
    dataset = client.post(
        "/cohere/v1/datasets",
        json={"name": "chat-input", "rows": [{"custom_id": "c-1", "messages": [{"role": "user", "content": "Hello"}]}]},
    ).json()
    batch = client.post(
        "/cohere/v2/batches",
        json={"input_dataset_id": dataset["id"], "model": "command-r-plus"},
    ).json()
    data = _wait(client, f"/cohere/v2/batches/{batch['id']}", lambda item: item["status"] in {"BATCH_STATUS_COMPLETED", "BATCH_STATUS_CANCELLED", "BATCH_STATUS_FAILED"})

    assert data["status"] == "BATCH_STATUS_COMPLETED"
    output = client.get(f"/cohere/v1/datasets/{data['output_dataset_id']}").json()
    assert output["rows"][0]["response"]["message"]["role"] == "assistant"


def test_xai_batch_add_requests_and_results(monkeypatch, client):
    import llmock.routers.batch as batch_mod

    monkeypatch.setattr(batch_mod, "_BATCH_DELAY", 0.05)
    batch = client.post("/xai/v1/batches", json={"name": "xai-batch"}).json()
    batch_id = batch["batch_id"]
    added = client.post(
        f"/xai/v1/batches/{batch_id}/requests",
        json={
            "requests": [
                {
                    "custom_id": "x-1",
                    "url": "/v1/chat/completions",
                    "body": {"model": "grok-3", "messages": [{"role": "user", "content": "Hello"}]},
                }
            ]
        },
    ).json()
    assert added["added_requests"] == 1

    data = _wait(client, f"/xai/v1/batches/{batch_id}", lambda item: item["state"]["status"] in {"COMPLETED", "FAILED", "CANCELLED"})
    assert data["state"]["status"] == "COMPLETED"
    results = client.get(f"/xai/v1/batches/{batch_id}/results").json()["data"]
    assert results[0]["custom_id"] == "x-1"
    assert results[0]["response"]["choices"][0]["message"]["role"] == "assistant"
