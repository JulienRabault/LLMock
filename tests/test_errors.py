"""Tests for provider-specific error formats."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llmock.main import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# OpenAI error format (all /v1/* routes)
# ---------------------------------------------------------------------------

def test_openai_404_error_format(client):
    """Unknown model or resource returns OpenAI error envelope."""
    resp = client.get("/v1/batches/batch_nonexistent")
    assert resp.status_code == 404
    body = resp.json()
    assert "error" in body
    assert "message" in body["error"]
    assert "type" in body["error"]


def test_openai_400_unknown_file(client):
    resp = client.post(
        "/v1/batches",
        json={"input_file_id": "file-doesnotexist", "endpoint": "/v1/chat/completions", "completion_window": "24h"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert "error" in body
    assert "message" in body["error"]
    assert "file-doesnotexist" in body["error"]["message"]


def test_openai_422_validation_error(client):
    """Missing required field returns 422 in OpenAI format."""
    resp = client.post("/v1/chat/completions", json={"messages": []})  # missing model
    assert resp.status_code == 422
    body = resp.json()
    assert "error" in body
    assert "message" in body["error"]
    assert body["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Anthropic error format (/anthropic/v1/*)
# ---------------------------------------------------------------------------

def test_anthropic_404_error_format(client):
    resp = client.get("/anthropic/v1/messages/batches/batch_nonexistent")
    assert resp.status_code == 404
    body = resp.json()
    assert body.get("type") == "error"
    assert "error" in body
    assert "type" in body["error"]
    assert "message" in body["error"]


def test_anthropic_422_validation_error(client):
    resp = client.post(
        "/anthropic/v1/messages",
        json={"messages": [{"role": "user", "content": "hi"}]},  # missing model, max_tokens
        headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body.get("type") == "error"
    assert body["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Gemini error format (/gemini/v1beta/*)
# ---------------------------------------------------------------------------

def test_gemini_404_error_format(client):
    resp = client.get("/gemini/v1beta/batches/nonexistent")
    assert resp.status_code == 404
    body = resp.json()
    assert "error" in body
    assert "code" in body["error"]
    assert "message" in body["error"]
    assert "status" in body["error"]
    assert body["error"]["status"] == "NOT_FOUND"


def test_gemini_422_validation_error(client):
    resp = client.post("/gemini/v1beta/models/gemini-2.0-flash:batchGenerateContent", json=None)
    # FastAPI rejects non-dict body with 422 → Gemini maps 422 to FAILED_PRECONDITION
    assert resp.status_code == 422
    body = resp.json()
    assert "error" in body
    assert body["error"]["code"] == 422
    assert body["error"]["status"] == "FAILED_PRECONDITION"


# ---------------------------------------------------------------------------
# Mistral error format (/mistral/v1/*) — same as OpenAI
# ---------------------------------------------------------------------------

def test_mistral_404_error_format(client):
    """Mistral uses a flat error shape: {object, message, type, param, code}."""
    resp = client.get("/mistral/v1/batch/jobs/nonexistent")
    assert resp.status_code == 404
    body = resp.json()
    assert body.get("object") == "error"
    assert "message" in body
    assert "type" in body
    assert "code" in body


# ---------------------------------------------------------------------------
# Cohere error format (/cohere/v2/*)
# ---------------------------------------------------------------------------

def test_cohere_400_error_format(client):
    """Cohere uses flat {"message": "..."} for errors. Invalid dataset → 404."""
    resp = client.post(
        "/cohere/v2/batches",
        json={"input_dataset_id": "unknown-dataset", "model": "command-r-plus", "endpoint": "/v2/chat"},
    )
    # cohere.py raises 404 (not found) when dataset_id is unknown
    assert resp.status_code == 404
    body = resp.json()
    assert "message" in body
    assert "error" not in body  # Cohere uses flat {"message": "..."}


def test_cohere_404_error_format(client):
    resp = client.get("/cohere/v2/batches/nonexistent")
    assert resp.status_code == 404
    body = resp.json()
    assert "message" in body


# ---------------------------------------------------------------------------
# Groq and Together (OpenAI-compatible) streaming
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Error message content — not just shape
# ---------------------------------------------------------------------------

def test_error_message_reflects_detail(client):
    """The actual error detail string flows through to the response message."""
    resp = client.post(
        "/v1/batches",
        json={"input_file_id": "file-MYSPECIFICID", "endpoint": "/v1/chat/completions", "completion_window": "24h"},
    )
    assert resp.status_code == 400
    assert "MYSPECIFICID" in resp.json()["error"]["message"]


def test_anthropic_error_message_reflects_detail(client):
    resp = client.get("/anthropic/v1/messages/batches/MYSPECIFICBATCHID")
    assert resp.status_code == 404
    body = resp.json()
    assert "MYSPECIFICBATCHID" in body["error"]["message"]
