"""Tests for OpenAI-compatible endpoints."""

import pytest
from fastapi.testclient import TestClient

from llmock.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_models():
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert any(m["id"] == "gpt-4o" for m in body["data"])


def test_chat_completions():
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello!"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "usage" in body


def test_embeddings():
    r = client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "Hello world"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert len(body["data"][0]["embedding"]) == 1536


def test_image_generation_url():
    r = client.post(
        "/v1/images/generations",
        json={"prompt": "A tiny mock sunset", "n": 1, "response_format": "url"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 1
    assert body["data"][0]["url"].startswith("data:image/svg+xml;base64,")


def test_image_generation_b64():
    r = client.post(
        "/v1/images/generations",
        json={"prompt": "A tiny mock sunset", "n": 2, "response_format": "b64_json"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 2
    assert body["data"][0]["b64_json"]
