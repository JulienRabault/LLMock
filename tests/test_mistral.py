"""Tests for Mistral-compatible endpoints."""

import pytest
from fastapi.testclient import TestClient

from llmock.main import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_list_models(client):
    resp = client.get("/mistral/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert any(m["id"] == "mistral-large-latest" for m in data["data"])


def test_chat_completions(client):
    resp = client.post(
        "/mistral/v1/chat/completions",
        json={
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": "Hello!"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "mistral-large-latest"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] >= 0


def test_chat_completions_n_choices(client):
    resp = client.post(
        "/mistral/v1/chat/completions",
        json={
            "model": "open-mistral-7b",
            "messages": [{"role": "user", "content": "Hi"}],
            "n": 3,
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["choices"]) == 3
