"""Tests for Anthropic (Claude) compatible endpoints."""

import pytest
from fastapi.testclient import TestClient

from llmock.main import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_list_models(client):
    resp = client.get("/anthropic/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["data"], list)
    assert any(m["id"] == "claude-opus-4-6" for m in data["data"])


def test_create_message(client):
    resp = client.post(
        "/anthropic/v1/messages",
        json={
            "model": "claude-opus-4-6",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello!"}],
        },
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert len(data["content"]) == 1
    assert data["content"][0]["type"] == "text"
    assert data["stop_reason"] == "end_turn"
    assert "input_tokens" in data["usage"]
    assert "output_tokens" in data["usage"]


def test_create_message_with_system(client):
    resp = client.post(
        "/anthropic/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Say hi"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "claude-sonnet-4-6"
    assert data["usage"]["input_tokens"] > 0


def test_message_id_format(client):
    resp = client.post(
        "/anthropic/v1/messages",
        json={
            "model": "claude-opus-4-6",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Test"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["id"].startswith("msg_")
