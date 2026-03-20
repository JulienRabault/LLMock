"""Tests for configurable success payloads and provider-specific error shapes."""

from fastapi.testclient import TestClient

from llmock.chaos import ChaosSettings
from llmock.main import create_app
from llmock.simulation import MockResponseSettings


def test_response_style_hello_returns_greeting():
    client = TestClient(create_app(responses=MockResponseSettings(response_style="hello")))

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Tell me something"}],
        },
    )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"].startswith("Hello!")


def test_response_style_echo_reflects_prompt():
    client = TestClient(create_app(responses=MockResponseSettings(response_style="echo")))

    resp = client.post(
        "/anthropic/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Echo this mock prompt"}],
        },
    )

    assert resp.status_code == 200
    text = resp.json()["content"][0]["text"]
    assert "Echo this mock prompt" in text


def test_force_status_header_uses_gemini_error_shape():
    client = TestClient(create_app())

    resp = client.get(
        "/gemini/v1beta/models",
        headers={"x-llmock-force-status": "429"},
    )

    assert resp.status_code == 429
    body = resp.json()
    assert body["error"]["status"] == "RESOURCE_EXHAUSTED"
    assert body["error"]["code"] == 429


def test_force_status_header_uses_cohere_error_shape():
    client = TestClient(create_app())

    resp = client.post(
        "/cohere/v2/chat",
        json={"model": "command-r", "messages": [{"role": "user", "content": "Hi"}]},
        headers={"x-llmock-force-status": "401"},
    )

    assert resp.status_code == 401
    body = resp.json()
    assert body["type"] == "authentication_error"
    assert body["message"]


def test_anthropic_supports_provider_specific_529_overload():
    client = TestClient(create_app(chaos=ChaosSettings(error_rates={529: 1.0})))

    resp = client.post(
        "/anthropic/v1/messages",
        json={
            "model": "claude-opus-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert resp.status_code == 529
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "overloaded_error"
