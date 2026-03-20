"""Tests for configurable mock success payloads."""

from fastapi.testclient import TestClient

from llmock.main import create_app
from llmock.simulation import MockResponseSettings


def test_hello_response_style():
    client = TestClient(create_app(responses=MockResponseSettings(response_style="hello")))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"].startswith("Hello!")


def test_echo_response_style():
    client = TestClient(create_app(responses=MockResponseSettings(response_style="echo")))
    response = client.post(
        "/xai/v1/chat/completions",
        json={"model": "grok-3", "messages": [{"role": "user", "content": "Ping test"}]},
    )
    assert response.status_code == 200
    assert "Ping test" in response.json()["choices"][0]["message"]["content"]
