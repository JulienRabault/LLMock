"""Tests for provider-specific simulated error payloads."""

from fastapi.testclient import TestClient

from llmock.chaos import ChaosSettings
from llmock.main import create_app


def test_openai_error_shape():
    client = TestClient(create_app(chaos=ChaosSettings(error_rate_429=1.0)))
    response = client.get("/v1/models")
    assert response.status_code == 429
    body = response.json()
    assert body["error"]["type"] == "rate_limit_error"
    assert body["error"]["code"] == "rate_limit_exceeded"


def test_anthropic_error_shape():
    client = TestClient(create_app(chaos=ChaosSettings(error_rate_429=1.0)))
    response = client.get("/anthropic/v1/models")
    assert response.status_code == 429
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "rate_limit_error"


def test_cohere_error_shape():
    client = TestClient(create_app(chaos=ChaosSettings(error_rates={401: 1.0})))
    response = client.get("/cohere/v2/models")
    assert response.status_code == 401
    body = response.json()
    assert body["type"] == "authentication_error"
    assert "message" in body


def test_gemini_error_shape():
    client = TestClient(create_app(chaos=ChaosSettings(error_rate_429=1.0)))
    response = client.get("/gemini/v1beta/models")
    assert response.status_code == 429
    body = response.json()
    assert body["error"]["status"] == "RESOURCE_EXHAUSTED"
    assert body["error"]["code"] == 429
