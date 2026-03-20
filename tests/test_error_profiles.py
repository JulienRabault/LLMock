"""Tests for provider-specific error envelopes and configurable success styles."""

from fastapi.testclient import TestClient

from llmock.chaos import ChaosSettings
from llmock.main import create_app
from llmock.simulation import MockResponseSettings


def test_generic_error_rate_configuration_injects_401():
    client = TestClient(create_app(chaos=ChaosSettings(error_rates={401: 1.0})))

    response = client.get("/v1/models")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_openai_family_forced_status_uses_openai_error_shape():
    client = TestClient(create_app())

    response = client.get("/v1/models", headers={"x-llmock-force-status": "404"})

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "not_found"


def test_anthropic_forced_status_uses_anthropic_error_shape():
    client = TestClient(create_app())

    response = client.post(
        "/anthropic/v1/messages",
        headers={"x-llmock-force-status": "429"},
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 429
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "rate_limit_error"


def test_gemini_forced_status_uses_google_error_shape():
    client = TestClient(create_app())

    response = client.get("/gemini/v1beta/models", headers={"x-llmock-force-status": "503"})

    assert response.status_code == 503
    assert response.json()["error"]["status"] == "UNAVAILABLE"


def test_cohere_forced_status_uses_cohere_error_shape():
    client = TestClient(create_app())

    response = client.post(
        "/cohere/v2/chat",
        headers={"x-llmock-force-status": "422"},
        json={"model": "command-r-plus", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["type"] == "unprocessable_entity_error"
    assert body["message"]


def test_echo_response_style_echoes_prompt_text():
    client = TestClient(create_app(responses=MockResponseSettings(response_style="echo")))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Please echo this prompt"}],
        },
    )

    assert response.status_code == 200
    assert "Please echo this prompt" in response.json()["choices"][0]["message"]["content"]


def test_varied_response_style_changes_success_payload_by_prompt():
    client = TestClient(create_app(responses=MockResponseSettings(response_style="varied")))

    first = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "First prompt"}]},
    )
    second = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Second prompt"}]},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["choices"][0]["message"]["content"] != second.json()["choices"][0]["message"]["content"]
