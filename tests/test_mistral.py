"""Tests for Mistral-compatible endpoints."""

import json

import pytest
from fastapi.testclient import TestClient

from llmock.main import create_app
from llmock.scenarios import Reply, ToolCall


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


def test_mistral_streaming_tool_calls_arrive_whole(client):
    app = client.app
    app.state.llmock.scenarios.add(
        Reply(
            tool_calls=(
                ToolCall("calculator", {"expression": "12 * 7"}),
                ToolCall("calculator", {"expression": "5 * 9"}),
            )
        )
    )
    resp = client.post(
        "/mistral/v1/chat/completions",
        json={
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": "Calculate"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    events = []
    for line in resp.text.split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[len("data: ") :]))

    tool_call_deltas = [
        event["choices"][0]["delta"]["tool_calls"]
        for event in events
        if "tool_calls" in event["choices"][0].get("delta", {})
    ]
    assert len(tool_call_deltas) == 1, "Mistral tool calls should arrive whole in a single chunk"
    calls = tool_call_deltas[0]
    assert len(calls) == 2
    assert calls[0]["index"] == 0
    assert calls[0]["id"]
    assert calls[0]["function"]["name"] == "calculator"
    assert json.loads(calls[0]["function"]["arguments"]) == {"expression": "12 * 7"}
    assert calls[1]["index"] == 1
    assert calls[1]["id"]
    assert calls[1]["function"]["name"] == "calculator"
    assert json.loads(calls[1]["function"]["arguments"]) == {"expression": "5 * 9"}
