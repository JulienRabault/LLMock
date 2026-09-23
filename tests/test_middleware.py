"""The middleware: scripted errors, journaling, and the admin API."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from llmock.chaos import ChaosSettings
from llmock.journal import Journal, RequestRecord, fingerprint
from llmock.main import create_app
from llmock.scenarios import Delay, Fail, Match

CHAT = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
CLAUDE = {
    "model": "claude-sonnet-4",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Hi"}],
}


@pytest.fixture
def app():
    return create_app(chaos=ChaosSettings())


@pytest.fixture
def client(app):
    return TestClient(app)


def journal(app):
    return app.state.llmock.journal.records()


# -- scripted errors ----------------------------------------------------------


def test_scripted_fail_answers_with_the_error(app, client):
    app.state.llmock.scenarios.add(Fail(429))
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 429
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 200


def test_retry_after_is_sent_in_seconds_and_milliseconds(app, client):
    app.state.llmock.scenarios.add(Fail(429, retry_after=2.5))
    response = client.post("/v1/chat/completions", json=CHAT)
    assert response.headers["retry-after"] == "3"  # HTTP wants whole seconds
    assert response.headers["retry-after-ms"] == "2500"


def test_retryable_errors_default_to_one_second(app, client):
    app.state.llmock.scenarios.add(Fail(503))
    response = client.post("/v1/chat/completions", json=CHAT)
    assert response.headers["retry-after-ms"] == "1000"


def test_non_retryable_errors_carry_no_retry_after(app, client):
    app.state.llmock.scenarios.add(Fail(400))
    response = client.post("/v1/chat/completions", json=CHAT)
    assert "retry-after" not in response.headers


def test_scripted_fail_uses_the_provider_envelope(app, client):
    app.state.llmock.scenarios.add(Fail(529))
    response = client.post("/anthropic/v1/messages", json=CLAUDE)
    assert response.status_code == 529
    assert response.json() == {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Service overloaded."},
    }


def test_scripted_fail_can_override_message_and_code(app, client):
    app.state.llmock.scenarios.add(
        Fail(400, message="This model's maximum context length is 128000 tokens.",
             code="context_length_exceeded")
    )
    error = client.post("/v1/chat/completions", json=CHAT).json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert error["message"].startswith("This model's maximum context length")


def test_matching_is_applied_by_the_middleware(app, client):
    app.state.llmock.scenarios.add(Fail(529, match=Match(provider="anthropic")))
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 200
    assert client.post("/anthropic/v1/messages", json=CLAUDE).status_code == 529


def test_gemini_model_is_read_from_the_path(app, client):
    app.state.llmock.scenarios.add(Fail(503, match=Match(model="gemini-2.5-pro")))
    body = {"contents": [{"parts": [{"text": "Hi"}]}]}
    response = client.post("/gemini/v1beta/models/gemini-2.5-pro:generateContent", json=body)
    assert response.status_code == 503


def test_delay_holds_the_response(app, client):
    app.state.llmock.scenarios.add(Delay(0.3))
    start = time.monotonic()
    client.post("/v1/chat/completions", json=CHAT)
    # Windows timers tick every ~15.6 ms; allow a hair of slack.
    assert time.monotonic() - start >= 0.3 - 0.05


def test_forced_header_still_wins(app, client):
    app.state.llmock.scenarios.add(Fail(429))
    response = client.post(
        "/v1/chat/completions", json=CHAT, headers={"x-llmock-force-status": "503"}
    )
    assert response.status_code == 503
    # The forced status pre-empted the queue, so the scripted 429 still waits.
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 429


# -- journal ------------------------------------------------------------------


def test_every_request_is_journaled(app, client):
    client.post("/v1/chat/completions", json=CHAT)
    (record,) = journal(app)
    assert record.provider == "openai"
    assert record.model == "gpt-4o"
    assert record.status == 200
    assert record.completed
    assert record.fault is None
    assert record.body == CHAT


def test_journal_labels_where_an_error_came_from(app, client):
    app.state.llmock.scenarios.add(Fail(429, retry_after=2))
    client.post("/v1/chat/completions", json=CHAT)
    client.post("/v1/chat/completions", json=CHAT, headers={"x-llmock-force-status": "503"})
    app.state.chaos_settings.error_rate_500 = 1.0
    client.post("/v1/chat/completions", json=CHAT)
    faults = [(r.status, r.fault, r.retry_after) for r in journal(app)]
    assert faults == [(429, "scenario:429", 2.0), (503, "forced:503", 1.0), (500, "chaos:500", None)]


def test_retries_of_one_call_share_a_fingerprint(app, client):
    app.state.llmock.scenarios.add(Fail(429))
    client.post("/v1/chat/completions", json=CHAT)
    client.post("/v1/chat/completions", json=CHAT)
    client.post("/v1/chat/completions", json={**CHAT, "model": "gpt-4o-mini"})
    first, retry, other = journal(app)
    assert first.fingerprint == retry.fingerprint
    assert other.fingerprint != first.fingerprint


def test_sdk_retry_count_header_is_recorded(app, client):
    client.post("/v1/chat/completions", json=CHAT, headers={"x-stainless-retry-count": "2"})
    assert journal(app)[0].sdk_retry_count == 2


def test_records_are_numbered_in_arrival_order(app, client):
    for _ in range(3):
        client.post("/v1/chat/completions", json=CHAT)
    assert [r.seq for r in journal(app)] == [1, 2, 3]


@pytest.mark.parametrize("path", ["/health", "/_llmock/requests", "/openapi.json"])
def test_housekeeping_routes_are_not_journaled(app, client, path):
    client.get(path)
    assert journal(app) == []


def test_housekeeping_routes_bypass_chaos(app, client):
    app.state.chaos_settings.error_rate_500 = 1.0
    assert client.get("/health").status_code == 200
    assert client.get("/_llmock/requests").status_code == 200


def test_fingerprint_ignores_key_order():
    assert fingerprint("POST", "/v1/x", {"a": 1, "b": 2}) == fingerprint("post", "/v1/x", {"b": 2, "a": 1})


def test_journal_is_bounded():
    j = Journal(capacity=3)
    for seq in range(1, 6):
        j.add(RequestRecord(seq=seq, provider="openai", method="POST", path="/v1/x",
                            started_at=0.0, ended_at=0.1, status=200, fingerprint="f"))
    assert [r.seq for r in j.records()] == [3, 4, 5]


# -- admin API ----------------------------------------------------------------


def test_admin_queues_a_scenario_over_http(client):
    response = client.post(
        "/_llmock/scenario",
        json={"behaviors": [{"type": "fail", "status": 503, "times": 2, "retry_after": 0.5}]},
    )
    assert response.status_code == 201
    assert [client.post("/v1/chat/completions", json=CHAT).status_code for _ in range(3)] == [
        503, 503, 200,
    ]


def test_admin_accepts_match_and_permanent_behaviours(client):
    client.post(
        "/_llmock/scenario",
        json={"behaviors": [
            {"type": "fail", "status": 529, "times": None, "match": {"provider": "anthropic"}}
        ]},
    )
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 200
    assert all(
        client.post("/anthropic/v1/messages", json=CLAUDE).status_code == 529 for _ in range(5)
    )


def test_admin_lists_and_clears_pending_behaviours(client):
    client.post("/_llmock/scenario", json={"behaviors": [
        {"type": "stream_fault", "kind": "disconnect", "after_chunks": 3},
        {"type": "reply", "tool_calls": [{"name": "search", "arguments": {"q": "llmock"}}]},
    ]})
    pending = client.get("/_llmock/scenario").json()["pending"]
    assert [b["type"] for b in pending] == ["stream_fault", "reply"]
    assert pending[0]["match"] == {"stream": True}
    assert pending[1]["tool_calls"] == [{"name": "search", "arguments": {"q": "llmock"}}]
    client.delete("/_llmock/scenario")
    assert client.get("/_llmock/scenario").json()["pending"] == []


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({}, "behaviors"),
        ({"behaviors": []}, "behaviors"),
        ({"behaviors": [{"type": "explode"}]}, "Unknown behaviour type"),
        ({"behaviors": [{"type": "fail", "status": 200}]}, "400-599"),
        ({"behaviors": [{"type": "fail", "status": 429, "colour": "red"}]}, "colour"),
        ({"behaviors": [{"type": "fail", "status": 429, "match": {"planet": "mars"}}]}, "planet"),
        ({"behaviors": [{"type": "stream_fault", "kind": "explode"}]}, "explode"),
    ],
)
def test_admin_rejects_invalid_scenarios_with_a_clear_message(client, payload, fragment):
    response = client.post("/_llmock/scenario", json=payload)
    assert response.status_code == 400
    assert fragment in response.json()["error"]["message"]


def test_admin_exposes_the_journal(client):
    client.post("/v1/chat/completions", json=CHAT)
    data = client.get("/_llmock/requests").json()
    assert data["count"] == 1
    assert data["requests"][0]["model"] == "gpt-4o"
    assert "duration" in data["requests"][0]


def test_admin_reset_clears_journal_and_queue(client):
    client.post("/_llmock/scenario", json={"behaviors": [{"type": "fail", "status": 429}]})
    client.get("/v1/models")
    client.post("/_llmock/reset")
    assert client.get("/_llmock/requests").json()["count"] == 0
    assert client.get("/_llmock/scenario").json()["pending"] == []


def test_admin_exposes_the_verdict(client):
    client.post("/_llmock/scenario", json={"behaviors": [{"type": "fail", "status": 401}]})
    for _ in range(2):
        client.post("/v1/chat/completions", json=CHAT)
    data = client.get("/_llmock/verdict").json()
    assert data["passed"] is False
    assert data["findings"][0]["code"] == "retried_non_retryable"
    text = client.get("/_llmock/verdict?format=text").text
    assert "retried_non_retryable" in text and "FAIL" in text


def test_admin_accepts_json_without_a_json_content_type(client):
    """`curl -d '{...}'` sends form encoding; the README example must still work."""
    response = client.post(
        "/_llmock/scenario",
        content=b'{"behaviors": [{"type": "fail", "status": 429}]}',
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 201
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 429


def test_admin_rejects_a_body_that_is_not_json(client):
    response = client.post("/_llmock/scenario", content=b"not json")
    assert response.status_code == 400


async def test_a_client_that_leaves_mid_upload_is_not_journaled():
    """Found in code review: a disconnect while sending the body used to be served."""
    from llmock.middleware import LLMockMiddleware

    app = create_app(chaos=ChaosSettings())
    state = app.state.llmock
    reached = []

    async def downstream(scope, receive, send):
        reached.append(True)

    middleware = LLMockMiddleware(downstream, state)
    messages = iter([
        {"type": "http.request", "body": b'{"model": "gpt', "more_body": True},
        {"type": "http.disconnect"},
    ])

    async def receive():
        return next(messages)

    async def send(message):
        raise AssertionError("nothing should be sent")

    scope = {"type": "http", "path": "/v1/chat/completions", "method": "POST",
             "headers": [(b"content-type", b"application/json")]}
    await middleware(scope, receive, send)
    assert reached == [] and state.journal.records() == []
