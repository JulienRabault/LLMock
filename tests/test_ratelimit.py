"""Quota-based rate limits, context windows, and random stream chaos."""

import time
from datetime import datetime

import httpx
import pytest
from fastapi.testclient import TestClient

from llmock.chaos import ChaosSettings, StreamChaos
from llmock.main import create_app
from llmock.ratelimit import LimitSettings, RateLimiter, requested_tokens
from llmock.testing import LLMockServer

CHAT = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
CLAUDE = {"model": "claude-sonnet-4-6", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}


def app_with(**limits):
    return create_app(chaos=ChaosSettings(), limits=LimitSettings(**limits))


# -- the bucket ---------------------------------------------------------------


def test_requests_beyond_rpm_are_refused_with_the_real_wait():
    limiter = RateLimiter(LimitSettings(rpm=2))
    assert limiter.admit("openai", "k", 1).allowed
    assert limiter.admit("openai", "k", 1).allowed
    refused = limiter.admit("openai", "k", 1)
    assert not refused.allowed and refused.exhausted == "requests"
    assert refused.retry_after == pytest.approx(30, abs=0.5)  # 2/min refills one every 30 s


def test_buckets_refill_over_time(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("llmock.ratelimit.time.monotonic", lambda: clock[0])
    limiter = RateLimiter(LimitSettings(rpm=60))
    for _ in range(60):
        assert limiter.admit("openai", "k", 1).allowed
    assert not limiter.admit("openai", "k", 1).allowed
    clock[0] += 1.0  # 60/min is one per second
    assert limiter.admit("openai", "k", 1).allowed


def test_keys_and_providers_have_separate_buckets():
    limiter = RateLimiter(LimitSettings(rpm=1))
    assert limiter.admit("openai", "key-a", 1).allowed
    assert limiter.admit("openai", "key-b", 1).allowed
    assert limiter.admit("anthropic", "key-a", 1).allowed
    assert not limiter.admit("openai", "key-a", 1).allowed


def test_token_budget_counts_the_requested_output():
    limiter = RateLimiter(LimitSettings(tpm=1000))
    refused = limiter.admit("openai", "k", 1500)
    assert not refused.allowed and refused.exhausted == "tokens"


def test_requested_tokens_include_max_tokens():
    assert requested_tokens({"max_tokens": 500}, raw_size=400) == 600
    assert requested_tokens({"generationConfig": {"maxOutputTokens": 64}}, raw_size=40) == 74


@pytest.mark.parametrize("bad", [{"rpm": 0}, {"tpm": -5}, {"context_window": 0}])
def test_limits_must_be_positive(bad):
    with pytest.raises(ValueError):
        LimitSettings(**bad).validated()


# -- over HTTP ----------------------------------------------------------------


def test_openai_style_headers_on_every_response():
    client = TestClient(app_with(rpm=10, tpm=100_000))
    first = client.post("/v1/chat/completions", json=CHAT)
    second = client.post("/v1/chat/completions", json=CHAT)
    assert first.headers["x-ratelimit-limit-requests"] == "10"
    assert first.headers["x-ratelimit-remaining-requests"] == "9"
    assert second.headers["x-ratelimit-remaining-requests"] == "8"
    assert first.headers["x-ratelimit-reset-requests"].endswith(("ms", "s"))
    assert int(first.headers["x-ratelimit-remaining-tokens"]) < 100_000


def test_anthropic_style_headers_use_rfc3339_resets():
    response = TestClient(app_with(rpm=10)).post("/anthropic/v1/messages", json=CLAUDE)
    assert response.headers["anthropic-ratelimit-requests-remaining"] == "9"
    reset = response.headers["anthropic-ratelimit-requests-reset"]
    datetime.fromisoformat(reset.replace("Z", "+00:00"))


def test_exhausted_quota_answers_429_with_retry_after():
    client = TestClient(app_with(rpm=1))
    client.post("/v1/chat/completions", json=CHAT)
    refused = client.post("/v1/chat/completions", json=CHAT)
    assert refused.status_code == 429
    assert refused.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(refused.headers["retry-after"]) == 60
    assert refused.headers["x-ratelimit-remaining-requests"] == "0"


def test_quota_refusals_are_journaled_and_judged():
    app = app_with(rpm=1)
    client = TestClient(app)
    client.post("/v1/chat/completions", json=CHAT)
    client.post("/v1/chat/completions", json=CHAT)
    record = app.state.llmock.journal.records()[-1]
    assert record.fault == "ratelimit:requests" and record.retry_after == pytest.approx(60, abs=1)


def test_no_limit_means_no_headers():
    response = TestClient(create_app(chaos=ChaosSettings())).post("/v1/chat/completions", json=CHAT)
    assert not any(h.startswith("x-ratelimit") for h in response.headers)


# -- context window -----------------------------------------------------------

LONG = {"model": "gpt-4o", "messages": [{"role": "user", "content": "word " * 400}]}


def test_a_prompt_over_the_window_gets_context_length_exceeded():
    response = TestClient(app_with(context_window=100)).post("/v1/chat/completions", json=LONG)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert "maximum context length is 100 tokens" in error["message"]


def test_anthropic_says_prompt_is_too_long():
    body = {**CLAUDE, "messages": LONG["messages"]}
    response = TestClient(app_with(context_window=100)).post("/anthropic/v1/messages", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["message"].startswith("prompt is too long")


def test_prompts_within_the_window_pass():
    assert TestClient(app_with(context_window=100)).post("/v1/chat/completions", json=CHAT).status_code == 200


# -- random stream chaos --------------------------------------------------------


def test_stream_chaos_breaks_every_stream_at_rate_one():
    with LLMockServer() as server:
        server.state.stream_chaos = StreamChaos(fault_rates=(("truncate", 1.0),))
        url = server.base_url("openai") + "/chat/completions"
        for _ in range(3):
            with httpx.stream("POST", url, json={**CHAT, "stream": True}) as response:
                lines = [line for line in response.iter_lines() if line]
            assert "data: [DONE]" not in lines
        assert httpx.post(url, json=CHAT).status_code == 200  # non-streaming untouched
        faults = [r.fault for r in server.state.journal.records(wait=2)]
        assert all(f.startswith("stream:truncate@") for f in faults[:3]) and faults[3] is None


def test_chunk_delay_paces_the_stream():
    with LLMockServer() as server:
        server.state.stream_chaos = StreamChaos(chunk_delay_ms=50)
        started = time.monotonic()
        with httpx.stream("POST", server.base_url("openai") + "/chat/completions",
                          json={**CHAT, "stream": True}) as response:
            count = sum(1 for line in response.iter_lines() if line)
        assert time.monotonic() - started >= (count - 1) * 0.05 * 0.8


def test_scripted_stream_faults_win_over_random_ones():
    with LLMockServer() as server:
        from llmock.scenarios import StreamFault

        server.state.stream_chaos = StreamChaos(fault_rates=(("truncate", 1.0),))
        server.state.scenarios.add(StreamFault("disconnect", after_chunks=2))
        with pytest.raises(httpx.RemoteProtocolError):
            with httpx.stream("POST", server.base_url("openai") + "/chat/completions",
                              json={**CHAT, "stream": True}) as response:
                list(response.iter_lines())


def test_stream_chaos_validation():
    with pytest.raises(ValueError):
        StreamChaos(fault_rates=(("truncate", 0.7), ("disconnect", 0.7))).validated()


def test_a_fresh_bucket_is_full_even_if_the_clock_ticks_finely(monkeypatch):
    """Regression: on Linux the clock advances between calls, and the bucket
    used to read its creation time *after* `now`, starting slightly drained --
    so the very first request of an rpm=1 limit was refused."""
    ticks = iter(range(1, 1000))
    monkeypatch.setattr("llmock.ratelimit.time.monotonic", lambda: next(ticks) * 1e-9)
    limiter = RateLimiter(LimitSettings(rpm=1, tpm=1000))
    assert limiter.admit("openai", "k", 500).allowed
    assert not limiter.admit("openai", "k", 1).allowed


def test_the_bucket_ignores_a_clock_that_goes_backwards():
    from llmock.ratelimit import _Bucket

    bucket = _Bucket(10, 60, now=100.0)
    bucket.level = 4.0
    bucket.refill(now=99.0)
    assert bucket.level == 4.0
