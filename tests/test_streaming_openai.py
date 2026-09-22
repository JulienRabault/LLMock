"""Streaming through the real OpenAI SDK, against a real server.

Faults are transport-level, so they are only meaningful over a real socket:
these tests use :class:`llmock.testing.LLMockServer`, not an ASGI transport.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from llmock.scenarios import Fail, Reply, SlowFirstToken, StreamFault, ToolCall
from llmock.testing import LLMockServer

openai = pytest.importorskip("openai")

PROMPT = [{"role": "user", "content": "Tell me a short story"}]


@pytest.fixture(scope="module")
def server():
    with LLMockServer() as running:
        yield running


@pytest.fixture(autouse=True)
def clean(server):
    server.state.reset()
    yield


def client(server, **kwargs):
    kwargs.setdefault("max_retries", 0)
    kwargs.setdefault("timeout", 10.0)
    return openai.OpenAI(base_url=server.base_url("openai"), api_key="test", **kwargs)


def collect(stream):
    """Concatenate streamed text; return (text, finish_reason, usage)."""
    text, finish, usage = [], None, None
    with stream:
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            for choice in chunk.choices:
                if choice.delta.content:
                    text.append(choice.delta.content)
                if choice.finish_reason:
                    finish = choice.finish_reason
    return "".join(text), finish, usage


# -- a healthy stream ---------------------------------------------------------


def test_streamed_text_matches_the_non_streamed_answer(server):
    c = client(server)
    whole = c.chat.completions.create(model="gpt-4o", messages=PROMPT)
    text, finish, _ = collect(c.chat.completions.create(model="gpt-4o", messages=PROMPT, stream=True))
    assert text == whole.choices[0].message.content
    assert finish == "stop"


def test_usage_arrives_in_a_final_chunk_when_requested(server):
    _, _, usage = collect(client(server).chat.completions.create(
        model="gpt-4o", messages=PROMPT, stream=True, stream_options={"include_usage": True},
    ))
    assert usage is not None
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


def test_wire_format_is_sse_terminated_by_done(server):
    with httpx.stream(
        "POST", server.base_url("openai") + "/chat/completions",
        json={"model": "gpt-4o", "messages": PROMPT, "stream": True},
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [line for line in response.iter_lines() if line]
    assert all(line.startswith("data: ") for line in events)
    assert events[-1] == "data: [DONE]"
    first = json.loads(events[0][len("data: "):])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"


def test_scripted_reply_is_streamed_verbatim(server):
    server.state.scenarios.add(Reply(text="Once upon a time, a mock server saved the day."))
    text, finish, _ = collect(
        client(server).chat.completions.create(model="gpt-4o", messages=PROMPT, stream=True)
    )
    assert text == "Once upon a time, a mock server saved the day."
    assert finish == "stop"


def test_scripted_tool_call_is_streamed_in_fragments_and_reassembles(server):
    arguments = {"city": "Paris", "unit": "celsius", "days": 3}
    server.state.scenarios.add(Reply(tool_calls=(ToolCall("get_forecast", arguments),)))
    name, pieces, finish = None, [], None
    with client(server).chat.completions.create(
        model="gpt-4o", messages=PROMPT, stream=True
    ) as stream:
        for chunk in stream:
            choice = chunk.choices[0]
            for call in choice.delta.tool_calls or []:
                name = call.function.name or name
                if call.function.arguments:
                    pieces.append(call.function.arguments)
            finish = choice.finish_reason or finish
    assert name == "get_forecast"
    assert len(pieces) > 1, "arguments should arrive in several fragments"
    assert json.loads("".join(pieces)) == arguments
    assert finish == "tool_calls"


# -- faults -------------------------------------------------------------------


def test_disconnect_surfaces_as_a_connection_error(server):
    server.state.scenarios.add(StreamFault("disconnect", after_chunks=3))
    with pytest.raises(openai.APIConnectionError):
        collect(client(server).chat.completions.create(model="gpt-4o", messages=PROMPT, stream=True))
    (record,) = server.state.journal.records(wait=2)
    assert record.fault == "stream:disconnect@3"
    assert not record.completed
    assert record.chunks_sent == 3


def test_truncation_is_silent_for_the_client(server):
    """The dangerous case: no exception, just a partial answer and no finish reason."""
    full, _, _ = collect(
        client(server).chat.completions.create(model="gpt-4o", messages=PROMPT, stream=True)
    )
    server.state.reset()
    server.state.scenarios.add(StreamFault("truncate", after_chunks=3))
    partial, finish, _ = collect(
        client(server).chat.completions.create(model="gpt-4o", messages=PROMPT, stream=True)
    )
    assert finish is None
    assert partial and full.startswith(partial) and partial != full
    (record,) = server.state.journal.records(wait=2)
    assert record.fault == "stream:truncate@3"
    assert not record.completed


def test_malformed_chunk_escapes_as_a_raw_json_error(server):
    """Worth knowing: the SDK does not wrap this in an openai exception."""
    server.state.scenarios.add(StreamFault("malformed", after_chunks=2))
    with pytest.raises(json.JSONDecodeError):
        collect(client(server).chat.completions.create(model="gpt-4o", messages=PROMPT, stream=True))
    assert server.state.journal.records(wait=2)[0].fault == "stream:malformed@2"


def test_stall_trips_the_client_read_timeout(server):
    server.state.scenarios.add(StreamFault("stall", after_chunks=2, stall_seconds=5))
    started = time.monotonic()
    with pytest.raises(openai.APITimeoutError):
        collect(client(server, timeout=0.5).chat.completions.create(
            model="gpt-4o", messages=PROMPT, stream=True
        ))
    assert time.monotonic() - started < 3, "the client, not the stall, should end the wait"


def test_slow_first_token_delays_only_the_start(server):
    server.state.scenarios.add(SlowFirstToken(0.5))
    started = time.monotonic()
    first_at = None
    with client(server).chat.completions.create(model="gpt-4o", messages=PROMPT, stream=True) as s:
        for _ in s:
            first_at = first_at or time.monotonic() - started
    assert first_at >= 0.5
    assert time.monotonic() - started < first_at + 0.5


def test_fault_position_is_exact(server):
    server.state.scenarios.add(StreamFault("truncate", after_chunks=4))
    with httpx.stream(
        "POST", server.base_url("openai") + "/chat/completions",
        json={"model": "gpt-4o", "messages": PROMPT, "stream": True},
    ) as response:
        events = [line for line in response.iter_lines() if line]
    assert len(events) == 4
    assert "[DONE]" not in events[-1]


def test_scripted_error_on_a_streaming_request(server):
    server.state.scenarios.add(Fail(429, retry_after=0.1))
    with pytest.raises(openai.RateLimitError):
        collect(client(server).chat.completions.create(model="gpt-4o", messages=PROMPT, stream=True))


def test_the_sdk_retries_a_503_and_the_journal_links_the_attempts(server):
    """With retries on, a transient 503 is invisible to the caller."""
    server.state.scenarios.add(Fail(503, retry_after=0.05))
    text, finish, _ = collect(
        client(server, max_retries=2).chat.completions.create(
            model="gpt-4o", messages=PROMPT, stream=True
        )
    )
    assert finish == "stop" and text
    first, retry = server.state.journal.records(wait=2)
    assert first.status == 503 and retry.status == 200
    assert first.fingerprint == retry.fingerprint
    assert retry.sdk_retry_count == 1
