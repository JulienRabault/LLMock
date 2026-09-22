"""Streaming on every provider, and stream faults on every provider.

Faults are applied by the middleware on the transport, so a single mechanism
must work for all four wire formats. The last section pins down how each real
SDK reacts to a broken stream -- which is exactly what users need to know.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from llmock.scenarios import Reply, StreamFault, ToolCall
from llmock.testing import LLMockServer

CHAT = [{"role": "user", "content": "Tell me a short story"}]


@pytest.fixture(scope="module")
def server():
    with LLMockServer() as running:
        yield running


@pytest.fixture(autouse=True)
def clean(server):
    server.state.reset()
    yield


# (provider, path under its base URL, request body, how to pull text out of an event)
def _openai_text(event):
    return "".join(c["delta"].get("content") or "" for c in event.get("choices", []))


def _anthropic_text(event):
    delta = event.get("delta") or {}
    return delta.get("text", "") if event.get("type") == "content_block_delta" else ""


def _gemini_text(event):
    parts = event["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def _cohere_text(event):
    if event.get("type") != "content-delta":
        return ""
    return event["delta"]["message"]["content"]["text"]


OPENAI_BODY = {"model": "m", "messages": CHAT}
PROVIDERS = [
    ("openai", "/chat/completions", OPENAI_BODY, _openai_text),
    ("groq", "/chat/completions", OPENAI_BODY, _openai_text),
    ("together", "/chat/completions", OPENAI_BODY, _openai_text),
    ("perplexity", "/chat/completions", OPENAI_BODY, _openai_text),
    ("xai", "/chat/completions", OPENAI_BODY, _openai_text),
    ("mistral", "/chat/completions", OPENAI_BODY, _openai_text),
    ("ai21", "/chat/completions", OPENAI_BODY, _openai_text),
    ("anthropic", "/v1/messages", {"model": "m", "max_tokens": 256, "messages": CHAT}, _anthropic_text),
    ("cohere", "/v2/chat", {"model": "m", "messages": CHAT}, _cohere_text),
]
GEMINI = ("gemini", "/v1beta/models/m:generateContent", {"contents": [{"parts": [{"text": "Tell me a short story"}]}]}, _gemini_text)


def _stream(server, provider, path, body):
    """POST a streaming request; return (content-type, parsed JSON events, raw data lines)."""
    url = server.base_url(provider) + path
    if provider == "gemini":
        url = url.replace(":generateContent", ":streamGenerateContent") + "?alt=sse"
        payload = body
    else:
        payload = {**body, "stream": True}
    with httpx.stream("POST", url, json=payload, timeout=10) as response:
        content_type = response.headers["content-type"]
        data = [line[len("data: "):] for line in response.iter_lines() if line.startswith("data: ")]
    events = [json.loads(d) for d in data if d != "[DONE]"]
    return content_type, events, data


def _plain(server, provider, path, body, extract):
    response = httpx.post(server.base_url(provider) + path, json=body, timeout=10).json()
    if provider == "anthropic":
        return response["content"][0]["text"]
    if provider == "gemini":
        return _gemini_text(response)
    if provider == "cohere":
        return response["message"]["content"][0]["text"]
    return response["choices"][0]["message"]["content"]


ALL = [*PROVIDERS, GEMINI]
IDS = [p[0] for p in ALL]


@pytest.mark.parametrize(("provider", "path", "body", "extract"), ALL, ids=IDS)
def test_every_provider_streams_the_same_text_as_its_json_answer(server, provider, path, body, extract):
    content_type, events, _ = _stream(server, provider, path, body)
    assert content_type.startswith("text/event-stream")
    streamed = "".join(extract(e) for e in events)
    assert streamed == _plain(server, provider, path, body, extract)
    assert streamed


@pytest.mark.parametrize(("provider", "path", "body", "extract"), PROVIDERS[:7], ids=IDS[:7])
def test_openai_compatible_streams_end_with_done(server, provider, path, body, extract):
    _, _, data = _stream(server, provider, path, body)
    assert data[-1] == "[DONE]"


@pytest.mark.parametrize(("provider", "path", "body", "extract"), ALL, ids=IDS)
def test_truncation_lands_on_the_exact_event_for_every_provider(server, provider, path, body, extract):
    server.state.scenarios.add(StreamFault("truncate", after_chunks=2))
    _, _, data = _stream(server, provider, path, body)
    assert len(data) == 2
    (record,) = server.state.journal.records(wait=2)
    assert record.provider == provider
    assert record.fault == "stream:truncate@2"


@pytest.mark.parametrize(("provider", "path", "body", "extract"), ALL, ids=IDS)
def test_disconnect_breaks_the_transport_for_every_provider(server, provider, path, body, extract):
    server.state.scenarios.add(StreamFault("disconnect", after_chunks=2))
    with pytest.raises(httpx.RemoteProtocolError):
        _stream(server, provider, path, body)


def test_gemini_without_alt_sse_returns_a_json_array(server):
    url = server.base_url("gemini") + "/v1beta/models/m:streamGenerateContent"
    chunks = httpx.post(url, json=GEMINI[2], timeout=10).json()
    assert isinstance(chunks, list) and chunks
    assert chunks[-1]["candidates"][0]["finishReason"] == "STOP"


# -- real SDKs ----------------------------------------------------------------


@pytest.fixture
def anthropic_client(server):
    anthropic = pytest.importorskip("anthropic")
    return anthropic, anthropic.Anthropic(
        base_url=server.base_url("anthropic"), api_key="test", max_retries=0, timeout=10
    )


@pytest.fixture
def gemini_client(server):
    genai = pytest.importorskip("google.genai")
    types = pytest.importorskip("google.genai.types")
    return genai.Client(
        api_key="test",
        http_options=types.HttpOptions(
            base_url=server.base_url("gemini"), api_version="v1beta", timeout=10_000
        ),
    )


@pytest.fixture
def cohere_client(server):
    cohere = pytest.importorskip("cohere")
    return cohere.ClientV2(api_key="test", base_url=server.base_url("cohere"), timeout=10)


class TestAnthropicSDK:
    def test_stream_helper_rebuilds_text_and_tool_use(self, server, anthropic_client):
        _, client = anthropic_client
        server.state.scenarios.add(
            Reply(text="Let me check.", tool_calls=(ToolCall("get_weather", {"city": "Paris"}),))
        )
        with client.messages.stream(model="claude-sonnet-4-6", max_tokens=256, messages=CHAT) as s:
            final = s.get_final_message()
        assert final.stop_reason == "tool_use"
        text, tool = final.content
        assert text.text == "Let me check."
        assert (tool.name, tool.input) == ("get_weather", {"city": "Paris"})

    def test_truncation_is_silent(self, server, anthropic_client):
        _, client = anthropic_client
        server.state.scenarios.add(StreamFault("truncate", after_chunks=4))
        with client.messages.stream(model="claude-sonnet-4-6", max_tokens=256, messages=CHAT) as s:
            final = s.get_final_message()
        assert final.stop_reason is None

    def test_disconnect_escapes_as_a_raw_transport_error(self, server, anthropic_client):
        """Not an anthropic.APIConnectionError: `except anthropic.APIError` misses it."""
        anthropic, client = anthropic_client
        server.state.scenarios.add(StreamFault("disconnect", after_chunks=4))
        with pytest.raises(Exception) as excinfo:
            with client.messages.stream(model="claude-sonnet-4-6", max_tokens=256, messages=CHAT) as s:
                s.get_final_message()
        assert not isinstance(excinfo.value, anthropic.APIError)
        assert "RemoteProtocolError" in type(excinfo.value).__name__


class TestGeminiSDK:
    def test_function_call_round_trip(self, server, gemini_client):
        types = pytest.importorskip("google.genai.types")
        server.state.scenarios.add(Reply(tool_calls=(ToolCall("get_weather", {"city": "Paris"}),)))
        first = gemini_client.models.generate_content(model="gemini-2.5-pro", contents="weather?")
        assert [(c.name, c.args) for c in first.function_calls] == [("get_weather", {"city": "Paris"})]
        # Second turn of an agent loop: the call and its result go back to the model.
        history = [
            types.Content(role="user", parts=[types.Part(text="weather?")]),
            first.candidates[0].content,
            types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
                name="get_weather", response={"temp_c": 21}))]),
        ]
        second = gemini_client.models.generate_content(model="gemini-2.5-pro", contents=history)
        assert second.text

    def test_streamed_text_matches(self, server, gemini_client):
        whole = gemini_client.models.generate_content(model="gemini-2.5-pro", contents="hi")
        chunks = gemini_client.models.generate_content_stream(model="gemini-2.5-pro", contents="hi")
        assert "".join(c.text or "" for c in chunks) == whole.text


class TestCohereSDK:
    def test_streamed_tool_call_reassembles(self, server, cohere_client):
        server.state.scenarios.add(
            Reply(tool_calls=(ToolCall("get_weather", {"city": "Paris", "unit": "celsius"}),))
        )
        name, args = None, ""
        for event in cohere_client.chat_stream(model="command-r-plus", messages=CHAT):
            if event.type == "tool-call-start":
                name = event.delta.message.tool_calls.function.name
            elif event.type == "tool-call-delta":
                args += event.delta.message.tool_calls.function.arguments
        assert name == "get_weather"
        assert json.loads(args) == {"city": "Paris", "unit": "celsius"}

    def test_malformed_chunk_is_dropped_silently(self, server, cohere_client, caplog):
        """The worst case of the four SDKs: text goes missing, finish says COMPLETE."""
        full = "".join(
            e.delta.message.content.text
            for e in cohere_client.chat_stream(model="command-r-plus", messages=CHAT)
            if e.type == "content-delta"
        )
        server.state.reset()
        server.state.scenarios.add(StreamFault("malformed", after_chunks=3))
        text, finish = "", None
        with caplog.at_level(logging.ERROR):
            for event in cohere_client.chat_stream(model="command-r-plus", messages=CHAT):
                if event.type == "content-delta":
                    text += event.delta.message.content.text
                elif event.type == "message-end":
                    finish = event.delta.finish_reason
        assert finish == "COMPLETE"
        assert text != full and len(text) < len(full)
