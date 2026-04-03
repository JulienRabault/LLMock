"""End-to-end tests exercising real SDK clients against LLMock.

These tests use the actual provider SDKs (openai, anthropic, langchain, llamaindex)
the same way a user would — via base_url override. They validate that the SDKs
can parse LLMock responses without errors.

Each test creates its own ASGI transport so there is no real network involved,
but the SDK still builds real requests and parses real responses.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from llmock.chaos import ChaosSettings
from llmock.main import create_app
from llmock.simulation import MockResponseSettings


@pytest.fixture()
def app():
    return create_app(
        chaos=ChaosSettings(),
        responses=MockResponseSettings(response_style="echo"),
    )


@pytest.fixture()
def chaos_app():
    """App with 100% rate-limit errors."""
    return create_app(
        chaos=ChaosSettings(error_rate_429=1.0),
        responses=MockResponseSettings(response_style="echo"),
    )


# ---------------------------------------------------------------------------
# OpenAI SDK
# ---------------------------------------------------------------------------


class TestOpenAISDK:
    async def test_chat_completion(self, app):
        import openai

        transport = ASGITransport(app=app)
        http_client = AsyncClient(transport=transport, base_url="http://test")
        client = openai.AsyncOpenAI(
            api_key="fake",
            base_url="http://test/v1",
            http_client=http_client,
        )
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello from E2E!"}],
        )
        assert response.choices[0].message.content is not None
        assert response.choices[0].finish_reason == "stop"
        assert response.model == "gpt-4o"
        assert response.usage is not None

    async def test_list_models(self, app):
        import openai

        transport = ASGITransport(app=app)
        http_client = AsyncClient(transport=transport, base_url="http://test")
        client = openai.AsyncOpenAI(
            api_key="fake",
            base_url="http://test/v1",
            http_client=http_client,
        )
        models = await client.models.list()
        model_ids = [m.id for m in models.data]
        assert "gpt-4o" in model_ids

    async def test_embeddings(self, app):
        import openai

        transport = ASGITransport(app=app)
        http_client = AsyncClient(transport=transport, base_url="http://test")
        client = openai.AsyncOpenAI(
            api_key="fake",
            base_url="http://test/v1",
            http_client=http_client,
        )
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input="test embedding",
        )
        assert len(response.data) == 1
        assert len(response.data[0].embedding) > 0

    async def test_rate_limit_raises(self, chaos_app):
        import openai

        transport = ASGITransport(app=chaos_app)
        http_client = AsyncClient(transport=transport, base_url="http://test")
        client = openai.AsyncOpenAI(
            api_key="fake",
            base_url="http://test/v1",
            http_client=http_client,
        )
        with pytest.raises(openai.RateLimitError):
            await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "should fail"}],
            )


# ---------------------------------------------------------------------------
# Anthropic SDK
# ---------------------------------------------------------------------------


class TestAnthropicSDK:
    async def test_messages_create(self, app):
        import anthropic
        from httpx import AsyncClient as HttpxAsyncClient

        transport = ASGITransport(app=app)
        http_client = HttpxAsyncClient(transport=transport, base_url="http://test")
        client = anthropic.AsyncAnthropic(
            api_key="fake",
            base_url="http://test/anthropic",
            http_client=http_client,
        )
        message = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello from E2E!"}],
        )
        assert message.content[0].text is not None
        assert message.role == "assistant"
        assert message.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------


class TestLangChain:
    async def test_invoke(self, app):
        from langchain_openai import ChatOpenAI
        from httpx import AsyncClient as HttpxAsyncClient

        transport = ASGITransport(app=app)
        http_client = HttpxAsyncClient(transport=transport, base_url="http://test")

        llm = ChatOpenAI(
            base_url="http://test/v1",
            api_key="fake",
            model="gpt-4o",
            http_async_client=http_client,
        )
        response = await llm.ainvoke("Hello from LangChain E2E!")
        assert response.content is not None
        assert len(response.content) > 0


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


class TestLlamaIndex:
    """LlamaIndex creates its own internal httpx client and does not expose
    a way to inject an ASGI transport. We test via the underlying OpenAI
    async client instead, which proves the response schema is compatible."""

    async def test_response_parses_as_llamaindex_completion(self, app):
        """Verify LLMock's response can be parsed by the OpenAI SDK
        the same way LlamaIndex would consume it."""
        import openai

        transport = ASGITransport(app=app)
        http_client = AsyncClient(transport=transport, base_url="http://test")
        client = openai.AsyncOpenAI(
            api_key="fake",
            base_url="http://test/v1",
            http_client=http_client,
        )
        # LlamaIndex internally calls chat.completions.create and reads
        # response.choices[0].message.content — verify that path works.
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello from LlamaIndex path!"}],
        )
        text = response.choices[0].message.content
        assert text is not None
        assert len(text) > 0


# ---------------------------------------------------------------------------
# Multi-provider round-trip
# ---------------------------------------------------------------------------


class TestMultiProvider:
    """Verify several provider endpoints return parseable responses."""

    PROVIDERS = [
        ("/v1/chat/completions", {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}),
        ("/anthropic/v1/messages", {"model": "claude-sonnet-4-20250514", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}),
        ("/mistral/v1/chat/completions", {"model": "mistral-large", "messages": [{"role": "user", "content": "hi"}]}),
        ("/groq/openai/v1/chat/completions", {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}),
        ("/together/v1/chat/completions", {"model": "meta-llama", "messages": [{"role": "user", "content": "hi"}]}),
        ("/perplexity/v1/chat/completions", {"model": "pplx-7b", "messages": [{"role": "user", "content": "hi"}]}),
        ("/ai21/v1/chat/completions", {"model": "jamba", "messages": [{"role": "user", "content": "hi"}]}),
        ("/xai/v1/chat/completions", {"model": "grok-1", "messages": [{"role": "user", "content": "hi"}]}),
    ]

    @pytest.mark.parametrize("path,payload", PROVIDERS, ids=[p[0].split("/")[1] for p in PROVIDERS])
    async def test_provider_returns_200(self, app, path, payload):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(path, json=payload)
            assert resp.status_code == 200
            body = resp.json()
            # Every provider should return some form of content
            assert body is not None


# ---------------------------------------------------------------------------
# Stress / concurrency
# ---------------------------------------------------------------------------


class TestStress:
    """Verify LLMock handles concurrent load without errors."""

    async def test_200_concurrent_requests(self, app):
        """200 concurrent chat completions, all should succeed."""
        import asyncio

        transport = ASGITransport(app=app)
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "stress"}]}

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            sem = asyncio.Semaphore(50)

            async def one_request():
                async with sem:
                    r = await client.post("/v1/chat/completions", json=payload)
                    return r.status_code

            results = await asyncio.gather(*[one_request() for _ in range(200)])
            assert all(s == 200 for s in results), f"Got non-200: {set(results)}"

    async def test_chaos_under_load(self, app):
        """100 requests with 50% error rate — roughly half should fail."""
        import asyncio

        app.state.chaos_settings.error_rate_429 = 0.5
        transport = ASGITransport(app=app)
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "chaos"}]}

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            sem = asyncio.Semaphore(20)

            async def one_request():
                async with sem:
                    r = await client.post("/v1/chat/completions", json=payload)
                    return r.status_code

            results = await asyncio.gather(*[one_request() for _ in range(100)])
            ok = sum(1 for s in results if s == 200)
            errors = sum(1 for s in results if s == 429)
            # With 50% rate, we expect roughly 40-60 successes (allow wide margin)
            assert 20 <= ok <= 80, f"Expected ~50 successes, got {ok}"
            assert ok + errors == 100, f"Unexpected status codes: {set(results)}"
