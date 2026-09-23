"""FastAPI application factory for LLMock."""

from __future__ import annotations

import os

from fastapi import FastAPI

from llmock import __version__
from llmock import admin
from llmock.chaos import ChaosSettings, StreamChaos, chaos_settings
from llmock.errors import register_error_handlers
from llmock.middleware import LLMockMiddleware, install_log_filter
from llmock.ratelimit import LimitSettings
from llmock.state import LLMockState

# Import every router module so their registry.register() calls fire at import time.
import llmock.routers.ai21  # noqa: F401
import llmock.routers.anthropic  # noqa: F401
import llmock.routers.cohere  # noqa: F401
import llmock.routers.gemini  # noqa: F401
import llmock.routers.groq  # noqa: F401
import llmock.routers.mistral  # noqa: F401
import llmock.routers.openai  # noqa: F401
import llmock.routers.openai_responses  # noqa: F401
import llmock.routers.perplexity  # noqa: F401
import llmock.routers.together  # noqa: F401
import llmock.routers.xai  # noqa: F401

from llmock.routers.registry import get_all_routers
from llmock.simulation import MockResponseSettings


def create_app(
    chaos: ChaosSettings | None = None,
    responses: MockResponseSettings | None = None,
    limits: LimitSettings | None = None,
    stream_chaos: StreamChaos | None = None,
) -> FastAPI:
    settings = (chaos or ChaosSettings.from_env()).validated()
    response_settings = (responses or MockResponseSettings.from_env()).validated()
    limit_settings = (limits or LimitSettings.from_env()).validated()
    stream_settings = (stream_chaos or StreamChaos.from_env()).validated()

    app = FastAPI(
        title="LLMock",
        description="Chaos engineering for LLM apps: mock 10 providers, break them on demand, judge your client.",
        version=__version__,
    )

    state = LLMockState(chaos=settings, limits=limit_settings, stream_chaos=stream_settings)
    app.state.llmock = state
    # Same object as state.chaos: mutating it changes behaviour live.
    app.state.chaos_settings = settings
    app.state.mock_response_settings = response_settings
    register_error_handlers(app)
    app.add_middleware(LLMockMiddleware, state=state)
    install_log_filter()

    for router in get_all_routers():
        app.include_router(router)
    app.include_router(admin.router)

    if os.getenv("LLMOCK_REPORT") == "1":
        app.router.add_event_handler("shutdown", lambda: _print_verdict(state))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    return app


app = create_app(chaos=chaos_settings)


def _print_verdict(state: LLMockState) -> None:
    from llmock.console import print_verdict
    from llmock.verdict import judge

    print()
    print_verdict(judge(state.journal.records()))
