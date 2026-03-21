"""FastAPI application factory for LLMock."""

from fastapi import FastAPI

from llmock import __version__
from llmock.chaos import ChaosMiddleware, ChaosSettings, chaos_settings
from llmock.errors import register_error_handlers

# Import every router module so their registry.register() calls fire at import time.
import llmock.routers.ai21  # noqa: F401
import llmock.routers.anthropic  # noqa: F401
import llmock.routers.cohere  # noqa: F401
import llmock.routers.gemini  # noqa: F401
import llmock.routers.groq  # noqa: F401
import llmock.routers.mistral  # noqa: F401
import llmock.routers.openai  # noqa: F401
import llmock.routers.perplexity  # noqa: F401
import llmock.routers.together  # noqa: F401
import llmock.routers.xai  # noqa: F401

from llmock.routers.registry import get_all_routers
from llmock.simulation import MockResponseSettings


def create_app(
    chaos: ChaosSettings | None = None,
    responses: MockResponseSettings | None = None,
) -> FastAPI:
    settings = (chaos or ChaosSettings.from_env()).validated()
    response_settings = (responses or MockResponseSettings.from_env()).validated()

    app = FastAPI(
        title="LLMock",
        description="OpenAI-compatible mock server for LLM API resilience testing",
        version=__version__,
    )

    app.state.chaos_settings = settings
    app.state.mock_response_settings = response_settings
    register_error_handlers(app)
    app.add_middleware(ChaosMiddleware, settings=settings)

    for router in get_all_routers():
        app.include_router(router)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    return app


app = create_app(chaos=chaos_settings)
