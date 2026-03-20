"""FastAPI application factory for LLMock."""

from fastapi import FastAPI

from llmock import __version__
from llmock.chaos import ChaosMiddleware, ChaosSettings, chaos_settings
from llmock.errors import register_error_handlers
from llmock.routers import ai21 as ai21_router
from llmock.routers import anthropic as anthropic_router
from llmock.routers import cohere as cohere_router
from llmock.routers import gemini as gemini_router
from llmock.routers import groq as groq_router
from llmock.routers import mistral as mistral_router
from llmock.routers import openai as openai_router
from llmock.routers import perplexity as perplexity_router
from llmock.routers import together as together_router
from llmock.routers import xai as xai_router
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
    app.include_router(openai_router.router)
    app.include_router(mistral_router.router)
    app.include_router(anthropic_router.router)
    app.include_router(gemini_router.router)
    app.include_router(cohere_router.router)
    app.include_router(cohere_router.legacy_router)
    app.include_router(groq_router.router)
    app.include_router(together_router.router)
    app.include_router(perplexity_router.router)
    app.include_router(ai21_router.router)
    app.include_router(xai_router.router)
    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    return app


app = create_app(chaos=chaos_settings)
