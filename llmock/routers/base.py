"""Base abstractions for LLMock provider routers.

How to add a new provider in 3 steps:
1. Create ``llmock/routers/<provider>.py`` with your ``APIRouter`` and endpoints.
2. At the bottom of that module call ``registry.register(router)``
   (or ``registry.register(router, extra_router)`` if you have more than one).
3. Done — ``create_app()`` picks it up automatically via the registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from fastapi import APIRouter


class MockProvider(ABC):
    """Interface that every LLMock provider implementation should satisfy.

    Most router modules do not need to subclass this directly — registering
    an ``APIRouter`` with :func:`llmock.routers.registry.register` is
    sufficient.  This class exists so that the contract is explicit and
    type-checkable.
    """

    prefix: str
    """URL prefix shared by all endpoints in this provider (e.g. ``"/v1"``)."""

    tags: list[str]
    """FastAPI OpenAPI tags for this provider."""

    mock_models: list[str]
    """Model IDs served by this provider's ``/models`` endpoint."""

    @abstractmethod
    def get_router(self) -> APIRouter:
        """Return the configured :class:`~fastapi.APIRouter` for this provider."""

    def count_tokens(self, text: str) -> int:
        """Estimate the token count for *text* (whitespace split heuristic)."""
        return max(1, len(text.split()))

    def mock_reply(self, model: str) -> str:
        """Return a short deterministic mock reply string for *model*."""
        return f"This is a mock response from {model}."
