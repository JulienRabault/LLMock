"""Router registry for LLMock.

Provider modules call :func:`register` with their ``APIRouter`` instance(s)
at import time.  :func:`create_app` then calls :func:`get_all_routers` once
to mount every provider in a single loop — no manual edits to ``main.py``
are ever needed when adding a new provider.

Example (in a new provider module)::

    from fastapi import APIRouter
    from llmock.routers import registry

    router = APIRouter(prefix="/myprovider/v1", tags=["myprovider"])

    @router.post("/chat/completions")
    def chat(...): ...

    # Register at module level so create_app() picks this up automatically.
    registry.register(router)
"""

from __future__ import annotations

from fastapi import APIRouter

_routers: list[APIRouter] = []


def register(*routers: APIRouter) -> None:
    """Add one or more :class:`~fastapi.APIRouter` instances to the registry."""
    _routers.extend(routers)


def get_all_routers() -> list[APIRouter]:
    """Return a snapshot of all registered :class:`~fastapi.APIRouter` instances."""
    return list(_routers)
