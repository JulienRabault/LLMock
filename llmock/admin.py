"""The ``/_llmock`` control API: script faults and read the journal over HTTP.

This is what makes LLMock usable from any language. A Node, Go or Rust test
suite can queue a scenario, run its code, then read back what happened::

    POST   /_llmock/scenario   {"behaviors": [{"type": "fail", "status": 429, "times": 2}]}
    GET    /_llmock/scenario   behaviours still waiting to fire
    DELETE /_llmock/scenario   drop them
    GET    /_llmock/requests   every request served, oldest first
    DELETE /_llmock/requests   forget them
    POST   /_llmock/reset      both of the above

These routes bypass chaos and are never journaled.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from llmock.scenarios import behavior_from_dict, behavior_to_dict
from llmock.state import LLMockState

__all__ = ["router"]

router = APIRouter(prefix="/_llmock", tags=["llmock-admin"])


def _state(request: Request) -> LLMockState:
    return request.app.state.llmock


@router.post("/scenario", status_code=201)
def add_scenario(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("behaviors")
    if not isinstance(raw, list) or not raw:
        raise HTTPException(400, "Expected a non-empty 'behaviors' list.")
    try:
        behaviors = [behavior_from_dict(item) for item in raw]
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    state = _state(request)
    state.scenarios.add(*behaviors)
    return {"queued": len(behaviors), "pending": len(state.scenarios.pending())}


@router.get("/scenario")
def list_scenario(request: Request) -> dict[str, Any]:
    pending = _state(request).scenarios.pending()
    return {"pending": [behavior_to_dict(b) for b in pending]}


@router.delete("/scenario")
def clear_scenario(request: Request) -> dict[str, Any]:
    _state(request).scenarios.clear()
    return {"pending": 0}


@router.get("/requests")
def list_requests(request: Request) -> dict[str, Any]:
    records = _state(request).journal.records()
    return {"count": len(records), "requests": [r.to_dict() for r in records]}


@router.delete("/requests")
def clear_requests(request: Request) -> dict[str, Any]:
    _state(request).journal.clear()
    return {"count": 0}


@router.post("/reset")
def reset(request: Request) -> dict[str, Any]:
    _state(request).reset()
    return {"reset": True}
