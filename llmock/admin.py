"""The ``/_llmock`` control API: script faults and read the journal over HTTP.

This is what makes LLMock usable from any language. A Node, Go or Rust test
suite can queue a scenario, run its code, then read back what happened::

    POST   /_llmock/scenario   {"behaviors": [{"type": "fail", "status": 429, "times": 2}]}
    GET    /_llmock/scenario   behaviours still waiting to fire
    DELETE /_llmock/scenario   drop them
    GET    /_llmock/requests   every request served, oldest first
    DELETE /_llmock/requests   forget them
    GET    /_llmock/verdict    how well the client coped (?format=text for a report)
    POST   /_llmock/reset      forget requests and queued behaviours

These routes bypass chaos and are never journaled.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from llmock.scenarios import behavior_from_dict, behavior_to_dict
from llmock.state import LLMockState
from llmock.verdict import judge

# How long a read waits for requests still in flight (e.g. a stream the
# client stopped reading at [DONE]) before answering with what it has.
_WAIT_SECONDS = 2.0

__all__ = ["router"]

router = APIRouter(prefix="/_llmock", tags=["llmock-admin"])


def _state(request: Request) -> LLMockState:
    return request.app.state.llmock


@router.post("/scenario", status_code=201)
async def add_scenario(request: Request) -> dict[str, Any]:
    # Parsed by hand rather than through a pydantic body parameter, so that a
    # `curl -d '{...}'` without a JSON content type still works.
    try:
        payload = json.loads(await request.body() or b"{}")
    except ValueError:
        raise HTTPException(400, "The body must be JSON.") from None
    if not isinstance(payload, dict):
        raise HTTPException(400, "Expected a JSON object with a 'behaviors' list.")
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
    records = _state(request).journal.records(wait=_WAIT_SECONDS)
    return {"count": len(records), "requests": [r.to_dict() for r in records]}


@router.delete("/requests")
def clear_requests(request: Request) -> dict[str, Any]:
    _state(request).journal.clear()
    return {"count": 0}


@router.get("/verdict", response_model=None)
def verdict(request: Request, format: str = "json"):
    result = judge(_state(request).journal.records(wait=_WAIT_SECONDS))
    if format == "text":
        return PlainTextResponse(result.render())
    return result.to_dict()


@router.post("/reset")
def reset(request: Request) -> dict[str, Any]:
    _state(request).reset()
    return {"reset": True}
