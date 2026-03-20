"""Provider-specific exception handlers for LLMock.

Overrides FastAPI's default `{"detail": "..."}` shape with the real
error envelope each provider uses so that clients can parse them normally.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from llmock.simulation import _build_error_content, provider_from_path


def _message_from_exc(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        return str(detail.get("msg") or detail.get("message") or exc.detail)
    return str(detail)


def _message_from_validation(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Request validation error."
    first = errors[0]
    loc = " → ".join(str(p) for p in first.get("loc", []))
    msg = first.get("msg", "validation error")
    return f"{loc}: {msg}" if loc else msg


async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    provider = provider_from_path(request.url.path)
    message = _message_from_exc(exc)
    content = _build_error_content(provider=provider, status_code=exc.status_code)

    # Inject the actual message instead of the generic label.
    _inject_message(content, message, provider)

    headers = dict(exc.headers or {})
    if exc.status_code in {429, 503, 504, 529} and "retry-after" not in {k.lower() for k in headers}:
        headers["retry-after"] = "1"

    return JSONResponse(status_code=exc.status_code, content=content, headers=headers or None)


async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    provider = provider_from_path(request.url.path)
    status_code = 422
    message = _message_from_validation(exc)
    content = _build_error_content(provider=provider, status_code=status_code)
    _inject_message(content, message, provider)
    return JSONResponse(status_code=status_code, content=content)


def _inject_message(content: dict, message: str, provider: str) -> None:
    """Mutate *content* in-place to use *message* instead of the generic label."""
    if provider == "anthropic":
        # {"type": "error", "error": {"type": "...", "message": "..."}}
        if isinstance(content.get("error"), dict):
            content["error"]["message"] = message
    elif provider == "gemini":
        # {"error": {"code": ..., "message": "...", "status": "..."}}
        if isinstance(content.get("error"), dict):
            content["error"]["message"] = message
    elif provider == "cohere":
        # {"message": "..."}
        content["message"] = message
    elif provider == "mistral":
        # {"object": "error", "message": "...", "type": "...", "param": null, "code": "..."}
        content["message"] = message
    elif provider == "xai":
        # {"error": {"message": "...", "type": "...", "code": "..."}}
        if isinstance(content.get("error"), dict):
            content["error"]["message"] = message
    else:
        # OpenAI-compatible: {"error": {"message": "...", "type": "...", ...}}
        if isinstance(content.get("error"), dict):
            content["error"]["message"] = message


def register_error_handlers(app: FastAPI) -> None:
    """Register provider-aware error handlers on *app*."""
    app.add_exception_handler(HTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)  # type: ignore[arg-type]
