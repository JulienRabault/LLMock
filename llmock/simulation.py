"""Shared simulation helpers for LLMock routers and middleware."""

from __future__ import annotations

import base64
import hashlib
import html
from http import HTTPStatus
import os
import random
from dataclasses import dataclass
from typing import Any

from starlette.responses import JSONResponse

SUPPORTED_RESPONSE_STYLES = {"static", "hello", "echo", "varied"}
DEFAULT_IMAGE_SIZE = "1024x1024"
ERROR_RATE_ENV_PREFIX = "LLMOCK_ERROR_RATE_"
SUPPORTED_ERROR_STATUS_CODES = tuple(range(400, 600))


def raise_if_streaming(stream: bool) -> None:
    """Raise HTTPException 501 when a client requests streaming mode.

    LLMock does not implement SSE streaming. Returning 501 instead of silently
    ignoring the flag prevents SDK streaming iterators from hanging forever.
    """
    if stream:
        from fastapi import HTTPException  # local import avoids circularity
        raise HTTPException(status_code=501, detail="Streaming is not supported by LLMock. Set stream=false.")


@dataclass
class MockResponseSettings:
    response_style: str = "varied"

    @classmethod
    def from_env(cls) -> "MockResponseSettings":
        return cls(
            response_style=os.getenv("LLMOCK_RESPONSE_STYLE", "varied"),
        ).validated()

    def validated(self) -> "MockResponseSettings":
        if self.response_style not in SUPPORTED_RESPONSE_STYLES:
            supported = ", ".join(sorted(SUPPORTED_RESPONSE_STYLES))
            raise ValueError(f"LLMOCK_RESPONSE_STYLE must be one of: {supported}.")
        return self

    def as_env(self) -> dict[str, str]:
        return {"LLMOCK_RESPONSE_STYLE": self.response_style}


def flatten_text(value: Any) -> str:
    """Flatten nested provider content blocks into a readable text string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(part for part in (flatten_text(item) for item in value) if part)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if "inline_data" in value or "image_url" in value or value.get("type") == "image_url":
            return "[image input]"
        for key in ("content", "parts", "input", "message"):
            if key in value:
                return flatten_text(value[key])
    return ""


def estimate_tokens(*values: Any) -> int:
    return sum(len(flatten_text(value)) // 4 for value in values)


def build_mock_text(
    *,
    settings: MockResponseSettings,
    model: str,
    prompt: str = "",
) -> str:
    prompt = " ".join(prompt.split()).strip()
    if settings.response_style == "static":
        return f"Mock response from {model}."
    if settings.response_style == "hello":
        return f"Hello! This is a mock response from {model}."
    if settings.response_style == "echo":
        if prompt:
            return f"Hello! You said: {prompt[:120]}"
        return f"Hello! This is a mock response from {model}."

    templates = [
        "Hello! LLMock handled this request for {model}.",
        "Mock response from {model}: everything is working as expected.",
        "Hi there. This is a simulated reply from {model}.",
        "LLMock says hello from {model}.",
        "Simulated success from {model}.",
    ]
    if prompt:
        templates.extend(
            [
                "Hello! {model} received: {prompt}",
                "{model} mock reply: I saw '{prompt}'.",
                "Simulated answer from {model} for '{prompt}'.",
            ]
        )

    # Use a cheap hash — we need determinism, not cryptographic strength.
    idx = hash(f"{model}|{prompt}") % len(templates)
    template = templates[idx]
    prompt_snippet = prompt[:80] if prompt else "your prompt"
    return template.format(model=model, prompt=prompt_snippet)


def build_mock_embedding(text: str, dimension: int = 1536) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    return [round(rng.uniform(-1.0, 1.0), 6) for _ in range(dimension)]


def _build_svg(prompt: str, size: str) -> str:
    width_text, _, height_text = size.partition("x")
    width = int(width_text) if width_text.isdigit() else 1024
    height = int(height_text) if height_text.isdigit() else 1024
    safe_prompt = html.escape(prompt[:64] or "LLMock image")
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>"
        "<defs><linearGradient id='g' x1='0' x2='1' y1='0' y2='1'>"
        "<stop offset='0%' stop-color='#111827'/>"
        "<stop offset='100%' stop-color='#2563eb'/>"
        "</linearGradient></defs>"
        f"<rect width='{width}' height='{height}' fill='url(#g)'/>"
        f"<text x='50%' y='48%' dominant-baseline='middle' text-anchor='middle' "
        "font-family='monospace' font-size='42' fill='white'>LLMock</text>"
        f"<text x='50%' y='58%' dominant-baseline='middle' text-anchor='middle' "
        "font-family='monospace' font-size='24' fill='#bfdbfe'>"
        f"{safe_prompt}</text>"
        "</svg>"
    )


def build_fake_image_data_uri(prompt: str, size: str = DEFAULT_IMAGE_SIZE) -> str:
    svg = _build_svg(prompt, size)
    payload = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{payload}"


def build_fake_image_payload(
    *,
    prompt: str,
    count: int,
    size: str = DEFAULT_IMAGE_SIZE,
    response_format: str = "url",
) -> list[dict[str, str]]:
    data_uri = build_fake_image_data_uri(prompt=prompt, size=size)
    if response_format == "b64_json":
        return [
            {"b64_json": data_uri.split(",", 1)[1], "revised_prompt": prompt}
            for _ in range(count)
        ]
    return [{"url": data_uri, "revised_prompt": prompt} for _ in range(count)]


def parse_error_rate_env() -> dict[int, float]:
    rates: dict[int, float] = {}
    for key, value in os.environ.items():
        if not key.startswith(ERROR_RATE_ENV_PREFIX):
            continue
        suffix = key[len(ERROR_RATE_ENV_PREFIX) :]
        if not suffix.isdigit():
            continue
        rates[int(suffix)] = float(value)
    return rates


def provider_from_path(path: str) -> str:
    if path.startswith("/anthropic/"):
        return "anthropic"
    if path.startswith("/gemini/"):
        return "gemini"
    if path.startswith("/cohere/"):
        return "cohere"
    if path.startswith("/mistral/"):
        return "mistral"
    if path.startswith("/groq/"):
        return "groq"
    if path.startswith("/together/"):
        return "together"
    if path.startswith("/perplexity/"):
        return "perplexity"
    if path.startswith("/ai21/"):
        return "ai21"
    if path.startswith("/xai/"):
        return "xai"
    return "openai"


def build_error_response(path: str, status_code: int) -> JSONResponse:
    provider = provider_from_path(path)
    content = _build_error_content(provider=provider, status_code=status_code)
    headers = {"retry-after": "1"} if status_code in {429, 503, 504, 529} else None
    return JSONResponse(status_code=status_code, content=content, headers=headers)


def _build_error_content(provider: str, status_code: int) -> dict[str, Any]:
    label = {
        400: "Bad request.",
        401: "Unauthorized.",
        402: "Payment required.",
        403: "Forbidden.",
        404: "Resource not found.",
        408: "Request timeout.",
        409: "Conflict.",
        413: "Payload too large.",
        422: "Unprocessable entity.",
        429: "Rate limit exceeded.",
        500: "Internal server error.",
        501: "Not implemented.",
        502: "Bad gateway.",
        503: "Service unavailable.",
        504: "Gateway timeout.",
        529: "Service overloaded.",
    }.get(status_code, _http_status_label(status_code))

    openai_types = {
        400: ("invalid_request_error", "bad_request"),
        401: ("authentication_error", "invalid_api_key"),
        402: ("billing_error", "billing_required"),
        403: ("permission_error", "forbidden"),
        404: ("invalid_request_error", "not_found"),
        408: ("request_timeout", "request_timeout"),
        409: ("conflict_error", "conflict"),
        413: ("invalid_request_error", "request_too_large"),
        422: ("invalid_request_error", "unprocessable_entity"),
        429: ("rate_limit_error", "rate_limit_exceeded"),
        500: ("server_error", "internal_error"),
        501: ("server_error", "not_implemented"),
        502: ("server_error", "bad_gateway"),
        503: ("server_error", "service_unavailable"),
        504: ("server_error", "gateway_timeout"),
        529: ("server_error", "overloaded"),
    }

    if provider == "anthropic":
        anthropic_type = {
            400: "invalid_request_error",
            401: "authentication_error",
            402: "billing_error",
            403: "permission_error",
            404: "not_found_error",
            408: "api_error",
            409: "api_error",
            413: "request_too_large",
            422: "invalid_request_error",
            429: "rate_limit_error",
            500: "api_error",
            501: "api_error",
            502: "api_error",
            503: "overloaded_error",
            504: "api_error",
            529: "overloaded_error",
        }.get(status_code, _fallback_anthropic_error_type(status_code))
        return {"type": "error", "error": {"type": anthropic_type, "message": label}}

    if provider == "gemini":
        gemini_status = {
            400: "INVALID_ARGUMENT",
            401: "UNAUTHENTICATED",
            402: "FAILED_PRECONDITION",
            403: "PERMISSION_DENIED",
            404: "NOT_FOUND",
            408: "DEADLINE_EXCEEDED",
            409: "ABORTED",
            413: "RESOURCE_EXHAUSTED",
            422: "FAILED_PRECONDITION",
            429: "RESOURCE_EXHAUSTED",
            500: "INTERNAL",
            501: "UNIMPLEMENTED",
            502: "UNAVAILABLE",
            503: "UNAVAILABLE",
            504: "DEADLINE_EXCEEDED",
            529: "UNAVAILABLE",
        }.get(status_code, _fallback_gemini_status(status_code))
        return {"error": {"code": status_code, "message": label, "status": gemini_status}}

    if provider == "cohere":
        cohere_type = {
            400: "invalid_request_error",
            401: "authentication_error",
            402: "billing_error",
            403: "permission_error",
            404: "not_found_error",
            408: "timeout_error",
            409: "conflict_error",
            413: "request_too_large",
            422: "unprocessable_entity_error",
            429: "rate_limit_error",
            500: "server_error",
            501: "server_error",
            502: "server_error",
            503: "server_error",
            504: "server_error",
            529: "server_error",
        }.get(status_code, _fallback_cohere_error_type(status_code))
        return {"message": label, "type": cohere_type}

    if provider == "mistral":
        error_type, error_code = openai_types.get(status_code, _fallback_openai_error(status_code))
        return {
            "object": "error",
            "message": label,
            "type": error_type,
            "param": None,
            "code": error_code,
        }

    error_type, error_code = openai_types.get(status_code, _fallback_openai_error(status_code))
    return {
        "error": {
            "message": label,
            "type": error_type,
            "param": None,
            "code": error_code,
        }
    }


def _http_status_label(status_code: int) -> str:
    try:
        phrase = HTTPStatus(status_code).phrase
    except ValueError:
        phrase = f"HTTP {status_code} error"
    return f"{phrase}."


def _fallback_openai_error(status_code: int) -> tuple[str, str]:
    if 400 <= status_code < 500:
        return ("invalid_request_error", f"http_{status_code}")
    return ("server_error", f"http_{status_code}")


def _fallback_anthropic_error_type(status_code: int) -> str:
    if 400 <= status_code < 500:
        return "invalid_request_error"
    return "api_error"


def _fallback_gemini_status(status_code: int) -> str:
    if 400 <= status_code < 500:
        return "INVALID_ARGUMENT"
    return "INTERNAL"


def _fallback_cohere_error_type(status_code: int) -> str:
    if 400 <= status_code < 500:
        return "invalid_request_error"
    return "server_error"
