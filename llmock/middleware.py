"""The ASGI middleware that decides, injects and records.

For every provider request it:

1. reads the body once and replays it to the app;
2. asks the scenario queue for a :class:`~llmock.scenarios.Plan`;
3. answers with an error when forced, scripted or sampled;
4. otherwise lets the provider handler run, while watching what goes out on
   the wire -- which is where stream faults are applied;
5. records the whole exchange in the journal.

Stream faults live here, on the transport, rather than in each provider's
streaming code. Every provider streams one SSE event per ASGI body message,
so cutting, stalling or corrupting that flow works for all of them at once.

This is a plain ASGI middleware on purpose: ``BaseHTTPMiddleware`` buffers
streaming bodies through an extra task and hides client disconnects.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from llmock.chaos import _sample_error_status
from llmock.journal import RequestRecord, fingerprint
from llmock.scenarios import Plan, RequestInfo, StreamFault
from llmock.simulation import build_error_response, provider_from_path
from llmock.state import LLMockState

__all__ = ["LLMockMiddleware", "StreamAborted", "install_log_filter"]

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

# Paths that are never chaos-tested nor journaled.
_BYPASS_PREFIXES = ("/_llmock",)
_BYPASS_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})

# Request bodies above this size are not kept in the journal (fingerprint still is).
_MAX_JOURNALED_BODY = 256 * 1024

_GEMINI_MODEL_RE = re.compile(r"/models/([^/:]+):")

# Set when LLMock drops a connection on purpose, so the log filter can tell a
# scripted disconnect from a real server bug.
_intentional_abort: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "llmock_intentional_abort", default=False
)


class StreamAborted(Exception):
    """Raised through the app's ``send`` to stop a stream on purpose."""


class LLMockMiddleware:
    def __init__(self, app: ASGIApp, state: LLMockState) -> None:
        self.app = app
        self.state = state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or _bypassed(scope["path"]):
            await self.app(scope, receive, send)
            return

        started_at = time.monotonic()
        raw_body = await _read_body(receive)
        replay = _replay(raw_body, receive)
        headers = _headers(scope)
        body = _parse_json(raw_body, headers)
        path: str = scope["path"]
        method: str = scope["method"]

        info = RequestInfo(
            provider=provider_from_path(path),
            method=method,
            path=path,
            model=_model(body, path),
            stream=_is_stream(body, path),
            has_tools=_has_tools(body),
        )
        # A forced status pre-empts the request, so it must not consume queued
        # behaviours: they are meant for requests that actually get handled.
        error = _forced_error(headers)
        plan = Plan() if error is not None else self.state.scenarios.plan_for(info)
        request_state = scope.setdefault("state", {})
        request_state["llmock_plan"] = plan
        request_state["llmock_info"] = info
        # Handlers read fields their pydantic model does not declare (e.g.
        # stream_options) from here, instead of every model redeclaring them.
        request_state["llmock_body"] = body if isinstance(body, dict) else {}

        journal = self.state.journal
        ticket = journal.begin()
        recorded = False

        def journal_once() -> None:
            # Called just *before* the last byte goes out, so a client can
            # never see a response end while its record is still missing.
            nonlocal recorded
            if recorded:
                return
            recorded = True
            journal.add(
                RequestRecord(
                    seq=ticket.seq,
                    provider=info.provider,
                    method=method,
                    path=path,
                    started_at=started_at,
                    ended_at=time.monotonic(),
                    status=tracker.status,
                    fingerprint=fingerprint(method, path, body if body is not None else raw_body),
                    model=info.model,
                    stream=info.stream,
                    fault=tracker.fault,
                    retry_after=tracker.retry_after,
                    completed=tracker.completed and tracker.fault_kind != "truncate",
                    chunks_sent=tracker.chunks,
                    sdk_retry_count=_int_or_none(headers.get("x-stainless-retry-count")),
                    stall_waited=tracker.stall_waited,
                    body=body if len(raw_body) <= _MAX_JOURNALED_BODY else None,
                ),
                ticket,
            )

        tracker = _Tracker(send, plan, on_final=journal_once)
        try:
            if error is None:
                await self._wait(plan)
                error = _planned_error(plan) or _sampled_error(self.state)
            if error is not None:
                status, retry_after, message, code, label = error
                tracker.fault = label
                response = build_error_response(
                    path, status, retry_after=retry_after, message=message, code=code
                )
                await response(scope, replay, tracker.send)
            else:
                await self.app(scope, replay, tracker.send)
        except BaseException as exc:
            if not _is_stream_abort(exc):
                raise
            if not tracker.completed:
                _intentional_abort.set(True)
        finally:
            # Disconnects and handler errors never reach a final message.
            journal_once()
            journal.end(ticket)

    async def _wait(self, plan: Plan) -> None:
        seconds = self.state.chaos.latency_ms / 1000.0
        if plan.delay is not None:
            seconds += plan.delay.seconds
        if seconds > 0:
            await asyncio.sleep(seconds)


class _Tracker:
    """Wraps ``send``: observes the response and applies stream faults."""

    def __init__(self, send: Send, plan: Plan, on_final: Callable[[], None]) -> None:
        self._send = send
        self._plan = plan
        self._on_final = on_final
        self._is_sse = False
        self._stalled = False
        self.status = 500
        self.retry_after: float | None = None
        self.chunks = 0
        self.completed = False
        self.fault: str | None = None
        self.fault_kind: str | None = None
        self.stall_waited: float | None = None

    async def send(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            self.status = message["status"]
            response_headers = {
                k.decode("latin-1").lower(): v.decode("latin-1")
                for k, v in message.get("headers", [])
            }
            self.retry_after = _retry_after_seconds(response_headers)
            self._is_sse = response_headers.get("content-type", "").startswith(
                "text/event-stream"
            )
            await self._send(message)
            return

        if message["type"] != "http.response.body":
            await self._send(message)
            return

        body: bytes = message.get("body", b"")
        more_body: bool = message.get("more_body", False)

        if body and self._is_sse:
            await self._before_chunk()
            if self._plan.stream_fault is not None and self.chunks == self._plan.stream_fault.after_chunks:
                message = await self._apply_fault(self._plan.stream_fault, message, body)
            self.chunks += 1
        elif body:
            self.chunks += 1

        if not more_body:
            self.completed = True
            self._on_final()
        await self._send(message)

    async def _before_chunk(self) -> None:
        slow = self._plan.slow_first_token
        if self.chunks == 0 and slow is not None and slow.seconds > 0:
            await asyncio.sleep(slow.seconds)

    async def _apply_fault(self, fault: StreamFault, message: Message, body: bytes) -> Message:
        """Act on the chunk about to go out; return the message to send, if any."""
        if fault.kind == "stall":
            if not self._stalled:
                self._stalled = True
                self._record(fault)
                started = time.monotonic()
                try:
                    await asyncio.sleep(fault.stall_seconds)
                finally:
                    # Set even when cancelled: a client that hangs up early
                    # waited less than the full stall, which is the point.
                    self.stall_waited = time.monotonic() - started
            return message

        if fault.kind == "malformed":
            self._record(fault)
            return {**message, "body": _corrupt(body)}

        self._record(fault)
        if fault.kind == "truncate":
            # End the HTTP response cleanly: to the client this looks like a
            # normal end of stream, just without a finish reason or terminator.
            self.completed = True
            self._on_final()
            await self._send({"type": "http.response.body", "body": b"", "more_body": False})
        raise StreamAborted(fault.kind)

    def _record(self, fault: StreamFault) -> None:
        self.fault = f"stream:{fault.kind}@{self.chunks}"
        self.fault_kind = fault.kind


# -- error decisions ----------------------------------------------------------

_Error = tuple[int, "float | None", "str | None", "str | None", str]


def _forced_error(headers: dict[str, str]) -> _Error | None:
    forced = headers.get("x-llmock-force-status")
    if not forced:
        return None
    try:
        status = int(forced)
    except ValueError:
        return None
    if not 400 <= status < 600:
        return None
    return (status, None, None, None, f"forced:{status}")


def _planned_error(plan: Plan) -> _Error | None:
    fail = plan.fail
    if fail is None:
        return None
    return (fail.status, fail.retry_after, fail.message, fail.code, f"scenario:{fail.status}")


def _sampled_error(state: LLMockState) -> _Error | None:
    status = _sample_error_status(state.chaos)
    if status is None:
        return None
    return (status, None, None, None, f"chaos:{status}")


# -- request parsing ----------------------------------------------------------


def _bypassed(path: str) -> bool:
    return path in _BYPASS_PATHS or path.startswith(_BYPASS_PREFIXES)


async def _read_body(receive: Receive) -> bytes:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] != "http.request":
            break
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


def _replay(body: bytes, receive: Receive) -> Receive:
    """Hand the buffered body to the app once, then pass through.

    Later ``receive()`` calls must reach the real channel, so that a streaming
    response can still notice when the client goes away.
    """
    delivered = False

    async def replay() -> Message:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


def _headers(scope: Scope) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}


def _parse_json(raw: bytes, headers: dict[str, str]) -> Any:
    if not raw or "json" not in headers.get("content-type", "json"):
        return None
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None


def _model(body: Any, path: str) -> str | None:
    if isinstance(body, dict) and isinstance(body.get("model"), str):
        return body["model"]
    match = _GEMINI_MODEL_RE.search(path)
    return match.group(1) if match else None


def _is_stream(body: Any, path: str) -> bool:
    if ":streamGenerateContent" in path:
        return True
    return isinstance(body, dict) and body.get("stream") is True


def _has_tools(body: Any) -> bool:
    return isinstance(body, dict) and bool(body.get("tools"))


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _retry_after_seconds(headers: dict[str, str]) -> float | None:
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            return float(value) / scale
        except ValueError:
            continue
    return None


def _corrupt(chunk: bytes) -> bytes:
    """Cut the JSON payload of an SSE event in half, keeping the framing."""
    text = chunk.decode("utf-8", errors="replace")
    head, sep, payload = text.rpartition("data: ")
    if not sep:
        return b'data: {"llmock": "malformed\n\n'
    payload = payload.rstrip("\n")
    broken = payload[: max(1, len(payload) // 2)]
    return f"{head}data: {broken}\n\n".encode()


def _is_stream_abort(exc: BaseException) -> bool:
    """True for our own abort, including when wrapped by an exception group."""
    if isinstance(exc, StreamAborted):
        return True
    inner = getattr(exc, "exceptions", None)
    return bool(inner) and all(_is_stream_abort(e) for e in inner)


# -- logging ------------------------------------------------------------------


class _IntentionalAbortFilter(logging.Filter):
    """Hide uvicorn's error for connections LLMock dropped on purpose.

    A scripted disconnect makes the app return mid-response, which uvicorn
    reports as an error. It is the requested behaviour, not a bug, and a
    traceback in the user's terminal would suggest otherwise.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not _intentional_abort.get():
            return True
        return "without completing response" not in record.getMessage()


_FILTER = _IntentionalAbortFilter()


def install_log_filter() -> None:
    logger = logging.getLogger("uvicorn.error")
    if _FILTER not in logger.filters:
        logger.addFilter(_FILTER)
