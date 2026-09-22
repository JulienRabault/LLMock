"""What the mock model says, independently of any provider's wire format.

Routers turn a request into a :class:`Completion` with :func:`resolve_request`
(or :func:`resolve`), then render it: as a JSON body, or as a stream through
:mod:`llmock.streaming`. Keeping the decision in one place is what lets a
scripted reply, a tool call or a tool fault behave identically on every
provider.

The decision, in order:

1. a scripted :class:`~llmock.scenarios.Reply` wins;
2. otherwise, in ``tool_mode="auto"``, call an offered tool when a model
   would (see :mod:`llmock.tools`);
3. otherwise, when the request asks for structured output, JSON valid
   against its schema;
4. otherwise, text in the configured response style.

A queued :class:`~llmock.scenarios.ToolFault` then breaks the tool call.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from typing import Any, Literal

from llmock.scenarios import Plan, ToolFault
from llmock.schema import example_for, from_openapi
from llmock.simulation import MockResponseSettings, build_mock_text, estimate_tokens
from llmock.tools import ToolContext, auto_tool_call, tool_context

__all__ = [
    "Completion",
    "FinishReason",
    "ToolCallOut",
    "plan_of",
    "resolve",
    "resolve_request",
    "structured_output",
]

#: Provider-neutral finish reasons; each encoder maps them to its own vocabulary.
FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]

#: Name emitted by an ``unknown_tool`` fault: a tool that was never offered.
UNKNOWN_TOOL_NAME = "llmock_unknown_tool"

# Marker for "JSON requested, but no schema given" (json_object mode).
_ANY_JSON = object()


@dataclass(frozen=True)
class ToolCallOut:
    """A tool call as sent on the wire.

    ``arguments`` is the raw JSON string, because a :class:`ToolFault` can
    make it deliberately invalid.
    """

    id: str
    name: str
    arguments: str

    @property
    def parsed_arguments(self) -> Any:
        """The arguments as an object, or ``{}`` when they are malformed."""
        try:
            return json.loads(self.arguments)
        except ValueError:
            return {}


@dataclass(frozen=True)
class Completion:
    text: str
    tool_calls: tuple[ToolCallOut, ...]
    finish_reason: FinishReason | str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def plan_of(request: Any) -> Plan:
    """The plan the middleware attached to this request, or an empty one."""
    return getattr(request.state, "llmock_plan", None) or Plan()


def new_tool_call_id(prefix: str = "call") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def resolve_request(
    request: Any,
    *,
    model: str,
    prompt_text: str,
    prompt_tokens: int,
    tool_call_prefix: str = "call",
    dialect: str | None = None,
) -> Completion:
    """:func:`resolve`, reading tools and output format from the request.

    ``dialect`` overrides the provider when one provider exposes two request
    shapes (OpenAI chat vs. the Responses API).
    """
    info = getattr(request.state, "llmock_info", None)
    body = getattr(request.state, "llmock_body", None) or {}
    kind = dialect or (info.provider if info is not None else "openai")
    return resolve(
        plan=plan_of(request),
        settings=request.app.state.mock_response_settings,
        model=model,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
        tool_call_prefix=tool_call_prefix,
        tools=tool_context(kind, body, prompt_text),
        structured=structured_output(kind, body),
    )


def resolve(
    *,
    plan: Plan,
    settings: MockResponseSettings,
    model: str,
    prompt_text: str,
    prompt_tokens: int,
    tool_call_prefix: str = "call",
    tools: ToolContext | None = None,
    structured: Any = None,
) -> Completion:
    """Decide the completion for one request. See the module docstring."""
    reply = plan.reply
    calls: tuple[ToolCallOut, ...] = ()
    if reply is not None and (reply.text is not None or reply.tool_calls):
        text = reply.text or ""
        calls = tuple(_out(call.name, call.arguments, tool_call_prefix) for call in reply.tool_calls)
    else:
        auto = auto_tool_call(tools) if tools is not None and settings.tool_mode == "auto" else None
        if auto is not None:
            text = ""
            calls = (_out(auto.name, auto.arguments, tool_call_prefix),)
        elif structured is not None:
            text = _structured_text(structured, settings, model, prompt_text)
        else:
            text = build_mock_text(settings=settings, model=model, prompt=prompt_text)

    if plan.tool_fault is not None:
        calls = _break(plan.tool_fault, calls, tools, tool_call_prefix)

    if reply is not None and reply.finish_reason:
        finish: str = reply.finish_reason
    else:
        finish = "tool_calls" if calls else "stop"

    completion_tokens = estimate_tokens(text) + sum(estimate_tokens(c.arguments) for c in calls)
    return Completion(
        text=text,
        tool_calls=calls,
        finish_reason=finish,
        prompt_tokens=prompt_tokens,
        completion_tokens=max(1, completion_tokens),
    )


def _out(name: str, arguments: dict[str, Any], prefix: str) -> ToolCallOut:
    return ToolCallOut(id=new_tool_call_id(prefix), name=name, arguments=json.dumps(arguments))


def _break(
    fault: ToolFault,
    calls: tuple[ToolCallOut, ...],
    tools: ToolContext | None,
    prefix: str,
) -> tuple[ToolCallOut, ...]:
    """Corrupt the first tool call, making one first if there is none."""
    if not calls:
        forced = replace(tools, choice="required", answered=False) if tools else None
        call = auto_tool_call(forced) if forced is not None else None
        calls = (_out(call.name if call else UNKNOWN_TOOL_NAME, call.arguments if call else {}, prefix),)
    first, rest = calls[0], calls[1:]
    if fault.kind == "unknown_tool":
        first = replace(first, name=UNKNOWN_TOOL_NAME)
    else:  # malformed_arguments: cut the JSON in half, leaving it unparseable
        first = replace(first, arguments=first.arguments[: max(1, len(first.arguments) // 2)])
    return (first, *rest)


def _structured_text(structured: Any, settings: MockResponseSettings, model: str, prompt: str) -> str:
    if structured is _ANY_JSON:
        message = build_mock_text(settings=settings, model=model, prompt=prompt)
        return json.dumps({"message": message})
    return json.dumps(example_for(structured))


def structured_output(dialect: str, body: Any) -> Any:
    """The JSON Schema a request asks the answer to follow.

    Returns the schema, a marker for "any JSON object", or None for text.
    """
    if not isinstance(body, dict):
        return None
    if dialect == "gemini":
        config = body.get("generationConfig") or {}
        schema = config.get("responseJsonSchema") or config.get("responseSchema")
        if schema:
            return from_openapi(schema)
        return _ANY_JSON if config.get("responseMimeType") == "application/json" else None
    if dialect == "anthropic":
        fmt = body.get("output_format") or {}
        return fmt.get("schema") if isinstance(fmt, dict) and fmt.get("type") == "json_schema" else None
    if dialect == "openai-responses":
        fmt = ((body.get("text") or {}).get("format")) or {}
    else:
        fmt = body.get("response_format") or {}
    if not isinstance(fmt, dict):
        return None
    kind = fmt.get("type")
    if kind == "json_schema":
        nested = fmt.get("json_schema")
        schema = nested.get("schema") if isinstance(nested, dict) else fmt.get("schema")
        return schema if isinstance(schema, dict) else _ANY_JSON
    if kind == "json_object":
        schema = fmt.get("json_schema") or fmt.get("schema")  # Cohere puts it here
        return schema if isinstance(schema, dict) else _ANY_JSON
    return None
