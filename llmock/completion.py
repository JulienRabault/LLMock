"""What the mock model says, independently of any provider's wire format.

Routers turn a request into a :class:`Completion` with :func:`resolve`, then
render it: as a JSON body, or as a stream through :mod:`llmock.streaming`.
Keeping the decision in one place is what lets a scripted reply, a tool call
or a tool fault behave identically on OpenAI, Anthropic, Gemini and the
OpenAI-compatible providers.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from llmock.scenarios import Plan
from llmock.simulation import MockResponseSettings, build_mock_text, estimate_tokens

__all__ = ["Completion", "FinishReason", "ToolCallOut", "plan_of", "resolve"]

#: Provider-neutral finish reasons; each encoder maps them to its own vocabulary.
FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]


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


def resolve(
    *,
    plan: Plan,
    settings: MockResponseSettings,
    model: str,
    prompt_text: str,
    prompt_tokens: int,
    tool_call_prefix: str = "call",
) -> Completion:
    """Decide the completion for one request.

    A scripted :class:`~llmock.scenarios.Reply` wins; otherwise the configured
    response style produces the text.
    """
    reply = plan.reply
    if reply is not None and (reply.text is not None or reply.tool_calls):
        text = reply.text or ""
        calls = tuple(
            ToolCallOut(
                id=new_tool_call_id(tool_call_prefix),
                name=call.name,
                arguments=json.dumps(call.arguments),
            )
            for call in reply.tool_calls
        )
    else:
        text = build_mock_text(settings=settings, model=model, prompt=prompt_text)
        calls = ()

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
