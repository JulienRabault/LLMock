"""Tool calling: read the tools a request offers, and decide whether to call one.

In ``auto`` mode LLMock behaves like a well-mannered model in an agent loop:

- the request offers tools and the conversation does not yet contain a tool
  result -> call the tool that best matches the prompt, with arguments valid
  against its JSON Schema;
- the last turn carries a tool result -> answer in text, ending the loop.

``tool_choice`` is honoured: ``none`` never calls, ``required`` (or ``any``)
always calls, and a named tool is always the one called. That is enough to
run a real LangChain, LlamaIndex or OpenAI Agents SDK agent end to end: it
executes its real tools, receives an answer, and terminates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from llmock.schema import example_for, from_openapi
from llmock.scenarios import ToolCall

__all__ = ["ToolContext", "ToolSpec", "auto_tool_call", "tool_context"]

Choice = Literal["auto", "none", "required"]

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    parameters: Any = None


@dataclass(frozen=True)
class ToolContext:
    tools: tuple[ToolSpec, ...]
    choice: Choice = "auto"
    forced_name: str | None = None
    answered: bool = False
    """True when the last turn carries a tool result."""
    prompt: str = ""


def tool_context(provider: str, body: Any, prompt: str = "") -> ToolContext | None:
    """Extract the offered tools and the tool choice from a request body."""
    if not isinstance(body, dict):
        return None
    reader = _READERS.get(provider, _read_openai_chat)
    return reader(body, prompt)


def auto_tool_call(ctx: ToolContext) -> ToolCall | None:
    """The tool call a sensible model would make, or None to answer in text."""
    if not ctx.tools or ctx.choice == "none":
        return None
    if ctx.forced_name is not None:
        spec = next((t for t in ctx.tools if t.name == ctx.forced_name), None)
        return _call(spec) if spec else ToolCall(ctx.forced_name, {})
    if ctx.choice == "auto" and ctx.answered:
        return None
    return _call(_best_match(ctx.tools, ctx.prompt))


def _call(spec: ToolSpec) -> ToolCall:
    arguments = example_for(spec.parameters) if spec.parameters else {}
    return ToolCall(spec.name, arguments if isinstance(arguments, dict) else {})


def _best_match(tools: tuple[ToolSpec, ...], prompt: str) -> ToolSpec:
    """The tool whose name and description share most words with the prompt.

    Ties go to the first tool, so the choice is deterministic.
    """
    words = set(_WORD_RE.findall(prompt.lower()))
    if not words:
        return tools[0]

    def score(spec: ToolSpec) -> int:
        name_words = set(_WORD_RE.findall(spec.name.replace("_", " ").lower()))
        text = set(_WORD_RE.findall(spec.description.lower()))
        return 3 * len(words & name_words) + len(words & text)

    return max(tools, key=score)  # max() keeps the first of equal scores


# -- per-provider readers -----------------------------------------------------


def _read_openai_chat(body: dict[str, Any], prompt: str) -> ToolContext | None:
    tools = tuple(
        ToolSpec(
            name=str(f.get("name", "")),
            description=str(f.get("description") or ""),
            parameters=f.get("parameters"),
        )
        for t in body.get("tools") or []
        if isinstance(t, dict)
        for f in [t.get("function") if isinstance(t.get("function"), dict) else t]
        if f.get("name")
    )
    if not tools:
        return None
    choice, forced = _openai_choice(body.get("tool_choice"))
    messages = body.get("messages") or []
    answered = bool(messages) and isinstance(messages[-1], dict) and messages[-1].get("role") == "tool"
    return ToolContext(tools, choice, forced, answered, prompt)


def _read_openai_responses(body: dict[str, Any], prompt: str) -> ToolContext | None:
    tools = tuple(
        ToolSpec(
            name=str(t["name"]),
            description=str(t.get("description") or ""),
            parameters=t.get("parameters"),
        )
        for t in body.get("tools") or []
        if isinstance(t, dict) and t.get("type") == "function" and t.get("name")
    )
    if not tools:
        return None
    choice, forced = _openai_choice(body.get("tool_choice"))
    items = body.get("input")
    answered = (
        isinstance(items, list)
        and bool(items)
        and isinstance(items[-1], dict)
        and items[-1].get("type") == "function_call_output"
    )
    return ToolContext(tools, choice, forced, answered, prompt)


def _openai_choice(value: Any) -> tuple[Choice, str | None]:
    if value == "none":
        return "none", None
    if value == "required":
        return "required", None
    if isinstance(value, dict):
        name = value.get("name") or (value.get("function") or {}).get("name")
        if name:
            return "required", str(name)
    return "auto", None


def _read_anthropic(body: dict[str, Any], prompt: str) -> ToolContext | None:
    tools = tuple(
        ToolSpec(
            name=str(t["name"]),
            description=str(t.get("description") or ""),
            parameters=t.get("input_schema"),
        )
        for t in body.get("tools") or []
        # Server tools (web search, code execution...) carry a "type" and are
        # run by Anthropic, not the client: only client tools are callable here.
        if isinstance(t, dict) and t.get("name") and t.get("type") in (None, "custom")
    )
    if not tools:
        return None
    raw_choice = body.get("tool_choice") or {}
    kind = raw_choice.get("type") if isinstance(raw_choice, dict) else None
    choice: Choice = {"none": "none", "any": "required", "tool": "required"}.get(kind, "auto")  # type: ignore[assignment]
    forced = raw_choice.get("name") if kind == "tool" else None
    messages = body.get("messages") or []
    last = messages[-1] if messages and isinstance(messages[-1], dict) else {}
    content = last.get("content")
    answered = last.get("role") == "user" and isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content
    )
    return ToolContext(tools, choice, forced, answered, prompt)


def _read_gemini(body: dict[str, Any], prompt: str) -> ToolContext | None:
    tools = tuple(
        ToolSpec(
            name=str(d["name"]),
            description=str(d.get("description") or ""),
            parameters=(
                # google-genai sends the snake_case spelling; the REST docs
                # show camelCase. Accept both, then the OpenAPI-style schema.
                d.get("parametersJsonSchema")
                or d.get("parameters_json_schema")
                or from_openapi(d.get("parameters"))
            ),
        )
        for group in body.get("tools") or []
        if isinstance(group, dict)
        for d in group.get("functionDeclarations") or []
        if isinstance(d, dict) and d.get("name")
    )
    if not tools:
        return None
    config = ((body.get("toolConfig") or {}).get("functionCallingConfig")) or {}
    mode = str(config.get("mode", "AUTO")).upper()
    choice: Choice = {"NONE": "none", "ANY": "required"}.get(mode, "auto")  # type: ignore[assignment]
    allowed = config.get("allowedFunctionNames") or []
    forced = str(allowed[0]) if choice == "required" and len(allowed) == 1 else None
    contents = body.get("contents") or []
    last = contents[-1] if contents and isinstance(contents[-1], dict) else {}
    answered = any(
        isinstance(part, dict) and "functionResponse" in part for part in last.get("parts") or []
    )
    return ToolContext(tools, choice, forced, answered, prompt)




_READERS = {
    "anthropic": _read_anthropic,
    "gemini": _read_gemini,
    "openai-responses": _read_openai_responses,
}
