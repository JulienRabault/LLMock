"""Scripted behaviours that upcoming requests consume.

A scenario is a queue of behaviours. Each incoming request takes, *per
category*, the first behaviour whose :class:`Match` accepts it. Categories are
independent, so a scripted reply and a stream fault can both land on the same
request::

    queue.add(Reply(tool_calls=(ToolCall("get_weather", {"city": "Paris"}),)))
    queue.add(StreamFault("disconnect", after_chunks=3))
    # -> the next streaming request starts a tool call, then drops mid-stream

A :class:`Fail` short-circuits everything else: the request gets an error
response and never reaches the provider handler.
"""

from __future__ import annotations

import fnmatch
import threading
from dataclasses import dataclass, field
from typing import Any, Literal, Union

__all__ = [
    "Behavior",
    "Delay",
    "Fail",
    "Match",
    "Plan",
    "Reply",
    "RequestInfo",
    "ScenarioQueue",
    "SlowFirstToken",
    "StreamFault",
    "ToolCall",
    "ToolFault",
    "behavior_from_dict",
    "behavior_to_dict",
]

StreamFaultKind = Literal["disconnect", "truncate", "stall", "malformed"]
ToolFaultKind = Literal["malformed_arguments", "unknown_tool"]


@dataclass(frozen=True)
class RequestInfo:
    """What the scenario engine knows about a request before handling it."""

    provider: str
    method: str
    path: str
    model: str | None = None
    stream: bool = False
    has_tools: bool = False


@dataclass(frozen=True)
class Match:
    """Which requests a behaviour applies to. Unset fields match anything.

    ``path`` and ``model`` accept shell-style wildcards, e.g. ``"gpt-4*"``.
    """

    provider: str | None = None
    path: str | None = None
    model: str | None = None
    stream: bool | None = None
    tools: bool | None = None
    """Whether the request offers tools."""

    def accepts(self, info: RequestInfo) -> bool:
        if self.provider is not None and self.provider != info.provider:
            return False
        if self.path is not None and not fnmatch.fnmatchcase(info.path, self.path):
            return False
        if self.model is not None and not fnmatch.fnmatchcase(info.model or "", self.model):
            return False
        if self.tools is not None and self.tools != info.has_tools:
            return False
        return self.stream is None or self.stream == info.stream


ANY = Match()


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation the mock model should emit."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Fail:
    """Answer with an HTTP error instead of a completion."""

    status: int
    retry_after: float | None = None
    message: str | None = None
    code: str | None = None
    times: int | None = 1
    match: Match = ANY

    def __post_init__(self) -> None:
        if not 400 <= self.status <= 599:
            raise ValueError(f"Fail.status must be an HTTP error (400-599), got {self.status}")
        _check_times(self.times)


@dataclass(frozen=True)
class Delay:
    """Hold the whole response for ``seconds`` before answering."""

    seconds: float
    times: int | None = 1
    match: Match = ANY

    def __post_init__(self) -> None:
        if self.seconds < 0:
            raise ValueError("Delay.seconds must be >= 0")
        _check_times(self.times)


@dataclass(frozen=True)
class SlowFirstToken:
    """Wait ``seconds`` before the first streamed chunk (time to first token)."""

    seconds: float
    times: int | None = 1
    match: Match = field(default_factory=lambda: Match(stream=True))

    def __post_init__(self) -> None:
        if self.seconds < 0:
            raise ValueError("SlowFirstToken.seconds must be >= 0")
        _check_times(self.times)


@dataclass(frozen=True)
class StreamFault:
    """Break a streamed response after ``after_chunks`` chunks.

    - ``disconnect``: drop the connection. The client sees a transport error.
    - ``truncate``: end the stream cleanly but early, with no finish reason and
      no terminator. The nasty one: many clients accept it as a full answer.
    - ``stall``: stop sending for ``stall_seconds``, then carry on. Exercises
      read timeouts.
    - ``malformed``: emit a chunk that is not valid JSON.
    """

    kind: StreamFaultKind
    after_chunks: int = 1
    stall_seconds: float = 30.0
    times: int | None = 1
    match: Match = field(default_factory=lambda: Match(stream=True))

    def __post_init__(self) -> None:
        if self.kind not in ("disconnect", "truncate", "stall", "malformed"):
            raise ValueError(f"Unknown stream fault kind {self.kind!r}")
        if self.after_chunks < 0:
            raise ValueError("StreamFault.after_chunks must be >= 0")
        if self.stall_seconds < 0:
            raise ValueError("StreamFault.stall_seconds must be >= 0")
        _check_times(self.times)


@dataclass(frozen=True)
class Reply:
    """Script what the mock model says: plain text, tool calls, or both."""

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    times: int | None = 1
    match: Match = ANY

    def __post_init__(self) -> None:
        _check_times(self.times)


@dataclass(frozen=True)
class ToolFault:
    """Emit a broken tool call, to test how an agent loop copes.

    - ``malformed_arguments``: the arguments are not valid JSON.
    - ``unknown_tool``: the model calls a tool that was never offered.

    Waits for a request that offers tools, by default.
    """

    kind: ToolFaultKind
    times: int | None = 1
    match: Match = field(default_factory=lambda: Match(tools=True))

    def __post_init__(self) -> None:
        if self.kind not in ("malformed_arguments", "unknown_tool"):
            raise ValueError(f"Unknown tool fault kind {self.kind!r}")
        _check_times(self.times)


Behavior = Union[Fail, Delay, SlowFirstToken, StreamFault, Reply, ToolFault]

# Order in which categories are resolved for one request.
_CATEGORIES: tuple[type, ...] = (Fail, Delay, SlowFirstToken, StreamFault, Reply, ToolFault)


@dataclass(frozen=True)
class Plan:
    """Everything decided for one request, at most one behaviour per category."""

    fail: Fail | None = None
    delay: Delay | None = None
    slow_first_token: SlowFirstToken | None = None
    stream_fault: StreamFault | None = None
    reply: Reply | None = None
    tool_fault: ToolFault | None = None

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.fail,
                self.delay,
                self.slow_first_token,
                self.stream_fault,
                self.reply,
                self.tool_fault,
            )
        )


_PLAN_FIELD = {
    Fail: "fail",
    Delay: "delay",
    SlowFirstToken: "slow_first_token",
    StreamFault: "stream_fault",
    Reply: "reply",
    ToolFault: "tool_fault",
}


class ScenarioQueue:
    """Thread-safe FIFO of behaviours, consumed by matching requests.

    ``times=None`` makes a behaviour permanent until :meth:`clear`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Each entry is [behaviour, remaining]; remaining is None when permanent.
        self._entries: list[list[Any]] = []

    def add(self, *behaviors: Behavior) -> None:
        with self._lock:
            for behavior in behaviors:
                if type(behavior) not in _PLAN_FIELD:
                    raise TypeError(f"Not a scenario behaviour: {behavior!r}")
                self._entries.append([behavior, behavior.times])

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def pending(self) -> list[Behavior]:
        """Behaviours still waiting to fire, in queue order."""
        with self._lock:
            return [entry[0] for entry in self._entries]

    def plan_for(self, info: RequestInfo) -> Plan:
        """Consume, per category, the first behaviour that accepts ``info``."""
        chosen: dict[str, Behavior] = {}
        with self._lock:
            for category in _CATEGORIES:
                for entry in self._entries:
                    behavior = entry[0]
                    if type(behavior) is category and behavior.match.accepts(info):
                        chosen[_PLAN_FIELD[category]] = behavior
                        self._consume(entry)
                        break
                if category is Fail and "fail" in chosen:
                    # An error answer ends the request; keep the rest queued.
                    break
            self._entries = [e for e in self._entries if e[1] is None or e[1] > 0]
        return Plan(**chosen)

    @staticmethod
    def _consume(entry: list[Any]) -> None:
        if entry[1] is not None:
            entry[1] -= 1


def _check_times(times: int | None) -> None:
    if times is not None and times < 1:
        raise ValueError("times must be >= 1, or None for a permanent behaviour")


# -- JSON serialisation, for the admin HTTP API --------------------------------

_TYPE_NAMES: dict[str, type] = {
    "fail": Fail,
    "delay": Delay,
    "slow_first_token": SlowFirstToken,
    "stream_fault": StreamFault,
    "reply": Reply,
    "tool_fault": ToolFault,
}
_NAME_OF_TYPE = {cls: name for name, cls in _TYPE_NAMES.items()}


def behavior_from_dict(data: dict[str, Any]) -> Behavior:
    """Build a behaviour from its JSON form, e.g. ``{"type": "fail", "status": 429}``.

    Raises:
        ValueError: unknown type, unknown field, or an invalid value.
    """
    if not isinstance(data, dict):
        raise ValueError("A behaviour must be a JSON object")
    fields_ = dict(data)
    type_name = fields_.pop("type", None)
    cls = _TYPE_NAMES.get(type_name)  # type: ignore[arg-type]
    if cls is None:
        raise ValueError(
            f"Unknown behaviour type {type_name!r}; expected one of {sorted(_TYPE_NAMES)}"
        )
    if "match" in fields_:
        match = fields_["match"]
        if not isinstance(match, dict):
            raise ValueError("'match' must be a JSON object")
        try:
            fields_["match"] = Match(**match)
        except TypeError as exc:
            raise ValueError(f"Invalid match: {exc}") from None
    if "tool_calls" in fields_:
        calls = fields_["tool_calls"]
        if not isinstance(calls, list):
            raise ValueError("'tool_calls' must be a list")
        try:
            fields_["tool_calls"] = tuple(ToolCall(**call) for call in calls)
        except TypeError as exc:
            raise ValueError(f"Invalid tool call: {exc}") from None
    try:
        return cls(**fields_)
    except TypeError as exc:
        raise ValueError(f"Invalid {type_name!r} behaviour: {exc}") from None


def behavior_to_dict(behavior: Behavior) -> dict[str, Any]:
    """The JSON form of a behaviour; the inverse of :func:`behavior_from_dict`."""
    from dataclasses import asdict

    data = asdict(behavior)
    data["type"] = _NAME_OF_TYPE[type(behavior)]
    data["match"] = {k: v for k, v in data["match"].items() if v is not None}
    if "tool_calls" in data:
        data["tool_calls"] = [dict(call) for call in data["tool_calls"]]
    return data
