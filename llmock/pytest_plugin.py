"""The ``llmock`` pytest fixture. Registered automatically when llmock is installed.

    def test_my_agent_survives_rate_limits(llmock):
        llmock.rate_limit(times=2, retry_after=0.1)
        answer = my_app.ask("What is the weather in Paris?")   # unchanged app code
        assert answer
        llmock.assert_resilient()

The fixture starts one real LLMock server per test session, and for each
test resets it and points the provider SDKs at it through the environment
variables they read (``OPENAI_BASE_URL``, ``ANTHROPIC_BASE_URL``,
``GOOGLE_GEMINI_BASE_URL``, ``CO_API_URL``). API keys are replaced by dummy
ones, so a client that still targets a real provider fails fast instead of
spending money.

``pytest --llmock-report`` prints, at the end of the run, the verdict of
every test whose client mishandled a fault.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from llmock.journal import RequestRecord
    from llmock.scenarios import Behavior
    from llmock.testing import LLMockServer
    from llmock.verdict import Verdict

__all__ = ["LLMock"]

# How long inspection waits for requests still in flight, e.g. a stream the
# client stopped reading at [DONE] before the server sent its last byte.
_WAIT_SECONDS = 2.0

_DUMMY_KEY = "llmock-test-key"

_REPORT_KEY = pytest.StashKey[list]()


class LLMock:
    """Script faults and inspect traffic for one test. Methods chain."""

    def __init__(self, server: LLMockServer) -> None:
        self._server = server

    # -- where to point clients -------------------------------------------

    @property
    def url(self) -> str:
        """Root URL of the server, e.g. ``http://127.0.0.1:53412``."""
        return self._server.url

    def base_url(self, provider: str = "openai") -> str:
        """The ``base_url`` a provider's SDK needs, e.g. ``.../v1`` for OpenAI."""
        return self._server.base_url(provider)

    def env(self) -> dict[str, str]:
        """Environment variables that point every supported SDK at this server."""
        return {
            "OPENAI_BASE_URL": self.base_url("openai"),
            "OPENAI_API_KEY": _DUMMY_KEY,
            "ANTHROPIC_BASE_URL": self.base_url("anthropic"),
            "ANTHROPIC_API_KEY": _DUMMY_KEY,
            "GOOGLE_GEMINI_BASE_URL": self.base_url("gemini"),
            "GOOGLE_API_KEY": _DUMMY_KEY,
            "CO_API_URL": self.base_url("cohere"),
            "CO_API_KEY": _DUMMY_KEY,
        }

    # -- faults -----------------------------------------------------------

    def fail(
        self,
        status: int,
        *,
        times: int | None = 1,
        retry_after: float | None = None,
        message: str | None = None,
        code: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> LLMock:
        """Answer the next ``times`` requests with an HTTP error."""
        from llmock.scenarios import Fail

        return self.add(Fail(status, retry_after=retry_after, message=message, code=code,
                             times=times, match=_match(provider, model)))

    def rate_limit(self, *, times: int | None = 1, retry_after: float = 1.0,
                   provider: str | None = None) -> LLMock:
        """429 with a ``Retry-After``."""
        return self.fail(429, times=times, retry_after=retry_after, provider=provider)

    def outage(self, *, status: int = 503, provider: str | None = None) -> LLMock:
        """The provider is down until :meth:`reset`: every request fails."""
        return self.fail(status, times=None, provider=provider)

    def context_overflow(self, *, times: int | None = 1, provider: str | None = None) -> LLMock:
        """The provider's 'prompt too long' error."""
        return self.fail(400, times=times, provider=provider, code="context_length_exceeded",
                         message="This model's maximum context length was exceeded.")

    def delay(self, seconds: float, *, times: int | None = 1, provider: str | None = None) -> LLMock:
        from llmock.scenarios import Delay

        return self.add(Delay(seconds, times=times, match=_match(provider, None)))

    def disconnect(self, *, after_chunks: int = 1, times: int | None = 1,
                   provider: str | None = None) -> LLMock:
        """Drop the connection in the middle of the next stream."""
        return self._stream_fault("disconnect", after_chunks, times, provider)

    def truncate(self, *, after_chunks: int = 1, times: int | None = 1,
                 provider: str | None = None) -> LLMock:
        """End the next stream early, cleanly, with no finish reason."""
        return self._stream_fault("truncate", after_chunks, times, provider)

    def corrupt(self, *, after_chunks: int = 1, times: int | None = 1,
                provider: str | None = None) -> LLMock:
        """Send a chunk that is not valid JSON in the next stream."""
        return self._stream_fault("malformed", after_chunks, times, provider)

    def stall(self, *, after_chunks: int = 1, seconds: float = 30.0, times: int | None = 1,
              provider: str | None = None) -> LLMock:
        """Stop sending for ``seconds`` in the middle of the next stream."""
        from llmock.scenarios import Match, StreamFault

        return self.add(StreamFault("stall", after_chunks=after_chunks, stall_seconds=seconds,
                                    times=times, match=Match(provider=provider, stream=True)))

    def slow_first_token(self, seconds: float, *, times: int | None = 1,
                         provider: str | None = None) -> LLMock:
        from llmock.scenarios import Match, SlowFirstToken

        return self.add(SlowFirstToken(seconds, times=times,
                                       match=Match(provider=provider, stream=True)))

    # -- what the model says ----------------------------------------------

    def reply(self, text: str, *, times: int | None = 1, provider: str | None = None) -> LLMock:
        """Script the text of the next answer."""
        from llmock.scenarios import Reply

        return self.add(Reply(text=text, times=times, match=_match(provider, None)))

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None, *,
                  text: str | None = None, times: int | None = 1,
                  provider: str | None = None) -> LLMock:
        """Make the next answer call ``name`` with ``arguments``."""
        from llmock.scenarios import Reply, ToolCall

        return self.add(Reply(text=text, tool_calls=(ToolCall(name, arguments or {}),),
                              times=times, match=_match(provider, None)))

    def break_tool_call(self, kind: str = "malformed_arguments", *, times: int | None = 1) -> LLMock:
        """Corrupt the next tool call: ``malformed_arguments`` or ``unknown_tool``."""
        from llmock.scenarios import ToolFault

        return self.add(ToolFault(kind, times=times))  # type: ignore[arg-type]

    def tool_mode(self, mode: str) -> LLMock:
        """``auto`` calls offered tools; ``off`` always answers in text."""
        from dataclasses import replace

        settings = self._server.app.state.mock_response_settings
        self._server.app.state.mock_response_settings = replace(settings, tool_mode=mode).validated()
        return self

    # -- quotas and pacing ----------------------------------------------

    def limits(self, *, rpm: int | None = None, tpm: int | None = None,
               context_window: int | None = None) -> LLMock:
        """Enforce real quotas: 429s with the true wait, and rate-limit headers."""
        from llmock.ratelimit import LimitSettings

        self._server.state.limiter.configure(
            LimitSettings(rpm=rpm, tpm=tpm, context_window=context_window))
        return self

    def pace(self, chunk_delay_ms: int) -> LLMock:
        """Space streamed chunks out, like a model generating token by token."""
        from dataclasses import replace

        state = self._server.state
        state.stream_chaos = replace(state.stream_chaos, chunk_delay_ms=chunk_delay_ms).validated()
        return self

    def add(self, *behaviors: Behavior) -> LLMock:
        """Queue raw :mod:`llmock.scenarios` behaviours."""
        self._server.state.scenarios.add(*behaviors)
        return self

    def _stream_fault(self, kind: str, after_chunks: int, times: int | None,
                      provider: str | None) -> LLMock:
        from llmock.scenarios import Match, StreamFault

        return self.add(StreamFault(kind, after_chunks=after_chunks, times=times,  # type: ignore[arg-type]
                                    match=Match(provider=provider, stream=True)))

    # -- inspection -------------------------------------------------------

    @property
    def requests(self) -> list[RequestRecord]:
        """Every request of this test, in arrival order."""
        return self._server.state.journal.records(wait=_WAIT_SECONDS)

    @property
    def last_request(self) -> RequestRecord:
        records = self.requests
        if not records:
            raise AssertionError("LLMock received no request in this test")
        return records[-1]

    def verdict(self) -> Verdict:
        from llmock.verdict import judge

        return judge(self.requests)

    def assert_resilient(self, *, strict: bool = False) -> None:
        """Fail the test with the verdict report if the client mishandled a fault."""
        self.verdict().assert_ok(strict=strict)

    def reset(self) -> LLMock:
        """Forget requests and queued behaviours; restore default settings."""
        from llmock.chaos import ChaosSettings, StreamChaos
        from llmock.ratelimit import LimitSettings
        from llmock.simulation import MockResponseSettings

        state = self._server.state
        state.reset()
        # Swap whole objects rather than mutate them: the server thread may be
        # reading these for a request still in flight from the previous test.
        fresh_chaos = ChaosSettings()
        state.chaos = fresh_chaos
        self._server.app.state.chaos_settings = fresh_chaos
        state.limiter.configure(LimitSettings())
        state.stream_chaos = StreamChaos()
        self._server.app.state.mock_response_settings = MockResponseSettings()
        return self


def _match(provider: str | None, model: str | None):
    from llmock.scenarios import Match

    return Match(provider=provider, model=model)


# -- pytest wiring --------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("llmock")
    group.addoption(
        "--llmock-report",
        action="store_true",
        default=False,
        help="print the LLMock resilience verdict of every test that used the fixture "
             "and whose client mishandled a fault",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_REPORT_KEY] = []


@pytest.fixture(scope="session")
def llmock_server() -> Iterator[LLMockServer]:
    """One real LLMock server for the whole session, started on first use."""
    from llmock.testing import LLMockServer

    with LLMockServer() as server:
        yield server


@pytest.fixture
def llmock(llmock_server: LLMockServer, monkeypatch: pytest.MonkeyPatch,
           request: pytest.FixtureRequest) -> Iterator[LLMock]:
    """A clean LLMock for this test, with the provider SDKs pointed at it."""
    handle = LLMock(llmock_server).reset()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # else google-genai warns about two keys
    for name, value in handle.env().items():
        monkeypatch.setenv(name, value)
    yield handle
    if request.config.getoption("llmock_report") and handle.requests:
        verdict = handle.verdict()
        if verdict.findings:
            request.config.stash[_REPORT_KEY].append((request.node.nodeid, verdict))


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    reports = config.stash.get(_REPORT_KEY, [])
    if not config.getoption("llmock_report", default=False):
        return
    terminalreporter.section("LLMock resilience report")
    if not reports:
        terminalreporter.write_line("Every test's client handled the injected faults correctly.")
        return
    for nodeid, verdict in reports:
        terminalreporter.write_line(nodeid, bold=True)
        for line in verdict.render().splitlines():
            terminalreporter.write_line(line, **_markup(line, has_errors=bool(verdict.errors)))
        terminalreporter.write_line("")


def _markup(line: str, *, has_errors: bool) -> dict[str, bool]:
    """pytest's own colour markup for one line of a verdict report."""
    stripped = line.strip()
    if stripped.startswith("FAIL"):
        return {"red": True, "bold": True}
    if stripped.startswith("WARN"):
        return {"yellow": True, "bold": True}
    if stripped.startswith("PASS"):
        return {"green": True, "bold": True}
    if stripped.endswith("warning(s)"):  # the "N error(s), M warning(s)" total
        return {"red": True, "bold": True} if has_errors else {"yellow": True, "bold": True}
    if line.startswith("LLMock resilience verdict"):
        return {"cyan": True}
    return {}
