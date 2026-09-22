"""Per-app runtime state shared by the middleware, the admin API and pytest."""

from __future__ import annotations

from llmock.chaos import ChaosSettings, StreamChaos
from llmock.journal import Journal
from llmock.ratelimit import LimitSettings, RateLimiter
from llmock.scenarios import ScenarioQueue

__all__ = ["LLMockState"]


class LLMockState:
    """Everything one LLMock app instance knows at runtime.

    Attached to ``app.state.llmock``. The components are fixed for the life of
    the app; what changes is their content (queued behaviours, journal,
    buckets) -- and ``stream_chaos``, which is swapped as a whole.
    """

    __slots__ = ("chaos", "journal", "limiter", "scenarios", "stream_chaos")

    def __init__(
        self,
        chaos: ChaosSettings,
        journal: Journal | None = None,
        scenarios: ScenarioQueue | None = None,
        limits: LimitSettings | None = None,
        stream_chaos: StreamChaos | None = None,
    ) -> None:
        self.chaos = chaos
        self.journal = journal or Journal()
        self.scenarios = scenarios or ScenarioQueue()
        self.limiter = RateLimiter(limits or LimitSettings())
        self.stream_chaos = stream_chaos or StreamChaos()

    def reset(self) -> None:
        """Forget recorded requests, queued behaviours and spent quota. Settings stay."""
        self.journal.clear()
        self.scenarios.clear()
        self.limiter.reset()
