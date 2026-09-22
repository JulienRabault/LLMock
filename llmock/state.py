"""Per-app runtime state shared by the middleware, the admin API and pytest."""

from __future__ import annotations

from llmock.chaos import ChaosSettings
from llmock.journal import Journal
from llmock.scenarios import ScenarioQueue

__all__ = ["LLMockState"]


class LLMockState:
    """Everything one LLMock app instance knows at runtime.

    Attached to ``app.state.llmock``. The components are fixed for the life of
    the app; what changes is their content (queued behaviours, journal).
    """

    __slots__ = ("chaos", "journal", "scenarios")

    def __init__(
        self,
        chaos: ChaosSettings,
        journal: Journal | None = None,
        scenarios: ScenarioQueue | None = None,
    ) -> None:
        self.chaos = chaos
        self.journal = journal or Journal()
        self.scenarios = scenarios or ScenarioQueue()

    def reset(self) -> None:
        """Forget recorded requests and queued behaviours. Chaos settings stay."""
        self.journal.clear()
        self.scenarios.clear()
