"""A record of every request LLMock served, and how it answered.

The journal is what lets LLMock judge a client rather than only break it:
retries of the same call share a :attr:`RequestRecord.fingerprint`, and the
timestamps show whether the client waited as long as ``Retry-After`` asked.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["Journal", "RequestRecord", "Ticket", "fingerprint"]

DEFAULT_CAPACITY = 10_000


@dataclass(frozen=True)
class RequestRecord:
    """One request and its outcome. Times come from ``time.monotonic()``."""

    seq: int
    provider: str
    method: str
    path: str
    started_at: float
    ended_at: float
    status: int
    fingerprint: str
    model: str | None = None
    stream: bool = False
    fault: str | None = None
    retry_after: float | None = None
    completed: bool = True
    chunks_sent: int = 0
    sdk_retry_count: int | None = None
    body: Any = None

    @property
    def duration(self) -> float:
        return self.ended_at - self.started_at

    @property
    def failed(self) -> bool:
        return self.status >= 400 or not self.completed

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = round(self.duration, 6)
        return data


def fingerprint(method: str, path: str, body: Any) -> str:
    """Identify a logical call, so that its retries group together.

    A retry resends the same method, path and body, so hashing them in a
    canonical form -- key order does not matter -- ties every attempt of one
    call together.
    """
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(f"{method.upper()} {path}\n{canonical}".encode()).hexdigest()
    return digest[:16]


@dataclass(frozen=True)
class Ticket:
    """Handed out when a request starts; returned when it is recorded."""

    seq: int
    generation: int


class Journal:
    """Thread-safe, bounded log of :class:`RequestRecord`.

    Bounded so that a long-running ``llmock serve`` cannot grow without
    limit; the oldest records are dropped first.

    Two things make it safe to read right after a client call returns:

    - it counts requests still in flight, and :meth:`records` can wait for
      them. A client may stop reading at ``[DONE]`` before the server has
      sent its last byte, or give up on a stalled stream the server is still
      serving;
    - :meth:`clear` starts a new generation. A request that began before the
      clear but ends after it is discarded instead of leaking into the next
      test.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("Journal capacity must be >= 1")
        self._cond = threading.Condition()
        self._records: deque[RequestRecord] = deque(maxlen=capacity)
        self._next_seq = 1
        self._generation = 0
        # Counted per generation: a request stalled before clear() must not
        # make readers of the next generation wait for it.
        self._in_flight: dict[int, int] = {}

    def begin(self) -> Ticket:
        """Register a request as in flight."""
        with self._cond:
            ticket = Ticket(seq=self._next_seq, generation=self._generation)
            self._next_seq += 1
            self._in_flight[ticket.generation] = self._in_flight.get(ticket.generation, 0) + 1
            return ticket

    def end(self, ticket: Ticket) -> None:
        """Mark a request begun with :meth:`begin` as finished."""
        with self._cond:
            remaining = self._in_flight.get(ticket.generation, 0) - 1
            if remaining > 0:
                self._in_flight[ticket.generation] = remaining
            else:
                self._in_flight.pop(ticket.generation, None)
            self._cond.notify_all()

    def add(self, record: RequestRecord, ticket: Ticket | None = None) -> None:
        with self._cond:
            if ticket is not None and ticket.generation != self._generation:
                return  # started before the last clear(): belongs to a finished test
            self._records.append(record)
            self._cond.notify_all()

    def records(self, *, wait: float | None = None) -> list[RequestRecord]:
        """A snapshot, ordered by arrival.

        With ``wait``, first block up to that many seconds for in-flight
        requests to finish. Requests still running after that are left out.
        """
        with self._cond:
            if wait:
                self._cond.wait_for(lambda: self._current_in_flight() == 0, timeout=wait)
            return sorted(self._records, key=lambda r: r.seq)

    @property
    def in_flight(self) -> int:
        """Requests of the current generation still being served."""
        with self._cond:
            return self._current_in_flight()

    def _current_in_flight(self) -> int:
        return self._in_flight.get(self._generation, 0)

    def clear(self) -> None:
        with self._cond:
            self._records.clear()
            self._generation += 1

    def __len__(self) -> int:
        with self._cond:
            return len(self._records)
