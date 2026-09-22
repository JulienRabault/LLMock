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

__all__ = ["Journal", "RequestRecord", "fingerprint"]

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


class Journal:
    """Thread-safe, bounded log of :class:`RequestRecord`.

    Bounded so that a long-running ``llmock serve`` cannot grow without
    limit; the oldest records are dropped first.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("Journal capacity must be >= 1")
        self._lock = threading.Lock()
        self._records: deque[RequestRecord] = deque(maxlen=capacity)
        self._next_seq = 1

    def next_seq(self) -> int:
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            return seq

    def add(self, record: RequestRecord) -> None:
        with self._lock:
            self._records.append(record)

    def records(self) -> list[RequestRecord]:
        """A snapshot, oldest first, ordered by arrival."""
        with self._lock:
            return sorted(self._records, key=lambda r: r.seq)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
