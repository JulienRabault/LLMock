"""Quota-based rate limiting, the way providers actually enforce it.

A probability of 429 tests that a client *can* retry. A quota tests whether
it *regulates itself*: requests-per-minute and tokens-per-minute buckets that
refill continuously, a ``Retry-After`` equal to the real time until enough
capacity is back, and rate-limit headers on every response so that clients
reading them can slow down before hitting the wall.

Buckets are kept per provider and API key, like real organisation limits.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

__all__ = ["Admission", "LimitSettings", "RateLimiter", "ratelimit_headers"]


@dataclass(frozen=True)
class LimitSettings:
    rpm: int | None = None
    """Requests per minute per provider and API key. None disables the limit."""
    tpm: int | None = None
    """Tokens per minute (prompt estimate plus requested max tokens)."""
    context_window: int | None = None
    """Prompts estimated above this many tokens get a context-length error."""

    @classmethod
    def from_env(cls) -> LimitSettings:
        return cls(
            rpm=_int_env("LLMOCK_RPM"),
            tpm=_int_env("LLMOCK_TPM"),
            context_window=_int_env("LLMOCK_CONTEXT_WINDOW"),
        ).validated()

    def validated(self) -> LimitSettings:
        for name in ("rpm", "tpm", "context_window"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be a positive integer")
        return self

    def as_env(self) -> dict[str, str]:
        env = {}
        for name, var in (("rpm", "LLMOCK_RPM"), ("tpm", "LLMOCK_TPM"),
                          ("context_window", "LLMOCK_CONTEXT_WINDOW")):
            value = getattr(self, name)
            if value is not None:
                env[var] = str(value)
        return env

    @property
    def enabled(self) -> bool:
        return self.rpm is not None or self.tpm is not None


@dataclass(frozen=True)
class Quota:
    limit: int
    remaining: int
    reset_seconds: float
    """Seconds until the bucket is full again."""


@dataclass(frozen=True)
class Admission:
    allowed: bool
    requests: Quota | None
    tokens: Quota | None
    retry_after: float | None = None
    exhausted: str | None = None
    """``"requests"`` or ``"tokens"``: which bucket refused the request."""


class _Bucket:
    """Continuously refilling token bucket, full at start."""

    def __init__(self, capacity: int, per_minute: int) -> None:
        self.capacity = capacity
        self.rate = per_minute / 60.0
        self.level = float(capacity)
        self.updated = time.monotonic()

    def refill(self, now: float) -> None:
        self.level = min(self.capacity, self.level + (now - self.updated) * self.rate)
        self.updated = now

    def wait_for(self, amount: float) -> float:
        missing = amount - self.level
        return 0.0 if missing <= 0 else missing / self.rate

    def quota(self) -> Quota:
        return Quota(
            limit=self.capacity,
            remaining=max(0, math.floor(self.level)),
            reset_seconds=(self.capacity - self.level) / self.rate,
        )


class RateLimiter:
    """Thread-safe RPM/TPM buckets keyed by (provider, API key)."""

    def __init__(self, settings: LimitSettings | None = None) -> None:
        self._lock = threading.Lock()
        self._settings = settings or LimitSettings()
        self._buckets: dict[tuple[str, str, str], _Bucket] = {}

    @property
    def settings(self) -> LimitSettings:
        return self._settings

    def configure(self, settings: LimitSettings) -> None:
        with self._lock:
            self._settings = settings.validated()
            self._buckets.clear()

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()

    def admit(self, provider: str, key: str, tokens: int) -> Admission:
        """Take one request and ``tokens`` tokens, or refuse with how long to wait."""
        with self._lock:
            now = time.monotonic()
            req = self._bucket(provider, key, "requests", self._settings.rpm, now)
            tok = self._bucket(provider, key, "tokens", self._settings.tpm, now)

            waits = []
            if req is not None:
                waits.append(("requests", req.wait_for(1)))
            if tok is not None:
                waits.append(("tokens", tok.wait_for(tokens)))
            blocking = [(name, wait) for name, wait in waits if wait > 0]
            if blocking:
                name, wait = max(blocking, key=lambda item: item[1])
                return Admission(False, req.quota() if req else None,
                                 tok.quota() if tok else None, retry_after=wait, exhausted=name)

            if req is not None:
                req.level -= 1
            if tok is not None:
                tok.level -= tokens
            return Admission(True, req.quota() if req else None, tok.quota() if tok else None)

    def _bucket(self, provider: str, key: str, kind: str, per_minute: int | None,
                now: float) -> _Bucket | None:
        if per_minute is None:
            return None
        slot = (provider, key, kind)
        bucket = self._buckets.get(slot)
        if bucket is None:
            bucket = self._buckets[slot] = _Bucket(per_minute, per_minute)
        bucket.refill(now)
        return bucket


def ratelimit_headers(provider: str, admission: Admission) -> dict[str, str]:
    """Rate-limit headers in the provider's own format."""
    headers: dict[str, str] = {}
    if provider == "anthropic":
        for kind, quota in (("requests", admission.requests), ("tokens", admission.tokens)):
            if quota is None:
                continue
            reset = datetime.now(timezone.utc) + timedelta(seconds=quota.reset_seconds)
            headers[f"anthropic-ratelimit-{kind}-limit"] = str(quota.limit)
            headers[f"anthropic-ratelimit-{kind}-remaining"] = str(quota.remaining)
            headers[f"anthropic-ratelimit-{kind}-reset"] = reset.isoformat(timespec="seconds").replace("+00:00", "Z")
        return headers
    # OpenAI's format, which Groq, Together and most compatible providers copy.
    for kind, quota in (("requests", admission.requests), ("tokens", admission.tokens)):
        if quota is None:
            continue
        headers[f"x-ratelimit-limit-{kind}"] = str(quota.limit)
        headers[f"x-ratelimit-remaining-{kind}"] = str(quota.remaining)
        headers[f"x-ratelimit-reset-{kind}"] = _duration(quota.reset_seconds)
    return headers


def _duration(seconds: float) -> str:
    """OpenAI's reset format: ``20ms``, ``1s``, ``6m0s``."""
    if seconds < 1:
        return f"{max(0, round(seconds * 1000))}ms"
    minutes, secs = divmod(seconds, 60)
    if minutes >= 1:
        return f"{int(minutes)}m{secs:.0f}s"
    return f"{secs:.3g}s"


def requested_tokens(body: Any, raw_size: int) -> int:
    """What a request charges against TPM: a prompt estimate plus the output budget."""
    prompt = max(1, raw_size // 4)
    budget = 0
    if isinstance(body, dict):
        for field in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            if isinstance(body.get(field), int):
                budget = body[field]
                break
        config = body.get("generationConfig")
        if not budget and isinstance(config, dict) and isinstance(config.get("maxOutputTokens"), int):
            budget = config["maxOutputTokens"]
    return prompt + budget


def _int_env(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else None
