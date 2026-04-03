"""Chaos Engineering middleware for LLMock."""

from __future__ import annotations

import asyncio
import os
import random

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from llmock.simulation import (
    ERROR_RATE_ENV_PREFIX,
    SUPPORTED_ERROR_STATUS_CODES,
    build_error_response,
)


class ChaosSettings:
    """Runtime settings for latency and injected HTTP errors."""

    def __init__(
        self,
        *,
        latency_ms: int = 0,
        error_rates: dict[int, float] | None = None,
        error_rate_429: float = 0.0,
        error_rate_500: float = 0.0,
        error_rate_503: float = 0.0,
        **named_error_rates: float,
    ) -> None:
        object.__setattr__(self, "latency_ms", latency_ms)
        object.__setattr__(self, "_error_rates", {})

        combined = dict(error_rates or {})
        if error_rate_429 > 0.0:
            combined[429] = error_rate_429
        if error_rate_500 > 0.0:
            combined[500] = error_rate_500
        if error_rate_503 > 0.0:
            combined[503] = error_rate_503

        for key, value in named_error_rates.items():
            status_code = self._status_code_from_attr(key)
            if status_code is None:
                raise TypeError(f"Unexpected chaos setting '{key}'.")
            combined[status_code] = value

        self.error_rates = combined

    def __getattr__(self, name: str) -> float:
        status_code = self._status_code_from_attr(name)
        if status_code is not None:
            return self._error_rates.get(status_code, 0.0)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: int | float) -> None:
        status_code = self._status_code_from_attr(name)
        if status_code is not None:
            self._error_rates[status_code] = float(value)
            # Keep the pre-sorted cache in sync.
            object.__setattr__(
                self, "_sorted_error_rates", sorted(self._error_rates.items())
            )
            return
        object.__setattr__(self, name, value)

    @property
    def error_rates(self) -> dict[int, float]:
        return dict(self._error_rates)

    @error_rates.setter
    def error_rates(self, rates: dict[int, float]) -> None:
        normalized: dict[int, float] = {}
        for status, rate in rates.items():
            normalized[int(status)] = float(rate)
        object.__setattr__(self, "_error_rates", normalized)
        # Pre-sort for _sample_error_status so we don't sort on every request.
        object.__setattr__(
            self, "_sorted_error_rates", sorted(normalized.items())
        )

    @staticmethod
    def _status_code_from_attr(name: str) -> int | None:
        if not name.startswith("error_rate_"):
            return None
        suffix = name.removeprefix("error_rate_")
        if not suffix.isdigit():
            return None
        return int(suffix)

    @classmethod
    def from_env(cls) -> "ChaosSettings":
        parsed_rates: dict[int, float] = {}
        for key, value in os.environ.items():
            if not key.startswith(ERROR_RATE_ENV_PREFIX):
                continue
            suffix = key[len(ERROR_RATE_ENV_PREFIX) :]
            if not suffix.isdigit():
                continue
            parsed_rates[int(suffix)] = float(value)
        return cls(
            latency_ms=int(os.getenv("LLMOCK_LATENCY_MS", "0")),
            error_rates=parsed_rates,
        ).validated()

    def with_overrides(
        self,
        *,
        latency_ms: int | None = None,
        error_rates: dict[int, float] | None = None,
        error_rate_429: float | None = None,
        error_rate_500: float | None = None,
        error_rate_503: float | None = None,
    ) -> "ChaosSettings":
        combined = dict(self.error_rates)
        if error_rates:
            combined.update(error_rates)
        if error_rate_429 is not None:
            combined[429] = error_rate_429
        if error_rate_500 is not None:
            combined[500] = error_rate_500
        if error_rate_503 is not None:
            combined[503] = error_rate_503
        return type(self)(
            latency_ms=self.latency_ms if latency_ms is None else latency_ms,
            error_rates=combined,
        ).validated()

    def validated(self) -> "ChaosSettings":
        if self.latency_ms < 0:
            raise ValueError("LLMOCK_LATENCY_MS must be greater than or equal to 0.")

        total_probability = 0.0
        for status, rate in self.error_rates.items():
            if status not in SUPPORTED_ERROR_STATUS_CODES:
                raise ValueError(
                    f"LLMOCK_ERROR_RATE_{status} must target an HTTP error code between 400 and 599."
                )
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"LLMOCK_ERROR_RATE_{status} must be between 0.0 and 1.0.")
            total_probability += rate

        if total_probability > 1.0:
            raise ValueError("The sum of simulated error probabilities must be <= 1.0.")

        return self

    def as_env(self) -> dict[str, str]:
        env = {"LLMOCK_LATENCY_MS": str(self.latency_ms)}
        for status, rate in self.error_rates.items():
            env[f"{ERROR_RATE_ENV_PREFIX}{status}"] = str(rate)
        return env


chaos_settings = ChaosSettings.from_env()


class ChaosMiddleware(BaseHTTPMiddleware):
    """Middleware that injects latency and random errors based on ChaosSettings."""

    def __init__(self, app, settings: ChaosSettings | None = None) -> None:
        super().__init__(app)
        self.settings = (settings or ChaosSettings.from_env()).validated()

    async def dispatch(self, request: Request, call_next):
        cfg = self.settings

        if request.url.path == "/health":
            return await call_next(request)

        forced_status = request.headers.get("x-llmock-force-status")
        if forced_status:
            status_code = int(forced_status)
            if 400 <= status_code < 600:
                return build_error_response(request.url.path, status_code)

        if cfg.latency_ms > 0:
            await asyncio.sleep(cfg.latency_ms / 1000.0)

        sampled_status = _sample_error_status(cfg)
        if sampled_status is not None:
            return build_error_response(request.url.path, sampled_status)

        return await call_next(request)


def _sample_error_status(settings: ChaosSettings) -> int | None:
    r = random.random()
    cumulative = 0.0
    for status, rate in settings._sorted_error_rates:
        cumulative += rate
        if r < cumulative:
            return status
    return None
