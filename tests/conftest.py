"""Shared test fixtures."""

import os

import pytest

from llmock.routers import batch as batch_router
from llmock.simulation import ERROR_RATE_ENV_PREFIX


@pytest.fixture(autouse=True)
def reset_llmock_env():
    original = dict(os.environ)
    for key in list(os.environ):
        if key.startswith(ERROR_RATE_ENV_PREFIX) or key in {
            "LLMOCK_LATENCY_MS",
            "LLMOCK_RESPONSE_STYLE",
        }:
            del os.environ[key]
    yield
    os.environ.clear()
    os.environ.update(original)


@pytest.fixture(autouse=True)
def reset_batch_state():
    batch_router._files.clear()
    batch_router._batches.clear()
    batch_router._datasets.clear()
    yield
    batch_router._files.clear()
    batch_router._batches.clear()
    batch_router._datasets.clear()
