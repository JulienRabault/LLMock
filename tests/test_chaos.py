"""Tests for Chaos Engineering middleware."""

import time

import pytest
from httpx import ASGITransport, AsyncClient

from llmock.chaos import ChaosSettings
from llmock.main import create_app


@pytest.fixture
def app():
    return create_app(chaos=ChaosSettings())


async def test_no_chaos_passes_through(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/models")
    assert resp.status_code == 200


async def test_health_bypasses_chaos(app):
    """Health endpoint must always succeed regardless of error rates."""
    app.state.chaos_settings.error_rate_429 = 0.6
    app.state.chaos_settings.error_rate_500 = 0.4
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200


async def test_429_always_injected(app):
    app.state.chaos_settings.error_rate_429 = 1.0
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/models")
    assert resp.status_code == 429
    body = resp.json()
    assert body["error"]["code"] == "rate_limit_exceeded"


async def test_500_always_injected(app):
    app.state.chaos_settings.error_rate_500 = 1.0
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/models")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "internal_error"


async def test_503_always_injected(app):
    app.state.chaos_settings.error_rate_503 = 1.0
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/models")
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "service_unavailable"


async def test_arbitrary_4xx_error_is_injected_with_provider_shape(app):
    app.state.chaos_settings.error_rate_451 = 1.0
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/models")
    assert resp.status_code == 451
    body = resp.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "http_451"


async def test_arbitrary_5xx_error_is_injected_with_gemini_shape(app):
    app.state.chaos_settings.error_rate_507 = 1.0
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/gemini/v1beta/models")
    assert resp.status_code == 507
    body = resp.json()
    assert body["error"]["status"] == "INTERNAL"
    assert body["error"]["code"] == 507


async def test_multiple_error_rates_are_sampled(app, monkeypatch):
    app.state.chaos_settings.error_rate_429 = 0.6
    app.state.chaos_settings.error_rate_503 = 0.4
    monkeypatch.setattr("llmock.chaos.random.random", lambda: 0.7)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/models")
    assert resp.status_code == 503


async def test_latency_injected(app):
    app.state.chaos_settings.latency_ms = 200
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        start = time.monotonic()
        resp = await client.get("/v1/models")
        elapsed_ms = (time.monotonic() - start) * 1000
    assert resp.status_code == 200
    assert elapsed_ms >= 190  # allow 10ms tolerance


async def test_zero_error_rate_never_injects(app):
    """With all rates at 0, no error should ever be injected over many requests."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for _ in range(20):
            resp = await client.get("/v1/models")
            assert resp.status_code == 200


def test_chaos_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLMOCK_LATENCY_MS", "150")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_401", "0.2")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_429", "0.3")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_451", "0.15")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_500", "0.1")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_503", "0.05")
    cfg = ChaosSettings.from_env()
    assert cfg.latency_ms == 150
    assert cfg.error_rate_401 == 0.2
    assert cfg.error_rate_429 == 0.3
    assert cfg.error_rate_451 == 0.15
    assert cfg.error_rate_500 == 0.1
    assert cfg.error_rate_503 == 0.05


def test_create_app_uses_explicit_chaos_settings():
    cfg = ChaosSettings(latency_ms=500, error_rate_429=0.5)
    app = create_app(chaos=cfg)
    assert app.state.chaos_settings is cfg
    assert app.state.chaos_settings.latency_ms == 500
    assert app.state.chaos_settings.error_rate_429 == 0.5


def test_create_app_reads_chaos_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLMOCK_LATENCY_MS", "250")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_401", "0.25")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_429", "0.4")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_507", "0.05")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_500", "0.2")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_503", "0.05")
    app = create_app()
    cfg = app.state.chaos_settings
    assert cfg.latency_ms == 250
    assert cfg.error_rate_401 == 0.25
    assert cfg.error_rate_429 == 0.4
    assert cfg.error_rate_507 == 0.05
    assert cfg.error_rate_500 == 0.2
    assert cfg.error_rate_503 == 0.05


def test_invalid_total_error_probability_raises():
    with pytest.raises(ValueError):
        ChaosSettings(error_rate_429=0.8, error_rate_503=0.5).validated()


def test_chaos_settings_accept_arbitrary_status_kwargs():
    cfg = ChaosSettings(error_rate_401=0.2, error_rate_504=0.1)
    assert cfg.error_rate_401 == 0.2
    assert cfg.error_rate_504 == 0.1
    assert cfg.error_rates[401] == 0.2
    assert cfg.error_rates[504] == 0.1


def test_chaos_settings_support_dynamic_status_assignment():
    cfg = ChaosSettings()
    cfg.error_rate_422 = 0.35
    assert cfg.error_rate_422 == 0.35
    assert cfg.error_rates[422] == 0.35


def test_invalid_http_error_code_raises():
    with pytest.raises(ValueError):
        ChaosSettings(error_rate_200=0.2).validated()
