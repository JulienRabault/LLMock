"""Tests for the LLMock CLI."""

import json

from typer.testing import CliRunner

import llmock.cli as cli

runner = CliRunner()


def test_resolve_chaos_settings_prefers_explicit_overrides(monkeypatch):
    monkeypatch.setenv("LLMOCK_LATENCY_MS", "120")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_429", "0.2")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_500", "0.1")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_503", "0.05")

    settings = cli._resolve_chaos_settings(
        config={},
        latency_ms=250,
        error_rates=None,
        error_rate_429=0.6,
        error_rate_500=None,
        error_rate_503=0.3,
    )

    assert settings.latency_ms == 250
    assert settings.error_rate_429 == 0.6
    assert settings.error_rate_500 == 0.1
    assert settings.error_rate_503 == 0.3


def test_cli_uses_env_defaults_when_flags_are_omitted(monkeypatch):
    monkeypatch.setenv("LLMOCK_HOST", "0.0.0.0")
    monkeypatch.setenv("LLMOCK_PORT", "9001")
    monkeypatch.setenv("LLMOCK_LATENCY_MS", "175")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_404", "0.05")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_429", "0.25")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_500", "0.15")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_503", "0.05")

    captured: dict[str, object] = {}

    def fake_run(app_target, **kwargs):
        captured["app_target"] = app_target
        captured.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    result = runner.invoke(cli.app, ["serve"])

    assert result.exit_code == 0
    assert "Chaos: latency=175ms  errors=[404=5%, 429=25%, 500=15%, 503=5%]" in result.stdout
    assert "Responses: style=varied" in result.stdout
    assert captured["app_target"] == "llmock.main:create_app"
    assert captured["factory"] is True
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9001


def test_cli_flags_override_environment(monkeypatch):
    monkeypatch.setenv("LLMOCK_LATENCY_MS", "10")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_429", "0.1")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_500", "0.1")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_503", "0.1")

    def fake_run(*args, **kwargs):
        return None

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    result = runner.invoke(
        cli.app,
        [
            "serve",
            "--latency-ms",
            "300",
            "--error-rate-429",
            "0.5",
            "--error-rate-500",
            "0.2",
        ],
    )

    assert result.exit_code == 0
    assert "Chaos: latency=300ms  errors=[429=50%, 500=20%, 503=10%]" in result.stdout
    assert cli.os.environ["LLMOCK_LATENCY_MS"] == "300"
    assert cli.os.environ["LLMOCK_ERROR_RATE_429"] == "0.5"
    assert cli.os.environ["LLMOCK_ERROR_RATE_500"] == "0.2"
    assert cli.os.environ["LLMOCK_ERROR_RATE_503"] == "0.1"


def test_cli_accepts_generic_error_rate_option(monkeypatch):
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: None)

    result = runner.invoke(
        cli.app,
        [
            "serve",
            "--error-rate",
            "401=0.2",
            "--error-rate",
            "451=0.05",
            "--error-rate",
            "504=0.1",
            "--error-rate",
            "507=0.15",
        ],
    )

    assert result.exit_code == 0
    assert "errors=[401=20%, 451=5%, 504=10%, 507=15%]" in result.stdout
    assert cli.os.environ["LLMOCK_ERROR_RATE_401"] == "0.2"
    assert cli.os.environ["LLMOCK_ERROR_RATE_451"] == "0.05"
    assert cli.os.environ["LLMOCK_ERROR_RATE_504"] == "0.1"
    assert cli.os.environ["LLMOCK_ERROR_RATE_507"] == "0.15"


def test_cli_host_port_flags_override_environment(monkeypatch):
    monkeypatch.setenv("LLMOCK_HOST", "0.0.0.0")
    monkeypatch.setenv("LLMOCK_PORT", "9001")

    captured: dict[str, object] = {}

    def fake_run(app_target, **kwargs):
        captured["app_target"] = app_target
        captured.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    result = runner.invoke(cli.app, ["serve", "--host", "127.0.0.1", "--port", "8123"])

    assert result.exit_code == 0
    assert "Starting LLMock on http://127.0.0.1:8123" in result.stdout
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123


def test_cli_reads_json_config_file(tmp_path, monkeypatch):
    config_path = tmp_path / "llmock.json"
    config_path.write_text(
        json.dumps(
            {
                "server": {"host": "0.0.0.0", "port": 9100},
                "chaos": {"latency_ms": 120, "error_rates": {"404": 0.05, "507": 0.15}},
                "responses": {"style": "echo"},
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run(app_target, **kwargs):
        captured["app_target"] = app_target
        captured.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    result = runner.invoke(cli.app, ["serve", "--config", str(config_path)])

    assert result.exit_code == 0
    assert f"Config: {config_path}" in result.stdout
    assert "Chaos: latency=120ms  errors=[404=5%, 507=15%]" in result.stdout
    assert "Responses: style=echo" in result.stdout
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9100


def test_environment_overrides_config_file(tmp_path, monkeypatch):
    config_path = tmp_path / "llmock.yaml"
    config_path.write_text(
        "\n".join(
            [
                "server:",
                "  host: 0.0.0.0",
                "  port: 9100",
                "chaos:",
                "  latency_ms: 120",
                "  error_rates:",
                "    404: 0.05",
                "responses:",
                "  style: hello",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("LLMOCK_HOST", "127.0.0.1")
    monkeypatch.setenv("LLMOCK_PORT", "8123")
    monkeypatch.setenv("LLMOCK_LATENCY_MS", "300")
    monkeypatch.setenv("LLMOCK_ERROR_RATE_429", "0.2")
    monkeypatch.setenv("LLMOCK_RESPONSE_STYLE", "varied")

    captured: dict[str, object] = {}

    def fake_run(app_target, **kwargs):
        captured["app_target"] = app_target
        captured.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    result = runner.invoke(cli.app, ["serve", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Chaos: latency=300ms  errors=[404=5%, 429=20%]" in result.stdout
    assert "Responses: style=varied" in result.stdout
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
