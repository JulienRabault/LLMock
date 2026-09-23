"""CLI: the new serve flags, the report command and the shutdown verdict."""

import os

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import llmock.cli as cli
from llmock.main import create_app
from llmock.scenarios import Fail
from llmock.testing import LLMockServer

runner = CliRunner()
CHAT = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def captured_env(monkeypatch):
    """Run `serve` without starting uvicorn; return the env the app would see."""
    seen = {}

    def fake_run(*args, **kwargs):
        seen.update({k: v for k, v in os.environ.items() if k.startswith("LLMOCK_")})

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    return seen


def test_new_flags_reach_the_app(captured_env):
    result = runner.invoke(cli.app, [
        "serve", "--tool-mode", "off", "--rpm", "60", "--tpm", "90000",
        "--context-window", "8000", "--stream-fault", "truncate=0.1",
        "--stream-fault", "disconnect=0.05", "--stream-chunk-delay-ms", "20", "--report",
    ])
    assert result.exit_code == 0, result.output
    assert captured_env["LLMOCK_TOOL_MODE"] == "off"
    assert (captured_env["LLMOCK_RPM"], captured_env["LLMOCK_TPM"]) == ("60", "90000")
    assert captured_env["LLMOCK_CONTEXT_WINDOW"] == "8000"
    assert captured_env["LLMOCK_STREAM_FAULT_TRUNCATE"] == "0.1"
    assert captured_env["LLMOCK_STREAM_FAULT_DISCONNECT"] == "0.05"
    assert captured_env["LLMOCK_STREAM_CHUNK_DELAY_MS"] == "20"
    assert captured_env["LLMOCK_REPORT"] == "1"
    assert "rpm=60" in result.output and "truncate=10%" in result.output


def test_settings_come_from_the_config_file(captured_env, tmp_path):
    config = tmp_path / "llmock.yaml"
    config.write_text(
        "responses:\n  tool_mode: off\n"
        "limits:\n  rpm: 30\n  context_window: 4096\n"
        "stream:\n  faults:\n    stall: 0.2\n  stall_seconds: 3\n"
    )
    result = runner.invoke(cli.app, ["serve", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert captured_env["LLMOCK_TOOL_MODE"] == "off"
    assert captured_env["LLMOCK_RPM"] == "30"
    assert captured_env["LLMOCK_CONTEXT_WINDOW"] == "4096"
    assert captured_env["LLMOCK_STREAM_FAULT_STALL"] == "0.2"
    assert captured_env["LLMOCK_STREAM_STALL_SECONDS"] == "3.0"


def test_flags_override_the_config_file(captured_env, tmp_path):
    config = tmp_path / "llmock.yaml"
    config.write_text("limits:\n  rpm: 30\n")
    runner.invoke(cli.app, ["serve", "--config", str(config), "--rpm", "5"])
    assert captured_env["LLMOCK_RPM"] == "5"


@pytest.mark.parametrize("value", ["explode=0.1", "truncate", "truncate=lots"])
def test_bad_stream_fault_is_rejected(captured_env, value):
    result = runner.invoke(cli.app, ["serve", "--stream-fault", value])
    assert result.exit_code != 0


def test_bad_tool_mode_is_rejected(captured_env):
    assert runner.invoke(cli.app, ["serve", "--tool-mode", "sometimes"]).exit_code != 0


# -- llmock report --------------------------------------------------------------


def test_report_exits_zero_for_a_well_behaved_client():
    with LLMockServer() as server:
        import httpx

        httpx.post(server.base_url("openai") + "/chat/completions", json=CHAT)
        result = runner.invoke(cli.app, ["report", "--url", server.url])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_report_exits_one_when_the_client_misbehaved():
    with LLMockServer() as server:
        import httpx

        server.state.scenarios.add(Fail(401))
        url = server.base_url("openai") + "/chat/completions"
        httpx.post(url, json=CHAT)
        httpx.post(url, json=CHAT)
        result = runner.invoke(cli.app, ["report", "--url", server.url])
        as_json = runner.invoke(cli.app, ["report", "--url", server.url, "--json"])
    assert result.exit_code == 1
    assert "retried_non_retryable" in result.output
    assert '"passed": false' in as_json.output


def test_report_strict_fails_on_warnings():
    with LLMockServer() as server:
        import httpx

        server.state.scenarios.add(Fail(503))
        httpx.post(server.base_url("openai") + "/chat/completions", json=CHAT)
        lenient = runner.invoke(cli.app, ["report", "--url", server.url])
        strict = runner.invoke(cli.app, ["report", "--url", server.url, "--strict"])
    assert (lenient.exit_code, strict.exit_code) == (0, 1)


def test_report_exits_two_when_nothing_is_listening():
    result = runner.invoke(cli.app, ["report", "--url", "http://127.0.0.1:1"])
    assert result.exit_code == 2


# -- shutdown verdict -----------------------------------------------------------


def test_the_verdict_is_printed_on_shutdown(monkeypatch, capsys):
    monkeypatch.setenv("LLMOCK_REPORT", "1")
    app = create_app()
    app.state.llmock.scenarios.add(Fail(401))
    with TestClient(app) as client:  # the context manager runs startup and shutdown
        client.post("/v1/chat/completions", json=CHAT)
        client.post("/v1/chat/completions", json=CHAT)
    assert "retried_non_retryable" in capsys.readouterr().out
