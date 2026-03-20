"""Typer CLI entry point for LLMock."""

from __future__ import annotations

import json
import os
from typing import Any

import typer
import uvicorn
import yaml

from llmock.chaos import ChaosSettings
from llmock.simulation import ERROR_RATE_ENV_PREFIX, MockResponseSettings, SUPPORTED_ERROR_STATUS_CODES

app = typer.Typer(
    name="llmock",
    help="OpenAI-compatible mock server for LLM API resilience testing.",
    no_args_is_help=True,
)

ERROR_RATE_OPTION_HELP = (
    "Inject any HTTP error with STATUS=PROBABILITY. Repeat the option for as many 4xx/5xx "
    "statuses as needed, for example: --error-rate 400=0.05 --error-rate 429=0.2 --error-rate 503=0.1."
)


def _resolve_config_path(*, config: str | None) -> str | None:
    return config or os.getenv("LLMOCK_CONFIG")


def _load_config_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()

        lower_path = path.lower()
        if lower_path.endswith(".json"):
            data = json.loads(raw)
        elif lower_path.endswith((".yaml", ".yml")):
            data = yaml.safe_load(raw) or {}
        else:
            raise typer.BadParameter(
                "Unsupported config file format. Use a .json, .yaml, or .yml file.",
                param_hint="--config",
            )
    except OSError as exc:
        raise typer.BadParameter(
            f"Could not read config file: {exc}",
            param_hint="--config",
        ) from exc
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise typer.BadParameter(
            f"Could not parse config file: {exc}",
            param_hint="--config",
        ) from exc

    if not isinstance(data, dict):
        raise typer.BadParameter(
            "Config file root must be a JSON/YAML object.",
            param_hint="--config",
        )
    return data


def _config_scalar(config: dict[str, Any], key: str, *, section: str | None = None) -> Any:
    if section:
        nested = config.get(section)
        if isinstance(nested, dict) and key in nested:
            return nested[key]
    return config.get(key)


def _config_error_rates(config: dict[str, Any]) -> dict[int, float]:
    parsed: dict[int, float] = {}

    for source in (config, config.get("chaos", {})):
        if not isinstance(source, dict):
            continue

        mapping = source.get("error_rates")
        if mapping is not None:
            if not isinstance(mapping, dict):
                raise typer.BadParameter(
                    "Config key 'error_rates' must be an object mapping status codes to probabilities.",
                    param_hint="--config",
                )
            for status, rate in mapping.items():
                parsed[int(status)] = float(rate)

        for key, value in source.items():
            if not isinstance(key, str) or not key.startswith("error_rate_"):
                continue
            suffix = key.removeprefix("error_rate_")
            if suffix.isdigit():
                parsed[int(suffix)] = float(value)

    return parsed


def _parse_error_rate_options(values: list[str] | None) -> dict[int, float]:
    parsed: dict[int, float] = {}
    for value in values or []:
        try:
            status_str, rate_str = value.split("=", 1)
            status_code = int(status_str)
            rate = float(rate_str)
        except ValueError as exc:
            raise typer.BadParameter(
                "Each --error-rate value must use the form STATUS=PROBABILITY, e.g. 429=0.4."
            ) from exc

        if status_code not in SUPPORTED_ERROR_STATUS_CODES:
            raise typer.BadParameter(
                f"Unsupported status code {status_code}. Use an HTTP error code between 400 and 599."
            )

        parsed[status_code] = rate
    return parsed


def _set_server_env(*, chaos: ChaosSettings, responses: MockResponseSettings) -> None:
    for key in list(os.environ):
        if key.startswith(ERROR_RATE_ENV_PREFIX):
            del os.environ[key]

    for key, value in chaos.as_env().items():
        os.environ[key] = value
    for key, value in responses.as_env().items():
        os.environ[key] = value


def _resolve_chaos_settings(
    *,
    config: dict[str, Any],
    latency_ms: int | None,
    error_rates: list[str] | None = None,
    error_rate_429: float | None,
    error_rate_500: float | None,
    error_rate_503: float | None,
) -> ChaosSettings:
    base = ChaosSettings(
        latency_ms=int(_config_scalar(config, "latency_ms", section="chaos") or 0),
        error_rates=_config_error_rates(config),
    ).validated()

    env_settings = ChaosSettings.from_env()
    env_latency = os.getenv("LLMOCK_LATENCY_MS")
    return base.with_overrides(
        latency_ms=int(env_latency) if env_latency is not None else None,
        error_rates=env_settings.error_rates,
    ).with_overrides(
        latency_ms=latency_ms,
        error_rates=_parse_error_rate_options(error_rates),
        error_rate_429=error_rate_429,
        error_rate_500=error_rate_500,
        error_rate_503=error_rate_503,
    )


def _resolve_mock_response_settings(*, config: dict[str, Any], response_style: str | None) -> MockResponseSettings:
    settings = MockResponseSettings(
        response_style=str(
            _config_scalar(config, "response_style")
            or _config_scalar(config, "response_style", section="responses")
            or _config_scalar(config, "style", section="responses")
            or "varied"
        )
    ).validated()
    env_style = os.getenv("LLMOCK_RESPONSE_STYLE")
    if env_style is not None:
        settings.response_style = env_style
    if response_style is not None:
        settings.response_style = response_style
    return settings.validated()


def _resolve_server_host(*, config: dict[str, Any], host: str | None) -> str:
    return str(host or os.getenv("LLMOCK_HOST") or _config_scalar(config, "host", section="server") or "127.0.0.1")


def _resolve_server_port(*, config: dict[str, Any], port: int | None) -> int:
    if port is not None:
        return port
    return int(os.getenv("LLMOCK_PORT") or _config_scalar(config, "port", section="server") or "8000")


def _format_error_rates(settings: ChaosSettings) -> str:
    return ", ".join(
        f"{status}={rate:.0%}" for status, rate in sorted(settings.error_rates.items()) if rate
    )


@app.callback()
def main() -> None:
    """LLMock command group."""


@app.command()
def serve(
    config: str | None = typer.Option(
        None,
        "--config",
        help="Load startup settings from a JSON or YAML file. CLI flags override env vars, and env vars override the config file.",
    ),
    host: str | None = typer.Option(None, "--host", "-h", help="Override LLMOCK_HOST."),
    port: int | None = typer.Option(None, "--port", "-p", help="Override LLMOCK_PORT."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev mode)"),
    log_level: str = typer.Option("info", "--log-level", help="Uvicorn log level"),
    latency_ms: int | None = typer.Option(None, "--latency-ms", help="Override LLMOCK_LATENCY_MS."),
    error_rates: list[str] | None = typer.Option(
        None,
        "--error-rate",
        help=ERROR_RATE_OPTION_HELP,
    ),
    error_rate_429: float | None = typer.Option(
        None,
        "--error-rate-429",
        help="Shortcut for --error-rate 429=RATE.",
    ),
    error_rate_500: float | None = typer.Option(
        None,
        "--error-rate-500",
        help="Shortcut for --error-rate 500=RATE.",
    ),
    error_rate_503: float | None = typer.Option(
        None,
        "--error-rate-503",
        help="Shortcut for --error-rate 503=RATE.",
    ),
    response_style: str | None = typer.Option(
        None,
        "--response-style",
        help="Override LLMOCK_RESPONSE_STYLE (static, hello, echo, varied).",
    ),
) -> None:
    """Start the LLMock server."""
    config_path = _resolve_config_path(config=config)
    loaded_config = _load_config_file(config_path)
    resolved_host = _resolve_server_host(config=loaded_config, host=host)
    resolved_port = _resolve_server_port(config=loaded_config, port=port)
    chaos = _resolve_chaos_settings(
        config=loaded_config,
        latency_ms=latency_ms,
        error_rates=error_rates,
        error_rate_429=error_rate_429,
        error_rate_500=error_rate_500,
        error_rate_503=error_rate_503,
    )
    responses = _resolve_mock_response_settings(config=loaded_config, response_style=response_style)
    _set_server_env(chaos=chaos, responses=responses)

    configured_rates = _format_error_rates(chaos)
    if chaos.latency_ms or configured_rates:
        typer.echo(f"Chaos: latency={chaos.latency_ms}ms  errors=[{configured_rates or 'none'}]")

    if config_path:
        typer.echo(f"Config: {config_path}")
    typer.echo(f"Responses: style={responses.response_style}")
    typer.echo(f"Starting LLMock on http://{resolved_host}:{resolved_port}")
    uvicorn.run(
        "llmock.main:create_app",
        factory=True,
        host=resolved_host,
        port=resolved_port,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    app()
