"""The CLI's terminal output: logo fallback, network warning, coloured report."""

import io

from rich.console import Console

from llmock.chaos import ChaosSettings, StreamChaos
from llmock.console import logo, print_startup
from llmock.ratelimit import LimitSettings
from llmock.simulation import MockResponseSettings


def _console(encoding):
    stream = io.TextIOWrapper(io.BytesIO(), encoding=encoding)
    return Console(file=stream, force_terminal=False)


def test_block_logo_on_utf8_consoles():
    assert "██" in logo(_console("utf-8")).plain


def test_ascii_logo_where_block_letters_cannot_be_encoded():
    text = logo(_console("cp1252")).plain
    assert "██" not in text and "|_____|" in text


def _startup(capsys, host):
    print_startup(host=host, port=8000, chaos=ChaosSettings(), responses=MockResponseSettings(),
                  limits=LimitSettings(), stream=StreamChaos(), config_path=None, version="9.9.9")
    return capsys.readouterr().out


def test_startup_lists_the_base_urls(capsys):
    out = _startup(capsys, "127.0.0.1")
    assert "http://127.0.0.1:8000/anthropic" in out and "v9.9.9" in out
    assert "no authentication" not in out


def test_startup_warns_when_reachable_from_the_network(capsys):
    assert "no authentication" in _startup(capsys, "0.0.0.0")
