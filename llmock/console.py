"""Terminal output for the CLI: the startup banner and a coloured verdict.

Uses rich, which typer already depends on. Colour follows rich's rules: off
when stdout is not a terminal or when NO_COLOR is set, so CI logs stay clean.
The block-letter logo falls back to plain ASCII on consoles whose encoding
cannot draw it, such as a legacy Windows code page.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from llmock.chaos import ChaosSettings, StreamChaos
    from llmock.ratelimit import LimitSettings
    from llmock.simulation import MockResponseSettings
    from llmock.verdict import Verdict

__all__ = ["logo", "print_report", "print_startup", "print_verdict"]

_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})

_BLOCK_LOGO = r"""
██╗     ██╗     ███╗   ███╗ ██████╗  ██████╗██╗  ██╗
██║     ██║     ████╗ ████║██╔═══██╗██╔════╝██║ ██╔╝
██║     ██║     ██╔████╔██║██║   ██║██║     █████╔╝
██║     ██║     ██║╚██╔╝██║██║   ██║██║     ██╔═██╗
███████╗███████╗██║ ╚═╝ ██║╚██████╔╝╚██████╗██║  ██╗
╚══════╝╚══════╝╚═╝     ╚═╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝
""".strip("\n")

_ASCII_LOGO = r"""
 _     _     __  __            _
| |   | |   |  \/  | ___   ___| | __
| |   | |   | |\/| |/ _ \ / __| |/ /
| |___| |___| |  | | (_) | (__|   <
|_____|_____|_|  |_|\___/ \___|_|\_\
""".strip("\n")

# A vertical gradient, top to bottom, applied line by line.
_GRADIENT = ("#ff6b6b", "#ff8e53", "#ffb347", "#ffd166", "#f7a8ff", "#b69cff")

_PROVIDERS = (
    ("OpenAI", "openai"),
    ("Anthropic", "anthropic"),
    ("Gemini", "gemini"),
    ("Cohere", "cohere"),
)


def _console() -> Console:
    # Outside a terminal (CI logs, piped output) do not wrap at 80 columns.
    console = Console(highlight=False)
    if not console.is_terminal:
        console = Console(highlight=False, width=200)
    return console


def _can_draw(console: Console, sample: str) -> bool:
    try:
        sample.encode(console.encoding or "utf-8")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def logo(console: Console | None = None) -> Text:
    console = console or _console()
    art = _BLOCK_LOGO if _can_draw(console, _BLOCK_LOGO) else _ASCII_LOGO
    text = Text()
    for index, line in enumerate(art.splitlines()):
        text.append(line + "\n", style=f"bold {_GRADIENT[index % len(_GRADIENT)]}")
    return text


def print_startup(
    *,
    host: str,
    port: int,
    chaos: ChaosSettings,
    responses: MockResponseSettings,
    limits: LimitSettings,
    stream: StreamChaos,
    config_path: str | None,
    version: str,
    console: Console | None = None,
) -> None:
    console = console or _console()
    console.print(logo(console))
    dot = "·" if _can_draw(console, "·") else "-"
    console.print(Text.assemble(
        ("  chaos engineering for LLM apps", "italic"), (f"  {dot}  ", "dim"), (f"v{version}", "dim"),
    ))
    console.print()

    root = f"http://{host}:{port}"
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    table.add_row("Listening", Text(root, style="bold green"))
    for label, provider in _PROVIDERS:
        from llmock.testing import BASE_PATHS

        table.add_row(label, Text(root + BASE_PATHS[provider], style="green"))
    table.add_row("Control", Text.assemble((f"{root}/_llmock", "green"),
                                           (f"  scenario {dot} requests {dot} verdict", "dim")))
    table.add_row("", "")

    rates = ", ".join(f"{s}={r:.0%}" for s, r in sorted(chaos.error_rates.items()) if r)
    table.add_row("Chaos", _setting(
        f"latency={chaos.latency_ms}ms  errors=[{rates or 'none'}]", active=bool(rates or chaos.latency_ms)))
    faults = ", ".join(f"{k}={v:.0%}" for k, v in stream.fault_rates)
    table.add_row("Streams", _setting(
        f"faults=[{faults or 'none'}]  chunk_delay={stream.chunk_delay_ms}ms",
        active=bool(faults or stream.chunk_delay_ms)))
    table.add_row("Limits", _setting(
        f"rpm={limits.rpm or '-'}  tpm={limits.tpm or '-'}  "
        f"context_window={limits.context_window or '-'}",
        active=limits.enabled or bool(limits.context_window)))
    table.add_row("Responses", Text(f"style={responses.response_style}  tools={responses.tool_mode}"))
    if config_path:
        table.add_row("Config", Text(config_path))

    console.print(Panel(table, border_style="bright_black", padding=(1, 2), expand=False))
    if host not in _LOOPBACK:
        console.print(Text(
            f"  Warning: listening on {host}. The /_llmock control API has no authentication:\n"
            "  anyone who can reach this port can read requests and inject faults.",
            style="bold red",
        ))
    console.print(Text("  Point your SDK at a URL above. Ctrl+C to stop.", style="dim"))
    console.print()


def _setting(value: str, *, active: bool) -> Text:
    return Text(value, style="bold yellow" if active else "dim")


def print_verdict(verdict: Verdict) -> None:
    """The verdict report, with FAIL in red, WARN in yellow and PASS in green."""
    print_report(verdict.render())


def print_report(report: str) -> None:
    """Colour an already rendered verdict report, line by line."""
    console = _console()
    has_errors = "FAIL" in report
    for line in report.rstrip("\n").split("\n"):
        stripped = line.strip()
        if line.startswith("LLMock resilience verdict"):
            style = "bold cyan"
        elif stripped.startswith("FAIL"):
            style = "bold red"
        elif stripped.startswith("WARN"):
            style = "bold yellow"
        elif stripped.startswith("PASS"):
            style = "bold green"
        elif stripped.startswith("->"):
            style = "dim"
        elif stripped.endswith("warning(s)"):
            style = "bold red" if has_errors else "bold yellow"
        else:
            style = ""
        console.print(Text(line, style=style))
