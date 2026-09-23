"""Render docs/assets/demo.gif from real LLMock output.

Nothing in the animation is hand-written. Scene one is the real startup
screen of ``llmock serve``, recorded with its colours. Scene two runs a real
``pytest --llmock-report`` on a small demo app (``scripts/demo_app``) and
replays its output: every test passes, and LLMock still shows the client is
wrong.

    pip install pillow
    python scripts/render_demo_gif.py
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.segment import Segment

from llmock import __version__
from llmock.chaos import ChaosSettings, StreamChaos
from llmock.console import print_startup
from llmock.ratelimit import LimitSettings
from llmock.simulation import MockResponseSettings

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "demo.gif"
DEMO_APP = Path(__file__).resolve().parent / "demo_app"

COLS, ROWS = 96, 30
FONT_SIZE, LINE_H, PAD, TITLE_H = 15, 19, 18, 34
FONT_CANDIDATES = (
    "C:/Windows/Fonts/consola.ttf",
    "/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)
BOLD_CANDIDATES = (
    "C:/Windows/Fonts/consolab.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
)

BG, BAR = (13, 17, 23), (22, 27, 34)
FG, DIM = (201, 209, 217), (125, 133, 144)
GREEN, RED, CYAN, WHITE = (63, 185, 80), (248, 81, 73), (88, 166, 255), (240, 246, 252)

# How a dark terminal theme shows the 16 standard ANSI colours. rich's own
# defaults for them (e.g. olive for "yellow") look nothing like a real terminal.
ANSI = {
    0: (72, 79, 88), 1: (248, 81, 73), 2: (63, 185, 80), 3: (227, 179, 65),
    4: (88, 166, 255), 5: (188, 140, 255), 6: (57, 197, 207), 7: (201, 209, 217),
    8: (110, 118, 129), 9: (255, 123, 114), 10: (86, 211, 100), 11: (240, 198, 100),
    12: (121, 192, 255), 13: (210, 168, 255), 14: (86, 212, 221), 15: (240, 246, 252),
}

Span = tuple[str, tuple[int, int, int], bool]
Line = list[Span]


# -- the two real sessions -------------------------------------------------------


def startup_lines() -> list[Line]:
    """The real `llmock serve` screen, with rich's own colours."""
    console = Console(record=True, width=COLS, force_terminal=True, color_system="truecolor",
                      file=io.StringIO(), legacy_windows=False)
    print_startup(
        host="127.0.0.1", port=8000,
        chaos=ChaosSettings(error_rates={429: 0.2}),
        responses=MockResponseSettings(),
        limits=LimitSettings(rpm=60),
        stream=StreamChaos(fault_rates=(("truncate", 0.1),)),
        config_path=None, version=__version__, console=console,
    )
    lines: list[Line] = []
    for segments in Segment.split_lines(console._record_buffer):
        spans: Line = []
        for segment in segments:
            if segment.control:
                continue
            style = segment.style
            colour, bold = FG, False
            if style is not None:
                if style.color is not None:
                    number = style.color.number
                    if style.color.triplet is None and number is not None and number < 16:
                        colour = ANSI[number]
                    else:
                        colour = tuple(style.color.get_truecolor())  # type: ignore[assignment]
                bold = bool(style.bold)
                if style.dim:
                    colour = DIM
            spans.append((segment.text, colour, bold))
        lines.append(spans)
    return lines


def pytest_lines() -> list[Line]:
    """Run pytest --llmock-report on the demo app and colour its real output."""
    env = {**os.environ, "NO_COLOR": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--llmock-report",
         "-q", "--no-header"],
        cwd=DEMO_APP, capture_output=True, text=True, env=env, check=False,
    )
    lines: list[Line] = []
    for raw in result.stdout.rstrip().splitlines():
        stripped = raw.strip()
        if set(stripped.split("[")[0].strip()) <= {"."} and stripped:
            colour, bold = GREEN, True
        elif stripped.startswith("===") and "LLMock" in stripped:
            colour, bold = CYAN, True
        elif stripped.startswith("test_") and "::" in stripped:
            colour, bold = WHITE, True
        elif raw.startswith("LLMock resilience verdict"):
            colour, bold = CYAN, False
        elif stripped.startswith("FAIL") or "error(s)" in stripped:
            colour, bold = RED, True
        elif stripped.startswith("->"):
            colour, bold = DIM, False
        elif " passed" in stripped:
            colour, bold = GREEN, True
        else:
            colour, bold = FG, False
        indent = len(raw) - len(raw.lstrip())
        for index, piece in enumerate(textwrap.wrap(stripped, COLS - indent - 3) or [""]):
            extra = "   " if index and stripped.startswith("->") else ""
            lines.append([(" " * indent + extra + piece, colour, bold)])
    return lines


# -- drawing ---------------------------------------------------------------------


def _font(candidates: tuple[str, ...]) -> ImageFont.FreeTypeFont:
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, FONT_SIZE)
    raise SystemExit("No monospace font found; add one to FONT_CANDIDATES")


REGULAR = _font(FONT_CANDIDATES)
BOLD = _font(BOLD_CANDIDATES) if any(Path(p).exists() for p in BOLD_CANDIDATES) else REGULAR
CHAR_W = REGULAR.getlength("M")
WIDTH = int(PAD * 2 + COLS * CHAR_W)
HEIGHT = TITLE_H + PAD + ROWS * LINE_H + PAD // 2


# Box-drawing characters, as (edges, double): which cell edges the lines reach.
# Terminals draw these geometrically so that neighbouring cells join up;
# font glyphs rarely do.
_BOX = {
    "─": ("lr", False), "│": ("tb", False), "┌": ("rb", False), "┐": ("lb", False),
    "└": ("rt", False), "┘": ("lt", False),
    "═": ("lr", True), "║": ("tb", True), "╔": ("rb", True), "╗": ("lb", True),
    "╚": ("rt", True), "╝": ("lt", True),
}


def _box(draw: ImageDraw.ImageDraw, char: str, x: float, y: float, colour) -> None:
    edges, double = _BOX[char]
    x0, x1, y0, y1 = x, x + CHAR_W, y - 1, y + LINE_H - 1
    cx, cy = x + CHAR_W / 2, y0 + LINE_H / 2
    offsets = (-CHAR_W * 0.17, CHAR_W * 0.17) if double else (0.0,)
    corner = len(edges) == 2 and edges not in ("lr", "tb")
    for d in offsets:
        # For a double corner, the outer line turns outside the inner one.
        dx = dy = d
        if corner and double:
            dx = d if "l" in edges else -d
            dy = d if "t" in edges else -d
        if "l" in edges:
            draw.line((x0, cy + dy, cx + (dx if corner else 0), cy + dy), fill=colour)
        if "r" in edges:
            draw.line((cx + (dx if corner else 0), cy + dy, x1, cy + dy), fill=colour)
        if "t" in edges:
            draw.line((cx + dx, y0, cx + dx, cy + (dy if corner else 0)), fill=colour)
        if "b" in edges:
            draw.line((cx + dx, cy + (dy if corner else 0), cx + dx, y1), fill=colour)
        if edges == "lr":
            draw.line((x0, cy + d, x1, cy + d), fill=colour)
        if edges == "tb":
            draw.line((cx + d, y0, cx + d, y1), fill=colour)


def frame(lines: list[Line], cursor: tuple[int, int] | None = None) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, TITLE_H), fill=BAR)
    for index, colour in enumerate(((255, 95, 88), (255, 189, 46), (24, 193, 50))):
        x = 20 + index * 20
        draw.ellipse((x - 6, TITLE_H // 2 - 6, x + 6, TITLE_H // 2 + 6), fill=colour)
    title = "llmock"
    draw.text(((WIDTH - REGULAR.getlength(title)) / 2, TITLE_H / 2 - 9), title, font=REGULAR, fill=DIM)

    visible = lines[-ROWS:]
    offset = len(lines) - len(visible)
    for row, spans in enumerate(visible):
        x, y = PAD, TITLE_H + PAD + row * LINE_H
        for text, colour, bold in spans:
            font = BOLD if bold else REGULAR
            for char in text:
                if char == "█":  # a full cell, drawn exactly rather than with the glyph
                    draw.rectangle((x, y - 1, x + CHAR_W, y + LINE_H - 1), fill=colour)
                elif char in _BOX:
                    _box(draw, char, x, y, colour)
                elif char != " ":
                    draw.text((x, y), char, font=font, fill=colour)
                x += CHAR_W
    if cursor is not None:
        row, col = cursor[0] - offset, cursor[1]
        if 0 <= row < ROWS:
            x, y = PAD + col * CHAR_W, TITLE_H + PAD + row * LINE_H
            draw.rectangle((x, y + 1, x + CHAR_W - 1, y + LINE_H - 3), fill=FG)
    return image


def prompt(command: str) -> Line:
    return [("~/myapp ", CYAN, False), ("$ ", DIM, False), (command, WHITE, True)]


class Film:
    def __init__(self) -> None:
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []

    def add(self, image: Image.Image, ms: int) -> None:
        self.frames.append(image)
        self.durations.append(ms)

    def type(self, screen: list[Line], command: str) -> None:
        for end in range(0, len(command) + 1, 2):
            line = prompt(command[:end])
            self.add(frame([*screen, line], cursor=(len(screen), 10 + end)), 45)
        self.add(frame([*screen, prompt(command)], cursor=(len(screen), 10 + len(command))), 500)

    def reveal(self, screen: list[Line], output: list[Line], *, per_frame: int, ms: int,
               hold: int) -> list[Line]:
        for end in range(per_frame, len(output) + per_frame, per_frame):
            self.add(frame([*screen, *output[:end]]), ms)
        self.durations[-1] = hold
        return [*screen, *output]


def main() -> None:
    serve_output = startup_lines()
    pytest_output = pytest_lines()
    film = Film()

    film.add(frame([prompt("")], cursor=(0, 10)), 600)
    command = "llmock serve --rpm 60 --error-rate 429=0.2 --stream-fault truncate=0.1"
    film.type([], command)
    film.reveal([prompt(command)], serve_output, per_frame=2, ms=35, hold=2800)
    serve_done = film.frames[-1]

    film.add(frame([prompt("")], cursor=(0, 10)), 400)
    command = "pytest --llmock-report"
    film.type([], command)
    screen = [prompt(command)]
    first, rest = pytest_output[0], pytest_output[1:]
    dots = first[0][0].split()[0]
    for end in range(1, len(dots) + 1):  # tests turning green one by one
        film.add(frame([*screen, [(dots[:end], GREEN, True)]]), 350)
    film.add(frame([*screen, first]), 900)
    film.reveal([*screen, first], rest, per_frame=1, ms=70, hold=6500)

    palette = Image.new("RGB", (WIDTH, HEIGHT * 2))
    palette.paste(serve_done, (0, 0))
    palette.paste(film.frames[-1], (0, HEIGHT))
    palette = palette.quantize(colors=128, dither=Image.Dither.NONE)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in film.frames]
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=film.durations,
                   loop=0, optimize=False, disposal=1)
    print(f"wrote {OUT}: {len(frames)} frames, {OUT.stat().st_size / 1024:.0f} KiB, "
          f"{sum(film.durations) / 1000:.1f}s")


if __name__ == "__main__":
    main()
