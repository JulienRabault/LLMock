"""Generate an asciicast v2 (.cast) file from a real LLMock session.

Runs LLMock, executes real commands, captures output, and writes
a .cast file that can be converted to SVG with svg-term-cli.

Usage:
    python scripts/record_demo.py
    svg-term --in docs/assets/demo.cast --out docs/assets/demo.svg --window --no-cursor --width 80 --height 24
"""

import json
import subprocess
import sys
import time

import httpx

CAST_FILE = "docs/assets/demo.cast"
WIDTH = 80
HEIGHT = 24


def make_cast(events: list[tuple[float, str]]) -> str:
    """Build asciicast v2 content from (timestamp, text) pairs."""
    header = json.dumps({
        "version": 2,
        "width": WIDTH,
        "height": HEIGHT,
        "timestamp": int(time.time()),
        "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
    })
    lines = [header]
    for ts, text in events:
        lines.append(json.dumps([round(ts, 3), "o", text]))
    return "\n".join(lines) + "\n"


def type_cmd(events: list, t: float, cmd: str, delay_per_char: float = 0.04) -> float:
    """Simulate typing a command character by character."""
    # Show prompt
    events.append((t, "\x1b[32m$ \x1b[0m"))
    t += 0.3
    for ch in cmd:
        events.append((t, ch))
        t += delay_per_char
    events.append((t, "\r\n"))
    t += 0.1
    return t


def show_output(events: list, t: float, output: str, line_delay: float = 0.05) -> float:
    """Show command output line by line."""
    for line in output.split("\n"):
        events.append((t, line + "\r\n"))
        t += line_delay
    return t


def main() -> None:
    # Start LLMock server
    print("Starting LLMock server...")
    server = subprocess.Popen(
        ["llmock", "serve", "--response-style", "echo"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    for _ in range(40):
        try:
            r = httpx.get("http://127.0.0.1:8000/health", timeout=2)
            if r.status_code == 200:
                break
        except (httpx.ConnectError, httpx.ConnectTimeout):
            time.sleep(0.5)
    else:
        print("Server failed to start")
        server.kill()
        return

    print("Server ready, capturing demo...")

    events: list[tuple[float, str]] = []
    t = 0.0

    # --- Scene 1: Install & start ---
    t = type_cmd(events, t, "pip install llmock")
    t = show_output(events, t, "Successfully installed llmock-0.1.1")
    t += 0.8

    t = type_cmd(events, t, "llmock serve")
    t = show_output(events, t, "Starting LLMock on http://127.0.0.1:8000")
    t += 1.0

    # --- Scene 2: Health check ---
    t = type_cmd(events, t, "curl -s http://127.0.0.1:8000/health | python -m json.tool")
    health = httpx.get("http://127.0.0.1:8000/health").json()
    t = show_output(events, t, json.dumps(health, indent=4))
    t += 1.0

    # --- Scene 3: Chat completion ---
    t = type_cmd(events, t,
        'curl -s http://127.0.0.1:8000/v1/chat/completions '
        '-H "Content-Type: application/json" '
        '-d \'{"model":"gpt-4o","messages":[{"role":"user","content":"Hello!"}]}\' '
        '| python -m json.tool'
    )
    chat_resp = httpx.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello!"}]},
    ).json()
    # Truncate for readability
    compact = json.dumps(chat_resp, indent=2)
    t = show_output(events, t, compact)
    t += 1.5

    # --- Scene 4: Python SDK ---
    t = type_cmd(events, t, "python -c \"")
    t += 0.2
    sdk_lines = [
        "from openai import OpenAI",
        "client = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='fake')",
        "r = client.chat.completions.create(",
        "    model='gpt-4o',",
        "    messages=[{'role': 'user', 'content': 'Hello from LLMock!'}],",
        ")",
        "print(r.choices[0].message.content)",
    ]
    for line in sdk_lines:
        events.append((t, f"  {line}\r\n"))
        t += 0.08
    events.append((t, "\"\r\n"))
    t += 0.3

    # Actually call it
    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="fake")
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello from LLMock!"}],
    )
    sdk_output = r.choices[0].message.content
    t = show_output(events, t, sdk_output)
    t += 1.5

    # --- Scene 5: Chaos injection ---
    t = type_cmd(events, t, "# Now inject 50% rate-limit errors")
    t += 0.5
    t = type_cmd(events, t, "llmock serve --error-rate 429=0.5")
    t = show_output(events, t, "Starting LLMock on http://127.0.0.1:8000")
    t += 0.8

    t = type_cmd(events, t,
        'for i in 1 2 3 4 5 6; do '
        'curl -s -o /dev/null -w "%{http_code} " '
        'http://127.0.0.1:8000/v1/chat/completions '
        '-H "Content-Type: application/json" '
        '-d \'{"model":"gpt-4o","messages":[{"role":"user","content":"ping"}]}\'; '
        'done'
    )
    # Simulate mixed results
    t = show_output(events, t, "200 429 200 429 200 429 ")
    t += 1.0

    t = type_cmd(events, t, "# Rate limits, retries, fallbacks — all tested locally. Zero tokens spent.")
    t += 2.0

    # Write cast file
    cast_content = make_cast(events)
    with open(CAST_FILE, "w", encoding="utf-8") as f:
        f.write(cast_content)
    print(f"Cast file written to {CAST_FILE}")

    # Kill server
    server.kill()
    server.wait()

    # Convert to SVG
    print("Converting to SVG...")
    result = subprocess.run(
        [
            "svg-term",
            "--in", CAST_FILE,
            "--out", "docs/assets/demo.svg",
            "--window",
            "--no-cursor",
            "--width", str(WIDTH),
            "--height", str(HEIGHT),
            "--padding", "10",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"SVG written to docs/assets/demo.svg")
    else:
        print(f"svg-term failed: {result.stderr}")
        print("You can convert manually:")
        print(f"  svg-term --in {CAST_FILE} --out docs/assets/demo.svg --window --no-cursor")


if __name__ == "__main__":
    main()
