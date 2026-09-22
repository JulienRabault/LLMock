"""Mock output must be identical across processes.

Python randomises ``hash()`` for strings per process (PYTHONHASHSEED), so any
"deterministic" choice built on it silently changes between two CI runs. These
tests pin the seed to different values in fresh interpreters and require the
same answer every time.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_SNIPPET = (
    "from llmock.simulation import build_mock_text, MockResponseSettings;"
    "print(build_mock_text(settings=MockResponseSettings('varied'),"
    " model={model!r}, prompt={prompt!r}))"
)


def _mock_text_in_fresh_process(model: str, prompt: str, hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    result = subprocess.run(
        [sys.executable, "-c", _SNIPPET.format(model=model, prompt=prompt)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize(
    ("model", "prompt"),
    [
        ("gpt-4o", "Hello"),
        ("claude-sonnet-4", "Summarise this document"),
        ("gemini-2.5-pro", ""),
    ],
)
def test_varied_style_is_stable_across_processes(model: str, prompt: str) -> None:
    outputs = {_mock_text_in_fresh_process(model, prompt, seed) for seed in ("0", "1", "42", "1337")}
    assert len(outputs) == 1, f"mock text changed between processes: {outputs}"
