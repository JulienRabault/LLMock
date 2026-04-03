"""Docs integrity checks for local markdown references."""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = (
    ROOT / "README.md",
    ROOT / "ARCHITECTURE.md",
    ROOT / "examples" / "README.md",
)

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
INLINE_REPO_PATH_RE = re.compile(r"`((?:examples|docs|llmock|scripts|tests)/[^`\s]+)`")


def _iter_local_targets(text: str) -> set[str]:
    targets: set[str] = set()

    for raw_target in MARKDOWN_LINK_RE.findall(text):
        target = raw_target.strip()
        if "://" in target or target.startswith("#") or target.startswith("mailto:"):
            continue
        targets.add(target.split("#", 1)[0])

    for raw_target in INLINE_REPO_PATH_RE.findall(text):
        targets.add(raw_target.split("#", 1)[0])

    return targets


def test_local_doc_references_exist():
    missing: list[str] = []

    for doc_path in DOC_FILES:
        text = doc_path.read_text(encoding="utf-8")
        for target in sorted(_iter_local_targets(text)):
            resolved = (doc_path.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{doc_path.relative_to(ROOT)} -> {target}")

    assert not missing, "Missing local doc references:\n" + "\n".join(missing)
