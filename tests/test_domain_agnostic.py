"""Mechanical guard: the alphagate source must stay domain-agnostic.

alphagate is a generic model-governance package. Vocabulary from any specific
application domain (here: the one that happens to depend on it) must never
leak into the package source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FORBIDDEN = ("trade", "persona", "alpaca", "ticker", "portfolio")
SRC = Path(__file__).resolve().parent.parent / "src" / "alphagate"


def _source_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def test_source_tree_is_found():
    # Guard against a vacuous pass if the package moves.
    py_files = [p for p in _source_files() if p.suffix == ".py"]
    assert len(py_files) >= 7, py_files


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_domain_vocabulary(path: Path):
    text = path.read_text(encoding="utf-8").lower()
    hits = [
        f"{path.name}:{lineno}: {word!r}"
        for lineno, line in enumerate(text.splitlines(), 1)
        for word in FORBIDDEN
        if word in line
    ]
    assert not hits, "domain-specific vocabulary in alphagate source:\n" + "\n".join(hits)
