"""Mechanical guard: the alphagate source must stay domain-agnostic.

alphagate is a generic model-governance package. Vocabulary from any specific
application domain (here: the one that happens to depend on it) must never
leak into the package source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FORBIDDEN = (
    "trade", "persona", "alpaca", "ticker", "portfolio",
    "stock", "broker", "sharpe", "xgboost", "wolfpack",
)
SRC = Path(__file__).resolve().parent.parent / "src" / "alphagate"


def _source_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def test_source_tree_is_found():
    # Guard against a vacuous pass if the package moves.
    py_files = [p for p in _source_files() if p.suffix == ".py"]
    assert len(py_files) >= 7, py_files


def test_imports_are_stdlib_or_package_relative_only():
    """alphagate has zero runtime dependencies: every import in the source is
    either the standard library or alphagate itself. This is the mechanical
    guarantee that no application-domain library is pulled in."""
    import ast
    import sys

    allowed = set(sys.stdlib_module_names) | {"__future__", "alphagate"}
    offenders = []
    for path in _source_files():
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import within the package
                    continue
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            offenders += [f"{path.name}: {r}" for r in roots if r not in allowed]
    assert not offenders, offenders


def test_version_is_consistent():
    import re

    import alphagate

    pyproject = (SRC.parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    assert m and m.group(1) == alphagate.__version__ == "0.2.0"


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
