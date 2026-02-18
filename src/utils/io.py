"""I/O helpers for reading and writing lightweight artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory path if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON with stable formatting for easy diffing."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return target


def read_json(path: str | Path) -> Any:
    """Read JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str | Path, text: str) -> Path:
    """Write UTF-8 text file."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as f:
        f.write(text)
    return target
