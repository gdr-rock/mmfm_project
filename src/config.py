"""Configuration utilities for CLI scripts.

This module keeps config handling lightweight so the project can run in smoke
mode without heavy dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path | None) -> Dict[str, Any]:
    """Load a YAML config file if it exists, otherwise return an empty dict."""
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def build_run_metadata(stage: str, args: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Create structured metadata saved by each stage script."""
    return {
        "stage": stage,
        "args": args,
        "config": config,
    }
