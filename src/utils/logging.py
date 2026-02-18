"""Logging setup helpers with timestamped output."""

from __future__ import annotations

import logging
import sys


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Create or reuse a logger with a consistent formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level.upper())
    logger.propagate = False
    return logger
