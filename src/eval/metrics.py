"""Metrics used by placeholder evaluation scripts."""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple


def success_rate(successes: int, total: int) -> float:
    """Compute task success rate."""
    if total <= 0:
        return 0.0
    return float(successes / total)


def mean(values: Iterable[float]) -> float:
    """Compute arithmetic mean."""
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def mean_ci95(values: List[float]) -> Tuple[float, float]:
    """Return (mean, half-width of approximate 95% CI)."""
    if not values:
        return 0.0, 0.0
    m = mean(values)
    if len(values) == 1:
        return m, 0.0
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    stderr = math.sqrt(variance / len(values))
    return m, 1.96 * stderr


def retrieval_at_k(ranks: List[int], k: int) -> float:
    """Compute retrieval@k from 1-indexed rank list."""
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r <= k)
    return float(hits / len(ranks))
