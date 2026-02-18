"""Placeholder VPA evaluation driver."""

from __future__ import annotations

from typing import Dict, List

from src.eval.metrics import mean_ci95, success_rate


def evaluate_vpa_stub(outcomes: List[bool]) -> Dict[str, float]:
    """Compute placeholder VPA metrics from boolean outcomes."""
    successes = sum(1 for x in outcomes if x)
    total = len(outcomes)
    sr = success_rate(successes, total)
    m, ci = mean_ci95([1.0 if x else 0.0 for x in outcomes])
    return {
        "task_success": sr,
        "mean": m,
        "ci95": ci,
        "num_samples": total,
    }
