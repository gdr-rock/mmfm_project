"""Placeholder retrieval benchmark evaluation helpers."""

from __future__ import annotations

from typing import Dict, List

from src.eval.metrics import retrieval_at_k


def evaluate_retrieval_stub(ranks: List[int]) -> Dict[str, float]:
    """Compute retrieval metrics from synthetic rank list."""
    return {
        "r_at_1": retrieval_at_k(ranks, 1),
        "r_at_5": retrieval_at_k(ranks, 5),
        "num_queries": float(len(ranks)),
    }
