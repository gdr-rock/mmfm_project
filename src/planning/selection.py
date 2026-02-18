"""Candidate selection utilities."""

from __future__ import annotations

from typing import Dict, List


def select_best_candidate(candidates: List[Dict[str, object]], scores: List[float]) -> Dict[str, object]:
    """Select candidate with minimum score and return payload."""
    if not candidates or not scores or len(candidates) != len(scores):
        raise ValueError("Candidates and scores must be non-empty and aligned.")

    best_index = min(range(len(scores)), key=lambda idx: scores[idx])
    return {
        "best_index": best_index,
        "best_score": float(scores[best_index]),
        "best_candidate": candidates[best_index],
    }
