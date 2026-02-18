"""Text critic placeholder used for plan ranking."""

from __future__ import annotations

from typing import Dict, List


class TextCritic:
    """Simple heuristic critic returning lower-is-better costs.

    TODO: Replace with trained ranking model and calibration.
    """

    def __init__(self, length_penalty: float = 0.05) -> None:
        self.length_penalty = length_penalty

    def score_candidate(self, candidate: Dict[str, object]) -> float:
        """Compute a deterministic text cost from candidate steps."""
        steps = candidate.get("steps", [])
        if not isinstance(steps, list):
            return 10.0
        avg_len = sum(len(str(s)) for s in steps) / max(1, len(steps))
        return float(len(steps) * self.length_penalty + 1.0 / (1.0 + avg_len))

    def batch_score(self, candidates: List[Dict[str, object]]) -> List[float]:
        """Score a batch of candidates."""
        return [self.score_candidate(c) for c in candidates]
