"""Candidate plan generation placeholder for text goals."""

from __future__ import annotations

from typing import Dict, List


class PlannerLLM:
    """Generate synthetic candidates with varied step wording.

    TODO: Connect to an actual LLM backend and prompt templates.
    """

    def __init__(self, model_name: str = "placeholder-llm") -> None:
        self.model_name = model_name

    def generate_candidates(self, goal: str, k: int = 8) -> List[Dict[str, object]]:
        """Return K candidate plans as step lists."""
        templates = [
            [
                f"understand the goal: {goal}",
                "prepare required tools",
                "execute ordered procedure",
                "verify final outcome",
            ],
            [
                "gather ingredients/materials",
                "set up environment",
                "perform core steps carefully",
                "clean up and confirm success",
            ],
            [
                "check preconditions",
                "start process",
                "monitor progress",
                "finish and inspect result",
            ],
        ]
        candidates: List[Dict[str, object]] = []
        for idx in range(k):
            steps = templates[idx % len(templates)]
            candidates.append(
                {
                    "candidate_id": idx,
                    "goal": goal,
                    "steps": steps,
                    "source": self.model_name,
                }
            )
        return candidates
