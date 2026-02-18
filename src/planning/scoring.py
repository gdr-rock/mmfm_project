"""Score composition for System-2 plan selection."""

from __future__ import annotations

from typing import Dict, List


def combined_cost(
    critic_cost: float,
    transition_penalty: float,
    goal_distance: float,
    lambda_transition: float = 1.0,
    mu_goal: float = 1.0,
) -> float:
    """Weighted total cost (lower is better)."""
    return float(critic_cost + lambda_transition * transition_penalty + mu_goal * goal_distance)


def score_candidates(
    components: List[Dict[str, float]],
    lambda_transition: float,
    mu_goal: float,
) -> List[float]:
    """Compute weighted costs for a list of candidate component dicts."""
    scores: List[float] = []
    for item in components:
        scores.append(
            combined_cost(
                critic_cost=float(item["critic_cost"]),
                transition_penalty=float(item["transition_penalty"]),
                goal_distance=float(item["goal_distance"]),
                lambda_transition=lambda_transition,
                mu_goal=mu_goal,
            )
        )
    return scores
