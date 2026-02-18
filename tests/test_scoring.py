"""Unit tests for scoring and candidate selection."""

from src.planning.scoring import combined_cost, score_candidates
from src.planning.selection import select_best_candidate


def test_combined_cost_math() -> None:
    score = combined_cost(critic_cost=1.0, transition_penalty=2.0, goal_distance=3.0, lambda_transition=0.5, mu_goal=0.2)
    assert score == 1.0 + 0.5 * 2.0 + 0.2 * 3.0


def test_score_candidates_and_selection() -> None:
    components = [
        {"critic_cost": 0.5, "transition_penalty": 0.4, "goal_distance": 0.3},
        {"critic_cost": 0.7, "transition_penalty": 0.2, "goal_distance": 0.1},
    ]
    scores = score_candidates(components, lambda_transition=1.0, mu_goal=1.0)
    candidates = [{"candidate_id": 0}, {"candidate_id": 1}]
    best = select_best_candidate(candidates, scores)

    assert len(scores) == 2
    assert best["best_index"] in (0, 1)
    assert best["best_candidate"]["candidate_id"] == scores.index(min(scores))
