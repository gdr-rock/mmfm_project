#!/usr/bin/env python3
"""End-to-end smoke test for the placeholder JEPA-grounded pipeline."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
from pathlib import Path

from src.config import build_run_metadata, load_config
from src.models.bridges import GoalBridge, TransitionBridge
from src.models.critic import TextCritic
from src.models.jepa_wrapper import FrozenJEPAWrapper
from src.models.planner_llm import PlannerLLM
from src.planning.scoring import score_candidates
from src.planning.selection import select_best_candidate
from src.utils.io import ensure_dir, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke test with dummy data.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default="crosstask")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--goal", type=str, default="make tea")
    parser.add_argument("--k_candidates", type=int, default=4)
    parser.add_argument("--lambda_transition", type=float, default=0.6)
    parser.add_argument("--mu_goal", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("smoke_test")
    set_seed(args.seed)
    config = load_config(args.config)

    planner = PlannerLLM()
    critic = TextCritic()
    jepa = FrozenJEPAWrapper(latent_dim=64)
    bridge_f = TransitionBridge(latent_dim=64)
    bridge_g = GoalBridge(latent_dim=64)

    candidates = planner.generate_candidates(goal=args.goal, k=args.k_candidates)
    segment_ids = [f"{args.dataset}_seg_{i}" for i in range(5)]
    z_t = jepa.encode_segments(segment_ids)
    delta_z = jepa.transitions(z_t)
    predicted_final = z_t[-1]

    components = []
    for candidate in candidates:
        steps = [str(step) for step in candidate["steps"]]
        state_changes = [f"state shifts after: {step}" for step in steps]
        components.append(
            {
                "critic_cost": critic.score_candidate(candidate),
                "transition_penalty": bridge_f.transition_penalty(state_changes, delta_z),
                "goal_distance": bridge_g.goal_distance(predicted_final, args.goal),
            }
        )

    scores = score_candidates(components, args.lambda_transition, args.mu_goal)
    best = select_best_candidate(candidates, scores)

    smoke_dir = ensure_dir(Path(args.output_dir) / "smoke")
    report = {
        "metadata": build_run_metadata("smoke_test", vars(args), config),
        "num_candidates": len(candidates),
        "scores": scores,
        "best": best,
        "dry_run": args.dry_run,
        "dummy_data": args.use_dummy_data,
    }
    report_path = smoke_dir / "smoke_report.json"
    write_json(report_path, report)
    logger.info("Smoke test completed. Report: %s", report_path)


if __name__ == "__main__":
    main()
