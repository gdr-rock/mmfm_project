#!/usr/bin/env python3
"""Score candidate plans and select best via System-2 objective."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
from pathlib import Path

import numpy as np

from src.config import build_run_metadata, load_config
from src.models.bridges import GoalBridge, TransitionBridge
from src.models.critic import TextCritic
from src.models.jepa_wrapper import FrozenJEPAWrapper
from src.models.planner_llm import PlannerLLM
from src.planning.scoring import score_candidates
from src.planning.selection import select_best_candidate
from src.utils.io import ensure_dir, read_json, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="System-2 scoring and selection.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default="crosstask", choices=["crosstask", "coin"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--goal", type=str, default="make tea")
    parser.add_argument("--k_candidates", type=int, default=8)
    parser.add_argument("--lambda_transition", type=float, default=0.6)
    parser.add_argument("--mu_goal", type=float, default=0.8)
    return parser.parse_args()


def _load_or_generate_candidates(args: argparse.Namespace) -> list[dict]:
    cand_path = Path(args.output_dir) / "plans" / f"candidates_{args.dataset}.json"
    if cand_path.exists():
        payload = read_json(cand_path)
        candidates = payload.get("candidates", [])
        if candidates:
            return candidates
    return PlannerLLM().generate_candidates(args.goal, args.k_candidates)


def _load_state_changes(args: argparse.Namespace, candidates: list[dict]) -> dict[int, list[str]]:
    state_path = Path(args.output_dir) / "plans" / f"state_changes_{args.dataset}.json"
    if state_path.exists():
        payload = read_json(state_path)
        mapping = {}
        for item in payload.get("state_changes", []):
            cid = int(item.get("candidate_id", 0))
            mapping[cid] = [str(x) for x in item.get("delta_s", [])]
        return mapping

    fallback = {}
    for c in candidates:
        cid = int(c.get("candidate_id", 0))
        fallback[cid] = [f"state shifts after: {s}" for s in c.get("steps", [])]
    return fallback


def _load_delta_z(args: argparse.Namespace, latent_dim: int) -> np.ndarray:
    cache_path = Path(args.output_dir) / "cache" / f"jepa_latents_{args.dataset}.npz"
    if cache_path.exists():
        arr = np.load(cache_path)
        delta = arr["delta_z"]
        if delta.size > 0:
            return delta

    jepa = FrozenJEPAWrapper(latent_dim=latent_dim)
    z_t = jepa.encode_segments([f"{args.dataset}_seg_{i}" for i in range(5)])
    return jepa.transitions(z_t)


def main() -> None:
    args = parse_args()
    logger = setup_logging("score_and_select")
    set_seed(args.seed)
    config = load_config(args.config)

    latent_dim = 256
    candidates = _load_or_generate_candidates(args)
    delta_z = _load_delta_z(args, latent_dim)
    predicted_final = delta_z[-1] if len(delta_z) else np.zeros((latent_dim,), dtype=np.float32)

    critic = TextCritic()
    bridge_f = TransitionBridge(latent_dim=latent_dim)
    bridge_g = GoalBridge(latent_dim=latent_dim)

    state_change_map = _load_state_changes(args, candidates)
    components = []
    for candidate in candidates:
        cid = int(candidate.get("candidate_id", 0))
        changes = state_change_map.get(cid, [])
        components.append(
            {
                "candidate_id": cid,
                "critic_cost": critic.score_candidate(candidate),
                "transition_penalty": bridge_f.transition_penalty(changes, delta_z),
                "goal_distance": bridge_g.goal_distance(predicted_final, args.goal),
            }
        )

    score_inputs = [
        {
            "critic_cost": item["critic_cost"],
            "transition_penalty": item["transition_penalty"],
            "goal_distance": item["goal_distance"],
        }
        for item in components
    ]
    scores = score_candidates(score_inputs, args.lambda_transition, args.mu_goal)
    best = select_best_candidate(candidates, scores)

    plans_dir = ensure_dir(Path(args.output_dir) / "plans")
    out_path = plans_dir / f"selected_plan_{args.dataset}.json"
    payload = {
        "metadata": build_run_metadata("score_and_select", vars(args), config),
        "components": components,
        "scores": scores,
        "selection": best,
    }
    write_json(out_path, payload)
    logger.info("Saved selected plan: %s", out_path)


if __name__ == "__main__":
    main()
