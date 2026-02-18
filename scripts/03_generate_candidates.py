#!/usr/bin/env python3
"""Generate K candidate plans from goal text (placeholder LLM)."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
from pathlib import Path

from src.config import build_run_metadata, load_config
from src.models.planner_llm import PlannerLLM
from src.utils.io import ensure_dir, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate step plans.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default="crosstask", choices=["crosstask", "coin"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--goal", type=str, default="make tea")
    parser.add_argument("--k_candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("generate_candidates")
    set_seed(args.seed)
    config = load_config(args.config)

    planner = PlannerLLM(model_name="placeholder-llm")
    candidates = planner.generate_candidates(goal=args.goal, k=args.k_candidates)

    plans_dir = ensure_dir(Path(args.output_dir) / "plans")
    out_path = plans_dir / f"candidates_{args.dataset}.json"
    payload = {
        "metadata": build_run_metadata("generate_candidates", vars(args), config),
        "goal": args.goal,
        "k_candidates": args.k_candidates,
        "candidates": candidates,
    }
    write_json(out_path, payload)
    logger.info("Saved %d candidates: %s", len(candidates), out_path)


if __name__ == "__main__":
    main()
