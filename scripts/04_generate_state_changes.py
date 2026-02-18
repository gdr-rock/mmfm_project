#!/usr/bin/env python3
"""Generate state-change strings (ΔS) from candidate plan steps."""

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
from src.utils.io import ensure_dir, read_json, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive textual state changes from steps.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default="crosstask", choices=["crosstask", "coin"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--input_candidates", type=str, default="")
    parser.add_argument("--goal", type=str, default="make tea")
    return parser.parse_args()


def _load_candidates(args: argparse.Namespace) -> list[dict]:
    input_path = Path(args.input_candidates) if args.input_candidates else Path(args.output_dir) / "plans" / f"candidates_{args.dataset}.json"
    if input_path.exists():
        payload = read_json(input_path)
        return payload.get("candidates", [])

    planner = PlannerLLM()
    return planner.generate_candidates(goal=args.goal, k=4)


def main() -> None:
    args = parse_args()
    logger = setup_logging("generate_state_changes")
    set_seed(args.seed)
    config = load_config(args.config)

    candidates = _load_candidates(args)
    parsed_steps = []
    state_changes = []
    for item in candidates:
        steps = [str(s).strip() for s in item.get("steps", []) if str(s).strip()]
        parsed_steps.append({"candidate_id": item.get("candidate_id"), "steps": steps})
        changes = [f"after step '{step}', the environment advances toward completion" for step in steps]
        state_changes.append({"candidate_id": item.get("candidate_id"), "delta_s": changes})

    plans_dir = ensure_dir(Path(args.output_dir) / "plans")
    parsed_path = plans_dir / f"parsed_steps_{args.dataset}.json"
    state_path = plans_dir / f"state_changes_{args.dataset}.json"

    write_json(
        parsed_path,
        {
            "metadata": build_run_metadata("generate_state_changes", vars(args), config),
            "parsed_steps": parsed_steps,
        },
    )
    write_json(
        state_path,
        {
            "metadata": build_run_metadata("generate_state_changes", vars(args), config),
            "state_changes": state_changes,
        },
    )
    logger.info("Saved parsed steps: %s", parsed_path)
    logger.info("Saved state changes: %s", state_path)


if __name__ == "__main__":
    main()
