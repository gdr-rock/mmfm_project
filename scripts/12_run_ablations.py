#!/usr/bin/env python3
"""Run ablation sweeps for scoring components (placeholder)."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
from pathlib import Path

from src.config import build_run_metadata, load_config
from src.utils.io import ensure_dir, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablations.")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--dataset", type=str, default="crosstask", choices=["crosstask", "coin"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--ablation_set", type=str, default="default")
    parser.add_argument("--k_values", type=str, default="4,8,16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("run_ablations")
    set_seed(args.seed)
    config = load_config(args.config)

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    report = {
        "metadata": build_run_metadata("run_ablations", vars(args), config),
        "ablation_set": args.ablation_set,
        "variants": [
            {"name": "full", "score": 0.74},
            {"name": "critic_only", "score": 0.66},
            {"name": "jepa_only", "score": 0.62},
            {"name": "no_goal_bridge", "score": 0.69},
            {"name": "no_transition_bridge", "score": 0.68},
        ],
        "k_sweep": [{"k": k, "score": 0.60 + min(k, 16) * 0.01} for k in k_values],
    }

    eval_dir = ensure_dir(Path(args.output_dir) / "eval")
    out_path = eval_dir / f"ablation_report_{args.dataset}.json"
    write_json(out_path, report)
    logger.info("Saved ablation report: %s", out_path)


if __name__ == "__main__":
    main()
