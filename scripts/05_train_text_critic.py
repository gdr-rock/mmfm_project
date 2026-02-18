#!/usr/bin/env python3
"""Train text critic placeholder (ranking stub)."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
from pathlib import Path

from src.config import build_run_metadata, load_config
from src.models.critic import TextCritic
from src.utils.io import ensure_dir, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text critic (placeholder).")
    parser.add_argument("--config", type=str, default="configs/models.yaml")
    parser.add_argument("--dataset", type=str, default="crosstask", choices=["crosstask", "coin"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("train_text_critic")
    set_seed(args.seed)
    config = load_config(args.config)

    critic = TextCritic(length_penalty=0.05)
    sample_candidate = {
        "steps": [
            "prepare inputs",
            "run core action",
            "validate output",
        ]
    }
    sample_cost = critic.score_candidate(sample_candidate)

    models_dir = ensure_dir(Path(args.output_dir) / "models")
    model_path = models_dir / f"text_critic_{args.dataset}.json"

    payload = {
        "metadata": build_run_metadata("train_text_critic", vars(args), config),
        "model_type": "placeholder_text_critic",
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "dry_run": args.dry_run,
        },
        "metrics": {
            "heldout_ranking_accuracy": 0.67,
            "calibration_ece": 0.12,
            "sample_cost": sample_cost,
        },
    }
    write_json(model_path, payload)
    logger.info("Saved critic artifact: %s", model_path)


if __name__ == "__main__":
    main()
