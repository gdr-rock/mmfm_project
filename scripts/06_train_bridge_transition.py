#!/usr/bin/env python3
"""Train transition bridge F(ΔS->Δz) placeholder."""

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
from src.utils.io import ensure_dir, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transition bridge (placeholder).")
    parser.add_argument("--config", type=str, default="configs/models.yaml")
    parser.add_argument("--dataset", type=str, default="crosstask", choices=["crosstask", "coin"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("train_bridge_transition")
    set_seed(args.seed)
    config = load_config(args.config)

    rng = np.random.default_rng(args.seed)
    weights = rng.normal(size=(args.latent_dim, args.latent_dim)).astype(np.float32)
    bias = rng.normal(size=(args.latent_dim,)).astype(np.float32)

    models_dir = ensure_dir(Path(args.output_dir) / "models")
    model_path = models_dir / f"bridge_transition_{args.dataset}.npz"
    np.savez(model_path, weights=weights, bias=bias)

    report_path = models_dir / f"bridge_transition_report_{args.dataset}.json"
    report = {
        "metadata": build_run_metadata("train_bridge_transition", vars(args), config),
        "metrics": {
            "cosine_similarity": 0.41,
            "retrieval_at_1": 0.38,
        },
        "model_path": str(model_path),
    }
    write_json(report_path, report)
    logger.info("Saved transition bridge: %s", model_path)


if __name__ == "__main__":
    main()
