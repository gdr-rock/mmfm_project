#!/usr/bin/env python3
"""Cache placeholder JEPA latents for offline reuse."""

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
from src.data.datasets import load_dataset
from src.data.segmentation import segment_record, validate_segments
from src.models.bridges import GoalBridge
from src.models.jepa_wrapper import FrozenJEPAWrapper
from src.utils.io import ensure_dir, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen JEPA latent features.")
    parser.add_argument("--config", type=str, default="configs/models.yaml")
    parser.add_argument("--dataset", type=str, default="crosstask", choices=["crosstask", "coin"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--window_seconds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("cache_jepa_latents")
    set_seed(args.seed)
    config = load_config(args.config)

    records = load_dataset(dataset=args.dataset, use_dummy_data=args.use_dummy_data)
    jepa = FrozenJEPAWrapper(latent_dim=args.latent_dim)
    goal_bridge = GoalBridge(latent_dim=args.latent_dim)

    all_segment_ids = []
    all_goal_latents = []
    valid_segmentations = 0

    for record in records:
        segments = segment_record(record, window_seconds=args.window_seconds)
        if validate_segments(segments):
            valid_segmentations += 1
        all_segment_ids.extend(seg["segment_id"] for seg in segments)
        all_goal_latents.append(goal_bridge.encode_goal(record.goal))

    z_t = jepa.encode_segments(all_segment_ids)
    delta_z = jepa.transitions(z_t)
    z_goal = np.stack(all_goal_latents, axis=0) if all_goal_latents else np.zeros((0, args.latent_dim), dtype=np.float32)

    cache_dir = ensure_dir(Path(args.output_dir) / "cache")
    cache_path = cache_dir / f"jepa_latents_{args.dataset}.npz"
    np.savez(cache_path, z_t=z_t, delta_z=delta_z, z_goal=z_goal)

    report = {
        "metadata": build_run_metadata("cache_jepa_latents", vars(args), config),
        "num_records": len(records),
        "num_segments": int(len(all_segment_ids)),
        "valid_segmentations": valid_segmentations,
        "latent_shapes": {
            "z_t": list(z_t.shape),
            "delta_z": list(delta_z.shape),
            "z_goal": list(z_goal.shape),
        },
        "cache_path": str(cache_path),
    }
    report_path = cache_dir / f"jepa_cache_report_{args.dataset}.json"
    write_json(report_path, report)
    logger.info("Cached JEPA latents: %s", cache_path)


if __name__ == "__main__":
    main()
