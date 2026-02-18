#!/usr/bin/env python3
"""Prepare dataset manifests and basic split metadata."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
from pathlib import Path

from src.config import build_run_metadata, load_config
from src.data.datasets import load_dataset, save_manifest
from src.utils.io import ensure_dir, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CrossTask/COIN dataset manifests.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default="crosstask", choices=["crosstask", "coin"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_videos", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("prepare_dataset")
    set_seed(args.seed)
    config = load_config(args.config)

    records = load_dataset(dataset=args.dataset, root="data", use_dummy_data=args.use_dummy_data)
    records = records[: args.max_videos]

    out_dir = ensure_dir(Path(args.output_dir) / "data")
    manifest_path = out_dir / f"dataset_manifest_{args.dataset}.json"
    save_manifest(records, manifest_path)

    summary = {
        "metadata": build_run_metadata("prepare_dataset", vars(args), config),
        "num_records": len(records),
        "split": args.split,
        "manifest": str(manifest_path),
    }
    summary_path = out_dir / f"dataset_summary_{args.dataset}.json"
    write_json(summary_path, summary)
    logger.info("Prepared %d records. Manifest: %s", len(records), manifest_path)


if __name__ == "__main__":
    main()
