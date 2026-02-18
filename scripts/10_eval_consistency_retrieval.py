#!/usr/bin/env python3
"""Evaluate retrieval-style text-video consistency benchmark (placeholder)."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
from pathlib import Path

from src.config import build_run_metadata, load_config
from src.eval.retrieval import evaluate_retrieval_stub
from src.utils.io import ensure_dir, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate consistency retrieval metrics.")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--dataset", type=str, default="crosstask", choices=["crosstask", "coin"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num_queries", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("eval_consistency_retrieval")
    set_seed(args.seed)
    config = load_config(args.config)

    ranks = [(i % 7) + 1 for i in range(args.num_queries)]
    metrics = evaluate_retrieval_stub(ranks)
    metrics["requested_topk"] = args.topk

    eval_dir = ensure_dir(Path(args.output_dir) / "eval")
    out_path = eval_dir / f"retrieval_metrics_{args.dataset}.json"
    payload = {
        "metadata": build_run_metadata("eval_consistency_retrieval", vars(args), config),
        "metrics": metrics,
    }
    write_json(out_path, payload)
    logger.info("Saved retrieval metrics: %s", out_path)


if __name__ == "__main__":
    main()
