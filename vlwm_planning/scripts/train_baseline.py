from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer_runtime import run_training


def _resolve_config_path(config_arg: str) -> Path:
    p = Path(config_arg)
    if p.exists():
        return p
    candidate = PROJECT_ROOT / config_arg
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Config not found: {config_arg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline training (paper-faithful CE behavior cloning)")
    parser.add_argument("--config", default="configs/baseline_train.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run exactly one optimization step")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on total train steps")
    args = parser.parse_args()

    cfg_path = _resolve_config_path(args.config)
    cfg: Dict = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    print("[Baseline] Loaded config:")
    print(json.dumps(cfg, indent=2))
    print("Conditioning stream: [CONFIG/SYSTEM_PROMPT, VISUAL_CONTEXT, optional ASR, GOAL].")
    print("Target stream: action-only trajectory (behavior cloning).")
    print("Objective: autoregressive CE on action tokens only; conditioning tokens are masked.")

    metrics = run_training(mode="baseline", cfg=cfg, dry_run=args.dry_run, max_steps_override=args.max_steps)
    print("[Baseline] Training complete:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
