#!/usr/bin/env python3
"""Build COIN goal-model datasets for train/val/test."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from coin_utils import (
    canonical_remaining_steps,
    ensure_taxonomy_cache,
    format_system1_output,
    load_transition_rows,
    prefix_frontier,
    trajectory_from_transition_rows,
)


def format_prefix(instruction: str, prefix_steps: list[tuple[str, str]]) -> str:
    lines = ["Observed steps:"]
    for idx, (action, state_change) in enumerate(prefix_steps, start=1):
        lines.append(f"  {idx}) {action} | {state_change}")
    lines.append("")
    lines.append(instruction)
    return "\n".join(lines)


def build_samples(
    transition_csv: str,
    taxonomy: dict,
    samples_per_video: int,
    rng: random.Random,
) -> list[dict]:
    grouped_rows, _ = load_transition_rows(transition_csv)
    samples = []

    for (task_id, video_id), rows in grouped_rows.items():
        task = taxonomy["tasks"][str(task_id)]
        raw_steps = trajectory_from_transition_rows(rows)
        n_raw = len(raw_steps)
        if n_raw < 1:
            continue
        first_row = rows[0]

        for _ in range(samples_per_video):
            prefix_len = rng.randint(1, n_raw)
            prefix_raw = raw_steps[:prefix_len]
            prefix_steps = [(a, s) for a, s, _ in prefix_raw]
            frontier = prefix_frontier(prefix_raw)
            remaining = canonical_remaining_steps(task, frontier)

            samples.append(
                {
                    "task": "goal_prediction",
                    "input_text": format_prefix(
                        "Identify the overall goal of this task.",
                        prefix_steps,
                    ),
                    "output_text": task["goal"],
                    "meta": {
                        "task_id": str(task_id),
                        "task_name": task["task_name"],
                        "video_id": video_id,
                        "prefix_len": prefix_len,
                        "trajectory_len": n_raw,
                        "roi_start": float(first_row["roi_start"]),
                        "roi_end": float(first_row["roi_end"]),
                        "recipe_type": int(task_id),
                        "canonical_frontier": frontier,
                    },
                }
            )

            samples.append(
                {
                    "task": "goal_and_plan",
                    "input_text": format_prefix(
                        "Identify the overall goal and list the remaining steps to complete it.",
                        prefix_steps,
                    ),
                    "output_text": json.dumps(
                        {
                            "goal": task["goal"],
                            "remaining_plan": [
                                f"{action} | {state_change}"
                                for action, state_change in remaining
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    "meta": {
                        "task_id": str(task_id),
                        "task_name": task["task_name"],
                        "video_id": video_id,
                        "prefix_len": prefix_len,
                        "trajectory_len": n_raw,
                        "remaining_plan_len": len(remaining),
                        "roi_start": float(first_row["roi_start"]),
                        "roi_end": float(first_row["roi_end"]),
                        "recipe_type": int(task_id),
                        "canonical_frontier": frontier,
                    },
                }
            )

    return samples


def write_jsonl(path: str, samples: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def validate_samples(samples: list[dict]) -> dict:
    bad_goal = 0
    bad_json = 0
    for sample in samples:
        if sample["task"] == "goal_prediction":
            if not sample["output_text"].startswith("Complete the task: "):
                bad_goal += 1
        else:
            try:
                payload = json.loads(sample["output_text"])
                assert "goal" in payload and "remaining_plan" in payload
            except Exception:
                bad_json += 1
    return {"bad_goal": bad_goal, "bad_json": bad_json}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build COIN goal-model datasets")
    parser.add_argument("--coin_json", default="COIN_dataset/COIN.json")
    parser.add_argument("--taxonomy_xlsx", default="COIN_dataset/taxonomy.xlsx")
    parser.add_argument("--taxonomy_cache", default="data/coin/coin_taxonomy.json")
    parser.add_argument("--data_dir", default="data/coin")
    parser.add_argument("--samples_per_video", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    taxonomy = ensure_taxonomy_cache(
        args.coin_json, args.taxonomy_xlsx, args.taxonomy_cache
    )
    rng = random.Random(args.seed)

    outputs = {}
    for split in ("train", "val", "test"):
        transition_csv = os.path.join(args.data_dir, f"coin_state_change_transitions_{split}.csv")
        if not os.path.exists(transition_csv):
            raise FileNotFoundError(f"Missing transition CSV: {transition_csv}")
        split_rng = random.Random(rng.randint(0, 10**9))
        samples = build_samples(
            transition_csv=transition_csv,
            taxonomy=taxonomy,
            samples_per_video=args.samples_per_video,
            rng=split_rng,
        )
        stats = validate_samples(samples)
        out_path = os.path.join(args.data_dir, f"coin_goal_{split}.jsonl")
        write_jsonl(out_path, samples)
        outputs[split] = (out_path, len(samples), stats)

    for split, (path, count, stats) in outputs.items():
        print(f"{split}: {path}")
        print(
            f"  samples={count} bad_goal={stats['bad_goal']} bad_json={stats['bad_json']}"
        )


if __name__ == "__main__":
    main()
