#!/usr/bin/env python3
"""Build clean COIN full-plan evaluation datasets.

Outputs two JSONL files:
  1) Task-level text set:
       one row per task with goal + interpretation -> full canonical plan
  2) Video-level set:
       one row per test video with goal + interpretation -> full observed plan

These files are intended for full-plan generation evaluation with no prefix.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from coin_utils import (
    load_taxonomy_cache,
    load_transition_rows,
    make_interpretation,
    step_to_state_change,
    trajectory_from_transition_rows,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_task_level_rows(taxonomy: dict) -> list[dict]:
    rows = []
    tasks = sorted(taxonomy["tasks"].values(), key=lambda item: int(item["task_id"]))
    for task in tasks:
        actions = [step["action"] for step in task["canonical_steps"]]
        interpretation = make_interpretation(actions)
        gold_steps = [
            f"{action} | {step_to_state_change(action)}"
            for action in actions
        ]
        rows.append(
            {
                "goal": task["goal"],
                "interpretation": interpretation,
                "gold_steps": gold_steps,
                "input_mode": "goal+interpretation",
                "meta": {
                    "task_id": str(task["task_id"]),
                    "task_name": task["task_name"],
                    "recipe_type": int(task["recipe_type"]),
                    "trajectory_len": len(gold_steps),
                    "source": "coin_task_level_canonical",
                },
            }
        )
    return rows


def build_video_level_rows(taxonomy: dict, transitions_csv: str) -> list[dict]:
    grouped_rows, _ = load_transition_rows(transitions_csv)
    rows = []
    for (task_id, video_id), vid_rows in sorted(
        grouped_rows.items(),
        key=lambda item: (int(item[0][0]), item[0][1]),
    ):
        task = taxonomy["tasks"][str(task_id)]
        task_steps = [step["action"] for step in task["canonical_steps"]]
        interpretation = make_interpretation(task_steps)
        trajectory = trajectory_from_transition_rows(vid_rows)
        gold_steps = [f"{action} | {state_change}" for action, state_change, _ in trajectory]
        first_row = vid_rows[0]
        rows.append(
            {
                "goal": task["goal"],
                "interpretation": interpretation,
                "gold_steps": gold_steps,
                "input_mode": "goal+frames_or_interpretation",
                "meta": {
                    "task_id": str(task_id),
                    "task_name": task["task_name"],
                    "video_id": video_id,
                    "recipe_type": int(task_id),
                    "trajectory_len": len(gold_steps),
                    "roi_start": float(first_row["roi_start"]),
                    "roi_end": float(first_row["roi_end"]),
                    "source": "coin_video_level_test",
                },
            }
        )
    return rows


def compute_duplicate_summary(rows: list[dict], key_fields: tuple[str, ...]) -> dict:
    counts = defaultdict(int)
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        counts[key] += 1
    return {
        "rows": len(rows),
        "unique_keys": len(counts),
        "duplicate_rows": sum(count - 1 for count in counts.values() if count > 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean COIN full-plan test data")
    parser.add_argument("--taxonomy_cache", default="data/coin/coin_taxonomy.json")
    parser.add_argument(
        "--transitions_test_csv",
        default="data/coin/coin_state_change_transitions_test.csv",
    )
    parser.add_argument(
        "--task_output",
        default="data/coin/coin_full_plan_task_test.jsonl",
    )
    parser.add_argument(
        "--video_output",
        default="data/coin/coin_full_plan_video_test.jsonl",
    )
    args = parser.parse_args()

    taxonomy = load_taxonomy_cache(args.taxonomy_cache)
    task_rows = build_task_level_rows(taxonomy)
    video_rows = build_video_level_rows(taxonomy, args.transitions_test_csv)

    write_jsonl(Path(args.task_output), task_rows)
    write_jsonl(Path(args.video_output), video_rows)

    task_stats = compute_duplicate_summary(task_rows, ("goal", "interpretation"))
    video_stats = compute_duplicate_summary(video_rows, ("goal", "interpretation"))

    print(f"Task-level file: {args.task_output}")
    print(
        f"  rows={task_stats['rows']} unique_goal_interp={task_stats['unique_keys']} "
        f"duplicates={task_stats['duplicate_rows']}"
    )
    print(f"Video-level file: {args.video_output}")
    print(
        f"  rows={video_stats['rows']} unique_goal_interp={video_stats['unique_keys']} "
        f"duplicates={video_stats['duplicate_rows']}"
    )


if __name__ == "__main__":
    main()
