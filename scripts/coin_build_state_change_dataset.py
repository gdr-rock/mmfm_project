#!/usr/bin/env python3
"""Build COIN state-change transitions plus split CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from coin_utils import (
    canonical_remaining_steps,
    ensure_taxonomy_cache,
    load_split_manifest,
    read_coin_database,
    step_to_state_change,
)


FIELDNAMES = [
    "task_id",
    "task_name",
    "video_id",
    "transition_id",
    "seg_pos",
    "next_seg_pos",
    "step_idx",
    "action_id",
    "action",
    "state_change",
    "next_step_idx",
    "next_action_id",
    "next_action",
    "next_state_change",
    "action_start",
    "action_end",
    "next_action_start",
    "next_action_end",
    "roi_start",
    "roi_end",
    "canonical_order",
    "next_canonical_order",
    "is_canonical_next",
    "is_time_ordered",
    "task_goal",
    "remaining_plan",
    "plan_progress",
]


def build_rows(database: dict, taxonomy: dict, split_manifest: dict) -> list[dict]:
    rows = []
    for video_id, record in database.items():
        if video_id not in split_manifest:
            continue
        annotations = record.get("annotation", [])
        if len(annotations) < 2:
            continue

        task_id = str(int(record["recipe_type"]))
        task = taxonomy["tasks"][task_id]
        action_to_order = {
            int(step["action_id"]): int(step["step_idx"])
            for step in task["canonical_steps"]
        }
        frontier = 0
        n_canonical = len(task["canonical_steps"])

        for idx in range(len(annotations) - 1):
            cur = annotations[idx]
            nxt = annotations[idx + 1]
            cur_action_id = int(cur["id"])
            nxt_action_id = int(nxt["id"])
            cur_order = action_to_order[cur_action_id]
            nxt_order = action_to_order[nxt_action_id]

            frontier = max(frontier, cur_order)
            remaining = canonical_remaining_steps(task, frontier)

            row = {
                "task_id": task_id,
                "task_name": task["task_name"],
                "video_id": video_id,
                "transition_id": f"{task_id}_{video_id}_{idx}",
                "seg_pos": idx,
                "next_seg_pos": idx + 1,
                "step_idx": cur_order,
                "action_id": cur_action_id,
                "action": cur["label"],
                "state_change": step_to_state_change(cur["label"]),
                "next_step_idx": nxt_order,
                "next_action_id": nxt_action_id,
                "next_action": nxt["label"],
                "next_state_change": step_to_state_change(nxt["label"]),
                "action_start": float(cur["segment"][0]),
                "action_end": float(cur["segment"][1]),
                "next_action_start": float(nxt["segment"][0]),
                "next_action_end": float(nxt["segment"][1]),
                "roi_start": float(record["start"]),
                "roi_end": float(record["end"]),
                "canonical_order": cur_order,
                "next_canonical_order": nxt_order,
                "is_canonical_next": nxt_order >= cur_order,
                "is_time_ordered": float(nxt["segment"][0]) >= float(cur["segment"][0]),
                "task_goal": task["goal"],
                "remaining_plan": " -> ".join(action for action, _ in remaining) if remaining else "(task complete)",
                "plan_progress": round(frontier / max(n_canonical, 1), 3),
            }
            rows.append(row)

    rows.sort(key=lambda row: (int(row["task_id"]), row["video_id"], int(row["seg_pos"])))
    return rows


def write_split_csvs(rows: list[dict], split_manifest: dict, output_dir: str) -> None:
    buckets = {"train": [], "val": [], "test": []}
    for row in rows:
        split = split_manifest[row["video_id"]]["split"]
        buckets[split].append(row)

    out_dir = Path(output_dir)
    for split, split_rows in buckets.items():
        out_path = out_dir / f"coin_state_change_transitions_{split}.csv"
        with open(out_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(split_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build COIN state-change transitions")
    parser.add_argument("--coin_json", default="COIN_dataset/COIN.json")
    parser.add_argument("--taxonomy_xlsx", default="COIN_dataset/taxonomy.xlsx")
    parser.add_argument("--taxonomy_cache", default="data/coin/coin_taxonomy.json")
    parser.add_argument("--split_csv", default="data/coin/coin_video_splits.csv")
    parser.add_argument("--output", default="data/coin/coin_state_change_transitions.csv")
    args = parser.parse_args()

    taxonomy = ensure_taxonomy_cache(
        args.coin_json, args.taxonomy_xlsx, args.taxonomy_cache
    )
    database = read_coin_database(args.coin_json)
    split_manifest = load_split_manifest(args.split_csv)

    rows = build_rows(database, taxonomy, split_manifest)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    write_split_csvs(rows, split_manifest, str(out_path.parent))

    canonical_next = Counter(row["is_canonical_next"] for row in rows)
    print(f"Saved transitions: {args.output}")
    print(f"Rows: {len(rows)}")
    print(f"Canonical-next distribution: {dict(canonical_next)}")


if __name__ == "__main__":
    main()
