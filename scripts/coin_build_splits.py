#!/usr/bin/env python3
"""Build train/val/test video splits for COIN."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict

from coin_utils import build_video_split_rows, ensure_taxonomy_cache, read_coin_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Build COIN video-level splits")
    parser.add_argument("--coin_json", default="COIN_dataset/COIN.json")
    parser.add_argument("--taxonomy_xlsx", default="COIN_dataset/taxonomy.xlsx")
    parser.add_argument("--taxonomy_cache", default="data/coin/coin_taxonomy.json")
    parser.add_argument("--output", default="data/coin/coin_video_splits.csv")
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    taxonomy = ensure_taxonomy_cache(
        args.coin_json, args.taxonomy_xlsx, args.taxonomy_cache
    )
    database = read_coin_database(args.coin_json)
    rows = build_video_split_rows(database, taxonomy, args.val_frac, args.seed)

    fieldnames = [
        "task_id",
        "task_name",
        "video_id",
        "split",
        "official_subset",
        "recipe_type",
        "video_url",
        "roi_start",
        "roi_end",
        "duration",
    ]
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["split"] for row in rows)
    per_task = defaultdict(Counter)
    for row in rows:
        per_task[row["task_id"]][row["split"]] += 1
    tasks_missing_val = sum(1 for c in per_task.values() if c["val"] == 0)

    print(f"Saved split manifest: {args.output}")
    print(f"Videos: {len(rows)}")
    print(f"Split counts: {dict(counts)}")
    print(f"Tasks without val videos: {tasks_missing_val}")


if __name__ == "__main__":
    main()
