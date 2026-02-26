#!/usr/bin/env python3
"""
Split all generated datasets into train / val / test using the official
CrossTask video-level validation split.

Split strategy (video-level, no leakage):
  - val:   videos listed in  crosstask_release/videos_val.csv   (~13%)
  - test:  10% of remaining videos (per-task stratified)         (~9%)
  - train: everything else                                       (~78%)

Applied to:
  1. data/crosstask/state_change_transitions.csv  → {train,val,test}.csv
  2. data/crosstask/critic_train.jsonl            → critic_{train,val,test}.jsonl
  3. data/crosstask/system1_train.jsonl           → system1_{train,val,test}.jsonl
  4. data/crosstask/goal_train.jsonl              → goal_{train,val,test}.jsonl

Usage:
    python scripts/08_split_datasets.py [--seed 42] [--test_frac 0.10]
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, Set, Tuple


# ---------------------------------------------------------------------------
# 1.  Build the split mapping
# ---------------------------------------------------------------------------

def load_val_videos(val_csv: str) -> Set[Tuple[str, str]]:
    """Load official CrossTask validation video IDs."""
    val_vids: Set[Tuple[str, str]] = set()
    with open(val_csv) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                val_vids.add((parts[0], parts[1]))
    return val_vids


def build_split_map(
    transitions_csv: str,
    val_csv: str,
    test_frac: float,
    rng: random.Random,
) -> Dict[Tuple[str, str], str]:
    """
    Assign every (task_id, video_id) to 'train', 'val', or 'test'.

    1. val  ← official CrossTask val set
    2. test ← stratified 10% of remaining, per task
    3. train ← everything else
    """
    # Collect all (task_id, video_id) from the transition CSV
    all_vids: Dict[str, list] = defaultdict(list)  # task_id → [video_id, ...]
    with open(transitions_csv) as f:
        reader = csv.DictReader(f)
        seen = set()
        for row in reader:
            key = (row["task_id"], row["video_id"])
            if key not in seen:
                seen.add(key)
                all_vids[row["task_id"]].append(row["video_id"])

    val_set = load_val_videos(val_csv)

    split_map: Dict[Tuple[str, str], str] = {}

    for task_id, video_ids in all_vids.items():
        train_pool = []
        for vid in video_ids:
            key = (task_id, vid)
            if key in val_set:
                split_map[key] = "val"
            else:
                train_pool.append(vid)

        # From train_pool, hold out test_frac for test
        rng.shuffle(train_pool)
        n_test = max(1, int(len(train_pool) * test_frac))
        test_vids = set(train_pool[:n_test])

        for vid in train_pool:
            key = (task_id, vid)
            split_map[key] = "test" if vid in test_vids else "train"

    return split_map


# ---------------------------------------------------------------------------
# 2.  Split a CSV file
# ---------------------------------------------------------------------------

def split_csv(
    input_path: str,
    output_dir: str,
    basename: str,
    split_map: Dict[Tuple[str, str], str],
    task_id_col: str = "task_id",
    video_id_col: str = "video_id",
):
    """Split a CSV file into train/val/test based on video-level split."""
    with open(input_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Collect rows per split
        buckets: Dict[str, list] = {"train": [], "val": [], "test": []}
        skipped = 0
        for row in reader:
            key = (row[task_id_col], row[video_id_col])
            split = split_map.get(key)
            if split:
                buckets[split].append(row)
            else:
                skipped += 1

    for split_name, rows in buckets.items():
        out_path = os.path.join(output_dir, f"{basename}_{split_name}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return {k: len(v) for k, v in buckets.items()}, skipped


# ---------------------------------------------------------------------------
# 3.  Split a JSONL file
# ---------------------------------------------------------------------------

def split_jsonl(
    input_path: str,
    output_dir: str,
    basename: str,
    split_map: Dict[Tuple[str, str], str],
    meta_key: str = "meta",
    task_id_field: str = "task_id",
    video_id_field: str = "video_id",
):
    """Split a JSONL file into train/val/test based on video-level split."""
    buckets: Dict[str, list] = {"train": [], "val": [], "test": []}
    skipped = 0

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            meta = record.get(meta_key) or record.get("metadata", {})
            key = (meta.get(task_id_field, ""), meta.get(video_id_field, ""))
            split = split_map.get(key)
            if split:
                buckets[split].append(line)
            else:
                skipped += 1

    for split_name, lines in buckets.items():
        out_path = os.path.join(output_dir, f"{basename}_{split_name}.jsonl")
        with open(out_path, "w") as f:
            for l in lines:
                f.write(l + "\n")

    return {k: len(v) for k, v in buckets.items()}, skipped


# ---------------------------------------------------------------------------
# 4.  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Split all CrossTask datasets into train/val/test."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/crosstask",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="crosstask_release/videos_val.csv",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.10,
        help="Fraction of non-val videos to hold out as test (per task)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir = args.data_dir
    transitions_csv = os.path.join(data_dir, "state_change_transitions.csv")

    # ── Build split map ──────────────────────────────────────────────────
    print(f"[1/6] Building video-level split map...")
    print(f"       val source:  {args.val_csv}")
    print(f"       test_frac:   {args.test_frac}")

    split_map = build_split_map(
        transitions_csv, args.val_csv, args.test_frac, rng
    )

    from collections import Counter
    split_counts = Counter(split_map.values())
    total_vids = len(split_map)
    print(f"       Total videos: {total_vids}")
    for s in ("train", "val", "test"):
        c = split_counts.get(s, 0)
        print(f"         {s:5s}: {c:5d} videos ({100*c/total_vids:.1f}%)")

    # Check per-task coverage
    task_splits: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (tid, vid), split in split_map.items():
        task_splits[tid][split] += 1

    print(f"\n       Per-task split:")
    print(f"       {'task_id':>8s}  {'train':>6s}  {'val':>5s}  {'test':>5s}")
    for tid in sorted(task_splits.keys()):
        d = task_splits[tid]
        print(f"       {tid:>8s}  {d.get('train',0):>6d}  {d.get('val',0):>5d}  {d.get('test',0):>5d}")

    # ── Split transitions CSV ────────────────────────────────────────────
    print(f"\n[2/6] Splitting state_change_transitions.csv ...")
    counts, skip = split_csv(
        transitions_csv, data_dir, "state_change_transitions", split_map
    )
    print(f"       {counts}  (skipped {skip})")

    # ── Split critic JSONL ───────────────────────────────────────────────
    critic_path = os.path.join(data_dir, "critic_train.jsonl")
    if os.path.exists(critic_path):
        print(f"\n[3/6] Splitting critic_train.jsonl ...")
        counts, skip = split_jsonl(
            critic_path, data_dir, "critic", split_map,
            meta_key="metadata",
        )
        print(f"       {counts}  (skipped {skip})")
    else:
        print(f"\n[3/6] critic_train.jsonl not found — skipping")

    # ── Split system1 JSONL ──────────────────────────────────────────────
    sys1_path = os.path.join(data_dir, "system1_train.jsonl")
    if os.path.exists(sys1_path):
        print(f"\n[4/6] Splitting system1_train.jsonl ...")
        counts, skip = split_jsonl(
            sys1_path, data_dir, "system1", split_map,
            meta_key="meta",
        )
        print(f"       {counts}  (skipped {skip})")
    else:
        print(f"\n[4/6] system1_train.jsonl not found — skipping")

    # ── Split goal JSONL ─────────────────────────────────────────────────
    goal_path = os.path.join(data_dir, "goal_train.jsonl")
    if os.path.exists(goal_path):
        print(f"\n[5/6] Splitting goal_train.jsonl ...")
        counts, skip = split_jsonl(
            goal_path, data_dir, "goal", split_map,
            meta_key="meta",
        )
        print(f"       {counts}  (skipped {skip})")
    else:
        print(f"\n[5/6] goal_train.jsonl not found — skipping")

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n[6/6] Final file inventory:")
    for fname in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            # Count lines
            with open(fpath) as f:
                n_lines = sum(1 for _ in f)
            # Subtract 1 for CSV header
            label = ""
            if fname.endswith(".csv"):
                n_lines -= 1
                label = "rows"
            else:
                label = "samples"
            print(f"       {fname:45s}  {n_lines:>7d} {label:>7s}  {size_mb:>6.1f} MB")


if __name__ == "__main__":
    main()
