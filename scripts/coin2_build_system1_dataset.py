#!/usr/bin/env python3
"""Build a mixed COIN/CrossTask System-1 dataset under data/coin_2.

Design goals:
  - COIN val/test are split by task, not by video.
  - Test rows follow the full-plan task-level format (goal-only and
    goal+interpretation variants).
  - Train/val keep the useful prefix conditioning, but the target is the full
    remaining canonical plan rather than a fixed-k chunk.
  - Every target sequence ends with {"action": "<END>", "state_change": "<END>"}.
  - CrossTask is mixed into training for extra vocabulary/task coverage.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from coin2_utils import (
    build_full_plan_prompt,
    build_remaining_plan_prompt,
    clean_state_change_text,
    distinct_prefix_lengths,
    format_system1_output,
    tokenize_signature,
    write_jsonl,
)
from coin_utils import (
    ensure_taxonomy_cache,
    load_transition_rows as load_coin_transition_rows,
    make_interpretation,
    step_to_state_change as coin_step_to_state_change,
    trajectory_from_transition_rows,
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def normalize_crosstask_state_change(text: str) -> str:
    text = clean_state_change_text(text)
    replacements = {
        "The contents are now stirred and combined.": "The mixture is now stirred and combined.",
        "The steak is now repositioned on the grill.": "The steak is now placed in position on the grill.",
        "The bread is now placed in the pan.": "The bread is now placed in position in the pan.",
        "The steak is now placed on the grill.": "The steak is now placed in position on the grill.",
    }
    return replacements.get(text, text)


def canonical_remaining_steps(
    canonical_steps: list[tuple[str, str]],
    frontier: int,
) -> list[tuple[str, str]]:
    return [
        (action, state_change)
        for idx, (action, state_change) in enumerate(canonical_steps, start=1)
        if idx > frontier
    ]


def prefix_frontier(raw_steps: list[tuple[str, str, int]]) -> int:
    frontier = 0
    for _, _, order in raw_steps:
        frontier = max(frontier, int(order))
    return frontier


def parse_crosstask_tasks(tasks_file: str) -> dict[str, dict]:
    lines = Path(tasks_file).read_text().splitlines()
    tasks: dict[str, dict] = {}
    i = 0
    while i < len(lines):
        task_id = lines[i].strip()
        if not task_id:
            i += 1
            continue
        task_name = lines[i + 1].strip()
        steps = [step.strip() for step in lines[i + 4].split(",") if step.strip()]
        tasks[task_id] = {
            "task_id": task_id,
            "task_name": task_name,
            "steps": steps,
        }
        i += 6
    return tasks


def load_coin_task_records(
    *,
    coin_json: str,
    taxonomy_xlsx: str,
    taxonomy_cache: str,
    transition_csv: str,
) -> dict[tuple[str, str], dict]:
    taxonomy = ensure_taxonomy_cache(coin_json, taxonomy_xlsx, taxonomy_cache)
    grouped_rows, _ = load_coin_transition_rows(transition_csv)

    records: dict[tuple[str, str], dict] = {}
    for task_id, task in taxonomy["tasks"].items():
        canonical_actions = [step["action"] for step in task["canonical_steps"]]
        canonical_steps = [
            (action, clean_state_change_text(coin_step_to_state_change(action)))
            for action in canonical_actions
        ]
        records[("coin", task_id)] = {
            "source": "coin",
            "task_id": task_id,
            "task_name": task["task_name"],
            "goal": task["goal"],
            "interpretation": make_interpretation(canonical_actions),
            "canonical_steps": canonical_steps,
            "videos": {},
        }

    for (task_id, video_id), rows in grouped_rows.items():
        key = ("coin", str(task_id))
        if key not in records:
            continue
        first_row = rows[0]
        raw_steps = [
            (action, clean_state_change_text(state_change), int(order))
            for action, state_change, order in trajectory_from_transition_rows(rows)
        ]
        records[key]["videos"][video_id] = {
            "video_id": video_id,
            "raw_steps": raw_steps,
            "meta": {
                "video_id": video_id,
                "roi_start": float(first_row["roi_start"]),
                "roi_end": float(first_row["roi_end"]),
                "recipe_type": int(task_id),
            },
        }

    return records


def load_crosstask_task_records(
    *,
    repo_root: Path,
    tasks_file: str,
    transition_csv: str,
) -> dict[tuple[str, str], dict]:
    helper_mod = _load_module(
        "build_state_change_dataset",
        repo_root / "scripts" / "04_build_state_change_dataset.py",
    )
    crosstask_step_to_state_change = helper_mod.step_to_state_change
    canonical_tasks = parse_crosstask_tasks(tasks_file)

    grouped_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    task_info: dict[str, dict] = {}
    with open(transition_csv) as handle:
        for row in csv.DictReader(handle):
            key = (row["task_id"], row["video_id"])
            grouped_rows[key].append(row)
            task_info.setdefault(
                row["task_id"],
                {
                    "task_name": row["task_name"],
                    "goal": row["task_goal"],
                },
            )

    records: dict[tuple[str, str], dict] = {}
    for task_id, task in canonical_tasks.items():
        task_name = task["task_name"]
        goal = task_info.get(task_id, {}).get("goal", f"Complete the task: {task_name}.")
        canonical_steps = [
            (
                action,
                normalize_crosstask_state_change(crosstask_step_to_state_change(action)),
            )
            for action in task["steps"]
        ]
        records[("crosstask", task_id)] = {
            "source": "crosstask",
            "task_id": task_id,
            "task_name": task_name,
            "goal": goal,
            "interpretation": make_interpretation(task["steps"]),
            "canonical_steps": canonical_steps,
            "videos": {},
        }

    for (task_id, video_id), rows in grouped_rows.items():
        key = ("crosstask", task_id)
        if key not in records:
            continue
        ordered = sorted(rows, key=lambda row: int(row["seg_pos"]))
        raw_steps = [
            (
                row["action"],
                normalize_crosstask_state_change(row["state_change"]),
                int(row["canonical_order"] or row["step_idx"]),
            )
            for row in ordered
        ]
        last = ordered[-1]
        raw_steps.append(
            (
                last["next_action"],
                normalize_crosstask_state_change(last["next_state_change"]),
                int(last["next_canonical_order"] or last["next_step_idx"]),
            )
        )
        records[key]["videos"][video_id] = {
            "video_id": video_id,
            "raw_steps": raw_steps,
            "meta": {
                "video_id": video_id,
            },
        }

    return records


def task_signature(record: dict) -> set[str]:
    actions = [action for action, _ in record["canonical_steps"]]
    return tokenize_signature(record["goal"], record["task_name"], *actions)


def _coverage_ratio(signature: set[str], train_token_counts: Counter[str]) -> float:
    if not signature:
        return 1.0
    covered = sum(1 for token in signature if train_token_counts[token] > 0)
    return covered / len(signature)


def choose_task_splits(
    task_records: dict[tuple[str, str], dict],
    *,
    coin_test_task_frac: float,
    coin_val_task_frac: float,
    min_train_token_coverage: float,
    seed: int,
) -> dict[tuple[str, str], str]:
    rng = random.Random(seed)
    split_map: dict[tuple[str, str], str] = {}

    coin_keys = [key for key in task_records if key[0] == "coin"]
    crosstask_keys = [key for key in task_records if key[0] == "crosstask"]
    for key in crosstask_keys:
        split_map[key] = "train"

    signatures = {key: task_signature(record) for key, record in task_records.items()}
    train_token_counts: Counter[str] = Counter()
    for key, signature in signatures.items():
        if key in crosstask_keys or key in coin_keys:
            for token in signature:
                train_token_counts[token] += 1

    target_test = max(1, int(round(len(coin_keys) * coin_test_task_frac)))
    target_val = max(1, int(round(len(coin_keys) * coin_val_task_frac)))
    avg_videos = sum(len(task_records[key]["videos"]) for key in coin_keys) / max(len(coin_keys), 1)

    def select_holdout(
        available: list[tuple[str, str]],
        target_n: int,
    ) -> list[tuple[str, str]]:
        ranked = []
        for key in available:
            signature = signatures[key]
            coverage = _coverage_ratio(
                signature,
                Counter({token: max(train_token_counts[token] - 1, 0) for token in signature}),
            )
            ranked.append(
                (
                    coverage,
                    abs(len(task_records[key]["videos"]) - avg_videos),
                    rng.random(),
                    key,
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))

        selected: list[tuple[str, str]] = []
        skipped: list[tuple[float, tuple[str, str]]] = []
        for coverage, _, _, key in ranked:
            if len(selected) >= target_n:
                break
            signature = signatures[key]
            ratio = sum(
                1 for token in signature if (train_token_counts[token] - 1) > 0
            ) / max(len(signature), 1)
            if ratio >= min_train_token_coverage:
                selected.append(key)
                for token in signature:
                    train_token_counts[token] -= 1
            else:
                skipped.append((ratio, key))

        if len(selected) < target_n:
            skipped.sort(key=lambda item: (-item[0], item[1][1]))
            for _, key in skipped:
                if len(selected) >= target_n:
                    break
                if key in selected:
                    continue
                selected.append(key)
                for token in signatures[key]:
                    train_token_counts[token] -= 1

        return selected

    remaining = list(coin_keys)
    test_keys = select_holdout(remaining, target_test)
    remaining = [key for key in remaining if key not in test_keys]
    val_keys = select_holdout(remaining, target_val)
    train_keys = [key for key in remaining if key not in val_keys]

    for key in train_keys:
        split_map[key] = "train"
    for key in val_keys:
        split_map[key] = "val"
    for key in test_keys:
        split_map[key] = "test"

    return split_map


def build_prefix_rows(
    task_records: dict[tuple[str, str], dict],
    split_map: dict[tuple[str, str], str],
    *,
    split: str,
    samples_per_video: int,
    max_prefix: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    for key, record in sorted(task_records.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        if split_map.get(key) != split:
            continue
        canonical_steps = record["canonical_steps"]

        for video_id, video in sorted(record["videos"].items()):
            raw_steps = video["raw_steps"]
            valid_prefixes = []
            max_t = min(max_prefix, len(raw_steps))
            for prefix_len in range(1, max_t + 1):
                frontier = prefix_frontier(raw_steps[:prefix_len])
                remaining = canonical_remaining_steps(canonical_steps, frontier)
                if remaining:
                    valid_prefixes.append(prefix_len)

            for prefix_len in distinct_prefix_lengths(valid_prefixes, samples_per_video, rng):
                prefix_raw = raw_steps[:prefix_len]
                prefix_steps = [(action, state_change) for action, state_change, _ in prefix_raw]
                frontier = prefix_frontier(prefix_raw)
                remaining = canonical_remaining_steps(canonical_steps, frontier)
                if not remaining:
                    continue

                variants = [
                    ("goal_only", ""),
                    ("goal_plus_interpretation", record["interpretation"]),
                ]
                for prompt_variant, interpretation in variants:
                    rows.append(
                        {
                            "input_text": build_remaining_plan_prompt(
                                record["goal"],
                                prefix_steps,
                                interpretation=interpretation,
                            ),
                            "output_text": format_system1_output(remaining, append_end=True),
                            "meta": {
                                "source": record["source"],
                                "task_id": record["task_id"],
                                "task_name": record["task_name"],
                                "video_id": video_id,
                                "split": split,
                                "prompt_style": "remaining_plan_from_prefix",
                                "prompt_variant": prompt_variant,
                                "prefix_len": prefix_len,
                                "target_len": len(remaining),
                                "start_seg_pos": prefix_len,
                                "k": len(remaining),
                                "trajectory_len": len(raw_steps),
                                "has_end_token": True,
                                **video["meta"],
                            },
                        }
                    )

    return rows


def build_full_plan_rows(
    task_records: dict[tuple[str, str], dict],
    split_map: dict[tuple[str, str], str],
    *,
    split: str,
    include_gold_fields: bool,
) -> list[dict]:
    rows: list[dict] = []
    for key, record in sorted(task_records.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        if split_map.get(key) != split:
            continue
        gold_steps = [f"{action} | {state_change}" for action, state_change in record["canonical_steps"]]
        variants = [
            ("goal_only", ""),
            ("goal_plus_interpretation", record["interpretation"]),
        ]
        for prompt_variant, interpretation in variants:
            row = {
                "input_text": build_full_plan_prompt(
                    record["goal"],
                    interpretation=interpretation,
                ),
                "output_text": format_system1_output(record["canonical_steps"], append_end=True),
                "goal": record["goal"],
                "interpretation": record["interpretation"],
                "meta": {
                    "source": record["source"],
                    "task_id": record["task_id"],
                    "task_name": record["task_name"],
                    "split": split,
                    "prompt_style": "full_plan_from_task",
                    "prompt_variant": prompt_variant,
                    "prefix_len": 0,
                    "target_len": len(record["canonical_steps"]),
                    "trajectory_len": len(record["canonical_steps"]),
                    "has_end_token": True,
                },
            }
            if include_gold_fields:
                row["gold_steps"] = gold_steps
                row["input_mode"] = prompt_variant
            rows.append(row)
    return rows


def summarize_rows(rows: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        meta = row.get("meta", {})
        key = f"{meta.get('source', 'unknown')}::{meta.get('prompt_style', 'unknown')}::{meta.get('prompt_variant', 'unknown')}"
        counts[key] += 1
    return dict(sorted(counts.items()))


def build_split_manifest(
    task_records: dict[tuple[str, str], dict],
    split_map: dict[tuple[str, str], str],
) -> list[dict]:
    train_keys = {key for key, split in split_map.items() if split == "train"}
    train_token_counts: Counter[str] = Counter()
    for key in train_keys:
        for token in task_signature(task_records[key]):
            train_token_counts[token] += 1

    rows = []
    for key, record in sorted(task_records.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        signature = task_signature(record)
        rows.append(
            {
                "source": record["source"],
                "task_id": record["task_id"],
                "task_name": record["task_name"],
                "split": split_map[key],
                "n_videos": len(record["videos"]),
                "n_canonical_steps": len(record["canonical_steps"]),
                "train_token_coverage": round(_coverage_ratio(signature, train_token_counts), 4),
            }
        )
    return rows


def write_manifest_csv(path: str | Path, rows: list[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_report(
    task_records: dict[tuple[str, str], dict],
    split_map: dict[tuple[str, str], str],
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
) -> dict:
    manifest_rows = build_split_manifest(task_records, split_map)
    by_source_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in manifest_rows:
        by_source_split[row["source"]][row["split"]] += 1

    coverage_by_split: dict[str, list[float]] = defaultdict(list)
    for row in manifest_rows:
        coverage_by_split[row["split"]].append(float(row["train_token_coverage"]))

    return {
        "tasks_by_source_and_split": {
            source: dict(sorted(counter.items()))
            for source, counter in sorted(by_source_split.items())
        },
        "rows": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "row_breakdown": {
            "train": summarize_rows(train_rows),
            "val": summarize_rows(val_rows),
            "test": summarize_rows(test_rows),
        },
        "train_token_coverage": {
            split: {
                "min": round(min(values), 4) if values else 0.0,
                "mean": round(sum(values) / len(values), 4) if values else 0.0,
                "count": len(values),
            }
            for split, values in sorted(coverage_by_split.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed COIN/CrossTask System-1 data in data/coin_2")
    parser.add_argument("--coin_json", default="COIN_dataset/COIN.json")
    parser.add_argument("--taxonomy_xlsx", default="COIN_dataset/taxonomy.xlsx")
    parser.add_argument("--taxonomy_cache", default="data/coin/coin_taxonomy.json")
    parser.add_argument("--coin_transition_csv", default="data/coin/coin_state_change_transitions.csv")
    parser.add_argument("--crosstask_transition_csv", default="data/crosstask/state_change_transitions.csv")
    parser.add_argument("--crosstask_tasks_file", default="crosstask_release/tasks_primary.txt")
    parser.add_argument("--output_dir", default="data/coin_2")
    parser.add_argument("--samples_per_video", type=int, default=3)
    parser.add_argument("--max_prefix", type=int, default=6)
    parser.add_argument("--coin_test_task_frac", type=float, default=0.15)
    parser.add_argument("--coin_val_task_frac", type=float, default=0.10)
    parser.add_argument("--min_train_token_coverage", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    coin_records = load_coin_task_records(
        coin_json=args.coin_json,
        taxonomy_xlsx=args.taxonomy_xlsx,
        taxonomy_cache=args.taxonomy_cache,
        transition_csv=args.coin_transition_csv,
    )
    crosstask_records = load_crosstask_task_records(
        repo_root=repo_root,
        tasks_file=str(repo_root / args.crosstask_tasks_file),
        transition_csv=str(repo_root / args.crosstask_transition_csv),
    )
    task_records = {**coin_records, **crosstask_records}

    split_map = choose_task_splits(
        task_records,
        coin_test_task_frac=args.coin_test_task_frac,
        coin_val_task_frac=args.coin_val_task_frac,
        min_train_token_coverage=args.min_train_token_coverage,
        seed=args.seed,
    )

    train_rows = build_prefix_rows(
        task_records,
        split_map,
        split="train",
        samples_per_video=args.samples_per_video,
        max_prefix=args.max_prefix,
        seed=args.seed,
    )
    train_rows.extend(
        build_full_plan_rows(
            task_records,
            split_map,
            split="train",
            include_gold_fields=False,
        )
    )

    val_rows = build_prefix_rows(
        task_records,
        split_map,
        split="val",
        samples_per_video=max(1, min(args.samples_per_video, 2)),
        max_prefix=args.max_prefix,
        seed=args.seed + 1,
    )
    val_rows.extend(
        build_full_plan_rows(
            task_records,
            split_map,
            split="val",
            include_gold_fields=False,
        )
    )

    test_rows = build_full_plan_rows(
        task_records,
        split_map,
        split="test",
        include_gold_fields=True,
    )
    test_goal_only = [row for row in test_rows if row["meta"]["prompt_variant"] == "goal_only"]
    test_goal_plus_interpretation = [
        row for row in test_rows
        if row["meta"]["prompt_variant"] == "goal_plus_interpretation"
    ]

    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "coin2_system1_train.jsonl", train_rows)
    write_jsonl(out_dir / "coin2_system1_val.jsonl", val_rows)
    write_jsonl(out_dir / "coin2_system1_test_prompt_variants.jsonl", test_rows)
    write_jsonl(out_dir / "coin2_system1_test_goal_only.jsonl", test_goal_only)
    write_jsonl(
        out_dir / "coin2_system1_test_goal_plus_interpretation.jsonl",
        test_goal_plus_interpretation,
    )

    manifest_rows = build_split_manifest(task_records, split_map)
    write_manifest_csv(out_dir / "coin2_task_split_manifest.csv", manifest_rows)

    report = compute_report(task_records, split_map, train_rows, val_rows, test_rows)
    (out_dir / "coin2_dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )

    print(f"train rows: {len(train_rows)}")
    print(f"val rows: {len(val_rows)}")
    print(f"test rows: {len(test_rows)}")
    print(f"goal-only test rows: {len(test_goal_only)}")
    print(f"goal+interpretation test rows: {len(test_goal_plus_interpretation)}")
    print(f"manifest: {out_dir / 'coin2_task_split_manifest.csv'}")
    print(f"report:   {out_dir / 'coin2_dataset_report.json'}")


if __name__ == "__main__":
    main()

