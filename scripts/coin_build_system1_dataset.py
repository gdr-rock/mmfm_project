#!/usr/bin/env python3
"""Build COIN System-1 datasets for train/val/test."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from coin_utils import (
    canonical_remaining_steps,
    ensure_taxonomy_cache,
    format_system1_output,
    format_system1_prompt,
    load_transition_rows,
    make_interpretation,
    prefix_frontier,
    trajectory_from_transition_rows,
)


def build_samples(
    transition_csv: str,
    taxonomy: dict,
    samples_per_video: int,
    max_prefix: int,
    k_values: list[int],
    rng: random.Random,
    zero_prefix_prob: float,
    interpretation_dropout_prob: float,
    allow_zero_prefix_fallback: bool,
) -> list[dict]:
    grouped_rows, _ = load_transition_rows(transition_csv)
    samples = []

    for (task_id, video_id), rows in grouped_rows.items():
        task = taxonomy["tasks"][str(task_id)]
        raw_steps = trajectory_from_transition_rows(rows)
        n_raw = len(raw_steps)
        task_steps = [step["action"] for step in task["canonical_steps"]]
        base_interpretation = make_interpretation(task_steps)

        valid_prefixes = []
        max_t = min(max_prefix, n_raw - 1) if n_raw > 0 else 0
        for prefix_len in range(1, max_t + 1):
            frontier = prefix_frontier(raw_steps[:prefix_len])
            if canonical_remaining_steps(task, frontier):
                valid_prefixes.append(prefix_len)

        for _ in range(samples_per_video):
            use_zero_prefix = zero_prefix_prob > 0 and rng.random() < zero_prefix_prob
            if use_zero_prefix:
                prefix_len = 0
            elif not valid_prefixes:
                if not allow_zero_prefix_fallback:
                    continue
                prefix_len = 0
            else:
                prefix_len = rng.choice(valid_prefixes)

            prefix_steps = [(a, s) for a, s, _ in raw_steps[:prefix_len]]
            frontier = prefix_frontier(raw_steps[:prefix_len])
            remaining = canonical_remaining_steps(task, frontier)
            if not remaining:
                continue

            k = min(rng.choice(k_values), len(remaining))
            interpretation = base_interpretation
            if interpretation_dropout_prob > 0 and rng.random() < interpretation_dropout_prob:
                interpretation = ""

            input_text = format_system1_prompt(
                task["goal"],
                interpretation,
                prefix_steps,
                k,
            )
            output_text = format_system1_output(remaining[:k])
            first_row = rows[0]
            samples.append(
                {
                    "input_text": input_text,
                    "output_text": output_text,
                    "meta": {
                        "task_id": str(task_id),
                        "task_name": task["task_name"],
                        "video_id": video_id,
                        "prefix_len": prefix_len,
                        "k": k,
                        "start_seg_pos": prefix_len,
                        "trajectory_len": n_raw,
                        "roi_start": float(first_row["roi_start"]),
                        "roi_end": float(first_row["roi_end"]),
                        "recipe_type": int(task_id),
                        "canonical_frontier": frontier,
                    },
                }
            )

    return samples


def validate_samples(samples: list[dict], expect_zero_prefix: bool | None = None) -> dict:
    bad_json = 0
    interp_present = 0
    interp_missing = 0
    zero_prefix = 0
    for sample in samples:
        try:
            payload = json.loads(sample["output_text"])
            assert "next_steps" in payload
        except Exception:
            bad_json += 1
        if "Interpretation:" in sample["input_text"]:
            interp_present += 1
        else:
            interp_missing += 1
        if sample["meta"]["prefix_len"] == 0:
            zero_prefix += 1

    if expect_zero_prefix is True and zero_prefix != len(samples):
        raise ValueError("Expected all samples to be zero-prefix")
    if expect_zero_prefix is False and zero_prefix != 0:
        raise ValueError("Expected no zero-prefix samples")

    return {
        "bad_json": bad_json,
        "interp_present": interp_present,
        "interp_missing": interp_missing,
        "zero_prefix": zero_prefix,
    }


def write_jsonl(path: str, samples: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build COIN System-1 datasets")
    parser.add_argument("--coin_json", default="COIN_dataset/COIN.json")
    parser.add_argument("--taxonomy_xlsx", default="COIN_dataset/taxonomy.xlsx")
    parser.add_argument("--taxonomy_cache", default="data/coin/coin_taxonomy.json")
    parser.add_argument("--data_dir", default="data/coin")
    parser.add_argument("--samples_per_video", type=int, default=6)
    parser.add_argument("--max_prefix", type=int, default=8)
    parser.add_argument("--k_values", default="1,2,3")
    parser.add_argument("--zero_prefix_prob", type=float, default=0.30)
    parser.add_argument("--interpretation_dropout_prob", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    taxonomy = ensure_taxonomy_cache(
        args.coin_json, args.taxonomy_xlsx, args.taxonomy_cache
    )
    rng = random.Random(args.seed)
    k_values = [int(value) for value in args.k_values.split(",") if value.strip()]

    outputs = {}
    for split in ("train", "val", "test"):
        transition_csv = os.path.join(args.data_dir, f"coin_state_change_transitions_{split}.csv")
        if not os.path.exists(transition_csv):
            raise FileNotFoundError(f"Missing transition CSV: {transition_csv}")
        split_rng = random.Random(rng.randint(0, 10**9))
        zero_prefix_prob = args.zero_prefix_prob if split == "train" else 0.0
        interp_dropout = args.interpretation_dropout_prob if split == "train" else 0.0
        samples = build_samples(
            transition_csv=transition_csv,
            taxonomy=taxonomy,
            samples_per_video=args.samples_per_video,
            max_prefix=args.max_prefix,
            k_values=k_values,
            rng=split_rng,
            zero_prefix_prob=zero_prefix_prob,
            interpretation_dropout_prob=interp_dropout,
            allow_zero_prefix_fallback=(split == "train"),
        )
        stats = validate_samples(
            samples,
            expect_zero_prefix=False if split in ("val", "test") else None,
        )
        out_path = os.path.join(args.data_dir, f"coin_system1_{split}.jsonl")
        write_jsonl(out_path, samples)
        outputs[split] = (out_path, len(samples), stats)

    for split, (path, count, stats) in outputs.items():
        print(f"{split}: {path}")
        print(
            f"  samples={count} bad_json={stats['bad_json']} "
            f"zero_prefix={stats['zero_prefix']} "
            f"interp_present={stats['interp_present']} interp_missing={stats['interp_missing']}"
        )


if __name__ == "__main__":
    main()
