#!/usr/bin/env python3
"""
Build a VLWM-style critic training dataset from state_change_transitions.csv.

Follows the self-supervised critic training from:
  Chen et al., "Planning with Reasoning using Vision Language World Model" (2025)

Three training signal types (ranking constraints):
  1.  C_good  < C_base            (valid continuation is better than stopping)
  2.  C_base  < C_bad             (stopping is better than adding distractors)
  3.  C_base  < C_shuffled        (correct order is better than shuffled order)

Each JSONL row contains:
  - goal:       task goal string
  - base:       list of step strings (prefix of length t)
  - good:       list of step strings (prefix + k valid next steps)
  - bad:        list of step strings (prefix + k distractor steps from another task)
  - shuffled:   list of step strings (base steps in random order)
  - metadata:   {task_id, video_id, prefix_len, k, distractor_task_id}

Step format:  "action | state_change"
  e.g. "add onion | Onion is now added to the mixture."

NO supervision-leaking fields (timestamps, canonical_order, remaining_plan,
plan_progress, is_canonical_next, etc.) appear in the training text.

Usage:
    python scripts/05_build_critic_dataset.py \
        --input  data/crosstask/state_change_transitions.csv \
        --output data/crosstask/critic_train.jsonl \
        --samples_per_video 5 \
        --seed 42
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


def load_trajectories(csv_path: str) -> Tuple[Dict, Dict]:
    """
    Load the state-change CSV and reconstruct per-video trajectories.

    Returns:
        trajectories: {(task_id, video_id): [step_str, ...]}
            where step_str = "action | state_change"
            ordered by seg_pos (chronological).
        task_info: {task_id: {"name": str, "goal": str}}
    """
    video_rows: Dict[Tuple[str, str], list] = defaultdict(list)
    task_info: Dict[str, dict] = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["task_id"], row["video_id"])
            video_rows[key].append(row)
            if row["task_id"] not in task_info:
                task_info[row["task_id"]] = {
                    "name": row["task_name"],
                    "goal": row["task_goal"],
                }

    trajectories: Dict[Tuple[str, str], List[str]] = {}

    for key, rows in video_rows.items():
        # Sort by seg_pos to guarantee chronological order
        rows.sort(key=lambda r: int(r["seg_pos"]))

        # Build the step sequence.
        # Each transition row gives us one step (the "current" action).
        # The very last row also gives us the "next" action (the final step).
        steps = []
        for r in rows:
            step_str = f"{r['action']} | {r['state_change']}"
            steps.append(step_str)

        # Append the final "next" step from the last transition row
        last = rows[-1]
        final_step = f"{last['next_action']} | {last['next_state_change']}"
        steps.append(final_step)

        trajectories[key] = steps

    return trajectories, task_info


def sample_distractors(
    task_id: str,
    k: int,
    all_trajectories: Dict[Tuple[str, str], List[str]],
    rng: random.Random,
) -> Tuple[List[str], str]:
    """
    Sample k distractor steps from a different task.

    Returns (distractor_steps, distractor_task_id).
    """
    # Collect all steps from other tasks
    other_task_ids = set()
    other_steps_by_task: Dict[str, List[str]] = defaultdict(list)

    for (tid, vid), steps in all_trajectories.items():
        if tid != task_id:
            other_task_ids.add(tid)
            other_steps_by_task[tid].extend(steps)

    # Pick a random different task
    distractor_tid = rng.choice(list(other_task_ids))
    pool = other_steps_by_task[distractor_tid]

    # Sample k steps (with replacement if pool is small)
    distractor_steps = rng.choices(pool, k=k)

    return distractor_steps, distractor_tid


def build_critic_samples(
    trajectories: Dict[Tuple[str, str], List[str]],
    task_info: Dict[str, dict],
    samples_per_video: int,
    k_values: List[int],
    rng: random.Random,
) -> List[dict]:
    """
    Build critic training samples.

    For each video, we sample `samples_per_video` prefixes, each with a
    random k ∈ k_values. For each prefix we produce:
      - base:     steps[:t]
      - good:     steps[:t+k]
      - bad:      steps[:t] + k distractors from another task
      - shuffled: random permutation of steps[:t]
    """
    samples = []

    for (task_id, video_id), steps in trajectories.items():
        n = len(steps)
        if n < 3:
            # Need at least 2 base steps + 1 continuation
            continue

        goal = task_info[task_id]["goal"]

        for _ in range(samples_per_video):
            k = rng.choice(k_values)

            # prefix length t: need at least 2 steps for base,
            # and at least k steps remaining for "good"
            max_t = n - k
            if max_t < 2:
                continue
            t = rng.randint(2, max_t)

            base = steps[:t]
            good = steps[:t + k]

            # Bad: base + k distractor steps from another task
            distractor_steps, distractor_tid = sample_distractors(
                task_id, k, trajectories, rng
            )
            bad = base + distractor_steps

            # Shuffled: random permutation of the base steps
            shuffled = list(base)
            rng.shuffle(shuffled)
            # Ensure shuffled is actually different from base
            # (for very short sequences, shuffling may produce the same order)
            attempts = 0
            while shuffled == base and attempts < 10:
                rng.shuffle(shuffled)
                attempts += 1

            sample = {
                "goal": goal,
                "base": base,
                "good": good,
                "bad": bad,
                "shuffled": shuffled,
                "metadata": {
                    "task_id": task_id,
                    "task_name": task_info[task_id]["name"],
                    "video_id": video_id,
                    "prefix_len": t,
                    "k": k,
                    "trajectory_len": n,
                    "distractor_task_id": distractor_tid,
                },
            }
            samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Build VLWM-style critic training dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/crosstask/state_change_transitions.csv",
        help="Input state-change transitions CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/crosstask/critic_train.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--samples_per_video",
        type=int,
        default=5,
        help="Number of critic samples to generate per video",
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="1,2,3",
        help="Comma-separated continuation lengths to sample from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    rng = random.Random(args.seed)

    print(f"[1/4] Loading trajectories from: {args.input}")
    trajectories, task_info = load_trajectories(args.input)
    print(f"       {len(trajectories)} videos, {len(task_info)} tasks")

    # Print trajectory length stats
    lengths = [len(s) for s in trajectories.values()]
    print(f"       Trajectory lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.1f}")

    print(f"[2/4] Building critic samples (samples_per_video={args.samples_per_video}, "
          f"k∈{{{args.k_values}}})")
    samples = build_critic_samples(
        trajectories, task_info, args.samples_per_video, k_values, rng
    )
    print(f"       Generated {len(samples)} critic training samples")

    print(f"[3/4] Validating samples...")
    # Validation checks
    n_good_extends_base = sum(
        1 for s in samples if s["good"][:s["metadata"]["prefix_len"]] == s["base"]
    )
    n_bad_extends_base = sum(
        1 for s in samples if s["bad"][:s["metadata"]["prefix_len"]] == s["base"]
    )
    n_shuffled_diff = sum(
        1 for s in samples if s["shuffled"] != s["base"]
    )
    n_no_leakage = sum(
        1 for s in samples
        if not any(
            field in json.dumps(s)
            for field in ["remaining_plan", "plan_progress", "canonical_order",
                          "action_start", "action_end", "is_canonical_next",
                          "is_time_ordered", "step_idx", "seg_pos"]
        )
    )

    print(f"       ✓ good extends base:   {n_good_extends_base}/{len(samples)}")
    print(f"       ✓ bad extends base:    {n_bad_extends_base}/{len(samples)}")
    print(f"       ✓ shuffled ≠ base:     {n_shuffled_diff}/{len(samples)}")
    print(f"       ✓ no leakage fields:   {n_no_leakage}/{len(samples)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"[4/4] Writing to: {args.output}")
    with open(args.output, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\n--- Summary ---")
    print(f"  Total samples:   {len(samples)}")
    print(f"  Videos used:     {len(set(s['metadata']['video_id'] for s in samples))}")
    print(f"  Tasks covered:   {len(set(s['metadata']['task_id'] for s in samples))}")

    # k distribution
    from collections import Counter
    k_dist = Counter(s["metadata"]["k"] for s in samples)
    print(f"  k distribution:  {dict(sorted(k_dist.items()))}")

    # prefix length distribution
    prefix_lens = [s["metadata"]["prefix_len"] for s in samples]
    print(f"  prefix_len:      min={min(prefix_lens)}, max={max(prefix_lens)}, "
          f"mean={sum(prefix_lens)/len(prefix_lens):.1f}")

    # Show one example
    print(f"\n--- Example sample ---")
    ex = samples[0]
    print(f"  goal:     {ex['goal']}")
    print(f"  base ({len(ex['base'])} steps):")
    for s in ex["base"]:
        print(f"    • {s}")
    print(f"  good ({len(ex['good'])} steps) — extends with {ex['metadata']['k']} valid step(s):")
    for s in ex["good"][len(ex["base"]):]:
        print(f"    + {s}")
    print(f"  bad ({len(ex['bad'])} steps) — extends with {ex['metadata']['k']} distractor(s) from task {ex['metadata']['distractor_task_id']}:")
    for s in ex["bad"][len(ex["base"]):]:
        print(f"    ✗ {s}")
    print(f"  shuffled ({len(ex['shuffled'])} steps):")
    for s in ex["shuffled"]:
        print(f"    ~ {s}")


if __name__ == "__main__":
    main()
