#!/usr/bin/env python3
"""
Build a VLWM System-1 (rollout generator) training dataset from
state_change_transitions.csv.

Follows the VLWM formulation (Chen et al., 2025, Eq. 1):
  [config, context] → [goal, interpretation, ⟨A0,ΔS0⟩, ..., ⟨AN,ΔSN⟩]

For our grounded variant the training task is:
  Given   : goal + interpretation + prefix trajectory ⟨A0,ΔS0⟩…⟨At-1,ΔSt-1⟩
  Predict : next k steps  ⟨At,ΔSt⟩…⟨At+k-1,ΔSt+k-1⟩  as JSON

Each JSONL row:
  {
    "input_text":  <structured prompt>,
    "output_text": <gold JSON>,
    "meta": {task_id, task_name, video_id, prefix_len, k, start_seg_pos,
             trajectory_len}
  }

NO leakage of: remaining_plan, plan_progress, canonical_order, timestamps,
seg_pos, is_canonical_next, is_time_ordered.

Usage:
    python scripts/06_build_system1_dataset.py \
        --input  data/crosstask/state_change_transitions.csv \
        --output data/crosstask/system1_train.jsonl \
        --samples_per_video 6 \
        --max_prefix 8 \
        --seed 42
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 1.  Trajectory reconstruction
# ---------------------------------------------------------------------------

def load_trajectories(csv_path: str) -> Tuple[Dict, Dict, Dict]:
    """
    Reconstruct per-video step sequences from the transition CSV.

    Returns:
        trajectories: {(task_id, video_id): [(action, state_change), ...]}
        task_info:    {task_id: {"name": str, "goal": str}}
        task_steps:   {task_id: [canonical_step_text, ...]}  (for interpretation)
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

    # Also parse canonical steps for interpretation generation
    task_steps = _load_canonical_steps()

    trajectories: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

    for key, rows in video_rows.items():
        rows.sort(key=lambda r: int(r["seg_pos"]))

        steps: List[Tuple[str, str]] = []
        for r in rows:
            steps.append((r["action"], r["state_change"]))

        # Append the final "next" step from the last transition row
        last = rows[-1]
        steps.append((last["next_action"], last["next_state_change"]))

        trajectories[key] = steps

    return trajectories, task_info, task_steps


def _load_canonical_steps() -> Dict[str, List[str]]:
    """Load canonical step lists from tasks_primary.txt for interpretation."""
    path = os.path.join("crosstask_release", "tasks_primary.txt")
    if not os.path.exists(path):
        return {}
    lines = open(path).read().strip().split("\n")
    task_steps = {}
    i = 0
    while i < len(lines):
        tid = lines[i].strip()
        steps = [s.strip() for s in lines[i + 4].split(",")]
        task_steps[tid] = steps
        i += 6
    return task_steps


# ---------------------------------------------------------------------------
# 2.  Interpretation generation (rule-based, no LLM)
# ---------------------------------------------------------------------------

def make_interpretation(
    task_name: str, task_id: str, task_steps: Dict[str, List[str]]
) -> str:
    """
    Generate a 1-sentence interpretation describing initial and final state.

    Mirrors VLWM's "interpretation" field which describes
    "the initial and expected final states" (Chen et al., 2025, §2.1).
    """
    if task_id in task_steps:
        steps = task_steps[task_id]
        first_step = steps[0]
        last_step = steps[-1]
        return (
            f"The task begins with '{first_step}' and is considered complete "
            f"once '{last_step}' is done."
        )
    # Fallback
    return f"The task is to {task_name.lower()} from start to finish."


# ---------------------------------------------------------------------------
# 3.  Input/output text formatting
# ---------------------------------------------------------------------------

def format_input_text(
    goal: str,
    interpretation: str,
    prefix_steps: List[Tuple[str, str]],
    k: int,
) -> str:
    """
    Build the structured prompt for the System-1 model.

    Format follows VLWM Eq.1:
      [config, context] → [goal, interpretation, ⟨A,ΔS⟩ pairs]

    We provide goal + interpretation + observed prefix, and ask the model
    to generate the next k steps.
    """
    lines = []
    lines.append(f"Goal: {goal}")
    lines.append(f"Interpretation: {interpretation}")
    lines.append("")
    lines.append("Progress so far:")

    for i, (action, state_change) in enumerate(prefix_steps, 1):
        lines.append(f"  {i}) {action} | {state_change}")

    lines.append("")
    lines.append(
        f"Predict the next {k} step(s). "
        f"Output JSON only: {{\"next_steps\": [{{\"action\": \"...\", \"state_change\": \"...\"}}, ...]}}"
    )

    return "\n".join(lines)


def format_output_text(target_steps: List[Tuple[str, str]]) -> str:
    """
    Build the gold JSON output.

    Deterministic formatting: sorted keys, no trailing whitespace.
    """
    obj = {
        "next_steps": [
            {"action": action, "state_change": state_change}
            for action, state_change in target_steps
        ]
    }
    return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 4.  Sample generation
# ---------------------------------------------------------------------------

def build_system1_samples(
    trajectories: Dict[Tuple[str, str], List[Tuple[str, str]]],
    task_info: Dict[str, dict],
    task_steps: Dict[str, List[str]],
    samples_per_video: int,
    max_prefix: int,
    k_values: List[int],
    rng: random.Random,
) -> List[dict]:
    """Generate System-1 training samples."""
    samples = []

    for (task_id, video_id), steps in trajectories.items():
        n = len(steps)
        if n < 3:
            # Need at least 1 prefix step + 1 target step + margin
            continue

        info = task_info[task_id]
        goal = info["goal"]
        interpretation = make_interpretation(info["name"], task_id, task_steps)

        for _ in range(samples_per_video):
            k = rng.choice(k_values)

            # prefix_len t: at least 1, at most min(n - k, max_prefix)
            max_t = min(n - k, max_prefix)
            if max_t < 1:
                continue
            t = rng.randint(1, max_t)

            prefix_steps = steps[:t]
            target_steps = steps[t : t + k]

            input_text = format_input_text(goal, interpretation, prefix_steps, k)
            output_text = format_output_text(target_steps)

            sample = {
                "input_text": input_text,
                "output_text": output_text,
                "meta": {
                    "task_id": task_id,
                    "task_name": info["name"],
                    "video_id": video_id,
                    "prefix_len": t,
                    "k": k,
                    "start_seg_pos": t,  # target starts at this seg position
                    "trajectory_len": n,
                },
            }
            samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# 5.  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build VLWM System-1 rollout training dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/crosstask/state_change_transitions.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/crosstask/system1_train.jsonl",
    )
    parser.add_argument(
        "--samples_per_video",
        type=int,
        default=6,
        help="Training examples to sample per video",
    )
    parser.add_argument(
        "--max_prefix",
        type=int,
        default=8,
        help="Maximum prefix length (cap to avoid very long inputs)",
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="1,2,3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    rng = random.Random(args.seed)

    print(f"[1/5] Loading trajectories from: {args.input}")
    trajectories, task_info, task_steps = load_trajectories(args.input)
    print(f"       {len(trajectories)} videos, {len(task_info)} tasks")

    lengths = [len(s) for s in trajectories.values()]
    print(f"       Trajectory lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.1f}")

    print(f"[2/5] Building System-1 samples "
          f"(samples_per_video={args.samples_per_video}, "
          f"max_prefix={args.max_prefix}, k∈{{{args.k_values}}})")
    samples = build_system1_samples(
        trajectories, task_info, task_steps,
        args.samples_per_video, args.max_prefix, k_values, rng,
    )
    print(f"       Generated {len(samples)} training samples")

    # --- Validation ---
    print(f"[3/5] Validating...")

    # Check output is valid JSON
    bad_json = 0
    for s in samples:
        try:
            parsed = json.loads(s["output_text"])
            assert "next_steps" in parsed
            assert len(parsed["next_steps"]) == s["meta"]["k"]
        except Exception:
            bad_json += 1

    # Check no leakage
    leak_fields = [
        "remaining_plan", "plan_progress", "canonical_order",
        "action_start", "action_end", "next_action_start", "next_action_end",
        "is_canonical_next", "is_time_ordered", "seg_pos", "next_seg_pos",
    ]
    leakage = 0
    for s in samples:
        if any(field in s["input_text"] for field in leak_fields):
            leakage += 1

    print(f"       ✓ valid JSON output:    {len(samples) - bad_json}/{len(samples)}")
    print(f"       ✓ no leakage fields:    {len(samples) - leakage}/{len(samples)}")

    # --- Write ---
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"[4/5] Writing to: {args.output}")
    with open(args.output, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # --- Summary ---
    print(f"[5/5] Summary")
    from collections import Counter
    k_dist = Counter(s["meta"]["k"] for s in samples)
    prefix_lens = [s["meta"]["prefix_len"] for s in samples]
    print(f"  Total samples:   {len(samples)}")
    print(f"  Tasks:           {len(set(s['meta']['task_id'] for s in samples))}")
    print(f"  Videos:          {len(set(s['meta']['video_id'] for s in samples))}")
    print(f"  k distribution:  {dict(sorted(k_dist.items()))}")
    print(f"  prefix_len:      min={min(prefix_lens)}, max={max(prefix_lens)}, "
          f"mean={sum(prefix_lens)/len(prefix_lens):.1f}")

    # Show examples
    print(f"\n{'='*70}")
    print(f"EXAMPLE 1")
    print(f"{'='*70}")
    ex = samples[0]
    print(f"[INPUT TEXT]")
    print(ex["input_text"])
    print(f"\n[OUTPUT TEXT]")
    print(json.dumps(json.loads(ex["output_text"]), indent=2))
    print(f"\n[META] {ex['meta']}")

    # Show a longer example
    long_examples = [s for s in samples if s["meta"]["prefix_len"] >= 4 and s["meta"]["k"] >= 2]
    if long_examples:
        print(f"\n{'='*70}")
        print(f"EXAMPLE 2 (longer prefix)")
        print(f"{'='*70}")
        ex2 = long_examples[0]
        print(f"[INPUT TEXT]")
        print(ex2["input_text"])
        print(f"\n[OUTPUT TEXT]")
        print(json.dumps(json.loads(ex2["output_text"]), indent=2))
        print(f"\n[META] {ex2['meta']}")


if __name__ == "__main__":
    main()
