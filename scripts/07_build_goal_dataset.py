#!/usr/bin/env python3
"""
Build a goal-model training dataset from state_change_transitions.csv.

The goal model learns to infer the high-level task goal from partial
observations.  Two complementary tasks per sample:

  Task A – Goal Prediction
    Given:   a partial trajectory of ⟨action, state_change⟩ pairs
    Predict: the task goal string

  Task B – Goal + Plan Prediction
    Given:   a partial trajectory
    Predict: goal + remaining canonical plan (the steps still to be done)

Each JSONL row:
  {
    "task":        "goal_prediction" | "goal_and_plan",
    "input_text":  <structured prompt>,
    "output_text": <gold answer>,
    "meta":        {task_id, task_name, video_id, prefix_len, trajectory_len}
  }

NO leakage of: timestamps, canonical_order, seg_pos, is_canonical_next,
is_time_ordered, plan_progress.

Usage:
    python scripts/07_build_goal_dataset.py \
        --input  data/crosstask/state_change_transitions.csv \
        --output data/crosstask/goal_train.jsonl \
        --samples_per_video 4 \
        --seed 42
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# 1.  Load data
# ---------------------------------------------------------------------------

def load_trajectories(csv_path: str) -> Tuple[Dict, Dict, Dict]:
    """
    Reconstruct per-video step sequences from the transition CSV.

    Returns:
        trajectories: {(task_id, video_id): [(action, state_change, step_idx), ...]}
        task_info:    {task_id: {"name": str, "goal": str}}
        task_steps:   {task_id: [(step_idx, step_text), ...]}   (canonical order)
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

    task_steps = _load_canonical_steps()

    trajectories: Dict[Tuple[str, str], List[Tuple[str, str, int]]] = {}

    for key, rows in video_rows.items():
        rows.sort(key=lambda r: int(r["seg_pos"]))

        steps: List[Tuple[str, str, int]] = []
        for r in rows:
            steps.append((r["action"], r["state_change"], int(r["step_idx"])))

        # Append final step from last transition row
        last = rows[-1]
        steps.append((
            last["next_action"],
            last["next_state_change"],
            int(last["next_step_idx"]),
        ))

        trajectories[key] = steps

    return trajectories, task_info, task_steps


def _load_canonical_steps() -> Dict[str, List[Tuple[int, str]]]:
    """Load canonical step lists from tasks_primary.txt."""
    path = os.path.join("crosstask_release", "tasks_primary.txt")
    if not os.path.exists(path):
        return {}
    lines = open(path).read().strip().split("\n")
    task_steps: Dict[str, List[Tuple[int, str]]] = {}
    i = 0
    while i < len(lines):
        tid = lines[i].strip()
        step_texts = [s.strip() for s in lines[i + 4].split(",")]
        task_steps[tid] = [(idx + 1, text) for idx, text in enumerate(step_texts)]
        i += 6
    return task_steps


# ---------------------------------------------------------------------------
# 2.  State-change template (reuse from script 04)
# ---------------------------------------------------------------------------

def _build_state_change_fn():
    """
    Minimal copy of the state-change template engine from script 04.
    Only needed to convert canonical step texts → state_change strings
    for the remaining-plan portion of Task B output.
    """
    _EXACT_OVERRIDES = {
        "add egg to flour mixture": "Egg is now added to the flour mixture.",
        "add flour to bowl": "Flour is now added to the bowl.",
        "add oil to pan": "Oil is now added to the pan.",
        "add onion to pan": "Onion is now added to the pan.",
        "pour mixture into pan": "Mixture is now poured into the pan.",
        "put dough in pan": "Dough is now placed in the pan.",
        "put bread in toaster": "Bread is now placed in the toaster.",
        "add dressing": "Dressing is now added.",
        "add honey": "Honey is now added.",
        "add salt": "Salt is now added.",
        "add sugar": "Sugar is now added.",
        "add water": "Water is now added.",
        "add lemon": "Lemon is now added.",
        "add cream cheese": "Cream cheese is now added.",
        "add milk": "Milk is now added.",
        "add ground coffee to filter": "Ground coffee is now added to the filter.",
        "add peanut butter to bread": "Peanut butter is now added to the bread.",
        "add jelly to bread": "Jelly is now added to the bread.",
        "pour water over coffee": "Water is now poured over the coffee.",
        "pour coffee into cup": "Coffee is now poured into the cup.",
        "stir to dissolve": "Contents are now stirred until dissolved.",
        "squeeze lemon into glass": "Lemon juice is now squeezed into the glass.",
        "pour into glass": "Liquid is now poured into the glass.",
        "spin in salad spinner": "Greens are now spun in the salad spinner.",
        "cut into pieces": "Item is now cut into pieces.",
        "pull out tire": "Tire is now pulled out.",
        "take out spare tire": "Spare tire is now taken out.",
        "pour into mold": "Mixture is now poured into the mold.",
        "add oil to salad": "Oil is now added to the salad.",
        "add seasoning": "Seasoning is now added.",
        "add vinegar": "Vinegar is now added.",
        "add fruit to blender": "Fruit is now added to the blender.",
        "add yogurt to blender": "Yogurt is now added to the blender.",
        "add ice to blender": "Ice is now added to the blender.",
        "add protein powder to blender": "Protein powder is now added to the blender.",
        "pour smoothie into glass": "Smoothie is now poured into the glass.",
        "add sweetener": "Sweetener is now added.",
        "add spices": "Spices are now added.",
        "add strawberries": "Strawberries are now added.",
    }
    import re
    _VERB_TEMPLATES = [
        (re.compile(r"^place (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now placed."),
        (re.compile(r"^arrange (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now arranged."),
        (re.compile(r"^cut (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now cut."),
        (re.compile(r"^slice (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now sliced."),
        (re.compile(r"^chop (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now chopped."),
        (re.compile(r"^peel (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now peeled."),
        (re.compile(r"^dice (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now diced."),
        (re.compile(r"^mince (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now minced."),
        (re.compile(r"^grate (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now grated."),
        (re.compile(r"^spread (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now spread."),
        (re.compile(r"^mix (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now mixed."),
        (re.compile(r"^stir (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now stirred."),
        (re.compile(r"^whisk (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now whisked."),
        (re.compile(r"^beat (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now beaten."),
        (re.compile(r"^blend (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now blended."),
        (re.compile(r"^fold (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now folded in."),
        (re.compile(r"^cook (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now cooked."),
        (re.compile(r"^fry (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now fried."),
        (re.compile(r"^bake (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now baked."),
        (re.compile(r"^boil (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now boiled."),
        (re.compile(r"^toast (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now toasted."),
        (re.compile(r"^serve (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now served."),
        (re.compile(r"^remove (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now removed."),
        (re.compile(r"^rinse (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now rinsed."),
        (re.compile(r"^wash (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now washed."),
        (re.compile(r"^dry (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now dried."),
        (re.compile(r"^drain (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now drained."),
        (re.compile(r"^season (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now seasoned."),
        (re.compile(r"^wrap (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now wrapped."),
        (re.compile(r"^apply (.+)$", re.I), lambda m: f"{m.group(1).capitalize()} is now applied."),
    ]

    def step_to_state_change(step_text: str) -> str:
        lower = step_text.strip().lower()
        if lower in _EXACT_OVERRIDES:
            return _EXACT_OVERRIDES[lower]
        for pattern, template in _VERB_TEMPLATES:
            m = pattern.match(step_text.strip())
            if m:
                return template(m)
        return f"{step_text.strip().capitalize()} is now done."

    return step_to_state_change


# ---------------------------------------------------------------------------
# 3.  Sample generation
# ---------------------------------------------------------------------------

def build_goal_samples(
    trajectories: Dict[Tuple[str, str], List[Tuple[str, str, int]]],
    task_info: Dict[str, dict],
    task_steps: Dict[str, List[Tuple[int, str]]],
    samples_per_video: int,
    rng: random.Random,
) -> List[dict]:
    """
    Generate goal-model training samples.

    For each video we sample multiple prefix lengths and produce two task types:
      - goal_prediction:   observe prefix → predict goal
      - goal_and_plan:     observe prefix → predict goal + remaining plan
    """
    step_to_sc = _build_state_change_fn()
    samples = []

    for (task_id, video_id), steps in trajectories.items():
        n = len(steps)
        if n < 2:
            continue

        info = task_info[task_id]
        goal = info["goal"]
        canonical = task_steps.get(task_id, [])
        canonical_map = {idx: text for idx, text in canonical}

        for _ in range(samples_per_video):
            # Sample a prefix length: at least 1 step, at most n-1
            # (leave at least 1 unseen step so the task isn't trivial)
            max_t = n - 1
            t = rng.randint(1, max(1, max_t))

            prefix = steps[:t]
            seen_idxs: Set[int] = {s[2] for s in prefix}

            # --- Task A: Goal prediction ---
            input_a_lines = _format_prefix("Identify the overall goal of this task.", prefix)
            output_a = goal

            samples.append({
                "task": "goal_prediction",
                "input_text": "\n".join(input_a_lines),
                "output_text": output_a,
                "meta": {
                    "task_id": task_id,
                    "task_name": info["name"],
                    "video_id": video_id,
                    "prefix_len": t,
                    "trajectory_len": n,
                },
            })

            # --- Task B: Goal + remaining plan ---
            # Remaining plan = canonical steps that haven't been seen
            remaining = [
                (idx, text) for idx, text in canonical
                if idx not in seen_idxs
            ]

            if remaining:
                remaining_strs = [
                    f"{text} | {step_to_sc(text)}" for _, text in remaining
                ]
                output_b = json.dumps({
                    "goal": goal,
                    "remaining_plan": remaining_strs,
                }, ensure_ascii=False)
            else:
                # All canonical steps observed
                output_b = json.dumps({
                    "goal": goal,
                    "remaining_plan": [],
                }, ensure_ascii=False)

            input_b_lines = _format_prefix(
                "Identify the overall goal and list the remaining steps "
                "to complete it.",
                prefix,
            )
            samples.append({
                "task": "goal_and_plan",
                "input_text": "\n".join(input_b_lines),
                "output_text": output_b,
                "meta": {
                    "task_id": task_id,
                    "task_name": info["name"],
                    "video_id": video_id,
                    "prefix_len": t,
                    "trajectory_len": n,
                    "remaining_plan_len": len(remaining),
                },
            })

    return samples


def _format_prefix(
    instruction: str,
    prefix: List[Tuple[str, str, int]],
) -> List[str]:
    """Build the numbered-list input prompt."""
    lines = ["Observed steps:"]
    for i, (action, state_change, _) in enumerate(prefix, 1):
        lines.append(f"  {i}) {action} | {state_change}")
    lines.append("")
    lines.append(instruction)
    return lines


# ---------------------------------------------------------------------------
# 4.  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build goal-model training dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/crosstask/state_change_transitions.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/crosstask/goal_train.jsonl",
    )
    parser.add_argument(
        "--samples_per_video",
        type=int,
        default=4,
        help="Number of prefixes to sample per video (each yields 2 samples: "
             "goal_prediction + goal_and_plan)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"[1/4] Loading trajectories from: {args.input}")
    trajectories, task_info, task_steps = load_trajectories(args.input)
    print(f"       {len(trajectories)} videos, {len(task_info)} tasks")

    lengths = [len(s) for s in trajectories.values()]
    print(f"       Trajectory lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.1f}")

    print(f"[2/4] Building goal-model samples "
          f"(samples_per_video={args.samples_per_video} × 2 tasks)")
    samples = build_goal_samples(
        trajectories, task_info, task_steps,
        args.samples_per_video, rng,
    )
    print(f"       Generated {len(samples)} training samples")

    # --- Validation ---
    print(f"[3/4] Validating...")

    from collections import Counter
    task_dist = Counter(s["task"] for s in samples)
    print(f"       Task type distribution: {dict(task_dist)}")

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
    print(f"       ✓ no leakage in input_text: {len(samples) - leakage}/{len(samples)}")

    # Check goal_and_plan outputs are valid JSON
    gp_samples = [s for s in samples if s["task"] == "goal_and_plan"]
    bad_json = 0
    for s in gp_samples:
        try:
            parsed = json.loads(s["output_text"])
            assert "goal" in parsed
            assert "remaining_plan" in parsed
        except Exception:
            bad_json += 1
    print(f"       ✓ valid JSON (goal_and_plan): "
          f"{len(gp_samples) - bad_json}/{len(gp_samples)}")

    # Check goal_prediction outputs match known goals
    gp_pred = [s for s in samples if s["task"] == "goal_prediction"]
    known_goals = {v["goal"] for v in task_info.values()}
    bad_goal = sum(1 for s in gp_pred if s["output_text"] not in known_goals)
    print(f"       ✓ valid goals (goal_prediction): "
          f"{len(gp_pred) - bad_goal}/{len(gp_pred)}")

    # --- Write ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"[4/4] Writing to: {args.output}")
    with open(args.output, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # --- Summary ---
    print(f"\n--- Summary ---")
    print(f"  Total samples:     {len(samples)}")
    print(f"  Tasks covered:     {len(set(s['meta']['task_id'] for s in samples))}")
    print(f"  Videos used:       {len(set(s['meta']['video_id'] for s in samples))}")
    print(f"  Task A (goal):     {task_dist.get('goal_prediction', 0)}")
    print(f"  Task B (goal+plan):{task_dist.get('goal_and_plan', 0)}")

    prefix_lens = [s["meta"]["prefix_len"] for s in samples]
    print(f"  prefix_len:        min={min(prefix_lens)}, max={max(prefix_lens)}, "
          f"mean={sum(prefix_lens)/len(prefix_lens):.1f}")

    # Examples
    for task_type in ("goal_prediction", "goal_and_plan"):
        ex = next(s for s in samples if s["task"] == task_type)
        print(f"\n{'='*70}")
        print(f"EXAMPLE — {task_type}")
        print(f"{'='*70}")
        print(f"[INPUT TEXT]")
        print(ex["input_text"])
        print(f"\n[OUTPUT TEXT]")
        if task_type == "goal_and_plan":
            print(json.dumps(json.loads(ex["output_text"]), indent=2))
        else:
            print(ex["output_text"])
        print(f"\n[META] {ex['meta']}")


if __name__ == "__main__":
    main()
