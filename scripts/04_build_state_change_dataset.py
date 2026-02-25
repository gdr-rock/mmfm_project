#!/usr/bin/env python3
"""
Build a state-change transition dataset from CrossTask primary tasks.

Produces a CSV with consecutive action pairs, state-change descriptions,
and columns designed for three training regimes:

  1. System-1 (fast)  : predict next_state_change from (action, latent)
  2. System-2 (critic) : judge whether a transition is plausible
  3. Goal model        : given (task_name, current_step_idx, goal),
                         predict the full remaining plan

Output columns:
  task_id, task_name, video_id,
  step_idx, action, state_change,                          # current
  next_step_idx, next_action, next_state_change,           # next
  action_start, action_end, next_action_start, next_action_end,  # timestamps
  canonical_order, next_canonical_order,                   # canonical position in task recipe
  is_in_order,                                             # whether transition follows recipe
  task_goal,                                               # high-level task description
  remaining_plan,                                          # ordered steps still remaining after current
  plan_progress,                                           # fraction of canonical steps completed so far

Usage:
    python scripts/04_build_state_change_dataset.py \
        --crosstask_dir crosstask_release \
        --output data/crosstask/state_change_transitions.csv
"""

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 1.  State-change template engine
# ---------------------------------------------------------------------------

# Hand-written templates keyed by verb (or verb phrase).
# Each maps to a lambda:  object_str -> state_change_sentence
# We try the longest matching prefix first.

# --- Exact overrides for steps with prepositions that templates mangle ---
_EXACT_OVERRIDES = {
    "pour mixture into cup":       "The mixture is now poured into the cup.",
    "pour mixture into pan":       "The mixture is now poured into the pan.",
    "pour lemon juice":            "Lemon juice is now poured in.",
    "pour lemonade into glass":    "Lemonade is now poured into the glass.",
    "pour sesame oil":             "Sesame oil is now poured in.",
    "pour jello powder":           "Jello powder is now poured in.",
    "put steak on grill":          "The steak is now placed on the grill.",
    "put bread in pan":            "The bread is now placed in the pan.",
    "put bananas into blender":    "The bananas are now placed into the blender.",
    "put meringue into oven":      "The meringue is now placed into the oven.",
    "put mixture into bag":        "The mixture is now placed into a piping bag.",
    "put dough into form":         "The dough is now placed into the baking form.",
    "put vegetables in water":     "The vegetables are now placed in the water.",
    "put jar in water":            "The jar is now placed in the water bath.",
    "dip bread in mixture":        "The bread is now dipped and coated in the mixture.",
    "take pancake from pan":       "The pancake is now removed from the pan.",
    "take steak from grill":       "The steak is now removed from the grill.",
    "remove bread from pan":       "The bread is now removed from the pan.",
    "move steak on grill":         "The steak is now repositioned on the grill.",
    "pack cucumbers in jar":       "The cucumbers are now packed into the jar.",
    "add strawberries to cake":    "Strawberries are now added on top of the cake.",
    "spread creme upon cake":      "Creme is now spread evenly upon the cake.",
    "add vanilla extract":         "Vanilla extract is now added to the mixture.",
    "add whipped cream":           "Whipped cream is now added on top.",
    "add chili powder":            "Chili powder is now added to the pot.",
    "add mustard seeds":           "Mustard seeds are now added to the oil.",
    "add curry leaves":            "Curry leaves are now added to the pot.",
    "check temperature":           "The internal temperature is now checked.",
    "mix ingredients":             "The ingredients are now mixed together.",
    "start loose":                 "The lug nuts are now loosened.",
    "get things out":              "The tools and spare are now taken out of the trunk.",
    "put things back":             "The tools are now put back in the trunk.",
    "brake on":                    "The parking brake is now engaged.",
    "jack up":                     "The vehicle is now jacked up off the ground.",
    "jack down":                   "The vehicle is now lowered back to the ground.",
    "stir":                        "The contents are now stirred and combined.",
    "stir mixture":                "The mixture is now stirred and combined.",
    "add spices":                  "Spices are now added to the mixture.",
    "cut strawberries":            "Strawberries are now cut into pieces.",
}


_VERB_TEMPLATES = {
    # ---- add / pour / put -------------------------------------------------
    "add":          lambda obj: f"{_cap(obj)} is now added to the mixture.",
    "pour":         lambda obj: f"{_cap(obj)} is now poured in.",
    "put":          lambda obj: f"{_cap(obj)} is now placed in position.",

    # ---- cut / chop / slice -----------------------------------------------
    "cut":          lambda obj: f"{_cap(obj)} is now cut into pieces.",
    "chop":         lambda obj: f"{_cap(obj)} is now chopped into small pieces.",
    "slice":        lambda obj: f"{_cap(obj)} is now sliced.",
    "peel":         lambda obj: f"{_cap(obj)} is now peeled.",
    "squeeze":      lambda obj: f"{_cap(obj)} is now squeezed out.",

    # ---- mix / stir / whisk -----------------------------------------------
    "stir":         lambda obj: f"{_cap(obj) if obj else 'The mixture'} is now stirred and combined.",
    "mix":          lambda obj: f"{_cap(obj) if obj else 'The ingredients'} are now mixed together.",
    "whisk":        lambda obj: f"{_cap(obj) if obj else 'The mixture'} is now whisked until smooth.",

    # ---- heat / cook / melt -----------------------------------------------
    "melt":         lambda obj: f"{_cap(obj)} is now melted.",
    "steam":        lambda obj: f"{_cap(obj)} is now steamed.",
    "season":       lambda obj: f"{_cap(obj)} is now seasoned with spices.",
    "dip":          lambda obj: f"{_cap(obj)} is now dipped and coated.",
    "spread":       lambda obj: f"{_cap(obj)} is now spread evenly.",

    # ---- flip / move / remove ---------------------------------------------
    "flip":         lambda obj: f"{_cap(obj)} is now flipped over.",
    "move":         lambda obj: f"{_cap(obj)} is now repositioned.",
    "remove":       lambda obj: f"{_cap(obj)} is now removed.",
    "take":         lambda obj: f"{_cap(obj)} is now taken out.",
    "withdraw":     lambda obj: f"{_cap(obj)} is now withdrawn.",
    "pull out":     lambda obj: f"{_cap(obj)} is now pulled out.",
    "wipe off":     lambda obj: f"{_cap(obj)} is now wiped clean.",

    # ---- press / screw / unscrew ------------------------------------------
    "press":        lambda obj: f"{_cap(obj)} is now pressed down.",
    "screw":        lambda obj: f"{_cap(obj)} is now screwed on tight.",
    "unscrew":      lambda obj: f"{_cap(obj)} is now unscrewed and loosened.",
    "tight":        lambda obj: f"{_cap(obj)} is now tightened securely.",

    # ---- open / close / seal / attach -------------------------------------
    "open":         lambda obj: f"{_cap(obj)} is now opened.",
    "close":        lambda obj: f"{_cap(obj)} is now closed.",
    "seal":         lambda obj: f"{_cap(obj)} is now sealed shut.",
    "attach":       lambda obj: f"{_cap(obj)} is now attached and secured.",
    "insert":       lambda obj: f"{_cap(obj)} is now inserted back in place.",
    "pack":         lambda obj: f"{_cap(obj)} is now packed tightly.",

    # ---- raise / lower ----------------------------------------------------
    "raise":        lambda obj: f"{_cap(obj)} is now raised up.",
    "lower":        lambda obj: f"{_cap(obj)} is now lowered down.",

    # ---- assemble / sand / paint ------------------------------------------
    "assemble":     lambda obj: f"{_cap(obj)} is now assembled.",
    "sand":         lambda obj: f"{_cap(obj)} is now sanded smooth.",
    "paint":        lambda obj: f"{_cap(obj)} is now painted.",

    # ---- check / taste / top ----------------------------------------------
    "check":        lambda obj: f"{_cap(obj)} is now checked / verified.",
    "taste":        lambda obj: f"{_cap(obj)} is now tasted for flavour.",
    "top":          lambda obj: f"{_cap(obj)} is now topped with garnish.",
}


def _cap(s: str) -> str:
    """Capitalise first letter only."""
    return s[0].upper() + s[1:] if s else s


def step_to_state_change(step_text: str) -> str:
    """Convert a procedural step string into a state-change sentence."""
    step = step_text.strip().lower()

    # 1. Try hand-crafted exact override (highest quality)
    if step in _EXACT_OVERRIDES:
        return _EXACT_OVERRIDES[step]

    # 2. Try exact verb-template match (handles multi-word verbs)
    if step in _VERB_TEMPLATES:
        return _VERB_TEMPLATES[step]("")

    # 3. Try longest-prefix verb match
    for verb in sorted(_VERB_TEMPLATES.keys(), key=len, reverse=True):
        if step.startswith(verb + " "):
            obj = step[len(verb):].strip()
            return _VERB_TEMPLATES[verb](obj)
        elif step.startswith(verb):
            obj = step[len(verb):].strip()
            return _VERB_TEMPLATES[verb](obj)

    # Fallback: generic passive voice
    return f"The state changes: {step} is performed."


# ---------------------------------------------------------------------------
# 2.  Parse CrossTask release files
# ---------------------------------------------------------------------------

def parse_primary_tasks(path: str) -> Dict[str, dict]:
    """
    Parse tasks_primary.txt.

    Returns dict:  task_id -> {
        'name': str,
        'url': str,
        'n_steps': int,
        'steps': [str, ...],          # 1-indexed in annotations
        'goal': str,                   # high-level task description
    }
    """
    lines = Path(path).read_text().strip().split("\n")
    tasks = {}
    i = 0
    while i < len(lines):
        task_id = lines[i].strip()
        name = lines[i + 1].strip()
        url = lines[i + 2].strip()
        n_steps = int(lines[i + 3].strip())
        steps = [s.strip() for s in lines[i + 4].split(",")]
        assert len(steps) == n_steps, f"Task {task_id}: expected {n_steps} steps, got {len(steps)}"
        tasks[task_id] = {
            "name": name,
            "url": url,
            "n_steps": n_steps,
            "steps": steps,
            "goal": f"Complete the task: {name}.",
        }
        i += 6  # 5 lines + 1 blank
    return tasks


def get_primary_video_ids(videos_csv: str, primary_task_ids: set) -> Dict[str, List[str]]:
    """
    Parse videos.csv, return {task_id: [video_id, ...]} for primary tasks only.
    """
    mapping: Dict[str, List[str]] = {}
    with open(videos_csv) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            tid, vid = parts[0], parts[1]
            if tid in primary_task_ids:
                mapping.setdefault(tid, []).append(vid)
    return mapping


def load_annotations(ann_path: str) -> List[Tuple[int, float, float]]:
    """
    Load a single annotation CSV.
    Returns list of (step_index_1based, start_sec, end_sec) sorted by start_sec.
    """
    rows = []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            step_idx = int(parts[0])
            start = float(parts[1])
            end = float(parts[2])
            rows.append((step_idx, start, end))
    rows.sort(key=lambda r: r[1])  # sort by start time
    return rows


# ---------------------------------------------------------------------------
# 3.  Build the dataset
# ---------------------------------------------------------------------------

def build_dataset(crosstask_dir: str) -> List[dict]:
    """Build the full transition dataset."""
    tasks_file = os.path.join(crosstask_dir, "tasks_primary.txt")
    videos_file = os.path.join(crosstask_dir, "videos.csv")
    ann_dir = os.path.join(crosstask_dir, "annotations")

    tasks = parse_primary_tasks(tasks_file)
    primary_ids = set(tasks.keys())
    task_videos = get_primary_video_ids(videos_file, primary_ids)

    rows = []

    for tid, vids in task_videos.items():
        task = tasks[tid]
        steps = task["steps"]          # 0-indexed list
        n_canonical = task["n_steps"]

        for vid in vids:
            ann_path = os.path.join(ann_dir, f"{tid}_{vid}.csv")
            if not os.path.isfile(ann_path):
                continue

            annotations = load_annotations(ann_path)
            if len(annotations) < 2:
                continue

            # Track which canonical steps have been seen so far
            seen_canonical = set()
            # Track the highest canonical position reached (high-water mark)
            max_canonical_reached = 0

            for i in range(len(annotations) - 1):
                cur_step_idx, cur_start, cur_end = annotations[i]
                nxt_step_idx, nxt_start, nxt_end = annotations[i + 1]

                # Step indices in annotations are 1-based
                cur_text = steps[cur_step_idx - 1] if cur_step_idx <= len(steps) else f"step_{cur_step_idx}"
                nxt_text = steps[nxt_step_idx - 1] if nxt_step_idx <= len(steps) else f"step_{nxt_step_idx}"

                # Track progress
                seen_canonical.add(cur_step_idx)
                max_canonical_reached = max(max_canonical_reached, cur_step_idx)

                # Remaining plan: canonical steps that come AFTER the
                # furthest position reached so far, excluding those already
                # done.  Uses a high-water mark so going back to an earlier
                # step (out-of-order) does NOT re-expand the plan.
                remaining = [
                    steps[s - 1]
                    for s in range(max_canonical_reached + 1, n_canonical + 1)
                    if s not in seen_canonical
                ]

                # Progress = fraction of unique canonical steps seen so far
                progress = round(len(seen_canonical) / n_canonical, 3)

                # Whether the transition respects canonical ordering
                is_in_order = nxt_step_idx >= cur_step_idx

                row = {
                    # --- identifiers ---
                    "task_id":              tid,
                    "task_name":            task["name"],
                    "video_id":             vid,

                    # --- current action ---
                    "step_idx":             cur_step_idx,
                    "action":               cur_text,
                    "state_change":         step_to_state_change(cur_text),

                    # --- next action ---
                    "next_step_idx":        nxt_step_idx,
                    "next_action":          nxt_text,
                    "next_state_change":    step_to_state_change(nxt_text),

                    # --- timestamps (for V-JEPA latent extraction) ---
                    "action_start":         cur_start,
                    "action_end":           cur_end,
                    "next_action_start":    nxt_start,
                    "next_action_end":      nxt_end,

                    # --- canonical position ---
                    "canonical_order":      cur_step_idx,
                    "next_canonical_order": nxt_step_idx,
                    "is_in_order":          is_in_order,

                    # --- goal model ---
                    "task_goal":            task["goal"],
                    "remaining_plan":       " -> ".join(remaining) if remaining else "(task complete)",
                    "plan_progress":        progress,
                }
                rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# 4.  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build state-change transition dataset from CrossTask primary tasks."
    )
    parser.add_argument(
        "--crosstask_dir",
        type=str,
        default="crosstask_release",
        help="Path to crosstask_release folder",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/crosstask/state_change_transitions.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    print(f"[1/3] Parsing CrossTask release from: {args.crosstask_dir}")
    rows = build_dataset(args.crosstask_dir)
    print(f"[2/3] Built {len(rows)} transition rows")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Write CSV
    fieldnames = [
        "task_id", "task_name", "video_id",
        "step_idx", "action", "state_change",
        "next_step_idx", "next_action", "next_state_change",
        "action_start", "action_end", "next_action_start", "next_action_end",
        "canonical_order", "next_canonical_order", "is_in_order",
        "task_goal", "remaining_plan", "plan_progress",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[3/3] Saved to: {args.output}")

    # --- Summary statistics ---
    tasks_seen = set(r["task_id"] for r in rows)
    videos_seen = set((r["task_id"], r["video_id"]) for r in rows)
    in_order_count = sum(1 for r in rows if r["is_in_order"])
    print(f"\n--- Summary ---")
    print(f"  Tasks:          {len(tasks_seen)}")
    print(f"  Videos:         {len(videos_seen)}")
    print(f"  Transitions:    {len(rows)}")
    print(f"  In-order:       {in_order_count} ({100*in_order_count/len(rows):.1f}%)")
    print(f"  Out-of-order:   {len(rows)-in_order_count} ({100*(len(rows)-in_order_count)/len(rows):.1f}%)")

    # Print a few examples
    print(f"\n--- Sample rows ---")
    for r in rows[:5]:
        print(f"  [{r['task_name']}] {r['action']} -> {r['next_action']}")
        print(f"    ΔS:  {r['state_change']}")
        print(f"    ΔS': {r['next_state_change']}")
        print(f"    progress={r['plan_progress']}, in_order={r['is_in_order']}")
        print()


if __name__ == "__main__":
    main()
