#!/usr/bin/env python3
"""
Build task-level CrossTask evaluation files from the existing System-1 splits.

For each unique task_id covered by system1_train/val/test, emit:
  1. goal-only prompt -> full gold plan
  2. goal+interpretation prompt -> full gold plan

Outputs:
  data/crosstask/task_level_goal_only.jsonl
  data/crosstask/task_level_goal_plus_interpretation.jsonl
  data/crosstask/task_level_prompt_variants.jsonl
"""

import argparse
import importlib.util
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_state_change_helpers(repo_root: Path):
    mod_path = repo_root / "scripts" / "04_build_state_change_dataset.py"
    spec = importlib.util.spec_from_file_location("build_state_change_dataset", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.parse_primary_tasks, mod.step_to_state_change


def parse_interpretation(input_text: str) -> str:
    for line in input_text.splitlines():
        if line.startswith("Interpretation:"):
            return line[len("Interpretation:"):].strip()
    return ""


def build_input_text(goal: str, interpretation: str = "") -> str:
    lines = [f"Goal: {goal}"]
    if interpretation:
        lines.append(f"Interpretation: {interpretation}")
    lines.append("")
    lines.append(
        'Generate the full plan to complete the task. Output JSON only: '
        '{"next_steps": [{"action": "...", "state_change": "..."}, ...]}'
    )
    return "\n".join(lines)


def build_output_text(gold_steps: list[tuple[str, str]]) -> str:
    return json.dumps(
        {
            "next_steps": [
                {"action": action, "state_change": state_change}
                for action, state_change in gold_steps
            ]
        },
        ensure_ascii=False,
    )


def collect_task_records(split_paths: list[Path]) -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    for split_path in split_paths:
        split_name = split_path.stem.replace("system1_", "")
        for row in load_jsonl(split_path):
            meta = row["meta"]
            task_id = str(meta["task_id"])
            record = tasks.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "task_name": meta["task_name"],
                    "goal": f"Complete the task: {meta['task_name']}.",
                    "interpretation": parse_interpretation(row["input_text"]),
                    "source_splits": set(),
                },
            )
            record["source_splits"].add(split_name)

            interpretation = parse_interpretation(row["input_text"])
            if interpretation and not record["interpretation"]:
                record["interpretation"] = interpretation
    return tasks


def build_rows(task_records: dict[str, dict], canonical_tasks: dict, step_to_state_change) -> tuple[list[dict], list[dict], list[dict]]:
    combined_rows = []
    goal_only_rows = []
    goal_plus_interp_rows = []

    for task_id in sorted(task_records, key=int):
        record = task_records[task_id]
        canonical = canonical_tasks[task_id]
        gold_steps = [(step, step_to_state_change(step)) for step in canonical["steps"]]
        gold_step_strings = [f"{action} | {state_change}" for action, state_change in gold_steps]

        base_meta = {
            "task_id": task_id,
            "task_name": record["task_name"],
            "goal": record["goal"],
            "interpretation": record["interpretation"],
            "source_splits": sorted(record["source_splits"]),
            "num_gold_steps": len(gold_steps),
        }

        variants = [
            ("goal_only", build_input_text(record["goal"])),
            ("goal_plus_interpretation", build_input_text(record["goal"], record["interpretation"])),
        ]
        for prompt_variant, input_text in variants:
            row = {
                "input_text": input_text,
                "output_text": build_output_text(gold_steps),
                "goal": record["goal"],
                "interpretation": record["interpretation"],
                "gold_steps": gold_step_strings,
                "meta": {
                    **base_meta,
                    "prompt_variant": prompt_variant,
                },
            }
            combined_rows.append(row)
            if prompt_variant == "goal_only":
                goal_only_rows.append(row)
            else:
                goal_plus_interp_rows.append(row)

    return combined_rows, goal_only_rows, goal_plus_interp_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build task-level CrossTask full-plan evaluation files")
    parser.add_argument("--train", default="data/crosstask/system1_train.jsonl")
    parser.add_argument("--val", default="data/crosstask/system1_val.jsonl")
    parser.add_argument("--test", default="data/crosstask/system1_test.jsonl")
    parser.add_argument("--tasks_file", default="crosstask_release/tasks_primary.txt")
    parser.add_argument("--out_combined", default="data/crosstask/task_level_prompt_variants.jsonl")
    parser.add_argument("--out_goal_only", default="data/crosstask/task_level_goal_only.jsonl")
    parser.add_argument("--out_goal_plus_interpretation", default="data/crosstask/task_level_goal_plus_interpretation.jsonl")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    split_paths = [repo_root / args.train, repo_root / args.val, repo_root / args.test]
    task_records = collect_task_records(split_paths)

    parse_primary_tasks, step_to_state_change = load_state_change_helpers(repo_root)
    canonical_tasks = parse_primary_tasks(str(repo_root / args.tasks_file))

    missing = sorted(set(task_records) - set(canonical_tasks), key=int)
    if missing:
        raise ValueError(f"Tasks present in system1 splits but missing from tasks_primary.txt: {missing}")

    combined_rows, goal_only_rows, goal_plus_interp_rows = build_rows(
        task_records, canonical_tasks, step_to_state_change
    )

    write_jsonl(repo_root / args.out_combined, combined_rows)
    write_jsonl(repo_root / args.out_goal_only, goal_only_rows)
    write_jsonl(repo_root / args.out_goal_plus_interpretation, goal_plus_interp_rows)

    print(f"Wrote {len(combined_rows)} rows to {repo_root / args.out_combined}")
    print(f"Wrote {len(goal_only_rows)} rows to {repo_root / args.out_goal_only}")
    print(f"Wrote {len(goal_plus_interp_rows)} rows to {repo_root / args.out_goal_plus_interpretation}")


if __name__ == "__main__":
    main()
