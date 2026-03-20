#!/usr/bin/env python3
"""Utilities for the mixed COIN/CrossTask System-1 dataset in data/coin_2."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path


END_ACTION = "<END>"
END_STATE_CHANGE = "<END>"


def clean_state_change_text(text: str) -> str:
    """Normalize noisy state-change strings into a cleaner sentence form."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return text
    if text in {END_ACTION, END_STATE_CHANGE}:
        return text

    match = re.fullmatch(
        r"The state changes:\s*(.+?)\s+is performed\.",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        step = match.group(1).strip()
        if step:
            return f"{step[0].upper() + step[1:]} is now done."
        return "The step is now done."

    return text


def append_end_step(steps: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out = list(steps)
    out.append((END_ACTION, END_STATE_CHANGE))
    return out


def format_system1_output(
    steps: list[tuple[str, str]],
    *,
    append_end: bool = True,
) -> str:
    payload_steps = append_end_step(steps) if append_end else list(steps)
    payload = {
        "next_steps": [
            {
                "action": action,
                "state_change": state_change,
            }
            for action, state_change in payload_steps
        ]
    }
    return json.dumps(payload, ensure_ascii=False)


def build_remaining_plan_prompt(
    goal: str,
    prefix_steps: list[tuple[str, str]],
    *,
    interpretation: str = "",
) -> str:
    lines = [f"Goal: {goal}"]
    if interpretation:
        lines.append(f"Interpretation: {interpretation}")
    lines.append("")
    lines.append("Progress so far:")
    if prefix_steps:
        for idx, (action, state_change) in enumerate(prefix_steps, start=1):
            lines.append(f"  {idx}) {action} | {state_change}")
    else:
        lines.append("  (No steps observed yet.)")
    lines.append("")
    lines.append(
        'Generate the remaining plan to complete the task. Output JSON only: '
        '{"next_steps": [{"action": "...", "state_change": "..."}, ...]}. '
        'End the plan with {"action": "<END>", "state_change": "<END>"}.'
    )
    return "\n".join(lines)


def build_full_plan_prompt(goal: str, *, interpretation: str = "") -> str:
    lines = [f"Goal: {goal}"]
    if interpretation:
        lines.append(f"Interpretation: {interpretation}")
    lines.append("")
    lines.append(
        'Generate the full plan to complete the task. Output JSON only: '
        '{"next_steps": [{"action": "...", "state_change": "..."}, ...]}. '
        'End the plan with {"action": "<END>", "state_change": "<END>"}.'
    )
    return "\n".join(lines)


def tokenize_signature(*parts: str) -> set[str]:
    text = " ".join(part for part in parts if part).lower()
    return set(re.findall(r"[a-z0-9]+", text))


def distinct_prefix_lengths(
    valid_prefix_lengths: list[int],
    samples_per_video: int,
    rng: random.Random,
) -> list[int]:
    if samples_per_video <= 0 or not valid_prefix_lengths:
        return []
    if len(valid_prefix_lengths) <= samples_per_video:
        return sorted(valid_prefix_lengths)
    return sorted(rng.sample(valid_prefix_lengths, samples_per_video))


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
