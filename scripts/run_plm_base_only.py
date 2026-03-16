#!/usr/bin/env python3
"""
Run base Perception-LM inference without any fine-tuned adapter.

This is a minimal sanity-check script for comparing:
  - base model only
  - base model + LoRA adapter

It reuses the same System-1 prompt format used elsewhere in the repo.
"""

import argparse
import json
import re

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


CAUSAL_PROMPT_SUFFIX = "\n\nAssistant:"


def build_system1_prompt(goal: str, prefix_steps: list[str], k: int,
                         interpretation: str = "") -> str:
    lines = [f"Goal: {goal}"]
    if interpretation:
        lines.append(f"Interpretation: {interpretation}")
    lines.append("")
    lines.append("Progress so far:")
    if prefix_steps:
        for i, step in enumerate(prefix_steps, 1):
            lines.append(f"  {i}) {step}")
    else:
        lines.append("  (No steps observed yet.)")
    lines.append("")
    lines.append(
        f'Predict the next {k} step(s). Output JSON only: '
        '{"next_steps": [{"action": "...", "state_change": "..."}, ...]}'
    )
    return "\n".join(lines)


def parse_plan_json(text: str) -> list[str]:
    def _decode_json_string(s: str) -> str:
        try:
            return json.loads(f"\"{s}\"")
        except Exception:
            return s

    def _steps_from_list(step_list):
        if not isinstance(step_list, list):
            return []
        out = []
        for step in step_list:
            if not isinstance(step, dict):
                continue
            action = step.get("action")
            state_change = step.get("state_change")
            if isinstance(action, str) and isinstance(state_change, str):
                out.append(f"{action.strip()} | {state_change.strip()}")
        return out

    text = (text or "").strip()
    if not text:
        return []

    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()

    candidates = [text]
    if '"next_steps"' in text and not text.lstrip().startswith("{"):
        candidates.append("{" + text + "}")
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace:last_brace + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            steps = _steps_from_list(parsed.get("next_steps", []))
            if steps:
                return steps
        elif isinstance(parsed, list):
            steps = _steps_from_list(parsed)
            if steps:
                return steps

    action_vals = [
        _decode_json_string(v)
        for v in re.findall(r'"action"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    ]
    state_vals = [
        _decode_json_string(v)
        for v in re.findall(r'"state_change"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    ]
    n = min(len(action_vals), len(state_vals))
    if n > 0:
        return [
            f"{action_vals[i].strip()} | {state_vals[i].strip()}"
            for i in range(n)
        ]

    return []


def load_sample(jsonl_path: str, sample_idx: int):
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if sample_idx >= len(samples):
        raise IndexError(f"sample_idx {sample_idx} out of range for {jsonl_path}")

    sample = samples[sample_idx]
    lines = sample["input_text"].splitlines()
    goal = ""
    interpretation = ""
    prefix = []
    in_progress = False

    for line in lines:
        if line.startswith("Goal:"):
            goal = line[len("Goal:"):].strip()
        elif line.startswith("Interpretation:"):
            interpretation = line[len("Interpretation:"):].strip()
        elif line.strip() == "Progress so far:":
            in_progress = True
        elif line.startswith("Predict the next"):
            in_progress = False
        elif in_progress and line.strip().startswith("("):
            continue
        elif in_progress and line.strip():
            prefix.append(line.split(")", 1)[-1].strip())

    return sample, goal, interpretation, prefix


def main():
    parser = argparse.ArgumentParser(description="Run base Perception-LM without LoRA")
    parser.add_argument("--model_name", default="facebook/Perception-LM-1B")
    parser.add_argument("--goal", default="")
    parser.add_argument("--interpretation", default="")
    parser.add_argument("--prefix", nargs="*", default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--from_jsonl", default=None)
    parser.add_argument("--sample_idx", type=int, default=0)
    args = parser.parse_args()

    if args.from_jsonl:
        print(f"Loading sample {args.sample_idx} from {args.from_jsonl}")
        sample, goal, interpretation, prefix = load_sample(args.from_jsonl, args.sample_idx)
        args.goal = goal or args.goal
        args.interpretation = interpretation or args.interpretation
        args.prefix = prefix or args.prefix
        gold = parse_plan_json(sample.get("output_text", ""))
        print(f"GOLD STEPS ({len(gold)}):")
        for i, step in enumerate(gold, 1):
            print(f"  {i}) {step}")

    if not args.goal:
        raise ValueError("Provide --goal or use --from_jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading base model only: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    prompt = build_system1_prompt(
        goal=args.goal,
        interpretation=args.interpretation,
        prefix_steps=args.prefix or [],
        k=args.k,
    ) + CAUSAL_PROMPT_SUFFIX

    print(f"\n{'-' * 60}")
    print("PROMPT:")
    print(prompt[:-len(CAUSAL_PROMPT_SUFFIX)])
    print(f"{'-' * 60}\n")

    enc = tokenizer(
        prompt, max_length=512, truncation=True, return_tensors="pt"
    ).to(device)
    prompt_len = enc.input_ids.shape[1]
    do_sample = args.temperature > 0

    with torch.no_grad():
        gen_ids = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_ids = gen_ids[0][prompt_len:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    steps = parse_plan_json(raw)

    print("RAW OUTPUT:")
    print(raw)
    print(f"\nPARSED STEPS ({len(steps)}):")
    for i, step in enumerate(steps, 1):
        print(f"  {i}) {step}")
    if not steps:
        print("  (no valid JSON parsed)")


if __name__ == "__main__":
    main()
