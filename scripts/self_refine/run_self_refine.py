import argparse
import json
import os
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tree(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_tree(tree):
    if "nodes" not in tree or "root_id" not in tree:
        raise ValueError("Input must contain 'nodes' and 'root_id'.")
    if not isinstance(tree["nodes"], list) or not tree["nodes"]:
        raise ValueError("'nodes' must be a non-empty list.")


def build_node_map(nodes):
    return {n["id"]: n for n in nodes}


def dfs_linearize(node_id, node_map, prefix="", visited=None):
    if visited is None:
        visited = set()
    if node_id in visited:
        raise ValueError(f"Cycle detected in tree at node id: {node_id}")
    if node_id not in node_map:
        raise ValueError(f"Node id '{node_id}' referenced but missing from nodes.")

    visited.add(node_id)
    node = node_map[node_id]
    label = prefix if prefix else "1"
    start = node.get("start_sec")
    end = node.get("end_sec")
    caption = node.get("caption", "").strip()
    line = f"{label}. [{start}-{end}s] {caption}"

    lines = [line]
    children = node.get("children", [])
    for idx, child_id in enumerate(children, start=1):
        child_label = f"{label}.{idx}"
        lines.extend(dfs_linearize(child_id, node_map, child_label, visited))
    visited.remove(node_id)
    return lines


def linearize_tree(tree):
    validate_tree(tree)
    node_map = build_node_map(tree["nodes"])
    root_id = tree["root_id"]
    lines = dfs_linearize(root_id, node_map)
    return "\n".join(lines)


def extract_json_candidates(text):
    candidates = []
    stack = []
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if ch == '"' and not escape:
            in_string = not in_string
        if in_string:
            escape = (ch == "\\") and not escape
            continue
        escape = False

        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            candidate = text[start : i + 1]
            try:
                parsed = json.loads(candidate)
                candidates.append(parsed)
            except json.JSONDecodeError:
                pass
    return candidates


def validate_plan_schema(obj):
    if not isinstance(obj, dict):
        return False
    required = {
        "goal_description": str,
        "goal_interpretation": dict,
        "action_description": list,
        "world_states": list,
    }
    for key, expected_type in required.items():
        if key not in obj or not isinstance(obj[key], expected_type):
            return False

    gi = obj["goal_interpretation"]
    if "initial_world_state" not in gi or "final_world_state" not in gi:
        return False
    if not isinstance(gi["initial_world_state"], str) or not isinstance(gi["final_world_state"], str):
        return False

    if not all(isinstance(x, str) for x in obj["action_description"]):
        return False
    if not all(isinstance(x, str) for x in obj["world_states"]):
        return False
    return True


def extract_best_plan_json(text):
    candidates = extract_json_candidates(text)
    valid = [c for c in candidates if validate_plan_schema(c)]
    return valid[-1] if valid else None


def generate_text(tokenizer, model, prompt, max_new_tokens, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_initial_prompt(tree_text, task_name, video_id):
    return f"""You are generating supervision data for a Vision-Language World Model planner.
Input is a Tree of Captions linearized with DFS order from one CrossTask-like video.

Requirements:
- Output MUST be valid JSON only (no extra text).
- JSON schema:
  {{
    \"goal_description\": string,
    \"goal_interpretation\": {{
      \"initial_world_state\": string,
      \"final_world_state\": string
    }},
    \"action_description\": [string, ...],
    \"world_states\": [string, ...]
  }}
- `goal_description`: concise high-level achievement (single sentence).
- `goal_interpretation.initial_world_state`: tools/materials/dependencies before execution.
- `goal_interpretation.final_world_state`: concrete final state implied by goal.
- `action_description`: executable steps; avoid presentational/non-progress actions.
- `world_states`: information bottleneck state transitions caused by actions; low redundancy.
- Keep action_description and world_states aligned by progress ordering.
- Do not invent objects/actions unsupported by the tree.

Metadata:
- task_name: {task_name}
- video_id: {video_id}

Format example:
{{
  "goal_description": "Cook tomato and eggs.",
  "goal_interpretation": {{
    "initial_world_state": "Eggs, tomatoes, oil, and pan are available and uncooked.",
    "final_world_state": "Eggs are cooked and mixed with tomatoes, seasoned, and ready to serve."
  }},
  "action_description": ["..."],
  "world_states": ["..."]
}}

Tree of Captions (DFS order):
{tree_text}
"""


def build_feedback_prompt(draft_json, tree_text):
    return f"""You are a strict planning-data reviewer.
Evaluate the draft against the Tree of Captions and these dimensions:
1) Goal description quality and specificity.
2) Goal interpretation completeness (initial/final world state).
3) Action description executability, granularity, and relevance.
4) World states as non-redundant information bottleneck.
5) Alignment between actions and world-state transitions.

Return concise feedback as bullet points:
- [Critical] ...
- [Major] ...
- [Minor] ...

Tree of Captions:
{tree_text}

Draft:
{draft_json}
"""


def build_revision_prompt(draft_json, feedback):
    return f"""Revise the draft using the feedback.

Feedback:
{feedback}

Return ONLY the revised JSON (no extra text).
Before finalizing, check:
- JSON is valid and follows schema exactly.
- Action list excludes purely presentational steps.
- World states capture meaningful task-relevant consequences.
- No unsupported details are introduced.

Draft:
{draft_json}
"""


def ensure_output_dir(path):
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="feeltheAGI/Maverick-7B")
    parser.add_argument("--task-name", default="unknown_task")
    parser.add_argument("--video-id", default="video_1")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--feedback-max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    tree = load_tree(args.input)
    tree_text = linearize_tree(tree)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    ensure_output_dir(args.output)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    base_dir = os.path.join(output_dir, f"run_{stamp}")
    os.makedirs(base_dir, exist_ok=True)

    # Initial draft
    prompt = build_initial_prompt(tree_text, args.task_name, args.video_id)
    draft_raw = generate_text(tokenizer, model, prompt, args.max_new_tokens, args.temperature, args.top_p)
    with open(os.path.join(base_dir, "draft_0.txt"), "w", encoding="utf-8") as f:
        f.write(draft_raw)

    current = draft_raw

    # Self-refine loop
    for i in range(1, args.iterations + 1):
        feedback_prompt = build_feedback_prompt(current, tree_text)
        feedback_raw = generate_text(
            tokenizer,
            model,
            feedback_prompt,
            args.feedback_max_new_tokens,
            args.temperature,
            args.top_p,
        )
        with open(os.path.join(base_dir, f"feedback_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(feedback_raw)

        revision_prompt = build_revision_prompt(current, feedback_raw)
        revised_raw = generate_text(tokenizer, model, revision_prompt, args.max_new_tokens, args.temperature, args.top_p)
        with open(os.path.join(base_dir, f"draft_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(revised_raw)

        current = revised_raw

    # Extract JSON
    parsed = extract_best_plan_json(current)
    if parsed is None:
        # Fallback: write raw text to output
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(current)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)


if __name__ == "__main__":
    main()
