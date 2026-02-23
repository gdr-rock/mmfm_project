import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


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


def ensure_output_dir(path):
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline wrapper: Tree of Captions -> Self-Refine -> validated structured JSON"
    )
    parser.add_argument("--input", required=True, help="Path to Tree of Captions JSON")
    parser.add_argument("--output", required=True, help="Path to final structured JSON")
    parser.add_argument("--pipeline-dir", default="./output/pipeline_runs", help="Directory for pipeline artifacts")

    # forwarded self-refine arguments
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

    ensure_output_dir(args.output)
    os.makedirs(args.pipeline_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.abspath(args.pipeline_dir), f"run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    self_refine_output = os.path.join(run_dir, "self_refine_output.json")

    self_refine_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "self_refine",
        "run_self_refine.py",
    )

    cmd = [
        sys.executable,
        self_refine_script,
        "--input",
        args.input,
        "--output",
        self_refine_output,
        "--model",
        args.model,
        "--task-name",
        args.task_name,
        "--video-id",
        args.video_id,
        "--iterations",
        str(args.iterations),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--feedback-max-new-tokens",
        str(args.feedback_max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
    ]

    subprocess.run(cmd, check=True)

    with open(self_refine_output, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not validate_plan_schema(data):
        raise ValueError(
            "Self-refine output is not valid structured JSON with expected schema. "
            f"Check artifacts in: {run_dir}"
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    metadata = {
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "pipeline_run_dir": run_dir,
        "self_refine_output": self_refine_output,
        "model": args.model,
        "task_name": args.task_name,
        "video_id": args.video_id,
        "iterations": args.iterations,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Structured data written to: {args.output}")
    print(f"Pipeline artifacts: {run_dir}")


if __name__ == "__main__":
    main()
