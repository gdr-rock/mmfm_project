#!/usr/bin/env python3
"""
Frame-conditioned evaluation pipeline.

Flow:
  1) Read System-1 test JSONL
  2) For each sample, load initial frames from video/frames path
  3) Generate Interpretation text from frames (PLM vision)
  4) Rebuild prompt using Goal + Prefix + generated Interpretation
  5) Run existing evaluation code on rewritten dataset

Outputs:
  - rewritten_test_with_frame_interp.jsonl
  - frame_interpretations.jsonl
  - frame_conditioned_results.json
  - frame_conditioned_per_task.json
  - frame_conditioned_per_sample.jsonl
"""

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

import torch


def _load_module(file_name, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(os.path.dirname(__file__), file_name),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ri = _load_module("run_inference.py", "run_inference")
reval = _load_module("run_evaluation.py", "run_evaluation")


def parse_input_text(input_text):
    goal = ""
    interpretation = ""
    prefix_steps = []
    k = 3

    for line in input_text.split("\n"):
        s = line.strip()
        if s.startswith("Goal:"):
            goal = s[len("Goal:"):].strip()
        elif s.startswith("Interpretation:"):
            interpretation = s[len("Interpretation:"):].strip()
        elif s and s[0].isdigit() and ")" in s:
            prefix_steps.append(s.split(")", 1)[1].strip())
        elif s.startswith("Predict the next"):
            m = re.search(r"Predict the next\s+(\d+)\s+step", s)
            if m:
                k = int(m.group(1))

    return goal, interpretation, prefix_steps, k


def read_jsonl(path, max_samples=None):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_frames_path(args, task_id, video_id):
    candidates = []

    if args.frames_root:
        candidates.extend([
            Path(args.frames_root) / str(video_id),
            Path(args.frames_root) / str(task_id) / str(video_id),
        ])

    if args.video_root:
        for ext in (".mp4", ".webm", ".mkv", ".avi", ""):
            candidates.extend([
                Path(args.video_root) / f"{video_id}{ext}",
                Path(args.video_root) / str(task_id) / f"{video_id}{ext}",
            ])

    for c in candidates:
        if c.exists():
            return str(c)
    return None


def main():
    parser = argparse.ArgumentParser(description="Frame-conditioned System-1 evaluation")

    parser.add_argument("--test_data", default="data/coin/coin_system1_test.jsonl")
    parser.add_argument("--frames_root", default=None,
                        help="Root dir with per-video frame folders")
    parser.add_argument("--video_root", default=None,
                        help="Root dir with raw video files")
    parser.add_argument("--system1_model", required=True)
    parser.add_argument("--system1_type", choices=["t5", "plm"], default="t5")
    parser.add_argument("--plm_base_model", default=None)
    parser.add_argument("--interp_model", default="facebook/Perception-LM-1B")
    parser.add_argument("--interp_prompt", default=ri.DEFAULT_INTERP_PROMPT)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--strict_frames", action="store_true",
                        help="Skip samples with missing frames/video path")
    parser.add_argument("--output_dir", default="outputs/evaluation_frame_conditioned")

    args = parser.parse_args()

    if not args.frames_root and not args.video_root:
        raise ValueError("Provide at least one of --frames_root or --video_root")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(args.test_data, max_samples=args.max_samples)

    rewritten = []
    interp_rows = []

    for idx, sample in enumerate(samples):
        meta = sample.get("meta", {})
        task_id = str(meta.get("task_id", ""))
        video_id = str(meta.get("video_id", ""))

        goal, old_interp, prefix_steps, k = parse_input_text(sample["input_text"])
        frames_path = resolve_frames_path(args, task_id, video_id)

        status = "ok"
        interp = old_interp

        if frames_path:
            try:
                interp = ri.generate_interpretation_from_frames(
                    frames_path=frames_path,
                    device=device,
                    model_name=args.interp_model,
                    num_frames=args.num_frames,
                    prompt=args.interp_prompt,
                )
                if not interp:
                    interp = old_interp
                    status = "empty_interp_fallback"
            except Exception as exc:
                status = f"interp_error:{type(exc).__name__}"
                interp = old_interp
        else:
            status = "frames_not_found"

        if args.strict_frames and status == "frames_not_found":
            continue

        new_input = ri.build_system1_prompt(
            goal=goal,
            prefix_steps=prefix_steps,
            k=k,
            interpretation=interp,
        )

        rewritten_sample = dict(sample)
        rewritten_sample["input_text"] = new_input
        rewritten.append(rewritten_sample)

        interp_rows.append({
            "sample_idx": idx,
            "task_id": task_id,
            "video_id": video_id,
            "goal": goal,
            "frames_path": frames_path,
            "status": status,
            "old_interpretation": old_interp,
            "new_interpretation": interp,
        })

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{len(samples)} samples")

    rewritten_path = out_dir / "rewritten_test_with_frame_interp.jsonl"
    interp_path = out_dir / "frame_interpretations.jsonl"
    write_jsonl(rewritten_path, rewritten)
    write_jsonl(interp_path, interp_rows)

    print("\nRunning evaluation on frame-conditioned prompts...")
    results = reval.evaluate_system1(
        test_data_path=str(rewritten_path),
        model_path=args.system1_model,
        device=device,
        max_samples=None,
        model_type=args.system1_type,
        base_model_name=args.plm_base_model,
    )

    agg = reval.aggregate_metrics(results)
    per_task = reval.per_task_metrics(results)

    with open(out_dir / "frame_conditioned_results.json", "w") as handle:
        json.dump(agg, handle, indent=2)
    with open(out_dir / "frame_conditioned_per_task.json", "w") as handle:
        json.dump(per_task, handle, indent=2)
    with open(out_dir / "frame_conditioned_per_sample.jsonl", "w") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n============================================================")
    print("Frame-conditioned evaluation complete")
    print("============================================================")
    print(json.dumps(agg, indent=2))
    print(f"Saved rewritten dataset: {rewritten_path}")
    print(f"Saved interpretations:  {interp_path}")
    print(f"Saved metrics:          {out_dir / 'frame_conditioned_results.json'}")


if __name__ == "__main__":
    main()
