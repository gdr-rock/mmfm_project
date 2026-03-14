#!/usr/bin/env python3
"""Build COIN frame-aligned zero-prefix eval datasets."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from coin_utils import ensure_taxonomy_cache, format_system1_output, format_system1_prompt, read_coin_database, step_to_state_change


def _load_interp_fn():
    spec = importlib.util.spec_from_file_location(
        "run_inference",
        os.path.join(os.path.dirname(__file__), "run_inference.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_interpretation_from_frames


def download_video(video_id: str, cache_dir: str, max_height: int = 360) -> str:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(cache_dir, f"{video_id}.mp4")
    if os.path.exists(output_path):
        return output_path
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "-f",
        f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}]/best",
        "-o",
        output_path,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]
    subprocess.run(cmd, check=True)
    return output_path


def extract_roi_frames(video_path: str, roi_start: float, num_frames: int, output_dir: str) -> None:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, roi_start * 1000.0)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for idx in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(os.path.join(output_dir, f"frame_{idx:03d}.png"))
    cap.release()


def build_zero_prefix_samples(
    split_rows: list[dict],
    taxonomy: dict,
    interpretations: dict,
    k_values: list[int],
) -> list[dict]:
    samples = []
    for row in split_rows:
        task = taxonomy["tasks"][row["task_id"]]
        interpretation = interpretations[row["video_id"]]
        canonical = [
            (step["action"], step_to_state_change(step["action"]))
            for step in task["canonical_steps"]
        ]
        for k in k_values:
            target = canonical[: min(k, len(canonical))]
            if not target:
                continue
            samples.append(
                {
                    "input_text": format_system1_prompt(
                        task["goal"],
                        interpretation,
                        [],
                        len(target),
                    ),
                    "output_text": format_system1_output(target),
                    "meta": {
                        "task_id": row["task_id"],
                        "task_name": task["task_name"],
                        "video_id": row["video_id"],
                        "prefix_len": 0,
                        "k": len(target),
                        "start_seg_pos": 0,
                        "trajectory_len": 0,
                        "roi_start": float(row["roi_start"]),
                        "roi_end": float(row["roi_end"]),
                        "recipe_type": int(row["recipe_type"]),
                        "interpretation_source": "frames",
                    },
                }
            )
    return samples


def write_jsonl(path: str, samples: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build COIN frame-aligned eval datasets")
    parser.add_argument("--coin_json", default="COIN_dataset/COIN.json")
    parser.add_argument("--taxonomy_xlsx", default="COIN_dataset/taxonomy.xlsx")
    parser.add_argument("--taxonomy_cache", default="data/coin/coin_taxonomy.json")
    parser.add_argument("--split_csv", default="data/coin/coin_video_splits.csv")
    parser.add_argument("--data_dir", default="data/coin")
    parser.add_argument("--video_cache_dir", default="data/coin/video_cache")
    parser.add_argument("--interp_cache", default="data/coin/frame_eval/interpretations.json")
    parser.add_argument("--interp_model", default="facebook/Perception-LM-1B")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--k_values", default="1,2,3")
    parser.add_argument("--max_videos", type=int, default=None)
    args = parser.parse_args()

    taxonomy = ensure_taxonomy_cache(
        args.coin_json, args.taxonomy_xlsx, args.taxonomy_cache
    )
    interp_fn = _load_interp_fn()
    split_rows = list(csv.DictReader(open(args.split_csv)))
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cached = {}
    interp_cache_path = Path(args.interp_cache)
    if interp_cache_path.exists():
        cached = json.loads(interp_cache_path.read_text())

    k_values = [int(value) for value in args.k_values.split(",") if value.strip()]
    for split in ("val", "test"):
        selected = [row for row in split_rows if row["split"] == split]
        if args.max_videos is not None:
            selected = selected[: args.max_videos]
        for row in selected:
            video_id = row["video_id"]
            if video_id in cached:
                continue
            video_path = download_video(video_id, args.video_cache_dir)
            with tempfile.TemporaryDirectory() as tmpdir:
                extract_roi_frames(video_path, float(row["roi_start"]), args.num_frames, tmpdir)
                cached[video_id] = interp_fn(
                    frames_path=tmpdir,
                    device=device,
                    model_name=args.interp_model,
                    num_frames=args.num_frames,
                )
            interp_cache_path.parent.mkdir(parents=True, exist_ok=True)
            interp_cache_path.write_text(json.dumps(cached, indent=2, ensure_ascii=False) + "\n")

        samples = build_zero_prefix_samples(selected, taxonomy, cached, k_values)
        out_path = os.path.join(args.data_dir, f"coin_system1_{split}_frames.jsonl")
        write_jsonl(out_path, samples)
        print(f"{split}: {out_path} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
