#!/usr/bin/env python3
"""
Extract V-JEPA latents for COIN videos.

This script ports the workflow from JEPA_Latent_Generation.ipynb into a
re-runnable CLI tool and writes latents in the format expected by the rest of
the repository:

  data/coin/vjepa_latents/
    {task_id}/
      {video_id}/
        segment_000.pt
        segment_001.pt
        ...
        meta.json

Each ``segment_XXX.pt`` stores a pooled latent vector for one annotated COIN
segment. The final segment is also saved as ``goal.pt``. A ``meta.json`` file
keeps the segment mapping and extraction config.

By default the script downloads videos from ``COIN.json`` with ``yt-dlp``. If
you already have the videos locally, point ``--videos_dir`` at a directory
containing files named like ``{video_id}.mp4``.

Example:
    python3 scripts/coin_extract_vjepa_latents.py \
        --coin_json COIN_dataset/COIN.json \
        --split_csv data/coin/coin_video_splits.csv \
        --split train \
        --output_dir data/coin/vjepa_latents \
        --frames_per_segment 32 \
        --skip_existing
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from coin_utils import read_coin_database


DEFAULT_MODEL_NAME = "facebook/vjepa2-vitg-fpc64-384"
DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract V-JEPA latents for annotated COIN segments."
    )
    parser.add_argument("--coin_json", default="COIN_dataset/COIN.json")
    parser.add_argument(
        "--split_csv",
        default="data/coin/coin_video_splits.csv",
        help="Optional repo split manifest. Used with --split.",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "val", "test", "all"],
        help="train / val / test / all. Ignored if --split_csv is missing.",
    )
    parser.add_argument(
        "--official_subset",
        default="all",
        choices=["training", "testing", "all"],
        help="Fallback filter when --split_csv is not available.",
    )
    parser.add_argument("--output_dir", default="data/coin/vjepa_latents")
    parser.add_argument(
        "--videos_dir",
        default=None,
        help="Optional local COIN video directory containing files like {video_id}.mp4.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional Hugging Face cache dir for the V-JEPA model.",
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face V-JEPA model id.",
    )
    parser.add_argument("--frames_per_segment", type=int, default=32)
    parser.add_argument("--min_segments", type=int, default=2)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda / cpu. Defaults to auto.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model/input dtype used on the selected device.",
    )
    parser.add_argument(
        "--video_extensions",
        nargs="+",
        default=list(DEFAULT_VIDEO_EXTENSIONS),
        help="Extensions searched in --videos_dir.",
    )
    parser.add_argument(
        "--download_timeout_sec",
        type=int,
        default=600,
        help="yt-dlp timeout per video.",
    )
    parser.add_argument(
        "--save_bundle",
        action="store_true",
        help="Also save a combined latent bundle per video (z_segments + delta_z).",
    )
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
    )
    return parser.parse_args()


def resolve_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(device: str, dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    if device == "cuda":
        return torch.float16
    return torch.float32


def normalize_youtube_url(url: str) -> str:
    if "/embed/" in url:
        prefix, video_id = url.split("/embed/", 1)
        video_id = video_id.split("?", 1)[0].strip("/")
        return f"{prefix}/watch?v={video_id}"
    return url


def load_split_rows(split_csv_path: str) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with open(split_csv_path) as handle:
        for row in csv.DictReader(handle):
            rows[row["video_id"]] = row
    return rows


def select_coin_videos(
    database: Dict[str, dict],
    split_rows: Optional[Dict[str, dict]],
    split: str,
    official_subset: str,
    min_segments: int,
    max_videos: Optional[int],
) -> List[Tuple[str, dict, str]]:
    selected: List[Tuple[str, dict, str]] = []

    for video_id, entry in database.items():
        annotations = entry.get("annotation", [])
        if len(annotations) < min_segments:
            continue

        task_id = str(int(entry["recipe_type"]))

        if split_rows is not None:
            row = split_rows.get(video_id)
            if row is None:
                continue
            if split != "all" and row["split"] != split:
                continue
        else:
            if official_subset != "all" and entry.get("subset") != official_subset:
                continue

        selected.append((video_id, entry, task_id))
        if max_videos is not None and len(selected) >= max_videos:
            break

    return selected


def resolve_local_video(
    videos_dir: Optional[str],
    video_id: str,
    extensions: Sequence[str],
) -> Optional[Path]:
    if not videos_dir:
        return None

    base = Path(videos_dir)
    for ext in extensions:
        path = base / f"{video_id}{ext}"
        if path.exists():
            return path

    return None


def download_coin_video(video_url: str, out_path: Path, timeout_sec: int) -> bool:
    if out_path.exists():
        out_path.unlink()

    cmd = [
        "yt-dlp",
        "-f",
        "best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(out_path),
        "--no-playlist",
        normalize_youtube_url(video_url),
    ]

    try:
        subprocess.run(cmd, check=True, timeout=timeout_sec)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False

    return out_path.exists()


def sample_frames_from_time_window(vr, start_sec: float, end_sec: float, num_frames: int) -> torch.Tensor:
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    start_frame = max(0, int(start_sec * fps))
    end_frame = min(total_frames - 1, int(end_sec * fps))
    if end_frame <= start_frame:
        end_frame = min(total_frames - 1, start_frame + 1)

    frame_indices = np.linspace(start_frame, end_frame, num_frames).astype(int)
    video = vr.get_batch(frame_indices).asnumpy()
    return torch.from_numpy(video).permute(0, 3, 1, 2)


class VJEPAExtractor:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: torch.dtype,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            from transformers import AutoModel, AutoVideoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required. Install dependencies from requirements.txt."
            ) from exc

        self.device = device
        self.dtype = dtype
        self.model_name = model_name

        load_kwargs = {"cache_dir": cache_dir}
        if device == "cuda":
            load_kwargs["torch_dtype"] = dtype

        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        self.model.to(device)
        self.model.eval()
        self.processor = AutoVideoProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    def encode_segment(self, video: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(video, return_tensors="pt")
        pixel_values = inputs["pixel_values_videos"].to(self.device)
        if self.device == "cuda":
            pixel_values = pixel_values.to(dtype=self.dtype)

        autocast_enabled = self.device == "cuda" and self.dtype in (torch.float16, torch.bfloat16)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if autocast_enabled
            else contextlib.nullcontext()
        )

        with torch.inference_mode():
            with autocast_ctx:
                if not hasattr(self.model, "get_vision_features"):
                    raise AttributeError(
                        f"{self.model_name} does not expose get_vision_features()."
                    )
                tokens = self.model.get_vision_features(pixel_values)
                if tokens.dim() == 2:
                    pooled = tokens
                else:
                    pooled = tokens.mean(dim=1)

        pooled = pooled.detach().cpu().squeeze(0).float()

        del inputs, pixel_values, tokens
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return pooled


def ensure_decord():
    try:
        from decord import VideoReader
    except ImportError as exc:
        raise RuntimeError(
            "decord is required. Install dependencies from requirements.txt."
        ) from exc
    return VideoReader


def should_skip_video(video_dir: Path, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    if not (video_dir / "meta.json").exists():
        return False
    if not (video_dir / "goal.pt").exists():
        return False
    return any(video_dir.glob("segment_*.pt"))


def build_meta(
    *,
    task_id: str,
    task_name: str,
    video_id: str,
    source_path: str,
    model_name: str,
    frames_per_segment: int,
    segments: List[dict],
) -> dict:
    return {
        "task_id": task_id,
        "task_name": task_name,
        "video_id": video_id,
        "source_path": source_path,
        "model_name": model_name,
        "frames_per_segment": frames_per_segment,
        "num_segments": len(segments),
        "goal_segment_index": len(segments) - 1 if segments else None,
        "goal_file": "goal.pt" if segments else None,
        "segments": segments,
    }


def process_video(
    *,
    extractor: VJEPAExtractor,
    video_id: str,
    entry: dict,
    task_id: str,
    output_dir: Path,
    videos_dir: Optional[str],
    video_extensions: Sequence[str],
    download_timeout_sec: int,
    frames_per_segment: int,
    save_bundle: bool,
) -> Tuple[bool, str]:
    VideoReader = ensure_decord()

    local_path = resolve_local_video(videos_dir, video_id, video_extensions)
    tmp_path: Optional[Path] = None
    source_path: Optional[Path] = local_path

    if source_path is None:
        tmp = tempfile.NamedTemporaryFile(prefix=f"{video_id}_", suffix=".mp4", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        ok = download_coin_video(entry["video_url"], tmp_path, timeout_sec=download_timeout_sec)
        if not ok:
            if tmp_path.exists():
                tmp_path.unlink()
            return False, "download_failed"
        source_path = tmp_path

    try:
        vr = VideoReader(str(source_path))
    except Exception:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        return False, "video_read_failed"

    task_dir = output_dir / task_id / video_id
    task_dir.mkdir(parents=True, exist_ok=True)

    segments_meta: List[dict] = []
    pooled_segments: List[torch.Tensor] = []

    try:
        for seg_idx, ann in enumerate(entry["annotation"]):
            start_sec, end_sec = ann["segment"]
            video = sample_frames_from_time_window(
                vr,
                float(start_sec),
                float(end_sec),
                num_frames=frames_per_segment,
            )
            pooled = extractor.encode_segment(video)

            segment_path = task_dir / f"segment_{seg_idx:03d}.pt"
            torch.save(pooled, segment_path)
            pooled_segments.append(pooled)

            segments_meta.append(
                {
                    "segment_index": seg_idx,
                    "step_id": str(ann["id"]),
                    "step_label": ann["label"],
                    "start_sec": float(start_sec),
                    "end_sec": float(end_sec),
                    "latent_shape": list(pooled.shape),
                    "latent_file": segment_path.name,
                }
            )

        goal_latent = pooled_segments[-1]
        torch.save(goal_latent, task_dir / "goal.pt")

        meta = build_meta(
            task_id=task_id,
            task_name=entry["class"],
            video_id=video_id,
            source_path=str(source_path),
            model_name=extractor.model_name,
            frames_per_segment=frames_per_segment,
            segments=segments_meta,
        )
        (task_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

        if save_bundle:
            z_segments = torch.stack(pooled_segments)
            delta_z = z_segments[1:] - z_segments[:-1]
            torch.save(
                {
                    "task_id": task_id,
                    "task_name": entry["class"],
                    "video_id": video_id,
                    "z_segments": z_segments,
                    "delta_z": delta_z,
                    "segment_meta": segments_meta,
                },
                task_dir / "bundle.pt",
            )

        return True, "ok"
    except Exception as exc:
        shutil.rmtree(task_dir, ignore_errors=True)
        return False, f"segment_failed:{exc}"
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype)

    database = read_coin_database(args.coin_json)
    split_rows = None
    split_csv_path = Path(args.split_csv)
    if split_csv_path.exists():
        split_rows = load_split_rows(str(split_csv_path))
    elif args.split != "all":
        print(f"Warning: split CSV not found at {split_csv_path}. Falling back to official subsets.")

    selected = select_coin_videos(
        database=database,
        split_rows=split_rows,
        split=args.split,
        official_subset=args.official_subset,
        min_segments=args.min_segments,
        max_videos=args.max_videos,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"DType:  {dtype}")
    print(f"Model:  {args.model_name}")
    print(f"Videos: {len(selected)} selected")
    print(f"Output: {output_dir.resolve()}")

    extractor = VJEPAExtractor(
        model_name=args.model_name,
        device=device,
        dtype=dtype,
        cache_dir=args.cache_dir,
    )

    n_done = 0
    n_skip = 0
    n_fail = 0

    for idx, (video_id, entry, task_id) in enumerate(selected, start=1):
        video_dir = output_dir / task_id / video_id
        if should_skip_video(video_dir, args.skip_existing):
            n_skip += 1
            continue

        print(
            f"[{idx}/{len(selected)}] "
            f"task={task_id} video={video_id} segments={len(entry['annotation'])}"
        )
        ok, status = process_video(
            extractor=extractor,
            video_id=video_id,
            entry=entry,
            task_id=task_id,
            output_dir=output_dir,
            videos_dir=args.videos_dir,
            video_extensions=args.video_extensions,
            download_timeout_sec=args.download_timeout_sec,
            frames_per_segment=args.frames_per_segment,
            save_bundle=args.save_bundle,
        )

        if ok:
            n_done += 1
        else:
            n_fail += 1
            print(f"  Failed: {status}")

    elapsed = time.time() - t0
    print("\nExtraction complete.")
    print(f"  done:   {n_done}")
    print(f"  skip:   {n_skip}")
    print(f"  fail:   {n_fail}")
    print(f"  time_s: {elapsed:.1f}")


if __name__ == "__main__":
    main()
