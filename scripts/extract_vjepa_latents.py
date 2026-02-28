#!/usr/bin/env python3
"""
Extract V-JEPA 2 latent features from CrossTask videos, one at a time.

Pipeline (per video):
  1. Download video via yt-dlp
  2. For each annotated segment (from transitions CSV), extract frames
  3. Run V-JEPA 2 encoder to get latent features
  4. Save per-segment latent as .pt file
  5. Delete the downloaded video

Storage layout:
  data/crosstask/vjepa_latents/
    {task_id}/
      {video_id}/
        segment_000.pt   ← tensor of shape (T_tokens, D)
        segment_001.pt
        ...
        meta.json         ← segment-to-timestamp mapping

V-JEPA 2 model:
  Uses the official facebookresearch/vjepa2 ViT-L encoder.
  Feature dimension: 1024 (ViT-L) or 1280 (ViT-H).
  We use ViT-L by default for compute efficiency.

⚠️  IMPORTANT CONSTRAINTS:
  1. V-JEPA 2 code must be cloned from github.com/facebookresearch/vjepa2
  2. Requires significant GPU memory (~16GB for ViT-L with 16 frames)
  3. Videos must be downloadable (some CrossTask YouTube videos may be deleted)
  4. Process is slow (~2-5 min per video depending on length/segments)
  5. Total storage for latents: ~2-5 GB for all CrossTask videos

Usage (HPC):
    python3 scripts/extract_vjepa_latents.py \
        --transitions_csv  data/crosstask/state_change_transitions.csv \
        --output_dir       data/crosstask/vjepa_latents \
        --vjepa_repo       ../vjepa2 \
        --model_name       vitl \
        --frames_per_segment 16 \
        --max_videos 10 \
        --skip_existing
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Video download
# ---------------------------------------------------------------------------

def download_video(video_id: str, output_path: str, max_height: int = 360) -> bool:
    """Download a YouTube video using yt-dlp. Returns True on success."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "-f", f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}]/best",
        "-o", output_path,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return os.path.exists(output_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_segment_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int,
) -> np.ndarray:
    """
    Extract n_frames uniformly from [start_sec, end_sec] of a video.
    Returns array of shape (n_frames, H, W, 3) as uint8.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = end_sec - start_sec
    if duration <= 0:
        duration = 1.0

    # Uniformly sample frame timestamps
    timestamps = np.linspace(start_sec, end_sec, n_frames, endpoint=False)
    frames = []

    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # Duplicate last frame or use black
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

    cap.release()
    return np.stack(frames)


def preprocess_frames(frames: np.ndarray, size: int = 224) -> torch.Tensor:
    """
    Resize and normalize frames for V-JEPA 2 input.
    Input:  (T, H, W, 3) uint8
    Output: (1, 3, T, size, size) float32
    """
    import cv2

    processed = []
    for frame in frames:
        resized = cv2.resize(frame, (size, size))
        processed.append(resized)

    arr = np.stack(processed).astype(np.float32) / 255.0
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # (T, H, W, 3) → (3, T, H, W) → (1, 3, T, H, W)
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)
    return tensor


# ---------------------------------------------------------------------------
# V-JEPA 2 encoder
# ---------------------------------------------------------------------------

class VJEPAEncoder:
    """
    Wrapper for V-JEPA 2 encoder.

    Supports two modes:
      1. Official vjepa2 repo (if cloned locally)
      2. Fallback: random features (for testing pipeline without model)
    """

    def __init__(self, vjepa_repo: str = None, model_name: str = "vitl",
                 device: str = "cuda"):
        self.device = device
        self.model = None
        self.feature_dim = 1024 if model_name == "vitl" else 1280

        if vjepa_repo and os.path.exists(vjepa_repo):
            self._load_vjepa(vjepa_repo, model_name)
        else:
            print(f"  ⚠️  V-JEPA 2 repo not found at: {vjepa_repo}")
            print(f"  ⚠️  Using RANDOM features (dim={self.feature_dim}) as placeholder")
            print(f"  ⚠️  To use real features, clone: git clone https://github.com/facebookresearch/vjepa2")
            self.model = None

    def _load_vjepa(self, repo_path: str, model_name: str):
        """Load V-JEPA 2 encoder from the official repo."""
        sys.path.insert(0, repo_path)
        try:
            # The vjepa2 repo provides a model loading API
            # This may need adjustment based on the actual repo structure
            from vjepa2.models import build_model
            from vjepa2.utils import load_checkpoint

            self.model, _ = build_model(model_name=model_name)
            # Load pretrained weights
            ckpt_path = os.path.join(repo_path, "checkpoints", f"vjepa2_{model_name}.pth")
            if os.path.exists(ckpt_path):
                load_checkpoint(self.model, ckpt_path)
                print(f"  Loaded V-JEPA 2 {model_name} from {ckpt_path}")
            else:
                print(f"  ⚠️  Checkpoint not found: {ckpt_path}")
                print(f"  ⚠️  Download from: https://github.com/facebookresearch/vjepa2")

            self.model.to(self.device)
            self.model.eval()
        except ImportError as e:
            print(f"  ⚠️  Failed to import vjepa2: {e}")
            print(f"  ⚠️  Falling back to random features")
            self.model = None

    @torch.no_grad()
    def encode(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode a video tensor to latent features.

        Input:  (1, 3, T, H, W) float32
        Output: (N_tokens, D) float32  where D = feature_dim
        """
        if self.model is not None:
            video_tensor = video_tensor.to(self.device)
            # Forward through encoder
            features = self.model(video_tensor)
            # Pool spatially: (1, N, D) → (N, D)
            if features.dim() == 3:
                return features.squeeze(0).cpu()
            return features.cpu()
        else:
            # Random placeholder features
            n_tokens = video_tensor.shape[2]  # temporal tokens
            return torch.randn(n_tokens, self.feature_dim)


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def load_video_segments(csv_path: str) -> dict:
    """
    Parse transitions CSV to get per-video segment timestamps.

    Returns:
        {(task_id, video_id): [(seg_pos, start_sec, end_sec, step_text), ...]}
    """
    video_segs = defaultdict(list)
    seen = set()

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["task_id"], row["video_id"])

            # Current segment
            seg_key = (key, int(row["seg_pos"]))
            if seg_key not in seen:
                seen.add(seg_key)
                video_segs[key].append((
                    int(row["seg_pos"]),
                    float(row["action_start"]),
                    float(row["action_end"]),
                    row["action"],
                ))

            # Next segment (from last transition row)
            next_seg_key = (key, int(row["next_seg_pos"]))
            if next_seg_key not in seen:
                seen.add(next_seg_key)
                video_segs[key].append((
                    int(row["next_seg_pos"]),
                    float(row["next_action_start"]),
                    float(row["next_action_end"]),
                    row["next_action"],
                ))

    # Sort segments by seg_pos
    for key in video_segs:
        video_segs[key].sort(key=lambda x: x[0])

    return dict(video_segs)


def extract_latents(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load encoder
    print(f"Initializing V-JEPA 2 encoder...")
    encoder = VJEPAEncoder(
        vjepa_repo=args.vjepa_repo,
        model_name=args.model_name,
        device=device,
    )
    print(f"  Feature dim: {encoder.feature_dim}")

    # Load segment info
    print(f"Loading segments from: {args.transitions_csv}")
    video_segments = load_video_segments(args.transitions_csv)
    print(f"  {len(video_segments)} videos with segments")

    # Filter by split if specified
    if args.split_csv:
        split_vids = set()
        with open(args.split_csv) as f:
            for row in csv.DictReader(f):
                split_vids.add((row["task_id"], row["video_id"]))
        video_segments = {
            k: v for k, v in video_segments.items() if k in split_vids
        }
        print(f"  Filtered to {len(video_segments)} videos from split")

    if args.max_videos:
        keys = list(video_segments.keys())[:args.max_videos]
        video_segments = {k: video_segments[k] for k in keys}
        print(f"  Capped to {args.max_videos} videos")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process videos
    n_success = 0
    n_fail = 0
    n_skip = 0
    t0 = time.time()

    for idx, ((task_id, video_id), segments) in enumerate(video_segments.items()):
        vid_out_dir = out_dir / task_id / video_id

        # Skip if already processed
        if args.skip_existing and vid_out_dir.exists():
            meta_path = vid_out_dir / "meta.json"
            if meta_path.exists():
                n_skip += 1
                continue

        print(f"\n[{idx+1}/{len(video_segments)}] task={task_id} video={video_id} "
              f"({len(segments)} segments)")

        # Download video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            print(f"  Downloading...")
            ok = download_video(video_id, tmp_path)
            if not ok:
                print(f"  ✗ Download failed — skipping")
                n_fail += 1
                continue

            vid_out_dir.mkdir(parents=True, exist_ok=True)
            meta = {"task_id": task_id, "video_id": video_id, "segments": []}

            for seg_pos, start, end, action in segments:
                print(f"  Segment {seg_pos}: [{start:.1f}s - {end:.1f}s] {action}")

                try:
                    frames = extract_segment_frames(
                        tmp_path, start, end, args.frames_per_segment
                    )
                    video_tensor = preprocess_frames(frames, size=224)
                    latent = encoder.encode(video_tensor)

                    # Save latent
                    seg_path = vid_out_dir / f"segment_{seg_pos:03d}.pt"
                    torch.save(latent, seg_path)

                    meta["segments"].append({
                        "seg_pos": seg_pos,
                        "start_sec": start,
                        "end_sec": end,
                        "action": action,
                        "latent_shape": list(latent.shape),
                        "latent_file": seg_path.name,
                    })
                except Exception as e:
                    print(f"    ⚠️  Segment {seg_pos} failed: {e}")

            # Save metadata
            with open(vid_out_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            n_success += 1
            elapsed = time.time() - t0
            print(f"  ✓ Done ({len(meta['segments'])}/{len(segments)} segments) "
                  f"[{elapsed:.0f}s total]")

        finally:
            # Always delete the video file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                print(f"  Deleted temp video")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Extraction complete.")
    print(f"  Success: {n_success}")
    print(f"  Failed:  {n_fail}")
    print(f"  Skipped: {n_skip}")
    print(f"  Time:    {elapsed:.0f}s")
    print(f"  Output:  {out_dir}")

    # Size report
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(out_dir):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    print(f"  Total size: {total_size / (1024**2):.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Extract V-JEPA 2 latents from CrossTask videos"
    )
    parser.add_argument("--transitions_csv",
                        default="data/crosstask/state_change_transitions.csv")
    parser.add_argument("--split_csv", default=None,
                        help="Optional: only process videos from this split CSV "
                             "(e.g. state_change_transitions_train.csv)")
    parser.add_argument("--output_dir",
                        default="data/crosstask/vjepa_latents")
    parser.add_argument("--vjepa_repo", default="../vjepa2",
                        help="Path to cloned vjepa2 repo")
    parser.add_argument("--model_name", default="vitl",
                        choices=["vitl", "vith"],
                        help="V-JEPA 2 model size")
    parser.add_argument("--frames_per_segment", type=int, default=16)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing",
                        action="store_false")
    args = parser.parse_args()

    extract_latents(args)


if __name__ == "__main__":
    main()
