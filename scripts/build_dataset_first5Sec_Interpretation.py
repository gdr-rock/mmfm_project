#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def normalize_youtube_url(url: str) -> str:
    if not url:
        return url

    if "youtube.com/embed/" in url:
        vid = url.split("youtube.com/embed/")[-1].split("?")[0].strip("/")
        return f"https://www.youtube.com/watch?v={vid}"

    return url


def load_coin(coin_json, subset=None):
    with open(coin_json, "r", encoding="utf-8") as f:
        db = json.load(f)["database"]

    videos = []
    for vid, entry in db.items():
        if subset and entry.get("subset") != subset:
            continue

        videos.append({
            "video_id": vid,
            "task_name": entry.get("class", ""),
            "subset": entry.get("subset", ""),
            "recipe_type": entry.get("recipe_type", None),
            "video_url": normalize_youtube_url(entry.get("video_url", "")),
        })

    return videos


def load_processed_ids(output_path):
    processed = set()

    if not os.path.exists(output_path):
        return processed

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                vid = row.get("video_id")
                if vid:
                    processed.add(vid)
            except Exception:
                continue

    print(f"Found {len(processed)} already processed videos")
    return processed


def download_video_from_url(video_url, out_path):
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-warnings",
        "-o", str(out_path),
        video_url,
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print("Download error:", repr(e))
        return False


def extract_frames_window(video_path, out_dir, start_sec=0, duration_sec=5):
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration_sec),
        "-vf", "fps=1",
        os.path.join(out_dir, "frame_%03d.jpg"),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[:500]}")


class PerceptionLM:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

    def caption(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=30)

        text = self.processor.decode(out[0], skip_special_tokens=True)
        return text.strip()

    def describe_video_window(self, video_path, start_sec=0, duration_sec=5):
        with tempfile.TemporaryDirectory() as tmp:
            print(f"[DEBUG] Extracting frames from {video_path} at {start_sec}-{start_sec+duration_sec}s")

            extract_frames_window(video_path, tmp, start_sec=start_sec, duration_sec=duration_sec)

            frames = sorted(Path(tmp).glob("*.jpg"))[:5]
            print(f"[DEBUG] Found {len(frames)} frames")

            if len(frames) == 0:
                raise RuntimeError("No frames extracted")

            captions = []
            seen = set()

            for f in frames:
                try:
                    cap = self.caption(str(f))
                    print(f"[DEBUG] {f.name} -> {cap}")

                    norm = cap.lower().strip()
                    if norm and norm not in seen:
                        seen.add(norm)
                        captions.append(cap)
                except Exception as e:
                    print(f"[ERROR] Caption failed for {f}: {repr(e)}")

            if not captions:
                return None

            captions = captions[:3]
            return " ".join(captions)


def make_interpretation(obs, task):
    return (
        f"From this short video segment, the scene shows: {obs}. "
        f"This may indicate a task related to {task.lower()}."
    )


def process_and_write(
    videos,
    output_path,
    model,
    processed_ids,
    start_sec=0,
    duration_sec=5,
):
    written = 0
    skipped_download = 0
    skipped_interpret = 0
    skipped_existing = 0

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        for i, v in enumerate(videos, 1):
            vid = v["video_id"]
            task = v["task_name"]
            subset = v["subset"]
            recipe_type = v["recipe_type"]
            url = v["video_url"]

            if vid in processed_ids:
                skipped_existing += 1
                print(f"[SKIP EXISTING] {vid}")
                continue

            print(f"\n[{i}/{len(videos)}] {vid} | {task}")
            print(f"URL: {url}")

            if not url:
                print("[SKIP NO URL]")
                skipped_download += 1
                continue

            with tempfile.TemporaryDirectory() as tmp:
                tmp_video = os.path.join(tmp, f"{vid}.mp4")

                ok = download_video_from_url(url, tmp_video)
                if not ok or not os.path.exists(tmp_video):
                    print("[SKIP DOWNLOAD FAIL]")
                    skipped_download += 1
                    continue

                try:
                    obs = model.describe_video_window(
                        tmp_video,
                        start_sec=start_sec,
                        duration_sec=duration_sec,
                    )
                except Exception as e:
                    print(f"[SKIP INTERPRET FAIL] Reason: {repr(e)}")
                    skipped_interpret += 1
                    continue

                if not obs:
                    print("[SKIP EMPTY OBS]")
                    skipped_interpret += 1
                    continue

                interpretation = make_interpretation(obs, task)

                row = {
                    "video_id": vid,
                    "task_name": task,
                    "subset": subset,
                    "recipe_type": recipe_type,
                    "video_url": url,
                    "window_start_sec": start_sec,
                    "window_end_sec": start_sec + duration_sec,
                    "initial_observation": obs,
                    "interpretation": interpretation,
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                print(f"[OK] {vid}")

    return written, skipped_existing, skipped_download, skipped_interpret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--start_sec", type=int, default=0)
    parser.add_argument("--duration_sec", type=int, default=5)
    args = parser.parse_args()

    print("[1/4] Loading COIN...")
    videos = load_coin(args.coin_json, args.subset)

    if args.max_videos is not None:
        videos = videos[:args.max_videos]

    print(f"Videos loaded: {len(videos)}")

    print("[2/4] Loading existing progress...")
    processed_ids = load_processed_ids(args.output)

    print("[3/4] Loading interpretation model...")
    model = PerceptionLM()

    print("[4/4] Processing videos and writing one row per successful video...")
    written, skipped_existing, skipped_download, skipped_interpret = process_and_write(
        videos=videos,
        output_path=args.output,
        model=model,
        processed_ids=processed_ids,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
    )

    print("\nDone")
    print("Written:", written)
    print("Skipped existing:", skipped_existing)
    print("Skipped download/no-url:", skipped_download)
    print("Skipped interpretation:", skipped_interpret)


if __name__ == "__main__":
    main()