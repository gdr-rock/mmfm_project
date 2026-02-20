#!/usr/bin/env python3
"""CrossTask automation: download -> PerceptionLM captions/tree -> optional latent cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automatic one-by-one processing for N videos.")
    parser.add_argument("--url_list", type=str, default="data/crosstask/video_urls.txt")
    parser.add_argument("--max_videos", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="outputs/streaming")
    parser.add_argument("--download_dir", type=str, default="/tmp/crosstask_stream")
    parser.add_argument("--mode", type=str, default="captions_only", choices=["captions_only", "latents_only", "both"])
    parser.add_argument("--goal", type=str, default="complete the procedure")
    parser.add_argument("--num_caption_steps", type=int, default=10)
    parser.add_argument("--caption_model_id", type=str, default="")
    parser.add_argument("--caption_model_path", type=str, default="checkpoints/perceptionlm")
    parser.add_argument("--strict_caption_model", action="store_true")
    parser.add_argument("--encoder_checkpoint", type=str, default="checkpoints/perception_encoder.pt")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--yt_dlp_format", type=str, default="bv*[height<=360]+ba/b[height<=360]")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--keep_downloaded", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args()


def read_urls(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"URL list not found: {path}")
    urls: list[str] = []
    invalid: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if value in {"---", "--", "-"}:
            continue
        if "VIDEO_ID_" in value:
            invalid.append(value)
            continue
        match_watch = re.match(r"^https?://(www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]{11})", value)
        if match_watch:
            urls.append(f"https://www.youtube.com/watch?v={match_watch.group(2)}")
            continue
        match_short = re.match(r"^https?://youtu\.be/([A-Za-z0-9_-]{11})", value)
        if match_short:
            urls.append(f"https://www.youtube.com/watch?v={match_short.group(1)}")
            continue
        if re.match(r"^[A-Za-z0-9_-]{11}$", value):
            urls.append(f"https://www.youtube.com/watch?v={value}")
            continue
        invalid.append(value)

    if invalid:
        preview = ", ".join(invalid[:5])
        raise ValueError(
            "Invalid URL entries detected. Use real YouTube URLs or 11-char IDs. "
            f"Examples: {preview}"
        )
    return urls


def short_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_download_dir(path: Path) -> None:
    for item in path.glob("*"):
        if item.is_file():
            item.unlink()


def newest_file(path: Path) -> Path:
    files = [item for item in path.glob("*") if item.is_file()]
    if not files:
        raise FileNotFoundError(f"No downloaded file in {path}")
    return max(files, key=lambda item: item.stat().st_mtime)


def run_command(command: list[str], dry_run: bool) -> None:
    if dry_run:
        return
    subprocess.run(command, check=True)


def download_video(url: str, download_dir: Path, fmt: str, skip_download: bool, dry_run: bool) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    clear_download_dir(download_dir)
    if skip_download:
        placeholder = download_dir / f"{short_key(url)}.mp4"
        placeholder.write_bytes(b"placeholder-video")
        return placeholder
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp not found. Install with `python3 -m pip install yt-dlp`.")
    command = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        fmt,
        "-o",
        str(download_dir / "%(id)s.%(ext)s"),
        url,
    ]
    run_command(command, dry_run)
    return newest_file(download_dir)


def parse_numbered_steps(text: str, target_count: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    steps: list[str] = []
    for line in lines:
        cleaned = re.sub(r"^\s*(\d+[\).\:-]|[-*•])\s*", "", line).strip()
        if cleaned and len(cleaned) > 2:
            steps.append(cleaned)
    if not steps:
        chunks = re.split(r"[\n\.]+", text)
        for chunk in chunks:
            cleaned = chunk.strip()
            if cleaned and len(cleaned) > 4:
                steps.append(cleaned)
    if not steps:
        steps = [
            "prepare materials",
            "set up workspace",
            "perform core action",
            "verify result",
        ]
    return steps[: max(1, target_count)]


def chunk_high_level(steps: list[str], group_count: int = 3) -> list[dict]:
    if not steps:
        return []
    groups: list[list[str]] = [[] for _ in range(group_count)]
    for index, step in enumerate(steps):
        groups[index * group_count // len(steps)].append(step)
    output: list[dict] = []
    for index, group in enumerate(groups, start=1):
        if not group:
            continue
        title = f"Phase {index}: {group[0]}"
        output.append({"phase_id": index, "title": title, "detailed_steps": group})
    return output


def deterministic_rng_seed(video_path: Path, seed: int) -> int:
    payload = f"{video_path.name}:{video_path.stat().st_size}:{seed}"
    return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8], 16)


def encode_video_placeholder(video_path: Path, latent_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(deterministic_rng_seed(video_path, seed))
    z_t = rng.normal(size=(8, latent_dim)).astype(np.float32)
    delta_z = z_t[1:] - z_t[:-1]
    return z_t, delta_z


class PerceptionLMCaptioner:
    def __init__(
        self,
        model_id: str,
        model_path: str,
        device: str,
        max_new_tokens: int,
        strict: bool,
    ) -> None:
        self.model_id = model_id.strip()
        self.model_path = Path(model_path)
        self.device_preference = device
        self.max_new_tokens = max_new_tokens
        self.strict = strict
        self.backend = "fallback"
        self.ready = False
        self._model = None
        self._processor = None
        self._torch = None
        self._device = "cpu"
        self._load()

    def _resolve_source(self) -> str:
        if self.model_id:
            return self.model_id
        if self.model_path.exists():
            return str(self.model_path)
        return ""

    def _load(self) -> None:
        source = self._resolve_source()
        if not source:
            if self.strict:
                raise RuntimeError(
                    "Caption model not found. "
                    f"--caption_model_path '{self.model_path}' does not exist. "
                    "Set --caption_model_id or download model files to that path."
                )
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor
        except Exception as error:  # noqa: BLE001
            if self.strict:
                raise RuntimeError("Transformers stack unavailable. Install requirements.") from error
            return

        if self.device_preference == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device_preference
        dtype = torch.float16 if device == "cuda" else torch.float32

        try:
            processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
            try:
                model = AutoModelForVision2Seq.from_pretrained(
                    source,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    source,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
            model.to(device)
            model.eval()
        except Exception as error:  # noqa: BLE001
            if self.strict:
                raise RuntimeError(f"Failed to load caption model from {source}") from error
            return

        self._processor = processor
        self._model = model
        self._torch = torch
        self._device = device
        self.backend = "perceptionlm"
        self.ready = True

    def _extract_frames(self, video_path: Path, frame_count: int = 6) -> list:
        try:
            import cv2
            from PIL import Image
        except Exception as error:  # noqa: BLE001
            if self.strict:
                raise RuntimeError("opencv-python-headless and Pillow are required for caption model inference.") from error
            return []

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            if self.strict:
                raise RuntimeError(f"Could not open video: {video_path}")
            return []

        total_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        indices = sorted(set(int(i * (total_frames - 1) / max(1, frame_count - 1)) for i in range(frame_count)))
        images = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
        capture.release()
        return images

    def _fallback_steps(self, goal: str, target_steps: int) -> tuple[list[str], str]:
        seed = int(hashlib.sha1(goal.encode("utf-8")).hexdigest()[:8], 16)
        random.seed(seed)
        verbs = [
            "prepare",
            "arrange",
            "perform",
            "inspect",
            "adjust",
            "finalize",
        ]
        nouns = [
            "materials",
            "workspace",
            "main action",
            "intermediate result",
            "quality checks",
            "final output",
        ]
        steps = []
        for _ in range(max(1, target_steps)):
            steps.append(f"{random.choice(verbs)} the {random.choice(nouns)} for goal '{goal}'")
        return steps, "fallback-generated"

    def caption_video(self, video_path: Path, goal: str, target_steps: int) -> tuple[list[str], str]:
        prompt = (
            f"You are PerceptionLM. Goal: {goal}. "
            f"Return a numbered list of {target_steps} detailed visual steps."
        )
        if not self.ready:
            return self._fallback_steps(goal, target_steps)

        frames = self._extract_frames(video_path)
        if not frames:
            if self.strict:
                raise RuntimeError("No frames extracted for caption inference.")
            return self._fallback_steps(goal, target_steps)

        try:
            inputs = self._processor(text=prompt, images=frames, return_tensors="pt")
            if isinstance(inputs, dict):
                moved = {}
                for key, value in inputs.items():
                    if hasattr(value, "to"):
                        moved[key] = value.to(self._device)
                    else:
                        moved[key] = value
                inputs = moved
            with self._torch.inference_mode():
                outputs = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
            caption_text = decoded[0] if decoded else ""
        except Exception as error:  # noqa: BLE001
            if self.strict:
                raise RuntimeError("PerceptionLM inference failed.") from error
            return self._fallback_steps(goal, target_steps)

        steps = parse_numbered_steps(caption_text, target_steps)
        return steps, caption_text


def build_caption_tree(goal: str, detailed_steps: list[str]) -> dict:
    phase_nodes = chunk_high_level(detailed_steps, group_count=3)
    detail_nodes = []
    for index, step in enumerate(detailed_steps, start=1):
        detail_nodes.append(
            {
                "step_index": index,
                "caption": step,
                "state_change": f"after '{step}', the environment advances toward goal completion",
            }
        )
    return {
        "goal": goal,
        "high_level_phases": phase_nodes,
        "detailed_nodes": detail_nodes,
    }


def main() -> None:
    args = parse_args()
    run_captions = args.mode in {"captions_only", "both"}
    run_latents = args.mode in {"latents_only", "both"}

    if run_latents and not Path(args.encoder_checkpoint).exists():
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {args.encoder_checkpoint}. "
            "Download it with scripts/01_download_model.py."
        )

    urls = read_urls(Path(args.url_list))
    selected_urls = urls[: args.max_videos]
    if not selected_urls:
        raise ValueError("No URLs found to process.")

    captioner = None
    if run_captions:
        captioner = PerceptionLMCaptioner(
            model_id=args.caption_model_id,
            model_path=args.caption_model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            strict=args.strict_caption_model,
        )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    download_dir = Path(args.download_dir)

    processed = []
    failed = []

    for index, url in enumerate(selected_urls, start=1):
        video_key = f"video_{index:02d}_{short_key(url)}"
        video_dir = output_root / video_key
        downloaded_file: Path | None = None

        try:
            downloaded_file = download_video(
                url=url,
                download_dir=download_dir,
                fmt=args.yt_dlp_format,
                skip_download=args.skip_download,
                dry_run=args.dry_run,
            )

            tree_path = ""
            cache_path = ""
            raw_caption_path = ""
            backend = captioner.backend if captioner is not None else ""

            if run_captions and captioner is not None:
                steps, raw_caption = captioner.caption_video(
                    video_path=downloaded_file,
                    goal=args.goal,
                    target_steps=args.num_caption_steps,
                )
                tree = build_caption_tree(goal=args.goal, detailed_steps=steps)
                plans_dir = video_dir / "plans"
                tree_path = str(plans_dir / "caption_tree_crosstask.json")
                raw_caption_path = str(plans_dir / "detailed_captions.txt")
                write_json(Path(tree_path), tree)
                Path(raw_caption_path).parent.mkdir(parents=True, exist_ok=True)
                Path(raw_caption_path).write_text(raw_caption, encoding="utf-8")

            if run_latents:
                z_t, delta_z = encode_video_placeholder(
                    video_path=downloaded_file,
                    latent_dim=args.latent_dim,
                    seed=args.seed,
                )
                cache_dir = video_dir / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = str(cache_dir / "perception_encoder_latents.npz")
                np.savez(cache_path, z_t=z_t, delta_z=delta_z)

            processed.append(
                {
                    "video_key": video_key,
                    "source_url": url,
                    "mode": args.mode,
                    "caption_backend": backend,
                    "tree_path": tree_path,
                    "raw_caption_path": raw_caption_path,
                    "latent_cache_path": cache_path,
                }
            )
            print(f"[OK] {video_key} processed")

        except Exception as error:  # noqa: BLE001
            failed.append({"video_key": video_key, "source_url": url, "error": str(error)})
            print(f"[FAIL] {video_key}: {error}")
            if args.fail_fast:
                raise

        finally:
            if downloaded_file is not None and downloaded_file.exists() and not args.keep_downloaded:
                downloaded_file.unlink()

    summary = {
        "requested_videos": args.max_videos,
        "processed_count": len(processed),
        "failed_count": len(failed),
        "processed": processed,
        "failed": failed,
    }
    summary_path = output_root / "streaming_summary.json"
    write_json(summary_path, summary)
    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
