#!/usr/bin/env python3
"""Process CrossTask videos one-by-one with download, encode, and caption tree export."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.io import ensure_dir, read_json, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download up to N CrossTask videos, run stages, and export caption trees."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default="crosstask", choices=["crosstask"])
    parser.add_argument("--output_dir", type=str, default="outputs/streaming")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--url_list", type=str, default="data/crosstask/video_urls.txt")
    parser.add_argument("--max_videos", type=int, default=5)
    parser.add_argument("--download_dir", type=str, default="/tmp/crosstask_stream")
    parser.add_argument("--keep_downloaded", action="store_true")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--goal", type=str, default="complete the procedure")
    parser.add_argument("--k_candidates", type=int, default=8)
    parser.add_argument("--models_config", type=str, default="configs/models.yaml")
    parser.add_argument("--planning_config", type=str, default="configs/default.yaml")
    parser.add_argument("--yt_dlp_format", type=str, default="bv*[height<=360]+ba/b[height<=360]")
    return parser.parse_args()


def load_urls(url_list_path: Path) -> list[str]:
    if not url_list_path.exists():
        raise FileNotFoundError(f"URL list not found: {url_list_path}")

    urls: list[str] = []
    with url_list_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            urls.append(value)
    return urls


def short_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def run_command(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def newest_file(path: Path) -> Path:
    files = [item for item in path.glob("*") if item.is_file()]
    if not files:
        raise FileNotFoundError(f"No downloaded file found in {path}")
    return max(files, key=lambda item: item.stat().st_mtime)


def download_one(url: str, args: argparse.Namespace, logger_name: str) -> Path:
    logger = setup_logging(logger_name)
    download_dir = ensure_dir(args.download_dir)
    if args.skip_download:
        placeholder = download_dir / f"{short_key(url)}.mp4"
        placeholder.write_bytes(b"")
        logger.info("Skip-download mode created placeholder: %s", placeholder)
        return placeholder

    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp not found. Install it first: pip install yt-dlp")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        args.yt_dlp_format,
        "-o",
        str(download_dir / "%(id)s.%(ext)s"),
        url,
    ]
    logger.info("Downloading: %s", url)
    run_command(cmd, dry_run=args.dry_run)
    video_path = newest_file(download_dir)
    logger.info("Downloaded file: %s", video_path)
    return video_path


def run_stage_script(script_name: str, stage_args: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, str(Path(ROOT) / "scripts" / script_name)]
    cmd.extend(stage_args)
    run_command(cmd, dry_run=dry_run)


def build_caption_tree(per_video_output: Path, video_key: str, source_url: str, goal: str) -> Path:
    parsed_path = per_video_output / "plans" / "parsed_steps_crosstask.json"
    state_path = per_video_output / "plans" / "state_changes_crosstask.json"
    parsed_payload = read_json(parsed_path)
    state_payload = read_json(state_path)

    state_map: dict[int, list[str]] = {}
    for item in state_payload.get("state_changes", []):
        candidate_id = int(item.get("candidate_id", 0))
        state_map[candidate_id] = [str(value) for value in item.get("delta_s", [])]

    tree_candidates = []
    for item in parsed_payload.get("parsed_steps", []):
        candidate_id = int(item.get("candidate_id", 0))
        steps = [str(step) for step in item.get("steps", [])]
        deltas = state_map.get(candidate_id, [])
        nodes = []
        for idx, step in enumerate(steps, start=1):
            nodes.append(
                {
                    "step_index": idx,
                    "step": step,
                    "delta_s": deltas[idx - 1] if idx - 1 < len(deltas) else "",
                }
            )
        tree_candidates.append({"candidate_id": candidate_id, "nodes": nodes})

    tree = {
        "video_key": video_key,
        "source_url": source_url,
        "goal": goal,
        "candidates": tree_candidates,
    }
    tree_path = per_video_output / "plans" / "caption_tree_crosstask.json"
    write_json(tree_path, tree)
    return tree_path


def main() -> None:
    args = parse_args()
    logger = setup_logging("stream_crosstask_pipeline")
    set_seed(args.seed)

    urls = load_urls(Path(args.url_list))
    selected_urls = urls[: args.max_videos]
    if not selected_urls:
        raise ValueError(f"No URLs found in {args.url_list}")

    output_root = ensure_dir(args.output_dir)
    processed: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []

    for index, url in enumerate(selected_urls, start=1):
        video_key = f"video_{index:02d}_{short_key(url)}"
        per_video_output = ensure_dir(output_root / video_key)
        downloaded_path: Path | None = None
        logger.info("Processing %d/%d: %s", index, len(selected_urls), video_key)

        try:
            downloaded_path = download_one(url, args, "stream_crosstask_download")

            run_stage_script(
                "02_cache_jepa_latents.py",
                [
                    "--config",
                    args.models_config,
                    "--dataset",
                    "crosstask",
                    "--output_dir",
                    str(per_video_output),
                    "--seed",
                    str(args.seed),
                ]
                + (["--dry_run"] if args.dry_run else [])
                + (["--use_dummy_data"] if args.use_dummy_data else []),
                dry_run=args.dry_run,
            )

            run_stage_script(
                "03_generate_candidates.py",
                [
                    "--config",
                    args.planning_config,
                    "--dataset",
                    "crosstask",
                    "--output_dir",
                    str(per_video_output),
                    "--seed",
                    str(args.seed),
                    "--goal",
                    args.goal,
                    "--k_candidates",
                    str(args.k_candidates),
                ]
                + (["--dry_run"] if args.dry_run else [])
                + (["--use_dummy_data"] if args.use_dummy_data else []),
                dry_run=args.dry_run,
            )

            run_stage_script(
                "04_generate_state_changes.py",
                [
                    "--config",
                    args.planning_config,
                    "--dataset",
                    "crosstask",
                    "--output_dir",
                    str(per_video_output),
                    "--seed",
                    str(args.seed),
                    "--goal",
                    args.goal,
                ]
                + (["--dry_run"] if args.dry_run else [])
                + (["--use_dummy_data"] if args.use_dummy_data else []),
                dry_run=args.dry_run,
            )

            tree_path = build_caption_tree(per_video_output, video_key, url, args.goal)
            processed.append(
                {
                    "video_key": video_key,
                    "source_url": url,
                    "tree_path": str(tree_path),
                    "output_dir": str(per_video_output),
                }
            )
            logger.info("Caption tree saved: %s", tree_path)

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed %s: %s", video_key, str(exc))
            failed.append({"video_key": video_key, "source_url": url, "error": str(exc)})
            if args.fail_fast:
                raise

        finally:
            if downloaded_path is not None and downloaded_path.exists() and not args.keep_downloaded:
                downloaded_path.unlink()
                logger.info("Deleted downloaded video: %s", downloaded_path)

    summary = {
        "requested_videos": args.max_videos,
        "processed_count": len(processed),
        "failed_count": len(failed),
        "processed": processed,
        "failed": failed,
    }
    summary_path = output_root / "streaming_summary.json"
    write_json(summary_path, summary)
    logger.info("Streaming run complete. Summary: %s", summary_path)


if __name__ == "__main__":
    main()
