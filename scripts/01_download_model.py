#!/usr/bin/env python3
"""Download model artifacts from URL or Hugging Face Hub."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch encoder/caption model artifacts.")
    parser.add_argument("--url", type=str, default="")
    parser.add_argument("--output", type=str, default="checkpoints/perception_encoder.pt")
    parser.add_argument("--hf_repo_id", type=str, default="")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--target_dir", type=str, default="checkpoints/perceptionlm")
    parser.add_argument("--allow_patterns", type=str, default="")
    parser.add_argument("--use_hf", action="store_true")
    parser.add_argument("--create_placeholder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_hf:
        if not args.hf_repo_id:
            raise SystemExit("Provide --hf_repo_id when using --use_hf.")
        try:
            from huggingface_hub import snapshot_download
        except Exception as error:  # noqa: BLE001
            raise SystemExit(
                "huggingface_hub is required for --use_hf. Install requirements first."
            ) from error

        target_dir = Path(args.target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        allow_patterns = None
        if args.allow_patterns.strip():
            allow_patterns = [value.strip() for value in args.allow_patterns.split(",") if value.strip()]

        print(f"Downloading {args.hf_repo_id}@{args.revision} to {target_dir}")
        snapshot_download(
            repo_id=args.hf_repo_id,
            revision=args.revision,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )
        print(f"Saved model directory: {target_dir}")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.url:
        print(f"Downloading checkpoint from {args.url}")
        urllib.request.urlretrieve(args.url, output_path)  # noqa: S310
        print(f"Saved checkpoint: {output_path}")
        return

    if args.create_placeholder:
        output_path.write_bytes(b"placeholder-checkpoint")
        print(f"Created placeholder checkpoint: {output_path}")
        return

    raise SystemExit("Provide --url <checkpoint_url> or use --create_placeholder.")


if __name__ == "__main__":
    main()
