#!/usr/bin/env python3
"""Normalize and validate YouTube URL lists for CrossTask runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


YOUTUBE_WATCH_RE = re.compile(r"^https?://(www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]{11})")
YOUTUBE_SHORT_RE = re.compile(r"^https?://youtu\.be/([A-Za-z0-9_-]{11})")
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare clean video URL list for pipeline.")
    parser.add_argument("--input", type=str, default="data/crosstask/video_urls.raw.txt")
    parser.add_argument("--output", type=str, default="data/crosstask/video_urls.txt")
    parser.add_argument("--max_videos", type=int, default=5)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def normalize_entry(value: str) -> str:
    line = value.strip()
    if not line or line.startswith("#"):
        return ""
    if line in {"---", "--", "-"}:
        return ""
    if "VIDEO_ID_" in line:
        raise ValueError(f"Placeholder found: {line}")

    watch_match = YOUTUBE_WATCH_RE.match(line)
    if watch_match:
        video_id = watch_match.group(2)
        return f"https://www.youtube.com/watch?v={video_id}"

    short_match = YOUTUBE_SHORT_RE.match(line)
    if short_match:
        video_id = short_match.group(1)
        return f"https://www.youtube.com/watch?v={video_id}"

    if YOUTUBE_ID_RE.match(line):
        return f"https://www.youtube.com/watch?v={line}"

    raise ValueError(f"Invalid YouTube entry: {line}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    normalized: list[str] = []
    invalid: list[str] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        try:
            cleaned = normalize_entry(line)
            if cleaned:
                normalized.append(cleaned)
        except ValueError as error:
            invalid.append(str(error))

    unique_urls: list[str] = []
    seen: set[str] = set()
    for url in normalized:
        if url in seen:
            continue
        seen.add(url)
        unique_urls.append(url)

    selected = unique_urls[: max(1, args.max_videos)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")

    print(f"Wrote {len(selected)} URL(s) to {output_path}")
    if invalid:
        print(f"Skipped {len(invalid)} invalid line(s)")
        for index, message in enumerate(invalid[:10], start=1):
            print(f"  {index}. {message}")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
