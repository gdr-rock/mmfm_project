#!/usr/bin/env python3
"""Build video URL list directly from official CrossTask release metadata."""

from __future__ import annotations

import argparse
import csv
import re
import urllib.request
import urllib.error
import zipfile
from pathlib import Path


DEFAULT_RELEASE_URL = "https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip"
WATCH_RE = re.compile(r"https?://(www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]{11})")
SHORT_RE = re.compile(r"https?://youtu\.be/([A-Za-z0-9_-]{11})")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CrossTask URL list from official release zip.")
    parser.add_argument("--release_url", type=str, default=DEFAULT_RELEASE_URL)
    parser.add_argument("--download_dir", type=str, default="data/crosstask")
    parser.add_argument("--zip_name", type=str, default="crosstask_release.zip")
    parser.add_argument("--extract_dir", type=str, default="data/crosstask/release")
    parser.add_argument("--output_url_list", type=str, default="data/crosstask/video_urls.txt")
    parser.add_argument("--max_videos", type=int, default=5)
    parser.add_argument("--prefer_val_split", action="store_true")
    parser.add_argument("--overwrite_zip", action="store_true")
    return parser.parse_args()


def download_release(url: str, zip_path: Path, overwrite: bool) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists() and not overwrite:
        print(f"Using cached release zip: {zip_path}")
        return
    print(f"Downloading official CrossTask release: {url}")
    try:
        urllib.request.urlretrieve(url, zip_path)  # noqa: S310
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Failed to download official CrossTask release. "
            "Check internet/DNS access or manually place the release zip at "
            f"{zip_path} and rerun."
        ) from error
    print(f"Saved: {zip_path}")


def extract_release(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting: {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as handle:
        handle.extractall(extract_dir)


def normalize_video_reference(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    watch = WATCH_RE.search(text)
    if watch:
        return f"https://www.youtube.com/watch?v={watch.group(2)}"
    short = SHORT_RE.search(text)
    if short:
        return f"https://www.youtube.com/watch?v={short.group(1)}"
    if ID_RE.match(text):
        return f"https://www.youtube.com/watch?v={text}"
    return ""


def extract_urls_from_csv(csv_path: Path) -> list[str]:
    urls: list[str] = []
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            for cell in row:
                cleaned = normalize_video_reference(cell)
                if cleaned:
                    urls.append(cleaned)
    return urls


def collect_csv_files(root: Path) -> list[Path]:
    candidates = [path for path in root.rglob("*.csv") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No CSV files found under {root}")
    return sorted(candidates)


def choose_csv_files(csv_files: list[Path], prefer_val_split: bool) -> list[Path]:
    if not prefer_val_split:
        return csv_files
    val_first = []
    others = []
    for path in csv_files:
        if "videos_val" in path.name.lower():
            val_first.append(path)
        else:
            others.append(path)
    return val_first + others


def unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    max_videos = max(1, int(args.max_videos))

    zip_path = Path(args.download_dir) / args.zip_name
    extract_dir = Path(args.extract_dir)
    output_path = Path(args.output_url_list)

    try:
        download_release(args.release_url, zip_path, overwrite=args.overwrite_zip)
        extract_release(zip_path, extract_dir)

        csv_files = collect_csv_files(extract_dir)
        csv_files = choose_csv_files(csv_files, prefer_val_split=args.prefer_val_split)

        all_urls: list[str] = []
        for csv_file in csv_files:
            all_urls.extend(extract_urls_from_csv(csv_file))

        urls = unique_keep_order(all_urls)
        if not urls:
            names = ", ".join(path.name for path in csv_files[:10])
            raise RuntimeError(
                "No YouTube references found in extracted CrossTask CSV files. "
                f"Checked examples: {names}"
            )

        selected = urls[:max_videos]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(selected) + "\n", encoding="utf-8")

        print(f"Found {len(urls)} official CrossTask video URLs.")
        print(f"Wrote {len(selected)} URL(s) to {output_path}")
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
