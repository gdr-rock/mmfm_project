from __future__ import annotations

import random
from pathlib import Path


def discover_videos(videos_root: str, exts: list[str]) -> list[Path]:
    root = Path(videos_root)
    allowed = {e.lower() for e in exts}
    videos = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed]
    videos.sort()
    return videos


def load_include_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    ids = {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}
    return ids


def select_subset(videos: list[Path], subset_size: int | None, seed: int, include_ids: set[str] | None) -> list[Path]:
    if include_ids is not None:
        videos = [v for v in videos if v.stem in include_ids]

    if subset_size is None or subset_size >= len(videos):
        return videos

    rnd = random.Random(seed)
    picked = videos[:]
    rnd.shuffle(picked)
    return sorted(picked[:subset_size])
