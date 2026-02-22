from __future__ import annotations

import argparse
import json
from pathlib import Path

from .caption import SegmentCaptioner
from .config import load_config
from .features import PerceptionFeatureEncoder, extract_temporal_features
from .subset import discover_videos, load_include_ids, select_subset
from .tree import build_hierarchical_tree


def _select_segment_frames(
    representative_frames: list,
    start_idx: int,
    end_idx: int,
    max_frames_per_segment: int,
) -> list:
    frames = representative_frames[start_idx : end_idx + 1]
    if not frames:
        return []
    if len(frames) <= max_frames_per_segment:
        return frames

    step = len(frames) / max_frames_per_segment
    picks = [frames[int(i * step)] for i in range(max_frames_per_segment)]
    return picks


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    videos_root = Path(cfg.dataset.videos_root)
    if not videos_root.exists():
        raise FileNotFoundError(f"videos_root does not exist: {videos_root}")

    out_root = Path(cfg.output.output_root)
    trees_dir = out_root / "trees"
    out_root.mkdir(parents=True, exist_ok=True)
    trees_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(cfg.dataset.videos_root, cfg.dataset.video_extensions)
    include_ids = load_include_ids(cfg.dataset.include_video_ids_file)
    subset = select_subset(videos, cfg.dataset.subset_size, cfg.dataset.seed, include_ids)
    print(f"Discovered videos: {len(videos)} | Selected subset: {len(subset)}")
    if not subset:
        raise RuntimeError(
            "No videos selected. Check videos_root, file extensions, and include_video_ids_file settings."
        )

    encoder = PerceptionFeatureEncoder(cfg.features.encoder_name)
    captioner = SegmentCaptioner(
        model_name=cfg.caption.model_name,
        fallback_model_name=cfg.caption.fallback_model_name,
        max_new_tokens=cfg.caption.max_new_tokens,
    )

    manifest_path = out_root / "subset_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for video_path in subset:
            stream = extract_temporal_features(
                video_path=str(video_path),
                encoder=encoder,
                temporal_item_seconds=cfg.features.temporal_item_seconds,
                frames_per_item=cfg.features.frame_sample_per_item,
            )
            if len(stream.segments) == 0:
                continue

            nodes, root_id = build_hierarchical_tree(stream.features, stream.segments)

            for node in nodes.values():
                dur = node.end_sec - node.start_sec
                if dur < cfg.segmentation.min_caption_seconds:
                    continue
                seg_frames = _select_segment_frames(
                    representative_frames=stream.representative_frames,
                    start_idx=node.start_idx,
                    end_idx=node.end_idx,
                    max_frames_per_segment=cfg.caption.max_frames_per_segment,
                )
                if seg_frames:
                    node.caption = captioner.caption_segment(seg_frames)

            payload = {
                "video_id": video_path.stem,
                "video_path": str(video_path),
                "duration_seconds": stream.duration_seconds,
                "root_id": root_id,
                "nodes": [
                    {
                        "id": n.id,
                        "start_idx": n.start_idx,
                        "end_idx": n.end_idx,
                        "start_sec": n.start_sec,
                        "end_sec": n.end_sec,
                        "children": n.children,
                        "level": n.level,
                        "caption": n.caption,
                    }
                    for n in sorted(nodes.values(), key=lambda x: x.id)
                ],
            }

            tree_file = trees_dir / f"{video_path.stem}.json"
            tree_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            mf.write(json.dumps({"video_id": video_path.stem, "tree_file": str(tree_file)}) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Tree-of-Captions subset dataset")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config)
