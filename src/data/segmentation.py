"""Step/window segmentation placeholders for procedural videos."""

from __future__ import annotations

from typing import Dict, List

from src.data.datasets import VideoRecord


def segment_record(record: VideoRecord, window_seconds: int = 5) -> List[Dict[str, object]]:
    """Generate synthetic contiguous segments from step labels.

    TODO: Replace with real timestamp-based segmentation from annotations.
    """
    segments: List[Dict[str, object]] = []
    current_start = 0
    for step_idx, step in enumerate(record.steps):
        start = current_start
        end = start + window_seconds
        segments.append(
            {
                "video_id": record.video_id,
                "segment_id": f"{record.video_id}_seg_{step_idx:03d}",
                "step_text": step,
                "start_sec": start,
                "end_sec": end,
            }
        )
        current_start = end
    return segments


def validate_segments(segments: List[Dict[str, object]]) -> bool:
    """Basic contiguity and non-negative duration checks."""
    prev_end = 0
    for seg in segments:
        start = int(seg["start_sec"])
        end = int(seg["end_sec"])
        if end <= start:
            return False
        if start < prev_end:
            return False
        prev_end = end
    return True
