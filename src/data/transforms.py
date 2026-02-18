"""Frame sampling and transform placeholders."""

from __future__ import annotations

from typing import List


def sample_frame_indices(num_frames: int, num_samples: int) -> List[int]:
    """Uniformly sample frame indices.

    TODO: Add temporal jitter and multi-view sampling for robustness experiments.
    """
    if num_frames <= 0 or num_samples <= 0:
        return []
    if num_samples >= num_frames:
        return list(range(num_frames))

    step = num_frames / float(num_samples)
    indices = [min(int(i * step), num_frames - 1) for i in range(num_samples)]
    return sorted(set(indices))
