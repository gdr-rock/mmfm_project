"""Frozen JEPA-style encoder wrapper placeholder."""

from __future__ import annotations

import hashlib
from typing import Iterable, List

import numpy as np


class FrozenJEPAWrapper:
    """Minimal deterministic encoder API used by stage scripts.

    TODO: Replace deterministic hashing with real V-JEPA2 forward passes.
    """

    def __init__(self, latent_dim: int = 256, frozen: bool = True) -> None:
        self.latent_dim = latent_dim
        self.frozen = frozen

    def _seed_from_text(self, text: str) -> int:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def encode_segments(self, segment_ids: Iterable[str]) -> np.ndarray:
        """Encode segment identifiers into deterministic latent vectors."""
        vectors: List[np.ndarray] = []
        for seg_id in segment_ids:
            rng = np.random.default_rng(self._seed_from_text(seg_id))
            vectors.append(rng.normal(size=(self.latent_dim,)).astype(np.float32))
        if not vectors:
            return np.zeros((0, self.latent_dim), dtype=np.float32)
        return np.stack(vectors, axis=0)

    def transitions(self, z_t: np.ndarray) -> np.ndarray:
        """Compute adjacent latent deltas."""
        if len(z_t) < 2:
            return np.zeros((0, self.latent_dim), dtype=np.float32)
        return z_t[1:] - z_t[:-1]
