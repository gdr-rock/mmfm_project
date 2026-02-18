"""Bridge placeholders for grounding text in latent transition space."""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np


def _seed(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class TransitionBridge:
    """Placeholder bridge F mapping textual state change to latent delta."""

    def __init__(self, latent_dim: int = 256) -> None:
        self.latent_dim = latent_dim

    def predict_delta(self, state_change: str) -> np.ndarray:
        """Produce deterministic pseudo-latent delta vector from text."""
        rng = np.random.default_rng(_seed(state_change))
        return rng.normal(size=(self.latent_dim,)).astype(np.float32)

    def transition_penalty(self, state_changes: List[str], true_deltas: np.ndarray) -> float:
        """Mean cosine-distance style penalty vs true delta sequence."""
        if len(state_changes) == 0 or len(true_deltas) == 0:
            return 1.0

        penalties = []
        for idx, change in enumerate(state_changes[: len(true_deltas)]):
            pred = self.predict_delta(change)
            truth = true_deltas[idx]
            denom = (np.linalg.norm(pred) * np.linalg.norm(truth)) + 1e-8
            cosine = float(np.dot(pred, truth) / denom)
            penalties.append(1.0 - cosine)
        return float(np.mean(penalties)) if penalties else 1.0


class GoalBridge:
    """Placeholder bridge G mapping goal text to latent goal vector."""

    def __init__(self, latent_dim: int = 256) -> None:
        self.latent_dim = latent_dim

    def encode_goal(self, goal_text: str) -> np.ndarray:
        """Deterministically encode goal text into pseudo-latent vector."""
        rng = np.random.default_rng(_seed(goal_text + "_goal"))
        return rng.normal(size=(self.latent_dim,)).astype(np.float32)

    def goal_distance(self, predicted_final: np.ndarray, goal_text: str) -> float:
        """Compute L2 distance to encoded goal latent."""
        goal_latent = self.encode_goal(goal_text)
        return float(np.linalg.norm(predicted_final - goal_latent))
