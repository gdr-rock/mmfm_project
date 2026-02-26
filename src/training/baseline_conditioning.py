from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.training.losses import IGNORE_INDEX

try:
    import torch
except ModuleNotFoundError:
    torch = None


@dataclass
class BaselineConditioning:
    """
    Paper-faithful baseline conditioning stream:
    [config/system prompt, visual context, optional ASR text, goal].
    """

    config_prompt: str
    visual_context: str
    goal_description: str
    asr_text: Optional[str] = None


def format_baseline_conditioning_prefix(conditioning: BaselineConditioning) -> str:
    parts = [
        "[CONFIG]",
        conditioning.config_prompt,
        "",
        "[VISUAL_CONTEXT]",
        conditioning.visual_context,
        "",
    ]
    if conditioning.asr_text:
        parts.extend(["[ASR]", conditioning.asr_text, ""])
    parts.extend(["[GOAL]", conditioning.goal_description, ""])
    return "\n".join(parts)


def build_labels_with_baseline_conditioning_mask(
    input_ids: "torch.Tensor",
    prefix_lengths: "torch.Tensor",
    ignore_index: int = IGNORE_INDEX,
) -> "torch.Tensor":
    """Masks conditioning tokens; CE is applied only to action-sequence target tokens."""
    if torch is None:
        raise ModuleNotFoundError("torch is required for label masking")

    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [B, T]")
    if prefix_lengths.ndim != 1 or prefix_lengths.size(0) != input_ids.size(0):
        raise ValueError("prefix_lengths must be [B]")

    labels = input_ids.clone()
    bsz, seq_len = labels.shape
    for b in range(bsz):
        p = int(prefix_lengths[b].item())
        p = max(0, min(p, seq_len))
        labels[b, :p] = ignore_index
    return labels
