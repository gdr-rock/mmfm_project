from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.training.losses import IGNORE_INDEX

try:
    import torch
except ModuleNotFoundError:
    torch = None


@dataclass
class System1Conditioning:
    """
    Conditioning stream for paper-aligned System-1 training:
    [config/system prompt, visual context, optional auxiliary text, goal].
    """

    config_prompt: str
    visual_context: str
    goal_description: str
    auxiliary_text: Optional[str] = None


def format_system1_conditioning_prefix(conditioning: System1Conditioning) -> str:
    parts = [
        "[CONFIG]",
        conditioning.config_prompt,
        "",
        "[VISUAL_CONTEXT]",
        conditioning.visual_context,
        "",
    ]
    if conditioning.auxiliary_text:
        parts.extend(["[AUX_TEXT]", conditioning.auxiliary_text, ""])
    parts.extend(["[GOAL]", conditioning.goal_description, ""])
    return "\n".join(parts)


def build_labels_with_conditioning_mask(
    input_ids: "torch.Tensor",
    prefix_lengths: "torch.Tensor",
    ignore_index: int = IGNORE_INDEX,
) -> "torch.Tensor":
    """
    Masks all conditioning tokens so CE is applied only on target tokens:
    [goal description, goal interpretation, <A_i, DeltaS_i>].
    """
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
