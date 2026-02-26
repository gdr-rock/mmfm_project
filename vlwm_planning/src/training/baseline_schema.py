from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from src.training.baseline_conditioning import BaselineConditioning, format_baseline_conditioning_prefix

TARGET_START = "[TARGET_START]"
TARGET_END = "[TARGET_END]"


@dataclass
class BaselineSample:
    config: str
    visual_context: str
    goal_description: str
    actions: List[str]
    asr_text: Optional[str] = None


def format_baseline_action_target(actions: Sequence[str]) -> str:
    lines = ["[ACTION_TRAJECTORY]"]
    lines.extend([f"<{i}:A> {action}" for i, action in enumerate(actions)])
    body = "\n".join(lines)
    return f"{TARGET_START}\n{body}\n{TARGET_END}"


def build_baseline_training_text(sample: BaselineSample) -> Tuple[str, str]:
    conditioning = BaselineConditioning(
        config_prompt=sample.config,
        visual_context=sample.visual_context,
        goal_description=sample.goal_description,
        asr_text=sample.asr_text,
    )
    prefix = format_baseline_conditioning_prefix(conditioning)
    target = format_baseline_action_target(sample.actions)
    return prefix, target
