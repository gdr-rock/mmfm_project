from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from src.training.system1_conditioning import System1Conditioning, format_system1_conditioning_prefix

TARGET_START = "[TARGET_START]"
TARGET_END = "[TARGET_END]"


@dataclass
class System1Sample:
    config: str
    context: str
    goal_description: str
    goal_interpretation: str
    actions: List[str]
    delta_states: List[str]


def interleaved_pairs(actions: Sequence[str], delta_states: Sequence[str]) -> str:
    """Builds <A_i, DeltaS_i> trajectory text used as the supervised System-1 target."""
    if len(actions) != len(delta_states):
        raise ValueError("actions and delta_states must have the same length")

    lines: List[str] = []
    for i, (action, state) in enumerate(zip(actions, delta_states)):
        lines.append(f"<{i}:A> {action}")
        lines.append(f"<{i}:DeltaS> {state}")
    return "\n".join(lines)


def format_system1_target(sample: System1Sample) -> str:
    """
    Paper-aligned target sequence:
    [goal description, goal interpretation, <A0, DeltaS0>, ..., <AN, DeltaSN>]
    """
    body = "\n".join(
        [
            f"[GOAL_DESCRIPTION] {sample.goal_description}",
            f"[GOAL_INTERPRETATION] {sample.goal_interpretation}",
            "[TRAJECTORY]",
            interleaved_pairs(sample.actions, sample.delta_states),
        ]
    )
    return f"{TARGET_START}\n{body}\n{TARGET_END}"


def format_system1_prefix(config: str, context: str, goal_description: str) -> str:
    """Conditioning prefix: [config, visual context, goal] (paper-aligned)."""
    conditioning = System1Conditioning(
        config_prompt=config,
        visual_context=context,
        goal_description=goal_description,
    )
    return format_system1_conditioning_prefix(conditioning)


def build_system1_training_text(sample: System1Sample) -> Tuple[str, str]:
    """Returns (prefix, target) so loss can be masked to target tokens only."""
    prefix = format_system1_prefix(sample.config, sample.context, sample.goal_description)
    target = format_system1_target(sample)
    return prefix, target
