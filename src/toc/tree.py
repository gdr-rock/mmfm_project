from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Node:
    id: int
    start_idx: int
    end_idx: int
    start_sec: float
    end_sec: float
    children: list[int]
    level: int
    caption: str | None = None


def _segment_sse(features: np.ndarray, i: int, j: int) -> float:
    seg = features[i : j + 1]
    if len(seg) == 0:
        return 0.0
    mu = seg.mean(axis=0)
    diff = seg - mu
    return float((diff * diff).sum())


def build_hierarchical_tree(features: np.ndarray, windows: list[tuple[float, float]]) -> tuple[dict[int, Node], int]:
    n = len(windows)
    nodes: dict[int, Node] = {}

    active: list[Node] = []
    next_id = 0
    for i, (s, e) in enumerate(windows):
        node = Node(id=next_id, start_idx=i, end_idx=i, start_sec=s, end_sec=e, children=[], level=0)
        nodes[next_id] = node
        active.append(node)
        next_id += 1

    while len(active) > 1:
        best_k = 0
        best_cost = float("inf")

        for k in range(len(active) - 1):
            a = active[k]
            b = active[k + 1]
            merged_i, merged_j = a.start_idx, b.end_idx
            merge_cost = _segment_sse(features, merged_i, merged_j)
            merge_cost -= _segment_sse(features, a.start_idx, a.end_idx)
            merge_cost -= _segment_sse(features, b.start_idx, b.end_idx)
            if merge_cost < best_cost:
                best_cost = merge_cost
                best_k = k

        left = active[best_k]
        right = active[best_k + 1]
        new_node = Node(
            id=next_id,
            start_idx=left.start_idx,
            end_idx=right.end_idx,
            start_sec=left.start_sec,
            end_sec=right.end_sec,
            children=[left.id, right.id],
            level=max(left.level, right.level) + 1,
        )
        nodes[next_id] = new_node
        next_id += 1

        active = active[:best_k] + [new_node] + active[best_k + 2 :]

    root_id = active[0].id if active else -1
    return nodes, root_id
