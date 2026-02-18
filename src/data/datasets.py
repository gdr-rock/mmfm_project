"""Dataset loaders for CrossTask/COIN (placeholder implementation)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.utils.io import ensure_dir, write_json


@dataclass
class VideoRecord:
    """Single procedural-video record used by placeholder pipeline."""

    dataset: str
    video_id: str
    task_id: str
    goal: str
    steps: List[str]


def dummy_records(dataset: str, count: int = 3) -> List[VideoRecord]:
    """Generate deterministic synthetic records for smoke tests."""
    base_goal = "make tea" if dataset.lower() == "crosstask" else "assemble shelf"
    records: List[VideoRecord] = []
    for idx in range(count):
        records.append(
            VideoRecord(
                dataset=dataset,
                video_id=f"{dataset}_video_{idx:03d}",
                task_id=f"task_{idx:03d}",
                goal=base_goal,
                steps=[
                    "prepare workspace",
                    "collect materials",
                    "execute core action",
                    "finalize and check outcome",
                ],
            )
        )
    return records


def load_dataset(dataset: str, root: str | Path = "data", use_dummy_data: bool = False) -> List[VideoRecord]:
    """Load dataset records.

    TODO: Replace this with full parsers for CrossTask and COIN metadata.
    """
    _ = Path(root)
    if use_dummy_data:
        return dummy_records(dataset)

    # Placeholder fallback until real loaders are implemented.
    return dummy_records(dataset)


def save_manifest(records: List[VideoRecord], output_path: str | Path) -> Path:
    """Save record manifests as JSON."""
    payload: Dict[str, Any] = {
        "num_records": len(records),
        "records": [asdict(r) for r in records],
    }
    output = Path(output_path)
    ensure_dir(output.parent)
    write_json(output, payload)
    return output
