from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DatasetConfig:
    videos_root: str
    subset_size: Optional[int]
    seed: int
    include_video_ids_file: Optional[str]
    video_extensions: list[str]


@dataclass
class FeaturesConfig:
    temporal_item_seconds: float
    frame_sample_per_item: int
    encoder_name: str


@dataclass
class SegmentationConfig:
    min_caption_seconds: float


@dataclass
class CaptionConfig:
    model_name: str
    fallback_model_name: Optional[str]
    max_new_tokens: int
    max_frames_per_segment: int


@dataclass
class OutputConfig:
    output_root: str


@dataclass
class AppConfig:
    dataset: DatasetConfig
    features: FeaturesConfig
    segmentation: SegmentationConfig
    caption: CaptionConfig
    output: OutputConfig


def load_config(path: str | Path) -> AppConfig:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    return AppConfig(
        dataset=DatasetConfig(**raw["dataset"]),
        features=FeaturesConfig(**raw["features"]),
        segmentation=SegmentationConfig(**raw["segmentation"]),
        caption=CaptionConfig(**raw["caption"]),
        output=OutputConfig(**raw["output"]),
    )
