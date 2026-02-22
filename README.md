# Tree of Captions (Paper-Style Pipeline)

This project implements a paper-style Tree of Captions pipeline:
1. Extract a temporal feature stream `Z = [z1, ..., zT]` from video using a Perception-style encoder.
2. Build a hierarchical tree with adjacent agglomerative merges minimizing within-segment variance increase.
3. Caption each segment node (except very short ones) with PerceptionLM.
4. Save one hierarchical caption tree JSON per video.

## Default models
- Feature encoder: `timm/PE-Core-B-16`
- Caption model: `facebook/Perception-LM-3B`

## Configs
- `configs/subset_example.yaml`: main paper-style run config.
- `configs/smoke_local.yaml`: smaller subset run, still paper-style structure.

## Run
```powershell
cd tree_of_captions
.\scripts\create_env.ps1
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
python -m src.toc.main --config configs/subset_example.yaml
```

## Cluster Download (One Video)
If your local internet is limited, run downloads over SSH so data uses cluster bandwidth:

```bash
bash scripts/setup_crosstask_one_video.sh "$HOME/datasets/crosstask"
```

Then set these values in `configs/subset_example.yaml`:
- `dataset.videos_root: "/home/<user>/datasets/crosstask/videos"`
- `dataset.subset_size: 1`
- `dataset.include_video_ids_file: "configs/video_ids_example.txt"`

Before running, edit `configs/subset_example.yaml`:
- Set `dataset.videos_root` to your local video root.
- Optional fixed subset: set `dataset.include_video_ids_file` to `configs/video_ids_example.txt`.
- Optional random subset: set `dataset.subset_size`.

## Key paper-style parameters
- `features.temporal_item_seconds`: base temporal granularity for feature stream items.
- `features.frame_sample_per_item`: sampled frames per temporal item for encoder features.
- `segmentation.min_caption_seconds`: minimum segment duration to caption.
- `caption.max_frames_per_segment`: keyframes sent to caption model per segment node.

## Output
- `output/subset_manifest.jsonl`
- `output/trees/<video_id>.json`
