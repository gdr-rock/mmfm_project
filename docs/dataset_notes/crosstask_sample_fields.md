# CrossTask Sample Field Check (One Video)

Sample file: `data/crosstask_samples/ldCg4aWd4mI.toc.json`

## What this sample contains
- Top-level keys: `video_id`, `video_path`, `duration_seconds`, `root_id`, `nodes`
- Per-node keys: `id`, `level`, `start_idx`, `end_idx`, `start_sec`, `end_sec`, `caption`, `children`

## Caption vs Action
- `caption`: Present (`nodes[*].caption`)
- `action`: Not present as an explicit field in this ToC sample

## Interpretation
- This file is a Tree-of-Captions representation.
- It is useful for caption-based planning pipelines.
- If action labels are needed, they must come from separate CrossTask annotations (not this ToC JSON field set).
