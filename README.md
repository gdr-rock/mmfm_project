# MMFM - Self-Refine Branch

This branch contains the Self-Refine pipeline for converting a Tree of Captions (ToC) JSON into structured planning data.

## Pipeline

1. Load ToC JSON (`nodes`, `root_id`).
2. Linearize the tree with DFS order.
3. Generate initial structured draft with Maverick (`feeltheAGI/Maverick-7B`).
4. Generate critique/feedback with the same model.
5. Revise using feedback.
6. Repeat for `N` iterations.
7. Parse and validate final JSON.

## Output Schema

```json
{
  "goal_description": "...",
  "goal_interpretation": {
    "initial_world_state": "...",
    "final_world_state": "..."
  },
  "action_description": ["...", "..."],
  "world_states": ["...", "..."]
}
```

## Main Script

- `scripts/self_refine/run_self_refine.py`

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch transformers accelerate sentencepiece
```

## Run

```bash
python3 scripts/self_refine/run_self_refine.py \
  --input ~/tree_of_captions/output/trees/Video_1.json \
  --output ./outputs/Video_1.plan.json \
  --model feeltheAGI/Maverick-7B \
  --task-name "cross_task_example" \
  --video-id "Video_1" \
  --iterations 2
```

## Notes

- This branch is for direct iterative self-refinement only.
- For a wrapped and validated end-to-end structured data pipeline, use branch `structured`.
