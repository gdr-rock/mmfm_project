# MMFM - Structured Data Branch

This branch contains a structured-data pipeline wrapper built on top of Self-Refine.

## Pipeline

1. Accept ToC JSON as input.
2. Call the Self-Refine stage to generate draft/feedback/revision iteratively.
3. Validate final output against strict schema.
4. Write validated structured JSON.
5. Save per-run metadata/artifacts for traceability.

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

## Main Scripts

- `scripts/self_refine/run_self_refine.py`
- `scripts/structured_data_pipeline/run_structured_data_pipeline.py`

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch transformers accelerate sentencepiece
```

## Run Structured Pipeline

```bash
python3 scripts/structured_data_pipeline/run_structured_data_pipeline.py \
  --input ~/tree_of_captions/output/trees/Video_1.json \
  --output ./outputs/Video_1.structured.json \
  --pipeline-dir ./outputs/pipeline_runs \
  --model feeltheAGI/Maverick-7B \
  --task-name "cross_task_example" \
  --video-id "Video_1" \
  --iterations 2
```

## Notes

- This branch is intended for producing validated structured outputs.
- For direct iterative loop only, see branch `self_refine`.
