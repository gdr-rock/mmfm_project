# VLWM Planning Training 

This module provides executable training pipelines for both tracks from the paper setup:

- System-1: language world-model training with goal + interpretation + `<action, DeltaState>` supervision.
- Baseline: action-only behavior cloning with the same data protocol for fair comparison.

## Requirements

Install dependencies from `requirements.txt`:

```powershell
cd vlwm_planning
python -m pip install -r requirements.txt
```

If `torch` install fails, install the correct wheel for your CPU/GPU first from:

- https://pytorch.org/get-started/locally/

Then reinstall:

```powershell
python -m pip install -r requirements.txt
```

## Folder Layout

- `configs/system1_train.yaml`: System-1 config
- `configs/baseline_train.yaml`: Baseline config
- `scripts/train_system1.py`: System-1 trainer entrypoint
- `scripts/train_baseline.py`: Baseline trainer entrypoint
- `src/training/trainer_runtime.py`: Shared end-to-end train loop
- `src/training/system1_conditioning.py`: System-1 conditioning + masking
- `src/training/baseline_conditioning.py`: Baseline conditioning + masking
- `src/training/dataset_schema.py`: System-1 target formatting
- `src/training/baseline_schema.py`: Baseline target formatting

## Data Format (JSONL)

### System-1 row format

```json
{
  "config": "predict goal-plan trajectory",
  "visual_context": "frame/caption summary text or serialized visual tokens",
  "goal_description": "make tea",
  "goal_interpretation": "initial: kettle empty; target: tea prepared",
  "actions": ["fill kettle", "boil water", "pour into cup"],
  "delta_states": ["kettle has water", "water is hot", "cup has hot tea"]
}
```

Accepted aliases used by loader:

- `system_prompt` for `config`
- `context` for `visual_context`
- `goal` for `goal_description`
- `states` for `delta_states`

### Baseline row format

```json
{
  "config": "predict procedural actions",
  "visual_context": "frame/caption summary text or serialized visual tokens",
  "goal_description": "make tea",
  "asr_text": "optional transcript text",
  "actions": ["fill kettle", "boil water", "pour into cup"]
}
```

Accepted aliases used by loader:

- `system_prompt` for `config`
- `context` for `visual_context`
- `goal` for `goal_description`

## Run Training

From repository root:

```powershell
python vlwm_planning\scripts\train_system1.py --config vlwm_planning\configs\system1_train.yaml
python vlwm_planning\scripts\train_baseline.py --config vlwm_planning\configs\baseline_train.yaml
```

Optional flags:

- `--dry-run`: run one optimization step
- `--max-steps N`: cap training steps

## Paper-Alignment Contract

### System-1
- Conditioning: `[CONFIG/SYSTEM_PROMPT, VISUAL_CONTEXT, optional AUX_TEXT, GOAL]`
- Target: `[GOAL_DESCRIPTION, GOAL_INTERPRETATION, <A_i, DeltaS_i>]`
- Loss: autoregressive next-token cross-entropy only
- Masking: conditioning tokens are excluded from CE (`ignore_index=-100`)
- No auxiliary reconstruction/contrastive loss

### Baseline
- Conditioning: `[CONFIG/SYSTEM_PROMPT, VISUAL_CONTEXT, optional ASR, GOAL]`
- Target: action-only trajectory
- Loss: autoregressive next-token cross-entropy only
- Masking: conditioning tokens are excluded from CE (`ignore_index=-100`)
- No critic/ranking/auxiliary losses

## Outputs

Checkpoints are written to:

- `outputs/system1/checkpoints/system1_last.pt`
- `outputs/baseline/checkpoints/baseline_last.pt`

## Notes

- Current environment must have `torch` + `transformers` available to run training.
- `decoder_context_tokens` (baseline) and `max_context_tokens` (System-1) are controlled by YAML configs.
