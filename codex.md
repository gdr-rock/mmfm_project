# Codex Working Notes

Last updated: 2026-03-13

## Purpose Of This File

This is the project-memory file I should read before making changes in this repository.
It records my current interpretation of the repository, the intended objective, the
actual implemented flow, and the gaps between the two. I should update this file when:

- my understanding of the pipeline changes,
- I verify or invalidate an assumption,
- I add or remove a required artifact or workflow,
- I discover a missing piece that affects training, inference, or evaluation.

## Working Interpretation

This repository is a research-style reproduction / adaptation of a dual-system video
planning pipeline:

- `System-1` generates future action/state-change steps from a task goal plus a partial trajectory.
- `System-2` reranks multiple candidate plans with a learned critic.
- Optional `V-JEPA` video latents ground the text model during training and can also
  contribute an energy term during reranking.
- A learned `goal-latent` model is available to map goal text and
  action/state-change trajectories into a shared latent space for energy scoring:
  `E = ||z_traj - z_goal||^2`.

The project is centered on `CrossTask` as the dataset and aims to bridge:

- the VLWM planning design from Chen et al. (fast generator + critic + latent grounding),
- V-JEPA 2 predictive video latents,
- a smaller, reproducible setup that can run on limited hardware.

In practice, this repo has two related but different tracks:

1. The main train / planning / evaluation track:
   dataset builders -> train models -> generate K plans -> rerank -> evaluate.
2. A separate streaming / smoke-demo track:
   download videos -> caption-tree outputs and placeholder latent caches.

The first track is the real research pipeline. The second looks like tooling for quick
checks and demos, not the main training path.

## Verified Repository Structure

Main project files:

- `README.md`: high-level project description and intended commands.
- `paper/main.tex`: paper draft that matches the intended architecture.
- `scripts/04_build_state_change_dataset.py` to `scripts/08_split_datasets.py`:
  dataset creation.
- `scripts/train_system1.py`: T5 System-1 training.
- `scripts/train_system1_plm_lora.py`: Perception-LM System-1 LoRA training.
- `scripts/train_goal_model.py`: goal model training.
- `scripts/train_goal_latent_model.py`: goal/trajectory latent alignment model
  (text -> V-JEPA-like latent space) for learned energy scoring.
- `scripts/train_critic.py`: MiniLM + MLP critic training.
- `scripts/train_critic_llm_lora.py`: Llama critic LoRA training.
- `scripts/extract_vjepa_latents.py`: per-segment latent extraction from video.
- `scripts/run_planning.py`: System-2 candidate generation + reranking.
- `scripts/run_evaluation.py`: VPA-style metrics.
- `scripts/run_inference.py`: standalone inference CLI for system1 / system2 / goal.
- `scripts/13_stream_crosstask_pipeline.py`: separate streaming demo pipeline.
- `scripts/coin_*.py`: parallel COIN dataset pipeline for canonical planning data.

Data currently present:

- `crosstask_release/`: raw CrossTask annotations and metadata are present locally.
- `data/crosstask/`: built train/val/test datasets are already present locally.
- `COIN_dataset/`: raw COIN metadata and taxonomy files are present locally.
- `data/coin/`: generated COIN taxonomy cache, split manifest, transitions, and
  System-1/goal datasets are present locally.

Model checkpoints currently present:

- Only `checkpoints/perception_encoder.pt` exists in this checkout.
- Trained model directories such as `checkpoints/system1`, `checkpoints/goal_model`,
  `checkpoints/critic`, `checkpoints/system1_plm_lora`, and
  `checkpoints/critic_llm_lora` do not currently exist.

This means the repo contains the data and training code, but not the trained models
needed for normal planning or inference.

## The Real End-To-End Flow

### 1. Build the transition dataset from CrossTask

`scripts/04_build_state_change_dataset.py`

Input:

- `crosstask_release/tasks_primary.txt`
- `crosstask_release/videos.csv`
- `crosstask_release/annotations/*.csv`

Output:

- `data/crosstask/state_change_transitions.csv`

What it does:

- reconstructs ordered action segments per video,
- maps each action to a state-change sentence,
- creates transition rows with current and next step information,
- stores metadata useful for downstream dataset generation.

### 2. Build task-specific training datasets

`scripts/05_build_critic_dataset.py`

- builds ranking samples for the critic:
  `base`, `good`, `bad`, `shuffled`.

`scripts/06_build_system1_dataset.py`

- builds structured prompts for next-step prediction,
- supports zero-prefix samples,
- stores prompt text in `input_text` and JSON target in `output_text`.

`scripts/07_build_goal_dataset.py`

- builds two tasks:
  `goal_prediction` and `goal_and_plan`.

### 3. Split datasets at the video level

`scripts/08_split_datasets.py`

Uses:

- `crosstask_release/videos_val.csv`

Outputs:

- `state_change_transitions_{train,val,test}.csv`
- `critic_{train,val,test}.jsonl`
- `system1_{train,val,test}.jsonl`
- `goal_{train,val,test}.jsonl`

Important point:

- the split is by video, not by sample, to avoid leakage across train/val/test.

### 4. Optional: extract V-JEPA latents

`scripts/extract_vjepa_latents.py`

Uses:

- the transition CSV,
- video downloads via `yt-dlp`,
- a separately cloned `vjepa2` repository,
- GPU inference over extracted video segments.

Outputs:

- `data/crosstask/vjepa_latents/{task_id}/{video_id}/segment_XXX.pt`
- `meta.json` per video.

Purpose:

- enables latent grounding during System-1 training,
- enables optional energy scoring during plan reranking.

### 5. Train models

Core models:

- `scripts/train_system1.py`: T5 seq2seq System-1.
- `scripts/train_goal_model.py`: T5 goal model.
- `scripts/train_critic.py`: MiniLM + MLP critic.

VLWM-faithful variants:

- `scripts/train_system1_plm_lora.py`: Perception-LM-1B + LoRA.
- `scripts/train_critic_llm_lora.py`: Llama-3.2-1B + LoRA.

Optional grounded training:

- both System-1 training scripts can consume `--latent_dir` and add InfoNCE loss.

### 6. Planning / inference

Offline planning:

- `scripts/run_planning.py`
- loads a System-1 model and a critic,
- generates `K` candidate plans,
- scores them with critic cost,
- optionally adds V-JEPA energy,
- writes JSONL outputs for later evaluation.

Standalone inference:

- `scripts/run_inference.py`
- supports:
  - `system1` mode,
  - `system2` mode,
  - `goal` mode.

Optional visual interpretation at inference time:

- `run_inference.py --frames ...`
- uses Perception-LM vision input to generate an `Interpretation:` sentence from
  initial frames and then conditions planning on that text.

### 7. Evaluation

`scripts/run_evaluation.py`

Measures:

- `SR` exact plan match,
- `mAcc` position-wise step accuracy,
- `mIoU` set overlap of predicted vs gold steps.

## What Is Required To Train

### Minimum local requirements

- Python environment with `requirements.txt`.
- `ffmpeg` installed.
- GPU strongly recommended for all model training.

### Required inputs already available here

- `crosstask_release/` exists.
- `data/crosstask/system1_{train,val,test}.jsonl` exists.
- `data/crosstask/critic_{train,val,test}.jsonl` exists.
- `data/crosstask/goal_{train,val,test}.jsonl` exists.

This means text-only training can start immediately without rebuilding datasets.

### Required to train text-only models

System-1 T5:

- `data/crosstask/system1_train.jsonl`
- `data/crosstask/system1_val.jsonl`
- download access to `google/flan-t5-small`

Goal model:

- `data/crosstask/goal_train.jsonl`
- `data/crosstask/goal_val.jsonl`
- download access to `google/flan-t5-small`

Critic:

- `data/crosstask/critic_train.jsonl`
- `data/crosstask/critic_val.jsonl`
- download access to `sentence-transformers/all-MiniLM-L6-v2`

### Additional requirements for LoRA / large-model variants

System-1 PLM LoRA:

- access to `facebook/Perception-LM-1B`
- acceptance of the FAIR noncommercial research license
- enough VRAM for LoRA or QLoRA
- `bitsandbytes`, `peft`, `accelerate`

Critic LLM LoRA:

- access to `meta-llama/Llama-3.2-1B`
- license acceptance on Hugging Face
- enough VRAM for LoRA or QLoRA
- `bitsandbytes`, `peft`, `accelerate`

### Additional requirements for latent-grounded training

- `data/crosstask/vjepa_latents/` must exist with real per-video segment latents.
- `scripts/extract_vjepa_latents.py` needs:
  - `yt-dlp`,
  - downloadable YouTube videos,
  - a local `vjepa2` clone,
  - suitable V-JEPA weights,
  - enough GPU memory.

If `--latent_dir` is missing or invalid, the training scripts fall back to text-only.

## What Is Required To Run Inference

### System-1 only inference

Required:

- a trained System-1 checkpoint:
  - T5 directory for `--system1_type t5`, or
  - LoRA adapter directory plus base model access for `--system1_type plm`.
- a goal string and optional prefix, or a JSONL sample via `--from_jsonl`.

Not required:

- critic checkpoint,
- V-JEPA latents.

### System-2 inference

Required:

- a trained System-1 checkpoint,
- a trained critic checkpoint,
- prompt inputs or a JSONL sample.

Optional:

- `--latent_dir` plus `task_id` and `video_id` for energy scoring.

### Goal-model inference

Required:

- a trained `checkpoints/goal_model/best_model` directory,
- observed prefix steps.

### Vision-grounded interpretation inference

Required:

- access to Perception-LM for visual interpretation,
- either:
  - a video file, or
  - a directory of initial frames.

Notes:

- this changes only the interpretation text at inference time,
- it does not retrain the planning model,
- it is additive, not part of the core training pipeline.

## Current Missing Aspects / Risks

### 1. The repository does not currently have trained checkpoints

This is the biggest practical gap for inference.

Current state:

- dataset files exist,
- training scripts exist,
- planning and evaluation scripts exist,
- trained System-1 / critic / goal checkpoints do not exist in `checkpoints/`.

Consequence:

- the repo is not inference-ready out of the box.

### 2. V-JEPA extraction contains a deliberate placeholder fallback

`scripts/extract_vjepa_latents.py` explicitly falls back to random features when:

- the `vjepa2` repo is not present,
- the import path does not match expectations,
- or the weights are missing.

Consequence:

- the pipeline can "run" without real V-JEPA latents,
- but those latents may be meaningless unless the external dependency is set up correctly.

This is acceptable for smoke tests, not for real grounded experiments.

### 3. The energy term in planning is heuristic, not a learned latent predictor

Both `README.md` and `run_planning.py` imply this already.

Current behavior:

- planning uses the latent at an index estimated from plan length,
- not a learned text-to-latent transition predictor.

Consequence:

- the energy score is a rough proxy,
- it should be treated as an experimental heuristic rather than a faithful world-model rollout.

### 4. The streaming pipeline is not the same as the main training pipeline

`scripts/13_stream_crosstask_pipeline.py`:

- uses placeholder videos when `--skip_download` is enabled,
- can use fallback-generated captions,
- can generate placeholder latent caches.

Observed output evidence:

- some output summaries contain placeholder `VIDEO_ID_*` URLs.

Consequence:

- those outputs are useful as smoke/demo artifacts,
- they should not be confused with the core training/evaluation artifacts.

### 5. Some project metadata appears stale or inconsistent

Example:

- `outputs/setup/prereq_report.json` reports a checkpoint path that does not match
  the actual current `checkpoints/` contents.

Consequence:

- any setup report or old output should be treated as historical, not authoritative.

### 6. Real end-to-end grounded training still depends on external access

Needed external pieces include:

- model downloads from Hugging Face,
- license-gated models,
- YouTube video availability,
- a working V-JEPA repo checkout.

Consequence:

- even though the local repo is structurally complete, reproducibility still depends on
  outside systems and artifacts.

## Current State Of Readiness

### What is ready now

- raw CrossTask annotations are present,
- derived train/val/test datasets are present,
- training scripts for all major components are present,
- planning and evaluation scripts are present,
- standalone inference CLI is present.

### What is not ready now

- no trained planning checkpoints are present,
- no goal model checkpoint is present,
- no critic checkpoint is present,
- no verified real V-JEPA latent directory is present,
- no evidence yet that the repo has been run end-to-end with real grounded training.

### My operational conclusion

This repository is best understood as:

- a mostly complete research codebase for a small-scale VLWM-style planning project,
- with usable dataset builders and training code,
- but with several important runtime dependencies left external,
- and with a few intentionally non-final placeholder / heuristic components.

## Recommended Default Path Forward

If the goal is to make the project actually runnable in a reliable way, the next
practical order should be:

1. Verify environment and package install.
2. Train text-only baseline models first:
   `train_system1.py`, `train_goal_model.py`, `train_critic.py`.
3. Validate baseline inference:
   `run_inference.py`, `run_planning.py`, `run_evaluation.py`.
4. Only after the baseline works, add:
   PLM LoRA, LLM LoRA, and then real V-JEPA latent extraction.
5. Treat the streaming pipeline as auxiliary unless explicitly needed.

## Maintenance Rule For Future Codex Turns

Before making project changes, I should:

- re-read this file,
- confirm whether the change affects data generation, training, inference, or evaluation,
- update this file if the repo state or my interpretation changed.

When updating this file, prefer:

- concrete facts over speculation,
- explicit artifact paths,
- a short note about what was verified versus assumed.
