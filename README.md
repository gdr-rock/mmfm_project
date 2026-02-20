# JEPA-Grounded Language Planning for Procedural Videos

## 1) Project Overview
This repository is a research scaffold for generating high-level, step-by-step plans from language goals for procedural videos (CrossTask/COIN).
It augments text-space planning with frozen predictive video latents (V-JEPA2-style) to reduce plausible-but-wrong plans.
The core idea is to score candidate plans with both linguistic quality and latent-space consistency.
The codebase is designed for single-GPU/small-cluster workflows (A40-compatible), with offline latent caching to reduce repeated compute.

## 2) Motivation
Text-only planning can produce fluent but visually inconsistent state changes. By grounding intermediate changes and end-goal progress in predictive video latents, we aim to improve reliability under domain shift, paraphrases, and noisy step annotations.

## 3) Method Overview
For each language goal, generate `K` candidate plans and choose the lowest-cost candidate:

`J(plan) = C_text(plan) + lambda * P_transition(plan) + mu * D_goal(plan)`

Where:
- `C_text`: text critic cost (ranking-based plausibility)
- `P_transition`: JEPA transition inconsistency penalty via bridge `F(ΔS -> Δz)`
- `D_goal`: latent distance to goal embedding via bridge `G(goal -> z_goal)`

Grounding components:
- State-change grounding: textual state-change spans map to latent transitions between adjacent video segments.
- Goal grounding: goal text maps to a goal latent and penalizes plans far from this target latent.
- System-2 inference: generate many candidates, score with all terms, then select best.

## 4) End-to-End Pipeline

Training-time:
1. Prepare dataset manifests/splits.
2. Segment videos into aligned windows/steps.
3. Extract and cache frozen JEPA latents (`z_t`, `Δz_t`, goal reference latents).
4. Build text planning artifacts (candidate steps, parsed steps, state-change spans `ΔS`).
5. Build critic training pairs/triples and train text critic.
6. Train transition bridge `F(ΔS -> Δz)`.
7. Train goal bridge `G(goal -> z_goal)`.

Inference-time:
1. Given goal text, generate `K` candidates.
2. Parse steps and derive `ΔS`.
3. Score each candidate with critic + transition penalty + goal distance.
4. Select best candidate plan.

ASCII pipeline:

```text
Goal text
  |
  v
Candidate LLM --> K candidate plans --> step parser --> ΔS spans
  |                                              |         |
  |                                              |         v
  +-----------------------> text critic cost     |    bridge F(ΔS->Δz)
                                                 |
Video -> segmentation -> frozen V-JEPA2 -> z_t, Δz_t -----+
                                                           |
Goal text -----------------------> bridge G(goal->z_goal)--+
                                                           |
                                                           v
                           score = C_text + lambda*P_transition + mu*D_goal
                                                           |
                                                           v
                                                     best plan
```

## 5) Installation and Environment
```bash
# Option A: pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate jepa-grounded-planning
```

Quick sanity run:
```bash
python scripts/00_smoke_test.py --use_dummy_data --dry_run
```

## 6) Data Layout
No proprietary data is included. Put public datasets under `data/`:

```text
data/
  crosstask/
    videos/
    annotations/
    splits/{train,val,test}.txt
  coin/
    videos/
    COIN.json
    splits/{train,val,test}.txt
```

See `data/README.md` for details.

## 7) Run the Pipeline (CrossTask-first)

```bash
# 1. Prepare dataset manifests
python scripts/01_prepare_dataset.py \
  --config configs/dataset_crosstask.yaml \
  --dataset crosstask --output_dir outputs --use_dummy_data

# 2. Segment/cache JEPA latents
python scripts/02_cache_jepa_latents.py \
  --config configs/models.yaml \
  --dataset crosstask --output_dir outputs --use_dummy_data

# 3. Generate K candidate plans
python scripts/03_generate_candidates.py \
  --config configs/default.yaml --dataset crosstask \
  --goal "make tea" --k_candidates 8 --output_dir outputs --use_dummy_data

# 4. Convert candidates to state changes
python scripts/04_generate_state_changes.py \
  --config configs/default.yaml --dataset crosstask \
  --output_dir outputs --use_dummy_data

# 5. Train text critic
python scripts/05_train_text_critic.py \
  --config configs/models.yaml --dataset crosstask \
  --epochs 3 --output_dir outputs --use_dummy_data

# 6. Train bridges
python scripts/06_train_bridge_transition.py \
  --config configs/models.yaml --dataset crosstask \
  --epochs 3 --output_dir outputs --use_dummy_data

python scripts/07_train_bridge_goal.py \
  --config configs/models.yaml --dataset crosstask \
  --epochs 3 --output_dir outputs --use_dummy_data

# 7. System-2 scoring + selection
python scripts/08_score_and_select.py \
  --config configs/default.yaml --dataset crosstask \
  --goal "make tea" --k_candidates 8 \
  --lambda_transition 0.6 --mu_goal 0.8 \
  --output_dir outputs --use_dummy_data

# 8. Evaluate
python scripts/09_eval_vpa.py --config configs/eval.yaml --dataset crosstask --output_dir outputs --use_dummy_data
python scripts/10_eval_consistency_retrieval.py --config configs/eval.yaml --dataset crosstask --output_dir outputs --use_dummy_data
python scripts/11_eval_robustness.py --config configs/eval.yaml --dataset crosstask --output_dir outputs --use_dummy_data

# 9. Ablations
python scripts/12_run_ablations.py --config configs/eval.yaml --dataset crosstask --output_dir outputs --use_dummy_data
```

Use the same commands with `--dataset coin` and `configs/dataset_coin.yaml` for COIN.

### 7.1) Streamed 5-video CrossTask run (download one-by-one, then delete)

This mode processes only `N=5` URLs at a time and avoids storing the full video set.

1. Put up to 5+ video URLs in:
   - `data/crosstask/video_urls.txt`
2. Run:

```bash
python3 scripts/13_stream_crosstask_pipeline.py \
  --dataset crosstask \
  --url_list data/crosstask/video_urls.txt \
  --max_videos 5 \
  --output_dir outputs/streaming \
  --goal "make tea" \
  --k_candidates 8
```

Optional flags:
- `--skip_download`: useful for local dry checks without network.
- `--use_dummy_data`: forces dummy dataset behavior for stage scripts.
- `--keep_downloaded`: do not delete each downloaded video after processing.
- `--dry_run`: prints/logs flow without executing subprocess stages.

Per-video outputs:
- `outputs/streaming/<video_key>/cache/jepa_latents_crosstask.npz`
- `outputs/streaming/<video_key>/plans/candidates_crosstask.json`
- `outputs/streaming/<video_key>/plans/state_changes_crosstask.json`
- `outputs/streaming/<video_key>/plans/caption_tree_crosstask.json`
- run summary: `outputs/streaming/streaming_summary.json`

## 8) Task List with Dependencies and Validation

### 1) Dataset setup & splits (CrossTask/COIN)
- Depends on: `[]`
- Why: All downstream modules assume consistent train/val/test IDs and annotation paths.
- Validate by:
  - Check per-split video counts and total counts match expected source metadata.
  - Inspect 3 sample manifest records for fields (`video_id`, `task_id`, timestamps/steps).
  - Expected artifact: `outputs/data/dataset_manifest_<dataset>.json`

### 2) Video preprocessing & segmentation
- Depends on: `[1]`
- Why: Latent extraction and step alignment require contiguous segments/windows.
- Validate by:
  - Visualize timestamps for sample videos and verify contiguous step ordering.
  - Sanity-check segment durations (no zero/negative windows).
  - Expected artifact: `outputs/data/segments_<dataset>.json`

### 3) V-JEPA2 wrapper integration (frozen)
- Depends on: `[1, 2]`
- Why: A stable encoder API is required before caching latents and bridge training.
- Validate by:
  - Forward pass returns deterministic latent shape `[num_segments, latent_dim]`.
  - Confirm gradients are disabled (frozen encoder).
  - Expected artifact: `outputs/models/jepa_wrapper_report.json`

### 4) Cache JEPA latents (`z_t`, `Δz_t`, `z_goal`)
- Depends on: `[2, 3]`
- Why: Offline caching reduces repeated encoding cost and enables reproducible experiments.
- Validate by:
  - Confirm latent tensor shapes and dtype.
  - Run nearest-neighbor latent retrieval sanity checks on a small sample.
  - Record caching speed (videos/sec) and cache size (MB/GB).
  - Expected artifact: `outputs/cache/jepa_latents_<dataset>.npz`

### 5) Candidate plan generation (text LLM; `K` candidates)
- Depends on: `[1]`
- Why: Candidate diversity is necessary for System-2 rescoring to matter.
- Validate by:
  - Ensure exactly `K` candidates per goal.
  - Verify extracted step lists are non-empty and ordered.
  - Check coverage of required canonical steps for sample tasks.
  - Expected artifact: `outputs/plans/candidates_<dataset>.json`

### 6) Robust step parsing (from messy output)
- Depends on: `[5]`
- Why: Real LLM outputs are noisy; parser robustness prevents cascading failures.
- Validate by:
  - Parse success rate on a stress set (numbering variants, bullets, malformed text).
  - Confirm parser recovers step ordering and removes duplicates.
  - Expected artifact: `outputs/plans/parsed_steps_<dataset>.json`

### 7) `ΔS` generation (templates or rewriter)
- Depends on: `[6]`
- Why: Transition bridge training needs explicit textual state-change spans for each step.
- Validate by:
  - Ensure 100% of parsed steps produce a `ΔS` string.
  - Produce template coverage report by task category.
  - Spot-check random samples for semantic quality.
  - Expected artifact: `outputs/plans/state_changes_<dataset>.json`

### 8) Text critic dataset generation (good/bad/shuffled)
- Depends on: `[1, 5, 6, 7]`
- Why: Ranking critic needs supervised contrastive examples tied to task goals.
- Validate by:
  - Verify class balance (good vs bad/shuffled) and no split leakage.
  - Check negative sampling strategy logs.
  - Expected artifact: `outputs/data/critic_pairs_<dataset>.jsonl`

### 9) Train text critic (ranking)
- Depends on: `[8]`
- Why: Critic provides the linguistic plausibility term in final scoring.
- Validate by:
  - Report held-out ranking accuracy / pairwise win rate.
  - Check score calibration summary (e.g., reliability bins).
  - Compare against random-scoring baseline.
  - Expected artifact: `outputs/models/text_critic_<dataset>.json`

### 10) Train transition bridge `F(ΔS)->Δz`
- Depends on: `[4, 7]`
- Why: Maps textual state changes to latent transitions for consistency penalties.
- Validate by:
  - Report cosine similarity/correlation with true `Δz` on held-out data.
  - Evaluate retrieval accuracy of the correct next segment latent.
  - Expected artifact: `outputs/models/bridge_transition_<dataset>.npz`

### 11) Train goal bridge `G(goal)->z_goal`
- Depends on: `[4, 5]`
- Why: Enables language-goal grounding in latent space.
- Validate by:
  - Measure within-task goal latent retrieval accuracy.
  - Report positive-vs-negative distance separation.
  - Expected artifact: `outputs/models/bridge_goal_<dataset>.npz`

### 12) Implement scoring + selection (System-2)
- Depends on: `[5, 9, 10, 11]`
- Why: Combines all learned signals into final plan decision logic.
- Validate by:
  - Show improvement over critic-only and JEPA-only scoring.
  - Plot sensitivity for `lambda` / `mu` and `K` vs performance.
  - Confirm deterministic tie-breaking behavior.
  - Expected artifact: `outputs/plans/selected_plan_<dataset>.json`

### 13) Evaluate VPA on CrossTask and COIN
- Depends on: `[1, 2, 4, 12]`
- Why: Core effectiveness metric on procedural planning tasks.
- Validate by:
  - Report task success/VPA per dataset and per task family.
  - Include confidence intervals for main metrics.
  - Expected artifact: `outputs/eval/vpa_metrics_<dataset>.json`

### 14) Build consistency retrieval benchmark + evaluate
- Depends on: `[2, 4, 10, 12]`
- Why: Directly tests text-to-video transition consistency independent of VPA.
- Validate by:
  - Build benchmark splits with positives/negatives and metadata checks.
  - Report retrieval `R@1/R@5` and hard-negative performance.
  - Expected artifact: `outputs/eval/retrieval_metrics_<dataset>.json`

### 15) Robustness evaluation (paraphrase, visual augmentations)
- Depends on: `[13, 14]`
- Why: Measures reliability under realistic distribution shifts.
- Validate by:
  - Report paraphrase success drop and augmentation success drop.
  - Provide confidence intervals and seed-variance summaries.
  - Expected artifact: `outputs/eval/robustness_<dataset>.json`

### 16) Compute/efficiency reporting
- Depends on: `[4, 12, 13, 15]`
- Why: Ensures method is practical on A40/small-cluster budgets.
- Validate by:
  - Record wall-clock time per video and peak GPU memory.
  - Report number of JEPA encodes (cached vs online).
  - Summarize cache hit rate and storage footprint.
  - Expected artifact: `outputs/eval/compute_report_<dataset>.json`

### 17) Ablations
- Depends on: `[12, 13, 14, 15, 16]`
- Why: Quantifies contribution of each component and verifies causal value.
- Validate by:
  - Run critic-only, JEPA-only, no-goal, no-transition, varying `K`.
  - Report delta vs full model with consistent seeds.
  - Expected artifact: `outputs/eval/ablation_report_<dataset>.json`

## 9) Metrics to Report
- Task success / VPA (CrossTask, COIN)
- Robustness drops under paraphrase + visual augmentation
- Compute efficiency: wall-clock/video, GPU memory, JEPA encodes, cache footprint
- Text-video consistency/calibration:
  - Retrieval metrics (`R@1`, `R@5`)
  - Critic calibration / score reliability
  - Agreement between text critic and latent consistency terms

## 10) Expected Output Artifacts
- `outputs/data/`: manifests, segment metadata, critic pairs
- `outputs/cache/`: cached JEPA latents (`z_t`, `Δz_t`, `z_goal`)
- `outputs/models/`: critic and bridge checkpoints/reports
- `outputs/plans/`: candidates, parsed steps, state changes, selected plans
- `outputs/eval/`: VPA, retrieval, robustness, compute, ablations

## 11) References
- Ha & Schmidhuber (2018), *World Models* — https://arxiv.org/abs/1803.10122
- Assran et al. (2025), *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning* — https://arxiv.org/abs/2506.09985
- Chen et al. (2025), *Planning with Reasoning using Vision-Language World Model (VLWM)* — https://arxiv.org/abs/2509.02722
- Tang et al. (2019), *COIN* — https://arxiv.org/abs/1903.02874
- Zhukov et al. (2019), *CrossTask* — https://arxiv.org/abs/1903.08225
- *WorldPrediction / WorldPrediction-PP* — https://arxiv.org/abs/2506.04363

## 12) Proposed Abstract (Verbatim)
High-level video planners that operate mainly in text space can generate plans whose step-by-step state changes sound correct (“water boiled”, “tea ready”) but are not visually consistent with what would actually happen in the video world. This mismatch becomes worse under domain shift (different camera styles, egocentric video) and noisy step annotations, causing the planner to pick plausible-but-wrong plans.

Current approach. VLWM-style planners compress video into language (actions + state descriptions) and use a learned text critic to score candidate plans. V-JEPA-style video models learn strong predictive video latents, but typically plan with visual goal images, not language goals.

Proposed solution. We combine both by grounding language planning in a predictive latent space: (1) State-change grounding: each predicted textual state-change span is trained to match the corresponding V-JEPA latent transition between adjacent video segments; (2) Goal grounding: we learn a small encoder that maps goal text → goal latent, enabling an energy objective that favors plans whose predicted final latent is close to the goal latent. During planning, we score candidates with a weighted sum of (text critic cost) + (transition inconsistency penalty) + (goal latent distance).

Evaluation. We evaluate plan success, robustness to domain/paraphrase shifts, compute overhead, and text–video consistency on COIN and CrossTask (VPA), WorldPrediction-PP, and a controlled retrieval-style consistency benchmark built from video segment pairs.
