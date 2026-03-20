# Grounding Language-Based Video Planning in Predictive Video Latents

**MMFM Final Project** — Bridges VLWM text planning (Chen et al., 2025) with V-JEPA 2 predictive video latents (Assran et al., 2025).

## Architecture Overview

| Component | Backbone | Params | Purpose |
|-----------|----------|--------|---------|
| **System-1** (Rollout Generator) | `google/flan-t5-small` | 60M | Seq2seq: prefix → next k steps (JSON) |
| **System-1 PLM+LoRA** (Rollout Generator) | `facebook/Perception-LM-1B` + LoRA | 1.2B (3M trainable) | Causal LM: prefix → trajectory (VLWM-faithful) |
| **Goal Model** | `google/flan-t5-small` | 60M | Seq2seq: partial trajectory → goal + remaining plan |
| **Critic** (System-2 Scorer) | `sentence-transformers/all-MiniLM-L6-v2` + MLP | 22M + 100K | Ranking loss: score trajectory quality |
| **Critic LLM+LoRA** (System-2 Scorer) | `meta-llama/Llama-3.2-1B` + LoRA | 1.2B (3M trainable) | Causal LM → last-token → scalar cost (VLWM-faithful) |
| **V-JEPA 2 Encoder** | `facebookresearch/vjepa2` ViT-L | 300M | Extract predictive video latents (1024-dim) |
| **Latent Projection Head** | MLP (2-layer + GELU + L2-norm) | ~2M | Project LLM hidden states → V-JEPA space (InfoNCE) |
| **PLM Vision Tower** (inference) | Built-in to PLM-1B/8B | — | Generate vision-grounded interpretation from frames (Optional, `--frames`) |

> **Scale note:** VLWM uses PerceptionLM-8B (System-1) and Llama-3.2-1B (Critic) trained on 180k videos.
> We have ~2.7k CrossTask videos (~12k training samples). We provide **two options** for each:
> 1. **System-1:** T5-small (60M full FT) vs. Perception-LM-1B + LoRA (~3M trainable)
> 2. **Critic:** MiniLM+MLP (22M full FT) vs. Llama-3.2-1B + LoRA (~3M trainable)
>
> The LoRA variants use the EXACT same backbone models as the VLWM paper.
>
> **Latent Grounding (VLWM §3.1.1, Eq. 3-4):** Both System-1 scripts support optional
> V-JEPA latent grounding via `--latent_dir`. When enabled, a learned projection head
> maps per-step hidden states to V-JEPA latent space, with symmetric InfoNCE loss:
> `L = L_text + α·L_latent`. This is disabled by default (text-only) and requires
> pre-extracted V-JEPA latents from `extract_vjepa_latents.py`.

## Installation

```bash
# System deps
sudo apt update && sudo apt install -y ffmpeg

# Python deps
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Data Pipeline (Scripts 04–08)

These scripts build the training datasets from CrossTask annotations. **Run them in order.**

```bash
# Step 1: Build state-change transition CSV (18,169 rows)
python3 scripts/04_build_state_change_dataset.py

# Step 2: Build Critic training data (11,762 samples)
python3 scripts/05_build_critic_dataset.py

# Step 3: Build System-1 training data (15,049 samples)
python3 scripts/06_build_system1_dataset.py

# Step 4: Build Goal Model training data (21,536 samples)
python3 scripts/07_build_goal_dataset.py

# Step 5: Split all datasets into train/val/test
#         Uses official CrossTask validation split (video-level, no leakage)
#         Train: 78.6% | Val: 13.1% | Test: 8.4%
python3 scripts/08_split_datasets.py
```

**Output files** (all in `data/crosstask/`):

| File | Rows | Size |
|------|------|------|
| `state_change_transitions_{train,val,test}.csv` | 14,120 / 2,492 / 1,557 | 4.0 / 0.7 / 0.4 MB |
| `critic_{train,val,test}.jsonl` | 9,259 / 1,538 / 965 | 11.7 / 2.0 / 1.3 MB |
| `system1_{train,val,test}.jsonl` | 11,852 / 1,953 / 1,244 | 9.6 / 1.6 / 1.0 MB |
| `goal_{train,val,test}.jsonl` | 16,920 / 2,816 / 1,800 | 10.5 / 1.8 / 1.1 MB |

---

## Training (Run on HPC — one at a time)

### 1. Train System-1 (Rollout Generator)

```bash
python3 scripts/train_system1.py \
    --train_data  data/crosstask/system1_train.jsonl \
    --val_data    data/crosstask/system1_val.jsonl \
    --output_dir  checkpoints/system1 \
    --model_name  google/flan-t5-small \
    --epochs 30 \
    --batch_size 16 \
    --lr 3e-4 \
    --seed 42
```

**Model:** `google/flan-t5-small` (60M params, T5 encoder-decoder)  
**Stored at:** `checkpoints/system1/best_model/` (HuggingFace format)  
**Log:** `checkpoints/system1/training_log.jsonl`  
**Metrics:** val loss, valid JSON rate, exact match rate  
**Estimated time:** ~30–60 min on 1× GPU (A100/V100)

### 1b. Train System-1 with PerceptionLM + LoRA (VLWM-faithful)

This uses the **same backbone family** as the VLWM paper (PerceptionLM), with LoRA
adapters to respect our data scale. Causal LM training (next-token prediction),
matching VLWM Eq. 1.

**Prereqs:** Accept the [FAIR Noncommercial Research License](https://huggingface.co/facebook/Perception-LM-1B) on HuggingFace.

```bash
# Standard LoRA (needs ~12GB VRAM)
python3 scripts/train_system1_plm_lora.py \
    --train_data  data/crosstask/system1_train.jsonl \
    --val_data    data/crosstask/system1_val.jsonl \
    --output_dir  checkpoints/system1_plm_lora \
    --model_name  facebook/Perception-LM-1B \
    --lora_r 16 \
    --lora_alpha 32 \
    --epochs 10 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr 2e-4 \
    --seed 42

# QLoRA variant (needs ~6GB VRAM — fits on consumer GPUs)
python3 scripts/train_system1_plm_lora.py \
    --use_qlora \
    --train_data  data/crosstask/system1_train.jsonl \
    --val_data    data/crosstask/system1_val.jsonl \
    --output_dir  checkpoints/system1_plm_qlora \
    --model_name  facebook/Perception-LM-1B \
    --lora_r 16 \
    --lora_alpha 32 \
    --epochs 10 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr 2e-4 \
    --seed 42
```

**Model:** `facebook/Perception-LM-1B` (1.2B params, ~3M trainable via LoRA)  
**Architecture:** Causal LM (next-token prediction, NOT seq2seq)  
**Stored at:** `checkpoints/system1_plm_lora/best_adapter/` (LoRA weights only, ~12MB)  
**Log:** `checkpoints/system1_plm_lora/training_log.jsonl`  
**Metrics:** val loss (perplexity), valid JSON rate, exact match rate  
**Estimated time:** ~2–3 hours on 1× A40/H100

**To load for inference:**
```python
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import PeftModel
base = AutoModelForImageTextToText.from_pretrained("facebook/Perception-LM-1B")
model = PeftModel.from_pretrained(base, "checkpoints/system1_plm_lora/best_adapter")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/system1_plm_lora/best_adapter")
```

### 1c. Latent Grounding with V-JEPA (VLWM §3.1.1, Eq. 3-4)

Both System-1 scripts support optional **V-JEPA latent grounding** via `--latent_dir`.
This adds a learned projection head + symmetric InfoNCE contrastive loss that aligns
per-step LLM hidden states with V-JEPA video latents:

$$L = L_\text{text} + \alpha \cdot L_\text{latent}$$

where $L_\text{latent}$ is symmetric InfoNCE between projected text representations and
mean-pooled V-JEPA segment features. **Requires pre-extracted V-JEPA latents** (see Step 4).

```bash
# T5-small + V-JEPA grounding:
python3 scripts/train_system1.py \
    --train_data    data/crosstask/system1_train.jsonl \
    --val_data      data/crosstask/system1_val.jsonl \
    --output_dir    checkpoints/system1_grounded \
    --latent_dir    data/crosstask/vjepa_latents \
    --vjepa_dim 1024 \
    --latent_weight 0.1 \
    --temperature 0.07 \
    --epochs 30 --batch_size 16 --lr 3e-4

# PLM + LoRA + V-JEPA grounding:
python3 scripts/train_system1_plm_lora.py \
    --train_data    data/crosstask/system1_train.jsonl \
    --val_data      data/crosstask/system1_val.jsonl \
    --output_dir    checkpoints/system1_plm_lora_grounded \
    --latent_dir    data/crosstask/vjepa_latents \
    --vjepa_dim 1024 \
    --latent_weight 0.1 \
    --temperature 0.07 \
    --model_name facebook/Perception-LM-1B \
    --lora_r 16 --lora_alpha 32 \
    --epochs 10 --batch_size 4 --gradient_accumulation_steps 8
```

**Projection Head:** MLP (LLM_dim → VJEPA_dim → VJEPA_dim) with GELU + L2-norm  
**Saved at:** `best_adapter/latent_head.pt` (or `best_model/latent_head.pt` for T5)  
**Without `--latent_dir`:** scripts fall back to text-only training (no grounding)

### 2. Train Goal Model

```bash
python3 scripts/train_goal_model.py \
    --train_data  data/crosstask/goal_train.jsonl \
    --val_data    data/crosstask/goal_val.jsonl \
    --output_dir  checkpoints/goal_model \
    --model_name  google/flan-t5-small \
    --epochs 20 \
    --batch_size 16 \
    --lr 3e-4 \
    --seed 42
```

**Model:** `google/flan-t5-small` (60M params)  
**Stored at:** `checkpoints/goal_model/best_model/` (HuggingFace format)  
**Log:** `checkpoints/goal_model/training_log.jsonl`  
**Metrics:** val loss, goal accuracy (18-class), plan valid JSON rate

### 3. Train Critic Model

```bash
python3 scripts/train_critic.py \
    --train_data  data/crosstask/critic_train.jsonl \
    --val_data    data/crosstask/critic_val.jsonl \
    --output_dir  checkpoints/critic \
    --encoder_name sentence-transformers/all-MiniLM-L6-v2 \
    --epochs 30 \
    --batch_size 32 \
    --lr 2e-5 \
    --margin 1.0 \
    --lambda_reg 0.01 \
    --seed 42
```

**Model:** `sentence-transformers/all-MiniLM-L6-v2` encoder (22M) + MLP head (~100K)  
**Stored at:** `checkpoints/critic/best_model.pt` (PyTorch state dict)  
**Log:** `checkpoints/critic/training_log.jsonl`  
**Loss:** Ranking loss (VLWM Eq. 2) with margin=1.0, λ=0.01  
**Metrics:** ranking accuracy (C_good < C_base, etc.)

### 3b. Train Critic — Llama-3.2-1B + LoRA (VLWM-faithful)

The VLWM paper uses Llama-3.2-1B as the critic backbone. We apply LoRA (r=16, α=32) to
train ~3M adapter parameters + a tiny Linear cost head, instead of fine-tuning all 1.2B.

```bash
python3 scripts/train_critic_llm_lora.py \
    --train_data  data/crosstask/critic_train.jsonl \
    --val_data    data/crosstask/critic_val.jsonl \
    --output_dir  checkpoints/critic_llm_lora \
    --model_name  meta-llama/Llama-3.2-1B \
    --lora_r 16 \
    --lora_alpha 32 \
    --epochs 5 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr 2e-4 \
    --margin 1.0 \
    --lambda_reg 0.01 \
    --max_len 512 \
    --seed 42

# QLoRA variant (fits on 16GB GPU):
python3 scripts/train_critic_llm_lora.py \
    --use_qlora \
    --train_data  data/crosstask/critic_train.jsonl \
    --val_data    data/crosstask/critic_val.jsonl \
    --output_dir  checkpoints/critic_llm_qlora \
    --batch_size 8 \
    --gradient_accumulation_steps 4
```

**Prereqs:** Accept [Llama-3.2 license](https://huggingface.co/meta-llama/Llama-3.2-1B) on HuggingFace.  
**Architecture:** Llama-3.2-1B → last-token hidden state → Linear(2048, 1) → scalar cost.  
**Stored at:** `checkpoints/critic_llm_lora/best_adapter/` (~12 MB LoRA weights + cost_head.pt)  
**Log:** `checkpoints/critic_llm_lora/training_log.jsonl`  
**Loss:** Same ranking loss (VLWM Eq. 2) with margin=1.0, λ=0.01  
**Metrics:** ranking accuracy, validated per epoch

### 4. Extract V-JEPA 2 Latents (Optional, requires video access)

```bash
# First, clone the V-JEPA 2 repo and download weights
git clone https://github.com/facebookresearch/vjepa2 ../vjepa2
# Follow vjepa2 README to download ViT-L checkpoint

# Then extract latents (processes one video at a time, deletes after)
python3 scripts/extract_vjepa_latents.py \
    --transitions_csv  data/crosstask/state_change_transitions.csv \
    --output_dir       data/crosstask/vjepa_latents \
    --vjepa_repo       ../vjepa2 \
    --model_name       vitl \
    --frames_per_segment 16 \
    --skip_existing

# To process only train split videos:
python3 scripts/extract_vjepa_latents.py \
    --transitions_csv  data/crosstask/state_change_transitions.csv \
    --split_csv        data/crosstask/state_change_transitions_train.csv \
    --output_dir       data/crosstask/vjepa_latents \
    --vjepa_repo       ../vjepa2 \
    --max_videos 100
```

**Model:** V-JEPA 2 ViT-L (300M params, 1024-dim features)  
**Stored at:** `data/crosstask/vjepa_latents/{task_id}/{video_id}/segment_NNN.pt`  
**⚠️  Without the repo/weights, random placeholder features are used**

---

## Inference & Evaluation

### 5. Generate & Plot Training Curves

```bash
python3 scripts/plot_training.py \
    --system1_log  checkpoints/system1/training_log.jsonl \
    --goal_log     checkpoints/goal_model/training_log.jsonl \
    --critic_log   checkpoints/critic/training_log.jsonl \
    --output_dir   plots
```

**Outputs** in `plots/`:
- `system1_loss_curve.png`, `system1_metrics.png`
- `goal_model_loss_curve.png`, `goal_model_metrics.png`
- `critic_loss_curve.png`, `critic_accuracy.png`
- `combined_overview.png`

### 6. System-2 Planning (K-plan Generation + Reranking)

```bash
# Critic-only reranking (no V-JEPA latents needed)
python3 scripts/run_planning.py \
    --test_data       data/crosstask/system1_test.jsonl \
    --system1_model   checkpoints/system1/best_model \
    --critic_model    checkpoints/critic/best_model.pt \
    --K 5 \
    --temperature 0.8 \
    --output_dir      outputs/planning

# With latent energy (when V-JEPA latents are available)
python3 scripts/run_planning.py \
    --test_data       data/crosstask/system1_test.jsonl \
    --system1_model   checkpoints/system1/best_model \
    --critic_model    checkpoints/critic/best_model.pt \
    --latent_dir      data/crosstask/vjepa_latents \
    --alpha 0.7 --beta 0.3 \
    --K 5 \
    --output_dir      outputs/planning
```

**Score:** `α × C_critic + β × Energy(z_pred, z_goal)` → select plan with LOWEST score

### 7. Evaluate (VPA Metrics: SR, mAcc, mIoU)

```bash
# System-1 only (greedy decode)
python3 scripts/run_evaluation.py \
    --mode system1 \
    --test_data       data/crosstask/system1_test.jsonl \
    --system1_model   checkpoints/system1/best_model \
    --output_dir      outputs/evaluation

# System-2 (reranked plans)
python3 scripts/run_evaluation.py \
    --mode system2 \
    --planning_output outputs/planning/<timestamp>_plans.jsonl \
    --output_dir      outputs/evaluation

# Both (side-by-side comparison table)
python3 scripts/run_evaluation.py \
    --mode both \
    --test_data       data/crosstask/system1_test.jsonl \
    --system1_model   checkpoints/system1/best_model \
    --planning_output outputs/planning/<timestamp>_plans.jsonl \
    --output_dir      outputs/evaluation
```

**Metrics** (from VPA benchmark, Patel et al. 2023):
- **SR** (Success Rate) — exact plan match (plan-level)
- **mAcc** (Mean Accuracy) — step-level accuracy at each position
- **mIoU** (Mean IoU) — set overlap between predicted and gold steps

### 7b. Clean Full-Plan Evaluation (No Prefix)

This is a separate evaluation mode for the question:

- input = `goal + interpretation`, or `goal + initial frames`
- model generates a **full plan**
- critic and optional goal-latent model score each candidate plan
- evaluation is done on the **full generated plan**, not next-`k` continuation

Build the clean test files:

```bash
python3 scripts/build_coin_full_plan_test_data.py
```

This creates:

- `data/coin/coin_full_plan_task_test.jsonl`
  one row per COIN task, `goal + interpretation -> canonical full plan`
- `data/coin/coin_full_plan_video_test.jsonl`
  one row per COIN test video, `goal + interpretation/frames -> full observed plan`

Run text-only full-plan evaluation on 5 rows:

```bash
python3 scripts/evaluate_full_plans_clean.py \
    --test_data data/coin/coin_full_plan_task_test.jsonl \
    --input_mode text \
    --system1_model checkpoints/system1_plm_lora/best_adapter \
    --system1_type plm \
    --plm_base_model facebook/Perception-LM-1B \
    --critic_model checkpoints/critic/best_model.pt \
    --goal_latent_model checkpoints/goal_latent/best_model.pt \
    --K 5 \
    --plan_generation_mode rollout \
    --rollout_chunk_size 3 \
    --max_plan_steps 12 \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_samples 5 \
    --output_dir outputs/full_plan_eval_clean
```

Run frame-based full-plan evaluation:

```bash
python3 scripts/evaluate_full_plans_clean.py \
    --test_data data/coin/coin_full_plan_video_test.jsonl \
    --input_mode frames \
    --frames_root path/to/initial_frames \
    --system1_model checkpoints/system1_plm_lora/best_adapter \
    --system1_type plm \
    --plm_base_model facebook/Perception-LM-1B \
    --critic_model checkpoints/critic/best_model.pt \
    --goal_latent_model checkpoints/goal_latent/best_model.pt \
    --K 5 \
    --plan_generation_mode rollout \
    --rollout_chunk_size 3 \
    --max_plan_steps 12 \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_samples 5 \
    --output_dir outputs/full_plan_eval_frames
```

The evaluator saves:

- `summary.json`
- `summary.csv`
- `per_sample.csv`
- `per_sample.jsonl`
- `summary_metrics.png`
- `length_scatter.png`
- `combined_score_hist.png`

#### What The Printed Values Mean

Example console output:

```text
[1/5] valid=5/5 exact=1.000 ord=1.000 iou=1.000
```

- `valid=5/5`
  all `K=5` generated candidates were valid JSON plans
- `exact=1.000`
  the selected best plan matched the gold step list exactly after normalization
- `ord=1.000`
  the selected plan preserved the full gold ordering
- `iou=1.000`
  predicted and gold step sets were identical

Summary fields:

- `samples`
  number of evaluated rows
- `valid_json_rate`
  fraction of samples where at least one valid plan was generated
- `exact_match`
  mean exact-match score across samples
- `ordered_ratio`
  mean longest-common-subsequence ratio against the gold plan
- `step_iou`
  mean set overlap between predicted and gold steps
- `step_accuracy`
  mean positional step accuracy
- `avg_length_delta`
  average `predicted_length - gold_length`
- `avg_pred_len`
  average predicted plan length
- `avg_gold_len`
  average gold plan length
- `avg_critic_score`
  mean critic cost of the selected plans; lower is better, negative values are fine
- `avg_goal_score`
  mean goal-latent energy of the selected plans; lower is better
- `avg_combined_score`
  mean reranking score used to select the best candidate; lower is better

#### Interpreting Perfect Scores

If you see:

```text
exact_match: 1.0
ordered_ratio: 1.0
step_iou: 1.0
step_accuracy: 1.0
```

that means the evaluator selected a plan that exactly matches the gold plan for every evaluated sample.

This is much easier to achieve on `coin_full_plan_task_test.jsonl` than on the original prefix-based evaluation because:

- the task-level file uses clean canonical plans
- the input includes a rule-based interpretation
- there is no partial-prefix ambiguity

For a more realistic test, prefer `coin_full_plan_video_test.jsonl`.

### 7c. Task-Disjoint COIN + CrossTask System-1 Data (`data/coin_2`)

The original COIN task-level file is useful, but it is still same-task evaluation.
`data/coin_2` adds a harder setup:

- COIN `test` is split by **task**, not by video
- train mixes COIN with CrossTask for extra task/context coverage
- train/val keep the useful **prefix** conditioning
- test uses **goal-only** and **goal+interpretation** full-plan prompts
- every target ends with `{"action":"<END>","state_change":"<END>"}` so the model can learn when to stop
- the noisy fallback text `The state changes: ...` is removed from the new data

Build it with:

```bash
python3 scripts/coin2_build_system1_dataset.py \
    --output_dir data/coin_2
```

This creates:

- `data/coin_2/coin2_system1_train.jsonl`
- `data/coin_2/coin2_system1_val.jsonl`
- `data/coin_2/coin2_system1_test_prompt_variants.jsonl`
- `data/coin_2/coin2_system1_test_goal_only.jsonl`
- `data/coin_2/coin2_system1_test_goal_plus_interpretation.jsonl`
- `data/coin_2/coin2_task_split_manifest.csv`
- `data/coin_2/coin2_dataset_report.json`

Train System-1 on the new split:

```bash
python3 scripts/train_system1_plm_lora.py \
    --train_data data/coin_2/coin2_system1_train.jsonl \
    --val_data data/coin_2/coin2_system1_val.jsonl \
    --output_dir checkpoints/system1_coin2_lora_grounded \
    --model_name facebook/Perception-LM-1B \
    --lora_r 16 \
    --lora_alpha 32 \
    --epochs 10 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr 1e-4 \
    --max_seq_len 1024 \
    --latent_dir /path/to/vjepa_latents \
    --vjepa_dim 1408 \
    --latent_weight 0.1 \
    --curriculum_text_fraction 0.7 \
    --latent_primary_batch_ratio 0.75 \
    --phase2_latent_weight 0.2 \
    --seed 42
```

The last three flags implement the simpler curriculum described above:

- first `70%` of epochs use the normal shuffled training mix
- final `30%` of epochs rebuild batches so they are latent-primary
- latent tensors are validated before use: they are pooled to `(vjepa_dim,)`, cast to `float32`, and rows with missing segments, wrong width, or non-finite values are skipped for grounding

Evaluate the held-out task split with direct full-plan generation:

```bash
python3 scripts/evaluate_full_plans_clean.py \
    --test_data data/coin_2/coin2_system1_test_goal_only.jsonl \
    --input_mode text \
    --system1_model checkpoints/system1_coin2/best_model \
    --critic_model checkpoints/critic/best_model.pt \
    --goal_latent_model checkpoints/goal_latent/best_model.pt \
    --plan_generation_mode direct \
    --output_dir outputs/coin2_goal_only
```

Use `coin2_system1_test_goal_plus_interpretation.jsonl` for the interpretation-assisted variant.

### 8. Standalone Inference (`run_inference.py`)

```bash
# System-1 greedy (text-only, rule-based interpretation)
python3 scripts/run_inference.py \
    --system1_model checkpoints/system1/best_model \
    --goal "Complete the task: Make Pancakes." \
    --prefix "pour egg | Egg is now poured in." \
    --k 3

# From a test JSONL sample (goal, prefix, interpretation auto-filled)
python3 scripts/run_inference.py \
    --from_jsonl data/crosstask/system1_test.jsonl \
    --sample_idx 0 \
    --system1_model checkpoints/system1/best_model

# Zero-prefix (no observed steps — plan from scratch)
python3 scripts/run_inference.py \
    --system1_model checkpoints/system1/best_model \
    --goal "Complete the task: Make Pancakes." \
    --interpretation "The task begins with 'add flour' and is done once 'take pancake from pan' is done." \
    --k 3
```

### 8b. Vision-Grounded Interpretation (Option 3 — PLM Frames → Text)

When `--frames` is provided, PLM's vision tower processes the initial video
frames (or a short clip) and generates a natural-language **Interpretation** line
that replaces the rule-based template. This is the closest our pipeline gets to
the VLWM paper's approach, where visual context informs the plan.

```bash
# From a video file (PLM-1B, 8 frames)
python3 scripts/run_inference.py \
    --system1_model checkpoints/system1/best_model \
    --goal "Complete the task: Make Pancakes." \
    --frames path/to/pancakes_video.mp4 \
    --num_frames 8 \
    --k 3

# From a directory of frame images
python3 scripts/run_inference.py \
    --system1_model checkpoints/system1/best_model \
    --goal "Complete the task: Make Pancakes." \
    --frames path/to/initial_frames/ \
    --k 3

# With PLM+LoRA System-1 (same PLM used for both interpretation and planning)
python3 scripts/run_inference.py \
    --system1_type plm \
    --system1_model checkpoints/system1_plm_lora/best_adapter \
    --plm_base_model facebook/Perception-LM-1B \
    --goal "Complete the task: Make Pancakes." \
    --frames path/to/video.mp4 \
    --interp_model facebook/Perception-LM-1B \
    --k 3
```

**How it works:**
1. **Initial frames only** — when given a video file, the script extracts the
   **first N frames** (default 8) using `decord`, NOT uniformly sampled across
   the video. This ensures PLM sees the **starting scene state** before any
   actions are performed.
2. Frames → PLM `AutoProcessor` (chat template with `<image>` tokens per frame)
3. PLM generates: *"The person has eggs and flour on the counter and is about to mix batter for pancakes."*
4. This text is injected as `Interpretation: ...` in the System-1 prompt
5. System-1 model plans forward conditioned on this richer, vision-grounded context

```
Inference flow:
                                                    
  Initial video frames           PLM Vision Tower             System-1
  (first 8 from .mp4     →    (frozen, bfloat16)    →    (T5 or PLM+LoRA)
   or frame directory)     generates Interpretation     predicts next k steps
                                                    
  ┌───────────────┐        ┌──────────────────────┐    ┌──────────────────────┐
  │ Frame 0       │        │ "The person has eggs  │    │ {"next_steps": [     │
  │ Frame 1       │───────▶│  and flour on the     │───▶│   {"action": "add    │
  │  ...          │        │  counter, preparing   │    │    flour", ...},     │
  │ Frame 7       │        │  to make pancakes."   │    │   ...                │
  └───────────────┘        └──────────────────────┘    └──────────────────────┘
                           Interpretation: line         {"next_steps": [...]}
```

**VRAM requirements:**
- PLM-1B: ~4GB (bfloat16) for interpretation alone
- PLM-1B + T5-small System-1: ~5GB total
- PLM-1B + PLM-1B+LoRA System-1: ~4GB (same model reused)

**Dependencies:** `pip install decord` (for video loading). Frame directory mode uses PIL only.

> **Impact on training/evaluation:** NONE. This feature is inference-time only.
> Training data still uses rule-based `make_interpretation()`. Evaluation scripts
> read `input_text` directly from JSONL (interpretation already baked in). The
> `--frames` option only affects `run_inference.py` when running standalone inference.

#### Interpretation Sources Comparison

| Source | When | Quality | Cost |
|--------|------|---------|------|
| **Rule-based** (`06_build`) | Training data, JSONL eval | Template from first/last canonical steps | Free |
| **`--interpretation` CLI** | Manual inference | User-provided text | Free |
| **`--from_jsonl`** | Test samples | Extracted from baked `input_text` | Free |
| **`--frames`** (Option 3) | Inference with video access | Vision-grounded, PLM-generated | ~4GB VRAM |

---

## Model Storage Summary

```
checkpoints/
├── system1/
│   ├── best_model/          ← HuggingFace T5 (config.json, model.safetensors, tokenizer)
│   ├── final_model/
│   └── training_log.jsonl
├── goal_model/
│   ├── best_model/          ← HuggingFace T5
│   ├── final_model/
│   └── training_log.jsonl
└── critic/
    ├── best_model.pt        ← PyTorch state dict {model_state_dict, encoder_name, epoch}
    ├── final_model.pt
    └── training_log.jsonl

data/crosstask/vjepa_latents/   ← V-JEPA segment features
    {task_id}/{video_id}/
        segment_000.pt          ← (N_tokens, 1024) float32
        meta.json

plots/                           ← Training curve PNGs
outputs/planning/                ← K-plan JSONL with scores
outputs/evaluation/              ← VPA metrics JSON + per-sample JSONL
```

## HPC Quick Reference (Sequential Execution)

```bash
# ─── Phase 1: Data Prep (CPU only, ~5 min total) ───
python3 scripts/04_build_state_change_dataset.py
python3 scripts/05_build_critic_dataset.py
python3 scripts/06_build_system1_dataset.py
python3 scripts/07_build_goal_dataset.py
python3 scripts/08_split_datasets.py

# ─── Phase 2: V-JEPA Latent Extraction (GPU, long) ───
#   MUST run before training if you plan to use --latent_dir
#   for latent grounding (VLWM §3.1.1 InfoNCE loss).
#   Without this, training falls back to text-only (still works).
python3 scripts/extract_vjepa_latents.py --max_videos 100

# ─── Phase 3: Training (GPU, run one at a time) ───
#   Text-only (no latent grounding):
python3 scripts/train_system1.py
python3 scripts/train_goal_model.py
python3 scripts/train_critic.py
#   With latent grounding (requires Phase 2):
python3 scripts/train_system1.py --latent_dir data/crosstask/vjepa_latents
#   VLWM-scale (PLM + LoRA System-1, LLM + LoRA Critic):
python3 scripts/train_system1_plm_lora.py --latent_dir data/crosstask/vjepa_latents
python3 scripts/train_critic_llm_lora.py

# ─── Phase 4: Plots (CPU, <1 min) ───
python3 scripts/plot_training.py

# ─── Phase 5: Inference (GPU, ~10-30 min) ───
python3 scripts/run_planning.py

# ─── Phase 6: Evaluation (GPU for system1, CPU for system2) ───
python3 scripts/run_evaluation.py --mode both \
    --planning_output outputs/planning/<latest>_plans.jsonl
```

## Known Constraints & Flags

1. **V-JEPA 2 integration**: The latent extraction script provides a working pipeline, but the actual V-JEPA 2 repo API may need minor adaptation once cloned. The script falls back to random features if the repo isn't available, so the rest of the pipeline works end-to-end regardless.

2. **CrossTask video availability**: Some YouTube videos may have been deleted since the dataset was published. The extraction script handles failures gracefully (skips and continues).

3. **Energy scoring is a heuristic**: The current `Energy(z_pred, z_goal)` uses direct latent lookup. A proper implementation would train a lightweight latent predictor that maps text state-changes to predicted latent transitions. This is a natural extension.

4. **Data scale vs VLWM**: Our models are 100–200× smaller than VLWM's. Expect lower absolute numbers but the relative improvements (System-2 > System-1) should still hold if the architecture is sound.

5. **Zero-prefix support**: Training data can include zero-prefix samples (controlled by `--zero_prefix_prob`, default 15%). At inference time, System-1 and System-2 modes accept empty `--prefix` (or omit it entirely) for planning from scratch.

6. **Vision-grounded interpretation (`--frames`)**: This is an inference-time-only feature. It does NOT change any training data, training loops, or evaluation metrics. The System-1 model was trained on rule-based interpretations, so it may respond differently to PLM-generated richer text. In practice, the interpretation line has a small effect (~0.6% in VLWM ablations, Table 6), so this is a nice-to-have rather than critical.

## References

- Chen et al., "Planning with Reasoning using Vision Language World Model" (2025) — [arXiv:2509.02722](https://arxiv.org/abs/2509.02722)
- Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning" (2025) — [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
- Zhukov et al., "Cross-task Weakly Supervised Learning from Instructional Videos" (2019) — [arXiv:1903.08225](https://arxiv.org/abs/1903.08225)
- Patel et al., "Visual Planning for Assistance" (2023)
