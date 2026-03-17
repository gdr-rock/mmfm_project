#!/usr/bin/env python3
"""
Train System-1 (rollout generator) using PerceptionLM + LoRA,
matching the VLWM paper's actual approach.

Background (VLWM, Chen et al. 2025):
  - System-1 is a causal language model initialized from PerceptionLM-8B.
  - It performs next-token prediction on structured trajectories:
      [config, context] → [goal, interpretation, ⟨A₀,ΔS₀⟩, ..., ⟨Aₙ,ΔSₙ⟩]
  - VLWM trains on 180k videos (5.7M steps) with the full 8B model.

  GROUNDING (VLWM §3.1.1, Eq. 3-4):
  - In addition to text rollout (Eq. 1), System-1 is grounded in video
    latents via a CONTRASTIVE (InfoNCE) objective.
  - A learned projection head maps per-step LLM hidden states to the
    V-JEPA 2 latent space:   z_text = proj(h_step)
  - InfoNCE loss aligns z_text with the corresponding V-JEPA segment
    latent z_video, using other segments in the batch as negatives:
      L_latent = -log( exp(sim(z_t, z_v) / τ) / Σ_j exp(sim(z_t, z_j) / τ) )
  - Combined loss:  L = L_text + α · L_latent

Our approach:
  - We use Perception-LM-1B (the LLM decoder only, text-only mode) because
    our CrossTask dataset has ~12k training samples — using 8B would overfit.
  - We apply LoRA (Low-Rank Adaptation) to the attention + MLP projections,
    freezing the base model and training only ~2-4M adapter parameters.
  - When V-JEPA latents are available (--latent_dir), we add a projection
    head + InfoNCE loss for video grounding. When not available, we train
    with text-only loss (pure rollout generator, no grounding).

Why LoRA instead of full fine-tuning?
  - Perception-LM-1B has ~1.2B parameters; full FT on 12k samples = massive
    overfitting risk, plus high GPU memory (>8GB just for optimizer states).
  - LoRA trains ~0.3% of params, acts as an implicit regularizer, and allows
    the model to retain pretrained knowledge while learning our task format.
  - The official PLM model card includes a LoRA example using the exact same
    target_modules we use here (q_proj, k_proj, v_proj, o_proj, gate_proj,
    up_proj, down_proj).

Why causal LM instead of seq2seq?
  - VLWM uses causal next-token prediction (Eq. 1 in the paper), NOT seq2seq.
  - The prompt (goal + interpretation + progress) is the prefix; the model
    completes with the trajectory (next action-state pairs as JSON).
  - This matches the paper's formulation exactly. Our previous T5 script was
    a pragmatic approximation; this is the faithful implementation.

Prereqs:
  - Accept the FAIR Noncommercial Research License for Perception-LM on HF.
  - pip install peft bitsandbytes accelerate

Saves:
  checkpoints/system1_plm_lora/best_adapter/  (LoRA adapter weights only)
  checkpoints/system1_plm_lora/latent_head.pt (projection head weights)
  checkpoints/system1_plm_lora/training_log.jsonl

Usage (HPC, single A40/H100 GPU):

  # Text-only (no V-JEPA grounding):
  python3 scripts/train_system1_plm_lora.py \\
      --train_data  data/crosstask/system1_train.jsonl \\
      --val_data    data/crosstask/system1_val.jsonl \\
      --output_dir  checkpoints/system1_plm_lora \\
      --model_name  facebook/Perception-LM-1B \\
      --lora_r 16 --lora_alpha 32 \\
      --epochs 10 --batch_size 4 --gradient_accumulation_steps 8 \\
      --lr 2e-4 --seed 42

  # With V-JEPA latent grounding (InfoNCE, VLWM Eq. 3-4):
  python3 scripts/train_system1_plm_lora.py \\
      --train_data  data/crosstask/system1_train.jsonl \\
      --val_data    data/crosstask/system1_val.jsonl \\
      --output_dir  checkpoints/system1_plm_lora_grounded \\
      --model_name  facebook/Perception-LM-1B \\
      --latent_dir  data/crosstask/vjepa_latents \\
      --vjepa_dim 1024 \\
      --latent_weight 0.1 \\
      --temperature 0.07 \\
      --lora_r 16 --lora_alpha 32 \\
      --epochs 10 --batch_size 4 --gradient_accumulation_steps 8 \\
      --lr 2e-4 --seed 42

  For QLoRA (4-bit quantized base, even lower memory):
  python3 scripts/train_system1_plm_lora.py \\
      --use_qlora \\
      --train_data  data/crosstask/system1_train.jsonl \\
      --val_data    data/crosstask/system1_val.jsonl \\
      --output_dir  checkpoints/system1_plm_qlora \\
      --model_name  facebook/Perception-LM-1B \\
      --lora_r 16 --lora_alpha 32 \\
      --epochs 10 --batch_size 8 --gradient_accumulation_steps 4 \\
      --lr 2e-4 --seed 42
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Lazy imports — script starts fast, heavy imports on demand
# ---------------------------------------------------------------------------
_LOADED = False


def _lazy_imports():
    global _LOADED
    global AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
    global BitsAndBytesConfig
    global LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    global get_linear_schedule_with_warmup
    if _LOADED:
        return

    from transformers import (
        AutoTokenizer as _AT,
        AutoModelForCausalLM as _ACLM,
        AutoModelForImageTextToText as _AITT,
    )

    AutoTokenizer = _AT
    AutoModelForCausalLM = _ACLM
    AutoModelForImageTextToText = _AITT

    try:
        from transformers import BitsAndBytesConfig as _BNB
        BitsAndBytesConfig = _BNB
    except ImportError:
        BitsAndBytesConfig = None

    from peft import (
        LoraConfig as _LC,
        get_peft_model as _GPM,
        prepare_model_for_kbit_training as _PMKBT,
        PeftModel as _PM,
    )
    LoraConfig = _LC
    get_peft_model = _GPM
    prepare_model_for_kbit_training = _PMKBT
    PeftModel = _PM

    try:
        from transformers import get_linear_schedule_with_warmup as _sched
        get_linear_schedule_with_warmup = _sched
    except ImportError:
        get_linear_schedule_with_warmup = None

    _LOADED = True


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# VLWM-style special tokens for structured output
PROMPT_SUFFIX = "\n\nAssistant:"
RESPONSE_PREFIX = ""

# LoRA target modules — same as official PLM fine-tuning example
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Step delimiters in output JSON — used to locate per-step boundaries
# in the generated sequence for hidden-state extraction
STEP_DELIMITERS = ['"action":', '"state_change":']


# ---------------------------------------------------------------------------
# Latent Projection Head + InfoNCE Loss (VLWM §3.1.1, Eq. 3-4)
# ---------------------------------------------------------------------------

class LatentProjectionHead(nn.Module):
    """
    Projects per-step LLM hidden states into V-JEPA latent space.

    VLWM grounds the text planner's internal representations in the video
    encoder's latent space via a learned projection:
        z_text = proj(h_step)    ∈ R^{vjepa_dim}
    where h_step is the LLM's hidden state at the boundary of each predicted
    step (specifically, at the token where each "state_change" value ends).

    Architecture: 2-layer MLP with GELU activation and L2-normalization.
    This is standard for contrastive learning heads (SimCLR, CLIP, etc.).
    """

    def __init__(self, llm_hidden_dim: int, vjepa_dim: int, proj_dim: int = 0):
        """
        Args:
            llm_hidden_dim: LLM hidden size (e.g. 2048 for PLM-1B)
            vjepa_dim: V-JEPA feature dimension (1024 for ViT-L, 1280 for ViT-H)
            proj_dim: Optional intermediate projection dimension.
                      If 0, projects directly to vjepa_dim.
        """
        super().__init__()
        mid_dim = proj_dim if proj_dim > 0 else vjepa_dim
        self.net = nn.Sequential(
            nn.Linear(llm_hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, vjepa_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (N, llm_hidden_dim) — hidden states at step boundaries
        Returns:
            z: (N, vjepa_dim) — L2-normalized projected embeddings
        """
        z = self.net(h)
        z = nn.functional.normalize(z, dim=-1)
        return z


def info_nce_loss(z_text: torch.Tensor, z_video: torch.Tensor,
                  temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE contrastive loss (VLWM Eq. 3-4).

    For a batch of N (text, video) pairs:
      L_t2v = -(1/N) Σ_i log( exp(sim(z_t_i, z_v_i)/τ) / Σ_j exp(sim(z_t_i, z_v_j)/τ) )
      L_v2t = -(1/N) Σ_i log( exp(sim(z_v_i, z_t_i)/τ) / Σ_j exp(sim(z_v_i, z_t_j)/τ) )
      L_latent = (L_t2v + L_v2t) / 2

    This is the standard symmetric InfoNCE used in CLIP, VLWM, etc.

    Args:
        z_text:  (N, D) L2-normalized text projections
        z_video: (N, D) L2-normalized video latents
        temperature: softmax temperature τ (VLWM uses 0.07)
    Returns:
        Scalar loss
    """
    # Cosine similarity matrix: (N, N)
    logits = torch.matmul(z_text, z_video.T) / temperature

    N = logits.size(0)
    labels = torch.arange(N, device=logits.device)

    # Symmetric: text→video and video→text
    loss_t2v = nn.functional.cross_entropy(logits, labels)
    loss_v2t = nn.functional.cross_entropy(logits.T, labels)

    return (loss_t2v + loss_v2t) / 2.0


# ---------------------------------------------------------------------------
# Dataset: Causal LM format
# ---------------------------------------------------------------------------

class CausalLMDataset(Dataset):
    """
    Converts System-1 JSONL to causal LM training format.

    For causal LM training, we concatenate:
        [prompt_tokens] + [completion_tokens] + [eos]
    And only compute loss on the completion tokens (labels = -100 for prompt).

    This matches VLWM Eq. 1: the model sees context as prefix and predicts
    the trajectory autoregressively.

    When latent_dir is provided, also loads V-JEPA latents for each predicted
    step to enable InfoNCE grounding loss (VLWM Eq. 3-4).
    V-JEPA latents are expected at:
        {latent_dir}/{task_id}/{video_id}/segment_{seg_pos:03d}.pt
    Each is a tensor of shape (T_tokens, D) — we mean-pool to (D,).

    Additionally, we record the token positions of each step boundary in the
    completion so the training loop can extract hidden states at those positions.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 768,
                 latent_dir: str = None, vjepa_dim: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.latent_dir = latent_dir
        self.vjepa_dim = vjepa_dim
        self.samples = []
        self.has_latents = latent_dir is not None and os.path.isdir(latent_dir or "")

        n_with_latents = 0
        n_total = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                meta = rec.get("meta", {})
                entry = {
                    "input_text": rec["input_text"],
                    "output_text": rec["output_text"],
                    "meta": meta,
                }

                # Load V-JEPA latents if available
                if self.has_latents:
                    latents = self._load_step_latents(meta)
                    if latents is not None:
                        entry["vjepa_latents"] = latents  # list of (D,) tensors
                        n_with_latents += 1

                self.samples.append(entry)
                n_total += 1

        print(f"  Loaded {len(self.samples)} samples from {jsonl_path}")
        if self.has_latents:
            print(f"  V-JEPA latents found for {n_with_latents}/{n_total} samples")

    def _load_step_latents(self, meta: dict):
        """
        Load V-JEPA latent for each predicted step.

        Each training sample predicts steps at positions
        [start_seg_pos, start_seg_pos+1, ..., start_seg_pos+k-1].
        The latent for segment i is at:
            {latent_dir}/{task_id}/{video_id}/segment_{i:03d}.pt
        """
        task_id = meta.get("task_id")
        video_id = meta.get("video_id")
        start_pos = meta.get("start_seg_pos")
        k = meta.get("k")

        if not all([task_id, video_id, start_pos is not None, k]):
            return None

        latents = []
        for step_offset in range(k):
            seg_pos = start_pos + step_offset
            pt_path = os.path.join(
                self.latent_dir, str(task_id), str(video_id),
                f"segment_{seg_pos:03d}.pt"
            )
            if not os.path.exists(pt_path):
                return None  # Missing any segment → skip latent for this sample

            # Load and mean-pool: (T_tokens, D) → (D,)
            seg_latent = torch.load(pt_path, map_location="cpu")
            if seg_latent.dim() == 2:
                seg_latent = seg_latent.mean(dim=0)  # (D,)
            elif seg_latent.dim() == 1:
                pass  # Already pooled
            else:
                seg_latent = seg_latent.reshape(-1, seg_latent.shape[-1]).mean(dim=0)
            latents.append(seg_latent)

        return latents  # list of k tensors, each (D,)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Build causal LM input: prompt + completion
        prompt = sample["input_text"] + PROMPT_SUFFIX
        completion = RESPONSE_PREFIX + sample["output_text"]

        # Tokenize separately to know where prompt ends
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        completion_ids = self.tokenizer.encode(
            completion, add_special_tokens=False
        )

        # Add EOS
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self.tokenizer.pad_token_id or 0

        full_ids = prompt_ids + completion_ids + [eos_id]

        # Truncate if too long (keep prompt, truncate completion)
        if len(full_ids) > self.max_seq_len:
            # Ensure we keep at least some completion
            max_completion = self.max_seq_len - len(prompt_ids) - 1
            if max_completion < 10:
                # Prompt itself is too long; truncate prompt
                prompt_ids = prompt_ids[: self.max_seq_len // 2]
                max_completion = self.max_seq_len - len(prompt_ids) - 1
            completion_ids = completion_ids[:max_completion]
            full_ids = prompt_ids + completion_ids + [eos_id]

        # Build labels: -100 for prompt tokens, actual ids for completion
        labels = [-100] * len(prompt_ids) + completion_ids + [eos_id]

        assert len(full_ids) == len(labels), (
            f"Length mismatch: {len(full_ids)} vs {len(labels)}"
        )

        attention_mask = [1] * len(full_ids)

        result = {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_len": len(prompt_ids),
        }

        # --- Step boundary positions for latent extraction ---
        # We find the token positions where each step's "state_change"
        # value ends in the completion. The LLM hidden state at these
        # positions encodes the full step context.
        if "vjepa_latents" in sample:
            # Decode the completion back to find step boundaries
            # Each step in the JSON is: {"action": "...", "state_change": "..."}
            # We locate the closing quote after each state_change value
            step_boundary_positions = self._find_step_boundaries(
                full_ids, len(prompt_ids)
            )
            k = len(sample["vjepa_latents"])
            if len(step_boundary_positions) >= k:
                result["step_positions"] = step_boundary_positions[:k]
                result["vjepa_latents"] = sample["vjepa_latents"]

        return result

    def _find_step_boundaries(self, full_ids, prompt_len):
        """
        Find token positions where each step ends in the generated sequence.

        Strategy: decode the completion tokens back to text, find the positions
        of each '}' that closes a step dict, and map back to token positions.
        These positions will have hidden states that encode the full step.
        """
        completion_ids = full_ids[prompt_len:]
        completion_text = self.tokenizer.decode(
            completion_ids, skip_special_tokens=False
        )

        # Find positions of each closing brace '}' that ends a step
        # In the JSON: {"action": "...", "state_change": "..."}, ...
        # Each '}' ends a step's information
        positions = []
        brace_depth = 0
        char_pos = 0
        step_end_chars = []

        for i, ch in enumerate(completion_text):
            if ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    step_end_chars.append(i)

        # Map character positions back to token positions
        # We'll use a simpler approach: tokenize incrementally
        for end_char in step_end_chars:
            # Tokenize text up to and including this closing brace
            prefix_text = completion_text[:end_char + 1]
            prefix_ids = self.tokenizer.encode(
                prefix_text, add_special_tokens=False
            )
            # Token position in full sequence = prompt_len + len(prefix_ids) - 1
            token_pos = prompt_len + len(prefix_ids) - 1
            if token_pos < len(full_ids):
                positions.append(token_pos)

        return positions


def collate_fn(batch, pad_token_id: int):
    """Pad batch to max length in batch (dynamic padding).

    Also collects V-JEPA latents and step boundary positions if present.
    """
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)

    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

    # --- Latent grounding data ---
    # Gather step positions and V-JEPA latents from samples that have them.
    # We flatten across the batch: collect all (step_position, vjepa_latent)
    # pairs with a batch index, so the training loop can extract hidden states
    # at those positions and compute InfoNCE.
    has_latent = any("step_positions" in item for item in batch)
    if has_latent:
        all_batch_idx = []     # which batch item this step belongs to
        all_step_pos = []      # token position in the sequence
        all_vjepa = []         # (D,) V-JEPA latent

        for b_idx, item in enumerate(batch):
            if "step_positions" not in item:
                continue
            for s_idx, (pos, lat) in enumerate(
                zip(item["step_positions"], item["vjepa_latents"])
            ):
                all_batch_idx.append(b_idx)
                all_step_pos.append(pos)
                all_vjepa.append(lat)

        if all_vjepa:
            result["latent_batch_idx"] = torch.tensor(
                all_batch_idx, dtype=torch.long
            )
            result["latent_step_pos"] = torch.tensor(
                all_step_pos, dtype=torch.long
            )
            result["latent_vjepa"] = torch.stack(all_vjepa, dim=0)  # (N_steps, D)

    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _infer_hidden_dim(model) -> int:
    """
    Infer decoder hidden size across different model/config layouts.

    PerceptionLM may store text-model dimensions under nested configs
    (e.g., config.text_config.hidden_size) instead of config.hidden_size.
    """
    cfg = getattr(model, "config", None)
    if cfg is not None:
        # Common single-config attributes
        for attr in ("hidden_size", "d_model", "n_embd", "dim", "model_dim"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val

        # Common nested text/decoder config attributes
        for sub_name in ("text_config", "language_config", "llm_config", "decoder_config"):
            sub_cfg = getattr(cfg, sub_name, None)
            if sub_cfg is None:
                continue
            for attr in ("hidden_size", "d_model", "n_embd", "dim", "model_dim"):
                val = getattr(sub_cfg, attr, None)
                if isinstance(val, int) and val > 0:
                    return val

    # Fallback to embedding matrices
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and emb.weight.ndim == 2:
            return int(emb.weight.shape[1])
    except Exception:
        pass

    try:
        out_emb = model.get_output_embeddings()
        if out_emb is not None and hasattr(out_emb, "weight") and out_emb.weight.ndim == 2:
            return int(out_emb.weight.shape[1])
    except Exception:
        pass

    raise AttributeError(
        "Could not infer model hidden dimension from config or embeddings. "
        f"Config type: {type(cfg).__name__ if cfg is not None else 'None'}"
    )


def load_model_and_tokenizer(args):
    """
    Load PerceptionLM as a causal language model with LoRA.

    PLM is an image-text-to-text model (vision encoder + LLM decoder).
    For our text-only task, we load via AutoModelForImageTextToText and the
    LoRA adapters are applied to the LLM decoder's projection layers.

    Alternative: if model architecture supports it, load just the text decoder
    via AutoModelForCausalLM. PLM's model card uses AutoModelForImageTextToText.
    """
    _lazy_imports()

    print(f"Loading model: {args.model_name}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True
    )
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Quantization config for QLoRA ---
    bnb_config = None
    if args.use_qlora:
        if BitsAndBytesConfig is None:
            raise ImportError(
                "bitsandbytes required for QLoRA. "
                "pip install bitsandbytes"
            )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("  Using QLoRA (4-bit quantized base)")

    # --- Load base model ---
    # PLM uses AutoModelForImageTextToText; for text-only we can still use this
    # The LoRA adapters target the LLM decoder's linear layers
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if not args.use_qlora else None,
            device_map="auto" if args.use_qlora else None,
            trust_remote_code=True,
        )
        print(f"  Loaded as AutoModelForImageTextToText")
    except Exception:
        # Fallback: try loading as standard causal LM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if not args.use_qlora else None,
            device_map="auto" if args.use_qlora else None,
            trust_remote_code=True,
        )
        print(f"  Loaded as AutoModelForCausalLM (fallback)")

    # Ensure output head is correctly tied to token embeddings when required.
    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
            print("  Tied input/output embeddings")
        except Exception as e:
            print(f"  Warning: could not tie weights ({e})")

    # --- Freeze vision encoder (we don't use vision) ---
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        for param in model.model.vision_tower.parameters():
            param.requires_grad = False
        print("  Froze vision_tower parameters")

    # --- Prepare for QLoRA ---
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=DEFAULT_TARGET_MODULES,
        # PerceptionLM reloads with a missing standalone lm_head, so we must
        # persist the trained output head alongside the LoRA adapters.
        modules_to_save=["lm_head"],
        bias="none",
        task_type="CAUSAL_LM",
        # Use DoRA for better quality unless QLoRA (where it can be unstable)
        use_dora=not args.use_qlora,
        init_lora_weights="gaussian",
    )

    model = get_peft_model(model, lora_config)

    # Print parameter counts
    trainable, total = model.get_nb_trainable_parameters()
    pct = trainable / total * 100
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable (LoRA):     {trainable:>12,}  ({pct:.2f}%)")

    # Get hidden dimension for latent projection head
    hidden_dim = _infer_hidden_dim(model)
    print(f"  Hidden dimension:     {hidden_dim}")

    return model, tokenizer, hidden_dim


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, device, latent_head=None,
             latent_weight=0.1, temperature=0.07):
    """Compute validation loss (text + optional latent grounding)."""
    model.eval()
    if latent_head is not None:
        latent_head.eval()
    total_loss = 0.0
    total_tokens = 0
    total_latent_loss = 0.0
    n_latent_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # Separate latent-specific keys before sending to model
            latent_batch_idx = batch.pop("latent_batch_idx", None)
            latent_step_pos = batch.pop("latent_step_pos", None)
            latent_vjepa = batch.pop("latent_vjepa", None)

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward with hidden states if we need latent loss
            need_hidden = (latent_head is not None and latent_batch_idx is not None)
            outputs = model(
                **batch,
                output_hidden_states=need_hidden,
            )

            n_tokens = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

            # Latent grounding loss
            if need_hidden and latent_batch_idx is not None and len(latent_batch_idx) > 1:
                hidden_states = outputs.hidden_states[-1]  # (B, T, H)
                h_steps = hidden_states[
                    latent_batch_idx.to(device),
                    latent_step_pos.to(device),
                ]  # (N_steps, H)
                z_text = latent_head(h_steps)
                z_video = nn.functional.normalize(
                    latent_vjepa.to(device, dtype=z_text.dtype), dim=-1
                )
                l_latent = info_nce_loss(z_text, z_video, temperature)
                total_latent_loss += l_latent.item()
                n_latent_batches += 1

    text_loss = total_loss / max(total_tokens, 1)
    latent_loss = total_latent_loss / max(n_latent_batches, 1) if n_latent_batches > 0 else 0.0
    combined = text_loss + latent_weight * latent_loss if n_latent_batches > 0 else text_loss

    return {
        "val_text_loss": text_loss,
        "val_latent_loss": latent_loss,
        "val_combined_loss": combined,
    }


def generate_samples(model, tokenizer, dataset, device, n_samples=50, max_new_tokens=256):
    """
    Generate completions for a few samples and compute structural metrics:
      - valid_json_rate: fraction of outputs parseable as JSON with next_steps
      - exact_match_rate: fraction matching gold output exactly
    """
    model.eval()
    n_valid = 0
    n_exact = 0
    n_total = 0

    indices = list(range(min(n_samples, len(dataset.samples))))

    for idx in indices:
        sample = dataset.samples[idx]
        prompt = sample["input_text"] + PROMPT_SUFFIX

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_ids = gen_ids[0][inputs["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        try:
            pred = json.loads(pred_text)
            if "next_steps" in pred:
                n_valid += 1
                gold = json.loads(sample["output_text"])
                if pred["next_steps"] == gold.get("next_steps"):
                    n_exact += 1
        except (json.JSONDecodeError, KeyError):
            pass

        n_total += 1

    return {
        "valid_json_rate": n_valid / max(n_total, 1),
        "exact_match_rate": n_exact / max(n_total, 1),
        "n_eval_gen_samples": n_total,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    _lazy_imports()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load model
    model, tokenizer, hidden_dim = load_model_and_tokenizer(args)

    # Determine device
    if not args.use_qlora:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        # QLoRA uses device_map="auto", model is already on device
        device = next(model.parameters()).device

    print(f"Device: {device}")

    # --- Latent Projection Head (VLWM §3.1.1, Eq. 3-4) ---
    latent_head = None
    use_latent_grounding = (
        args.latent_dir is not None
        and os.path.isdir(args.latent_dir or "")
    )

    if use_latent_grounding:
        latent_head = LatentProjectionHead(
            llm_hidden_dim=hidden_dim,
            vjepa_dim=args.vjepa_dim,
        ).to(device, dtype=torch.bfloat16)
        head_params = sum(p.numel() for p in latent_head.parameters())
        print(f"\n  Latent Projection Head:")
        print(f"    LLM hidden dim:  {hidden_dim}")
        print(f"    V-JEPA dim:      {args.vjepa_dim}")
        print(f"    Head params:     {head_params:,}")
        print(f"    InfoNCE τ:       {args.temperature}")
        print(f"    Latent weight α: {args.latent_weight}")
        print(f"    Latent dir:      {args.latent_dir}")
    else:
        if args.latent_dir:
            print(f"\n  ⚠️  --latent_dir={args.latent_dir} not found, "
                  f"training text-only (no grounding)")
        else:
            print(f"\n  Training text-only (no latent grounding)")

    # Data
    train_ds = CausalLMDataset(
        args.train_data, tokenizer, args.max_seq_len,
        latent_dir=args.latent_dir if use_latent_grounding else None,
        vjepa_dim=args.vjepa_dim,
    )
    val_ds = None
    if args.val_data and os.path.exists(args.val_data):
        val_ds = CausalLMDataset(
            args.val_data, tokenizer, args.max_seq_len,
            latent_dir=args.latent_dir if use_latent_grounding else None,
            vjepa_dim=args.vjepa_dim,
        )

    pad_id = tokenizer.pad_token_id or 0
    from functools import partial
    _collate = partial(collate_fn, pad_token_id=pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_collate,
            num_workers=2,
            pin_memory=True,
        )

    # Optimizer — LoRA parameters + latent head (if present)
    trainable_params = list(
        filter(lambda p: p.requires_grad, model.parameters())
    )
    if latent_head is not None:
        trainable_params += list(latent_head.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # Steps accounting for gradient accumulation
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(0.06 * total_steps)

    scheduler = None
    if get_linear_schedule_with_warmup is not None:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    best_val_loss = float("inf")
    log_entries = []

    print(f"\n{'='*60}")
    print(f"Training System-1 (PLM + LoRA)")
    print(f"  Base model:    {args.model_name}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  QLoRA:         {args.use_qlora}")
    print(f"  Grounding:     {'InfoNCE (VLWM Eq. 3-4)' if use_latent_grounding else 'None (text-only)'}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch (eff):   {effective_batch} "
          f"({args.batch_size} × {args.gradient_accumulation_steps})")
    print(f"  LR:            {args.lr}")
    print(f"  Total steps:   {total_steps}")
    print(f"  Warmup:        {warmup_steps}")
    print(f"  Max seq len:   {args.max_seq_len}")
    print(f"  Output:        {out_dir}")
    print(f"{'='*60}\n")

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if latent_head is not None:
            latent_head.train()

        epoch_text_loss = 0.0
        epoch_latent_loss = 0.0
        n_micro_steps = 0
        n_latent_steps = 0
        t0 = time.time()

        optimizer.zero_grad()

        for step_idx, batch in enumerate(train_loader, 1):
            # Pop latent-specific keys before sending to model
            latent_batch_idx = batch.pop("latent_batch_idx", None)
            latent_step_pos = batch.pop("latent_step_pos", None)
            latent_vjepa = batch.pop("latent_vjepa", None)

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass — request hidden states if grounding
            need_hidden = (
                latent_head is not None
                and latent_batch_idx is not None
            )
            outputs = model(
                **batch,
                output_hidden_states=need_hidden,
            )

            # Text loss (Eq. 1: next-token prediction)
            text_loss = outputs.loss

            # Latent grounding loss (Eq. 3-4: InfoNCE)
            latent_loss = torch.tensor(0.0, device=device)
            if need_hidden and latent_batch_idx is not None and len(latent_batch_idx) > 1:
                # Extract hidden states at step boundary positions
                hidden_states = outputs.hidden_states[-1]  # (B, T, H)
                h_steps = hidden_states[
                    latent_batch_idx.to(device),
                    latent_step_pos.to(device),
                ]  # (N_steps, H)

                # Project to V-JEPA space
                z_text = latent_head(h_steps)  # (N_steps, D_vjepa)

                # Normalize V-JEPA latents
                z_video = nn.functional.normalize(
                    latent_vjepa.to(device, dtype=z_text.dtype), dim=-1
                )  # (N_steps, D_vjepa)

                # Symmetric InfoNCE
                latent_loss = info_nce_loss(
                    z_text, z_video, args.temperature
                )
                epoch_latent_loss += latent_loss.item()
                n_latent_steps += 1

            # Combined loss: L = L_text + α · L_latent
            total_loss = text_loss + args.latent_weight * latent_loss
            scaled_loss = total_loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            epoch_text_loss += text_loss.item()
            n_micro_steps += 1

            if step_idx % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.max_grad_norm
                )
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_text_loss = epoch_text_loss / max(n_micro_steps, 1)
        avg_latent_loss = epoch_latent_loss / max(n_latent_steps, 1) if n_latent_steps > 0 else 0.0
        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train_text_loss": round(avg_text_loss, 5),
            "train_latent_loss": round(avg_latent_loss, 5),
            "train_loss": round(avg_text_loss + args.latent_weight * avg_latent_loss, 5),
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(elapsed, 1),
        }

        # Validation
        if val_loader is not None:
            val_metrics = evaluate(
                model, val_loader, device,
                latent_head=latent_head,
                latent_weight=args.latent_weight,
                temperature=args.temperature,
            )
            entry.update({
                "val_text_loss": round(val_metrics["val_text_loss"], 5),
                "val_latent_loss": round(val_metrics["val_latent_loss"], 5),
                "val_loss": round(val_metrics["val_combined_loss"], 5),
            })

            # Structural metrics every 2 epochs (generation is slow)
            if epoch % 2 == 0 or epoch == args.epochs:
                gen_metrics = generate_samples(
                    model, tokenizer, val_ds, device,
                    n_samples=min(100, len(val_ds)),
                )
                entry.update(gen_metrics)

            # Save best adapter (by combined val loss)
            val_loss = val_metrics["val_combined_loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                entry["is_best"] = True
                best_dir = out_dir / "best_adapter"
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                if latent_head is not None:
                    torch.save(
                        latent_head.state_dict(),
                        best_dir / "latent_head.pt",
                    )
                print(f"  ★ New best adapter saved (val_loss={best_val_loss:.4f})")
        else:
            best_dir = out_dir / "best_adapter"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            if latent_head is not None:
                torch.save(
                    latent_head.state_dict(),
                    best_dir / "latent_head.pt",
                )

        log_entries.append(entry)

        # Write log incrementally
        with open(log_path, "w") as f:
            for e in log_entries:
                f.write(json.dumps(e) + "\n")

        # Print progress
        val_str = ""
        if "val_loss" in entry:
            val_str = f"  val={entry['val_loss']:.4f}"
            if entry.get("val_latent_loss", 0) > 0:
                val_str += f"(t={entry['val_text_loss']:.3f}+l={entry['val_latent_loss']:.3f})"
        gen_str = ""
        if "valid_json_rate" in entry:
            gen_str = (
                f"  json={entry['valid_json_rate']:.1%}"
                f"  exact={entry['exact_match_rate']:.1%}"
            )
        latent_str = ""
        if avg_latent_loss > 0:
            latent_str = f"  L_lat={avg_latent_loss:.3f}"

        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  L_text={avg_text_loss:.4f}"
            f"{latent_str}"
            f"{val_str}{gen_str}"
            f"  lr={optimizer.param_groups[0]['lr']:.2e}"
            f"  [{elapsed:.0f}s]"
        )

    # Save final adapter
    final_dir = out_dir / "final_adapter"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    if latent_head is not None:
        torch.save(latent_head.state_dict(), final_dir / "latent_head.pt")

    print(f"\nTraining complete.")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Log:           {log_path}")
    print(f"  Best adapter:  {out_dir / 'best_adapter'}")
    print(f"  Final adapter: {final_dir}")
    if latent_head is not None:
        print(f"  Latent head:   {out_dir / 'best_adapter' / 'latent_head.pt'}")
    print(f"\nTo load for inference:")
    print(f"  from peft import PeftModel")
    print(f"  base = AutoModelForImageTextToText.from_pretrained('{args.model_name}')")
    print(f"  model = PeftModel.from_pretrained(base, '{out_dir / 'best_adapter'}')")
    if latent_head is not None:
        print(f"  # Load latent head for grounding:")
        print(f"  head = LatentProjectionHead({hidden_dim}, {args.vjepa_dim})")
        print(f"  head.load_state_dict(torch.load('{out_dir / 'best_adapter' / 'latent_head.pt'}'))")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train System-1 rollout generator (PerceptionLM + LoRA)"
    )

    # Data
    parser.add_argument("--train_data", type=str,
                        default="data/crosstask/system1_train.jsonl")
    parser.add_argument("--val_data", type=str,
                        default="data/crosstask/system1_val.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/system1_plm_lora")

    # Model
    parser.add_argument("--model_name", type=str,
                        default="facebook/Perception-LM-1B",
                        help="HF model ID. Options: "
                             "facebook/Perception-LM-{1B,3B,8B}")

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (higher = more capacity, more params)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (scaling = alpha/r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout on LoRA layers")
    parser.add_argument("--use_qlora", action="store_true",
                        help="Use 4-bit QLoRA for lower memory")

    # Latent Grounding (VLWM §3.1.1, Eq. 3-4)
    parser.add_argument("--latent_dir", type=str, default=None,
                        help="Path to V-JEPA latents "
                             "(e.g. data/crosstask/vjepa_latents). "
                             "If provided, enables InfoNCE grounding loss.")
    parser.add_argument("--vjepa_dim", type=int, default=1024,
                        help="V-JEPA feature dimension "
                             "(1024 for ViT-L, 1280 for ViT-H)")
    parser.add_argument("--latent_weight", type=float, default=0.1,
                        help="Weight α for latent grounding loss: "
                             "L = L_text + α·L_latent")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE temperature τ "
                             "(CLIP/VLWM default: 0.07)")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Micro-batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Effective batch = batch_size × this")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (LoRA typically uses 1e-4 to 3e-4)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_seq_len", type=int, default=768,
                        help="Max sequence length (prompt + completion)")

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
