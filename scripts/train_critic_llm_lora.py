#!/usr/bin/env python3
"""
Train Critic Model using Llama-3.2-1B + LoRA,
matching the VLWM paper's actual approach.

Background (VLWM, Chen et al. 2025, §3.1.2):
  - "The critic model is initialized from Llama-3.2-1B and trained for one
     epoch with a batch size of 128, maximum context length of 1536 tokens
     using a single node of 8×H100 GPUs."
  - The critic independently predicts cost C = critic(goal, trajectory),
     a scalar measuring semantic distance between the trajectory and the
     desired goal state.
  - Loss is a ranking loss with margin + cost centering regularization
     (Eq. 2): L = max(0, margin + C_pos − C_neg) + λ(C_pos² + C_neg²)
  - Three ranking pairs: (good,base), (base,bad), (base,shuffled)
  - Hyperparameters: margin=1.0, λ=0.01

Our approach:
  - We use Llama-3.2-1B (meta-llama/Llama-3.2-1B) — the EXACT same model
    as the VLWM paper, with LoRA adapters.
  - VLWM does full fine-tuning on 350k+ pairs (2.7k optimizer steps).
    We have ~9.3k training samples (27.7k ranking pairs).
    LoRA is the correct choice: trains ~2-4M params instead of 1.2B,
    prevents overfitting, and fits on a single GPU.

Architecture:
  Input:  "[GOAL] {goal} [TRAJ] {step_1} [SEP] {step_2} [SEP] ..."
  Model:  Llama-3.2-1B (causal LM) with LoRA adapters
  Pool:   Last non-pad token hidden state (like sequence classification)
  Head:   Linear(hidden_dim, 1) → scalar cost C
  Loss:   VLWM ranking loss (Eq. 2)

Why last-token pooling?
  For causal LMs, the last token has attended to all previous tokens.
  This is the standard approach for sequence classification/reward models
  with decoder-only transformers (used in RLHF reward models, DPO, etc.).

Prereqs:
  - pip install peft bitsandbytes accelerate
  - Accept Llama-3.2 license on HuggingFace (Meta Community License)

Saves:
  checkpoints/critic_llm_lora/best_adapter/  (LoRA weights + cost head)
  checkpoints/critic_llm_lora/training_log.jsonl

Usage (HPC):
  python3 scripts/train_critic_llm_lora.py \\
      --train_data  data/crosstask/critic_train.jsonl \\
      --val_data    data/crosstask/critic_val.jsonl \\
      --output_dir  checkpoints/critic_llm_lora \\
      --model_name  meta-llama/Llama-3.2-1B \\
      --lora_r 16 \\
      --lora_alpha 32 \\
      --epochs 5 \\
      --batch_size 4 \\
      --gradient_accumulation_steps 8 \\
      --lr 2e-4 \\
      --margin 1.0 \\
      --lambda_reg 0.01 \\
      --max_len 512 \\
      --seed 42

  QLoRA variant (lower memory):
  python3 scripts/train_critic_llm_lora.py \\
      --use_qlora \\
      --train_data  data/crosstask/critic_train.jsonl \\
      --val_data    data/crosstask/critic_val.jsonl \\
      --output_dir  checkpoints/critic_llm_qlora \\
      --model_name  meta-llama/Llama-3.2-1B \\
      --batch_size 8 \\
      --gradient_accumulation_steps 4 \\
      --seed 42
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
# Lazy imports
# ---------------------------------------------------------------------------
_LOADED = False


def _lazy_imports():
    global _LOADED
    global AutoTokenizer, AutoModelForCausalLM
    global BitsAndBytesConfig
    global LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    global get_linear_schedule_with_warmup
    if _LOADED:
        return

    from transformers import (
        AutoTokenizer as _AT,
        AutoModelForCausalLM as _ACLM,
    )
    AutoTokenizer = _AT
    AutoModelForCausalLM = _ACLM

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

# LoRA targets — same as standard Llama LoRA configs
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Critic model: LLM backbone + scalar cost head
# ---------------------------------------------------------------------------

class LLMCriticWithHead(nn.Module):
    """
    Wraps a causal LM (with LoRA) + a linear head that maps the
    last-token hidden state to a scalar cost.

    This is separate from the LoRA-adapted LM because PEFT only manages
    the adapter layers. The cost head is a small additional module that
    we train alongside the adapters.
    """

    def __init__(self, base_model, hidden_dim: int):
        super().__init__()
        self.base_model = base_model
        self.cost_head = nn.Linear(hidden_dim, 1, bias=True)
        # Initialize head near zero so initial costs are near 0
        nn.init.zeros_(self.cost_head.weight)
        nn.init.zeros_(self.cost_head.bias)

    def forward(self, input_ids, attention_mask):
        """
        Returns scalar cost for each item in the batch.
        Uses last non-pad token hidden state (standard for causal LM classification).
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Last layer hidden states: (batch, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states[-1]

        # Find position of last non-pad token for each sequence
        # attention_mask: 1 for real tokens, 0 for padding
        # Sum along seq dim, subtract 1 to get last real token index
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, seq_lengths]  # (batch, hidden_dim)

        cost = self.cost_head(last_hidden).squeeze(-1)  # (batch,)
        return cost

    def save_head(self, path):
        """Save just the cost head weights."""
        torch.save(self.cost_head.state_dict(), path)

    def load_head(self, path, device="cpu"):
        """Load cost head weights."""
        state = torch.load(path, map_location=device)
        self.cost_head.load_state_dict(state)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def format_trajectory(goal: str, steps: list) -> str:
    """
    Format goal + steps into a single string for the LLM.
    Matches the VLWM critic input format: goal + trajectory.
    """
    parts = [f"[GOAL] {goal}"]
    for step in steps:
        parts.append(step)
    return " [SEP] ".join(parts)


class CriticPairDataset(Dataset):
    """
    Each JSONL sample has goal + base/good/bad/shuffled trajectories.
    We produce 3 ranking pairs per sample:
      (good, base)     — good continuation has lower cost than stopping
      (base, bad)      — stopping has lower cost than distractor
      (base, shuffled) — correct order has lower cost than shuffled

    Returns (positive_text, negative_text) where C_pos should be < C_neg.
    """

    def __init__(self, jsonl_path: str):
        self.pairs = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                goal = rec["goal"]

                base_text = format_trajectory(goal, rec["base"])
                good_text = format_trajectory(goal, rec["good"])
                bad_text = format_trajectory(goal, rec["bad"])
                shuffled_text = format_trajectory(goal, rec["shuffled"])

                # Pair 1: C_good < C_base
                self.pairs.append((good_text, base_text))
                # Pair 2: C_base < C_bad
                self.pairs.append((base_text, bad_text))
                # Pair 3: C_base < C_shuffled
                self.pairs.append((base_text, shuffled_text))

        print(f"  {len(self.pairs)} ranking pairs "
              f"({len(self.pairs) // 3} samples × 3)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch, tokenizer, max_len):
    """Tokenize positive and negative texts with right-padding."""
    pos_texts = [b[0] for b in batch]
    neg_texts = [b[1] for b in batch]

    pos_enc = tokenizer(
        pos_texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    neg_enc = tokenizer(
        neg_texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return {
        "pos_input_ids": pos_enc.input_ids,
        "pos_attention_mask": pos_enc.attention_mask,
        "neg_input_ids": neg_enc.input_ids,
        "neg_attention_mask": neg_enc.attention_mask,
    }


# ---------------------------------------------------------------------------
# Loss (VLWM Eq. 2)
# ---------------------------------------------------------------------------

def ranking_loss(c_pos, c_neg, margin=1.0, lambda_reg=0.01):
    """
    L = max(0, margin + c_pos − c_neg) + λ(c_pos² + c_neg²)

    c_pos = cost of BETTER trajectory (should be lower)
    c_neg = cost of WORSE trajectory (should be higher)

    The hinge term pushes c_pos to be at least `margin` below c_neg.
    The regularization term prevents cost values from diverging.
    """
    hinge = torch.clamp(margin + c_pos - c_neg, min=0.0)
    reg = lambda_reg * (c_pos ** 2 + c_neg ** 2)
    return (hinge + reg).mean()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    """Load Llama-3.2-1B with LoRA + cost head."""
    _lazy_imports()

    print(f"Loading model: {args.model_name}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Right padding for last-token pooling
    tokenizer.padding_side = "right"

    # Quantization config
    bnb_config = None
    if args.use_qlora:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes required for QLoRA")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("  Using QLoRA (4-bit quantized base)")

    # Load base causal LM
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not args.use_qlora else None,
        device_map="auto" if args.use_qlora else None,
        trust_remote_code=True,
    )

    # Prepare for QLoRA
    if args.use_qlora:
        base_model = prepare_model_for_kbit_training(base_model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=DEFAULT_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=not args.use_qlora,
        init_lora_weights="gaussian",
    )
    base_model = get_peft_model(base_model, lora_config)

    trainable, total = base_model.get_nb_trainable_parameters()
    pct = trainable / total * 100
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable (LoRA):     {trainable:>12,}  ({pct:.2f}%)")

    # Get hidden dim from model config
    hidden_dim = base_model.config.hidden_size
    print(f"  Hidden dim:           {hidden_dim}")

    # Wrap with cost head
    critic = LLMCriticWithHead(base_model, hidden_dim)

    head_params = sum(p.numel() for p in critic.cost_head.parameters())
    print(f"  Cost head params:     {head_params:>12,}")
    print(f"  Total trainable:      {trainable + head_params:>12,}")

    return critic, tokenizer


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def compute_val_metrics(critic, val_loader, device, margin, lambda_reg):
    """Compute val loss and ranking accuracy."""
    critic.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for batch in val_loader:
            c_pos = critic(
                batch["pos_input_ids"].to(device),
                batch["pos_attention_mask"].to(device),
            )
            c_neg = critic(
                batch["neg_input_ids"].to(device),
                batch["neg_attention_mask"].to(device),
            )

            loss = ranking_loss(c_pos, c_neg, margin, lambda_reg)
            total_loss += loss.item()

            n_correct += (c_pos < c_neg).sum().item()
            n_total += c_pos.size(0)

    avg_loss = total_loss / max(len(val_loader), 1)
    accuracy = n_correct / max(n_total, 1)
    return {
        "val_loss": round(avg_loss, 5),
        "ranking_accuracy": round(accuracy, 4),
    }


# ---------------------------------------------------------------------------
# Training
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
    critic, tokenizer = load_model(args)

    # Device
    if not args.use_qlora:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        critic.to(device)
    else:
        device = next(critic.base_model.parameters()).device
        # Move cost head to same device
        critic.cost_head.to(device)

    print(f"Device: {device}")

    # Data
    print(f"Loading training data: {args.train_data}")
    train_ds = CriticPairDataset(args.train_data)

    val_ds = None
    val_loader = None
    if args.val_data and os.path.exists(args.val_data):
        print(f"Loading validation data: {args.val_data}")
        val_ds = CriticPairDataset(args.val_data)

    from functools import partial
    _collate = partial(collate_fn, tokenizer=tokenizer, max_len=args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_collate,
            num_workers=2,
            pin_memory=True,
        )

    # Optimizer: LoRA params + cost head, all at same LR
    # (LoRA already handles scaling via alpha/r)
    trainable_params = [
        p for p in critic.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # Steps with gradient accumulation
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

    # Output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    best_val_acc = 0.0
    log_entries = []

    print(f"\n{'='*60}")
    print(f"Training Critic (Llama-3.2-1B + LoRA)")
    print(f"  Base model:    {args.model_name}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  QLoRA:         {args.use_qlora}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch (eff):   {effective_batch} "
          f"({args.batch_size} × {args.gradient_accumulation_steps})")
    print(f"  LR:            {args.lr}")
    print(f"  Margin:        {args.margin}")
    print(f"  λ_reg:         {args.lambda_reg}")
    print(f"  Max len:       {args.max_len}")
    print(f"  Total steps:   {total_steps}")
    print(f"  Warmup:        {warmup_steps}")
    print(f"  Output:        {out_dir}")
    print(f"{'='*60}\n")

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        critic.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_micro = 0
        t0 = time.time()

        optimizer.zero_grad()

        for step_idx, batch in enumerate(train_loader, 1):
            # Forward pass for positive and negative
            c_pos = critic(
                batch["pos_input_ids"].to(device),
                batch["pos_attention_mask"].to(device),
            )
            c_neg = critic(
                batch["neg_input_ids"].to(device),
                batch["neg_attention_mask"].to(device),
            )

            loss = ranking_loss(c_pos, c_neg, args.margin, args.lambda_reg)
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            epoch_loss += loss.item()
            epoch_correct += (c_pos < c_neg).detach().sum().item()
            epoch_total += c_pos.size(0)
            n_micro += 1

            if step_idx % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.max_grad_norm
                )
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / max(n_micro, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": round(avg_loss, 5),
            "train_ranking_acc": round(train_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(elapsed, 1),
        }

        # Validation
        if val_loader:
            val_m = compute_val_metrics(
                critic, val_loader, device, args.margin, args.lambda_reg
            )
            entry.update(val_m)

            if val_m["ranking_accuracy"] > best_val_acc:
                best_val_acc = val_m["ranking_accuracy"]
                entry["is_best"] = True
                # Save LoRA adapter
                best_dir = out_dir / "best_adapter"
                critic.base_model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                # Save cost head separately
                critic.save_head(best_dir / "cost_head.pt")
                print(
                    f"  ★ New best adapter saved "
                    f"(ranking_acc={best_val_acc:.1%})"
                )
        else:
            best_dir = out_dir / "best_adapter"
            critic.base_model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            critic.save_head(best_dir / "cost_head.pt")

        log_entries.append(entry)
        with open(log_path, "w") as f:
            for e in log_entries:
                f.write(json.dumps(e) + "\n")

        # Print progress
        val_str = ""
        if "val_loss" in entry:
            val_str = (
                f"  val_loss={entry['val_loss']:.4f}"
                f"  val_acc={entry['ranking_accuracy']:.1%}"
            )
        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  loss={avg_loss:.4f}"
            f"  train_acc={train_acc:.1%}"
            f"{val_str}"
            f"  lr={optimizer.param_groups[0]['lr']:.2e}"
            f"  [{elapsed:.0f}s]"
        )

    # Save final
    final_dir = out_dir / "final_adapter"
    critic.base_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    critic.save_head(final_dir / "cost_head.pt")

    print(f"\nTraining complete.")
    print(f"  Best ranking acc: {best_val_acc:.1%}")
    print(f"  Log:              {log_path}")
    print(f"  Best adapter:     {out_dir / 'best_adapter'}")
    print(f"  Final adapter:    {final_dir}")
    print(f"\nTo load for inference:")
    print(f"  from peft import PeftModel")
    print(f"  base = AutoModelForCausalLM.from_pretrained('{args.model_name}')")
    print(f"  lora_model = PeftModel.from_pretrained(base, "
          f"'{out_dir / 'best_adapter'}')")
    print(f"  critic = LLMCriticWithHead(lora_model, "
          f"base.config.hidden_size)")
    print(f"  critic.load_head('{out_dir / 'best_adapter' / 'cost_head.pt'}')")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Critic Model (Llama-3.2-1B + LoRA, VLWM ranking loss)"
    )

    # Data
    parser.add_argument("--train_data", type=str,
                        default="data/crosstask/critic_train.jsonl")
    parser.add_argument("--val_data", type=str,
                        default="data/crosstask/critic_val.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/critic_llm_lora")

    # Model
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Llama-3.2-1B",
                        help="HF model ID. Default matches VLWM paper exactly.")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (scaling = alpha/r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_qlora", action="store_true",
                        help="Use 4-bit QLoRA for lower memory")

    # VLWM loss hyperparameters (Eq. 2)
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Ranking loss margin (VLWM default: 1.0)")
    parser.add_argument("--lambda_reg", type=float, default=0.01,
                        help="Cost centering regularization (VLWM default: 0.01)")

    # Training
    parser.add_argument("--epochs", type=int, default=5,
                        help="VLWM trains 1 epoch on 350k pairs; "
                             "we have ~28k pairs so use a few more")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Micro-batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Effective batch = batch_size × this")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512,
                        help="Max token length for trajectory encoding "
                             "(VLWM uses 1536; our trajectories are shorter)")

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
