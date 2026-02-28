#!/usr/bin/env python3
"""
Train Critic Model: score trajectory quality for System-2 planning.

Architecture:
  Encoder:  sentence-transformers/all-MiniLM-L6-v2  (22M params, 384-dim)
  Head:     MLP  384 → 256 → 1  (scalar cost)

  Each trajectory is encoded as:
    "[GOAL] {goal} [SEP] {step_1} [SEP] {step_2} [SEP] ..."
  and the encoder produces a single 384-dim vector → MLP → scalar cost C.

Loss (VLWM Eq. 2):
  L = max(0, margin + C_pos − C_neg) + λ (C_pos² + C_neg²)

  Three ranking pairs per sample:
    1. (C_good,  C_base)      — valid continuation < stopping
    2. (C_base,  C_bad)       — stopping < distractor
    3. (C_base,  C_shuffled)  — correct order < shuffled

  Hyperparameters from VLWM: margin=1.0, λ=0.01

Why not a full LLM?
  VLWM uses Llama-3.2-1B as critic backbone on 350k+ training pairs.
  We have ~9.3k training samples. A sentence-transformer + MLP is the
  right capacity — it encodes semantic meaning without 1B parameters.

Saves:
  checkpoints/critic/best_model.pt        (state dict)
  checkpoints/critic/training_log.jsonl   (per-epoch metrics)

Usage (HPC):
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
_HF_LOADED = False
def _lazy_hf():
    global _HF_LOADED, AutoTokenizer, AutoModel
    if _HF_LOADED:
        return
    from transformers import AutoTokenizer as _AT, AutoModel as _AM
    AutoTokenizer = _AT
    AutoModel = _AM
    _HF_LOADED = True


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CriticModel(nn.Module):
    """
    Encode a goal+trajectory string → scalar cost.

    Uses a frozen/fine-tunable sentence encoder + MLP head.
    """

    def __init__(self, encoder_name: str, freeze_encoder: bool = False):
        super().__init__()
        _lazy_hf()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_dim = self.encoder.config.hidden_size  # 384 for MiniLM

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _mean_pool(self, model_output, attention_mask):
        """Mean pooling over token embeddings, respecting attention mask."""
        token_embeddings = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(
            mask_expanded.sum(dim=1), min=1e-9
        )

    def forward(self, input_ids, attention_mask):
        """Returns scalar cost for each item in the batch."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(output, attention_mask)
        cost = self.head(pooled).squeeze(-1)  # (B,)
        return cost


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def format_trajectory_text(goal: str, steps: list) -> str:
    """
    Format a goal + step list into a single string for the encoder.
    "[GOAL] Make Pancakes [SEP] pour egg | Egg poured [SEP] add flour | Flour added"
    """
    parts = [f"[GOAL] {goal}"]
    for step in steps:
        parts.append(step)
    return " [SEP] ".join(parts)


class CriticDataset(Dataset):
    """
    Each sample produces 3 ranking pairs:
      (good, base), (base, bad), (base, shuffled)

    We store them as text pairs and yield them individually.
    """

    def __init__(self, jsonl_path: str):
        self.pairs = []  # (positive_text, negative_text)
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                goal = rec["goal"]

                base_text = format_trajectory_text(goal, rec["base"])
                good_text = format_trajectory_text(goal, rec["good"])
                bad_text = format_trajectory_text(goal, rec["bad"])
                shuffled_text = format_trajectory_text(goal, rec["shuffled"])

                # Pair 1: C_good < C_base → positive=good, negative=base
                self.pairs.append((good_text, base_text))
                # Pair 2: C_base < C_bad → positive=base, negative=bad
                self.pairs.append((base_text, bad_text))
                # Pair 3: C_base < C_shuffled → positive=base, negative=shuffled
                self.pairs.append((base_text, shuffled_text))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch, tokenizer, max_len):
    """Tokenize positive and negative texts."""
    pos_texts = [b[0] for b in batch]
    neg_texts = [b[1] for b in batch]

    pos_enc = tokenizer(
        pos_texts, max_length=max_len, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    neg_enc = tokenizer(
        neg_texts, max_length=max_len, padding="max_length",
        truncation=True, return_tensors="pt",
    )

    return {
        "pos_input_ids": pos_enc.input_ids,
        "pos_attention_mask": pos_enc.attention_mask,
        "neg_input_ids": neg_enc.input_ids,
        "neg_attention_mask": neg_enc.attention_mask,
    }


# ---------------------------------------------------------------------------
# Loss function (VLWM Eq. 2)
# ---------------------------------------------------------------------------

def ranking_loss(c_pos, c_neg, margin=1.0, lambda_reg=0.01):
    """
    L = max(0, margin + c_pos - c_neg) + λ (c_pos² + c_neg²)

    c_pos should be LOWER cost (better trajectory)
    c_neg should be HIGHER cost (worse trajectory)
    """
    hinge = torch.clamp(margin + c_pos - c_neg, min=0.0)
    reg = lambda_reg * (c_pos ** 2 + c_neg ** 2)
    return (hinge + reg).mean()


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

def compute_val_metrics(model, val_loader, device, margin, lambda_reg):
    """Compute val loss and ranking accuracy."""
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for batch in val_loader:
            c_pos = model(
                batch["pos_input_ids"].to(device),
                batch["pos_attention_mask"].to(device),
            )
            c_neg = model(
                batch["neg_input_ids"].to(device),
                batch["neg_attention_mask"].to(device),
            )

            loss = ranking_loss(c_pos, c_neg, margin, lambda_reg)
            total_loss += loss.item()

            # Ranking accuracy: how often c_pos < c_neg?
            n_correct += (c_pos < c_neg).sum().item()
            n_total += c_pos.size(0)

    avg_loss = total_loss / max(len(val_loader), 1)
    accuracy = n_correct / max(n_total, 1)
    return {"val_loss": round(avg_loss, 5), "ranking_accuracy": round(accuracy, 4)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    _lazy_hf()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)

    # Model
    print(f"Loading encoder: {args.encoder_name}")
    model = CriticModel(args.encoder_name, freeze_encoder=args.freeze_encoder)
    model.to(device)

    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {n_params_total:,}")
    print(f"Trainable params: {n_params_train:,}")

    # Data
    print(f"Loading training data: {args.train_data}")
    train_ds = CriticDataset(args.train_data)
    print(f"  {len(train_ds)} ranking pairs ({len(train_ds)//3} samples × 3 pairs)")

    val_ds = None
    val_loader = None
    if args.val_data and os.path.exists(args.val_data):
        val_ds = CriticDataset(args.val_data)
        print(f"  {len(val_ds)} val ranking pairs")

    from functools import partial
    _collate = partial(collate_fn, tokenizer=tokenizer, max_len=args.max_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=2, pin_memory=True, drop_last=True,
    )

    if val_ds:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=_collate, num_workers=2, pin_memory=True,
        )

    # Optimizer — different LR for encoder vs head
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * 10},  # head trains faster
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.06 * total_steps)

    # Cosine annealing with warmup (manual)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    best_val_acc = 0.0
    log_entries = []

    print(f"\n{'='*60}")
    print(f"Training Critic Model")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}  (head: {args.lr * 10})")
    print(f"  Margin:      {args.margin}")
    print(f"  λ_reg:       {args.lambda_reg}")
    print(f"  Max len:     {args.max_len}")
    print(f"  Output:      {out_dir}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for batch in train_loader:
            c_pos = model(
                batch["pos_input_ids"].to(device),
                batch["pos_attention_mask"].to(device),
            )
            c_neg = model(
                batch["neg_input_ids"].to(device),
                batch["neg_attention_mask"].to(device),
            )

            loss = ranking_loss(c_pos, c_neg, args.margin, args.lambda_reg)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_correct += (c_pos < c_neg).sum().item()
            epoch_total += c_pos.size(0)

        avg_loss = epoch_loss / max(len(train_loader), 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 5),
            "train_ranking_acc": round(train_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(elapsed, 1),
        }

        if val_loader:
            val_m = compute_val_metrics(
                model, val_loader, device, args.margin, args.lambda_reg
            )
            entry.update(val_m)

            if val_m["ranking_accuracy"] > best_val_acc:
                best_val_acc = val_m["ranking_accuracy"]
                entry["is_best"] = True
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "encoder_name": args.encoder_name,
                    "epoch": epoch,
                    "ranking_accuracy": best_val_acc,
                }, out_dir / "best_model.pt")
                print(f"  ★ New best model (ranking_acc={best_val_acc:.1%})")
        else:
            torch.save({"model_state_dict": model.state_dict(),
                        "encoder_name": args.encoder_name, "epoch": epoch},
                       out_dir / "best_model.pt")

        log_entries.append(entry)
        with open(log_path, "w") as f:
            for e in log_entries:
                f.write(json.dumps(e) + "\n")

        val_str = ""
        if "val_loss" in entry:
            val_str = (
                f"  val_loss={entry['val_loss']:.4f}"
                f"  val_rank_acc={entry['ranking_accuracy']:.1%}"
            )
        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  loss={avg_loss:.4f}"
            f"  train_acc={train_acc:.1%}"
            f"{val_str}"
            f"  [{elapsed:.0f}s]"
        )

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "encoder_name": args.encoder_name,
        "epoch": args.epochs,
    }, out_dir / "final_model.pt")

    print(f"\nTraining complete.")
    print(f"  Best ranking acc: {best_val_acc:.1%}")
    print(f"  Log:              {log_path}")
    print(f"  Best model:       {out_dir / 'best_model.pt'}")
    print(f"  Final model:      {out_dir / 'final_model.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train Critic Model (ranking loss)")
    parser.add_argument("--train_data", default="data/crosstask/critic_train.jsonl")
    parser.add_argument("--val_data", default="data/crosstask/critic_val.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/critic")
    parser.add_argument("--encoder_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder weights, only train MLP head")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=256,
                        help="Max token length for trajectory encoding")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
