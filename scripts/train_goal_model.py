#!/usr/bin/env python3
"""
Train Goal Model: infer the task goal (and optionally remaining plan)
from a partial observation trajectory.

Model : T5-small (60M params) — seq2seq fine-tuning with task prefixes
Tasks:
  A. goal_prediction:  partial trajectory → goal text  (18-class)
  B. goal_and_plan:    partial trajectory → JSON {goal, remaining_plan}

Both tasks share one T5 model — the task-type is embedded in the prompt
instruction line, so the model learns both jointly.

Saves:
  checkpoints/goal_model/best_model/         (HF model + tokenizer)
  checkpoints/goal_model/training_log.jsonl  (per-epoch loss/metrics)

Usage (HPC):
    python3 scripts/train_goal_model.py \
        --train_data  data/crosstask/goal_train.jsonl \
        --val_data    data/crosstask/goal_val.jsonl \
        --output_dir  checkpoints/goal_model \
        --model_name  google/flan-t5-small \
        --epochs 20 \
        --batch_size 16 \
        --lr 3e-4 \
        --max_input_len 384 \
        --max_target_len 512 \
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
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
_HF_LOADED = False
def _lazy_hf():
    global _HF_LOADED, AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
    if _HF_LOADED:
        return
    from transformers import AutoTokenizer as _AT, T5ForConditionalGeneration as _T5
    try:
        from transformers import get_linear_schedule_with_warmup as _sched
    except ImportError:
        _sched = None
    AutoTokenizer = _AT
    T5ForConditionalGeneration = _T5
    get_linear_schedule_with_warmup = _sched
    _HF_LOADED = True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GoalDataset(Dataset):
    """Loads goal_*.jsonl, returns (input_text, output_text, task_type)."""

    def __init__(self, jsonl_path: str):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    self.samples.append((
                        rec["input_text"],
                        rec["output_text"],
                        rec["task"],  # "goal_prediction" or "goal_and_plan"
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_input_len, max_target_len):
    inputs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    task_types = [b[2] for b in batch]

    input_enc = tokenizer(
        inputs,
        max_length=max_input_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with tokenizer.as_target_tokenizer():
        target_enc = tokenizer(
            targets,
            max_length=max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    labels = target_enc.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_enc.input_ids,
        "attention_mask": input_enc.attention_mask,
        "labels": labels,
        "_task_types": task_types,  # not a tensor, just for metrics
    }


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

def compute_metrics(model, val_ds, tokenizer, device, max_target_len):
    """
    Metrics on a sample of the val set (up to 300 samples for speed):
      - goal_accuracy:   correct goal string (Task A)
      - plan_valid_json: valid JSON with correct keys (Task B)
      - val_loss:        average cross-entropy on the full val set
    """
    model.eval()

    # ── Loss on full val set via batched forward ──
    from functools import partial
    _collate = partial(
        collate_fn, tokenizer=tokenizer,
        max_input_len=384, max_target_len=max_target_len,
    )
    loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=_collate)
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(**batch_gpu)
            total_loss += out.loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    # ── Generative metrics on a subset ──
    n_goal_correct = 0
    n_goal_total = 0
    n_plan_valid = 0
    n_plan_total = 0

    # Separate by task type
    subset = val_ds.samples[:300]
    for inp, tgt, task_type in subset:
        enc = tokenizer(inp, max_length=384, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(**enc, max_new_tokens=max_target_len,
                                     num_beams=1, do_sample=False)
        pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        if task_type == "goal_prediction":
            n_goal_total += 1
            if pred.strip() == tgt.strip():
                n_goal_correct += 1
        elif task_type == "goal_and_plan":
            n_plan_total += 1
            try:
                obj = json.loads(pred)
                if "goal" in obj and "remaining_plan" in obj:
                    n_plan_valid += 1
            except (json.JSONDecodeError, KeyError):
                pass

    return {
        "val_loss": round(avg_loss, 5),
        "goal_accuracy": round(n_goal_correct / max(n_goal_total, 1), 4),
        "plan_valid_json_rate": round(n_plan_valid / max(n_plan_total, 1), 4),
        "n_goal_eval": n_goal_total,
        "n_plan_eval": n_plan_total,
    }


# ---------------------------------------------------------------------------
# Training loop
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

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    print(f"Loading training data: {args.train_data}")
    train_ds = GoalDataset(args.train_data)
    print(f"  {len(train_ds)} training samples")

    val_ds = None
    if args.val_data and os.path.exists(args.val_data):
        val_ds = GoalDataset(args.val_data)
        print(f"  {len(val_ds)} validation samples")

    from functools import partial
    _collate = partial(
        collate_fn, tokenizer=tokenizer,
        max_input_len=args.max_input_len, max_target_len=args.max_target_len,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=2, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.06 * total_steps)

    scheduler = None
    if get_linear_schedule_with_warmup is not None:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    best_val_loss = float("inf")
    log_entries = []

    print(f"\n{'='*60}")
    print(f"Training Goal Model")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Total steps: {total_steps}")
    print(f"  Output:      {out_dir}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps = 0
        t0 = time.time()

        for batch in train_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            loss = model(**batch_gpu).loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            epoch_loss += loss.item()
            n_steps += 1

        avg_train_loss = epoch_loss / max(n_steps, 1)
        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(elapsed, 1),
        }

        if val_ds:
            val_m = compute_metrics(model, val_ds, tokenizer, device, args.max_target_len)
            entry.update(val_m)

            if val_m["val_loss"] < best_val_loss:
                best_val_loss = val_m["val_loss"]
                entry["is_best"] = True
                best_dir = out_dir / "best_model"
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"  ★ New best model saved (val_loss={best_val_loss:.4f})")
        else:
            best_dir = out_dir / "best_model"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

        log_entries.append(entry)
        with open(log_path, "w") as f:
            for e in log_entries:
                f.write(json.dumps(e) + "\n")

        val_str = ""
        if "val_loss" in entry:
            val_str = (
                f"  val_loss={entry['val_loss']:.4f}"
                f"  goal_acc={entry['goal_accuracy']:.1%}"
                f"  plan_json={entry['plan_valid_json_rate']:.1%}"
            )
        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  train_loss={avg_train_loss:.4f}"
            f"{val_str}"
            f"  [{elapsed:.0f}s]"
        )

    final_dir = out_dir / "final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\nTraining complete.")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Log:           {log_path}")
    print(f"  Best model:    {out_dir / 'best_model'}")
    print(f"  Final model:   {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Goal Model (T5 seq2seq)")
    parser.add_argument("--train_data", default="data/crosstask/goal_train.jsonl")
    parser.add_argument("--val_data", default="data/crosstask/goal_val.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/goal_model")
    parser.add_argument("--model_name", default="google/flan-t5-small")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_input_len", type=int, default=384)
    parser.add_argument("--max_target_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
