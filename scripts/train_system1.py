#!/usr/bin/env python3
"""
Train System-1 (rollout generator): given goal + interpretation + prefix,
predict the next k action/state-change steps.

Model : T5-small  (60M params) — seq2seq fine-tuning
Input : structured prompt from system1_{train,val,test}.jsonl
Output: JSON  {"next_steps": [{"action": "...", "state_change": "..."}, ...]}

Why T5-small and not a larger LLM?
  - VLWM uses PerceptionLM-8B trained on 180k videos (5.7M steps).
  - We have ~12k training samples from 2.1k CrossTask videos.
  - A 60M-param seq2seq model is the right capacity for this data scale.
  - Using an 8B model would massively overfit and violate research methodology.

Latent Grounding (VLWM §3.1.1, Eq. 3-4):
  When --latent_dir is provided with pre-extracted V-JEPA 2 latents, the
  training adds an InfoNCE contrastive loss that grounds the T5 decoder's
  per-step hidden states in the V-JEPA video latent space:
    L = L_text + α · L_latent
  A learned LatentProjectionHead maps decoder hidden states → V-JEPA dim.
  Without --latent_dir, training is text-only (pure seq2seq).

Saves:
  checkpoints/system1/best_model/         (HF model + tokenizer)
  checkpoints/system1/latent_head.pt      (projection head, if grounding)
  checkpoints/system1/training_log.jsonl  (per-epoch loss/metrics)

Usage (HPC):
    # Text-only:
    python3 scripts/train_system1.py \\
        --train_data  data/crosstask/system1_train.jsonl \\
        --val_data    data/crosstask/system1_val.jsonl \\
        --output_dir  checkpoints/system1 \\
        --model_name  google/flan-t5-small \\
        --epochs 30 --batch_size 16 --lr 3e-4 --seed 42

    # With V-JEPA latent grounding:
    python3 scripts/train_system1.py \\
        --train_data    data/crosstask/system1_train.jsonl \\
        --val_data      data/crosstask/system1_val.jsonl \\
        --output_dir    checkpoints/system1_grounded \\
        --model_name    google/flan-t5-small \\
        --latent_dir    data/crosstask/vjepa_latents \\
        --vjepa_dim 1024 --latent_weight 0.1 --temperature 0.07 \\
        --epochs 30 --batch_size 16 --lr 3e-4 --seed 42
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
# Lazy imports so the script fails fast on --help without needing transformers
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
# Latent Projection Head + InfoNCE (VLWM §3.1.1, Eq. 3-4)
# ---------------------------------------------------------------------------

class LatentProjectionHead(nn.Module):
    """
    Projects T5 decoder hidden states to V-JEPA latent space.

    For seq2seq models, we use the decoder hidden state at each step
    boundary (closing brace of each step dict in the JSON output).
    """

    def __init__(self, decoder_hidden_dim: int, vjepa_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(decoder_hidden_dim, vjepa_dim),
            nn.GELU(),
            nn.Linear(vjepa_dim, vjepa_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.net(h)
        return nn.functional.normalize(z, dim=-1)


def info_nce_loss(z_text, z_video, temperature=0.07):
    """Symmetric InfoNCE contrastive loss (VLWM Eq. 3-4)."""
    logits = torch.matmul(z_text, z_video.T) / temperature
    N = logits.size(0)
    labels = torch.arange(N, device=logits.device)
    loss_t2v = nn.functional.cross_entropy(logits, labels)
    loss_v2t = nn.functional.cross_entropy(logits.T, labels)
    return (loss_t2v + loss_v2t) / 2.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class System1Dataset(Dataset):
    """Loads JSONL, returns (input_text, output_text) pairs.

    When latent_dir is provided, also loads V-JEPA latents for each
    predicted step to enable InfoNCE grounding loss.
    """

    def __init__(self, jsonl_path: str, latent_dir: str = None,
                 vjepa_dim: int = 1024):
        self.samples = []
        self.latent_dir = latent_dir
        self.has_latents = latent_dir is not None and os.path.isdir(latent_dir or "")

        n_with_latents = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    entry = (rec["input_text"], rec["output_text"])
                    meta = rec.get("meta", {})

                    latents = None
                    if self.has_latents:
                        latents = self._load_step_latents(meta)
                        if latents is not None:
                            n_with_latents += 1

                    self.samples.append((rec["input_text"], rec["output_text"],
                                         meta, latents))

        print(f"  Loaded {len(self.samples)} samples from {jsonl_path}")
        if self.has_latents:
            print(f"  V-JEPA latents found for {n_with_latents}/{len(self.samples)} samples")

    def _load_step_latents(self, meta):
        """Load V-JEPA latent for each predicted step."""
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
                return None
            seg_latent = torch.load(pt_path, map_location="cpu")
            if seg_latent.dim() == 2:
                seg_latent = seg_latent.mean(dim=0)
            latents.append(seg_latent)

        return latents

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_input_len, max_target_len):
    """Tokenize a batch of (input_text, output_text, meta, latents) tuples.

    Also finds step boundary positions in decoder targets and collects
    V-JEPA latents for InfoNCE grounding.
    """
    inputs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    # meta and latents are at indices 2 and 3

    input_enc = tokenizer(
        list(inputs),
        max_length=max_input_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # transformers>=4.44 removes as_target_tokenizer() for some tokenizers.
    try:
        target_enc = tokenizer(
            text_target=list(targets),
            max_length=max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    except TypeError:
        with tokenizer.as_target_tokenizer():
            target_enc = tokenizer(
                list(targets),
                max_length=max_target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
    # Replace pad token id with -100 so loss ignores them
    labels = target_enc.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    result = {
        "input_ids": input_enc.input_ids,
        "attention_mask": input_enc.attention_mask,
        "labels": labels,
        "decoder_input_ids": target_enc.input_ids,
    }

    # --- Latent grounding data ---
    # Find step boundaries in decoder targets (positions of '}' tokens)
    # and collect corresponding V-JEPA latents
    has_latent = any(b[3] is not None for b in batch)
    if has_latent:
        all_batch_idx = []
        all_step_pos = []
        all_vjepa = []

        # Find the token id for '}'
        close_brace_ids = tokenizer.encode("}", add_special_tokens=False)

        for b_idx, (inp, tgt, meta, latents) in enumerate(batch):
            if latents is None:
                continue
            k = len(latents)

            # Find positions of '}' in the decoder target
            target_ids = target_enc.input_ids[b_idx].tolist()
            brace_positions = [
                pos for pos, tid in enumerate(target_ids)
                if tid in close_brace_ids
            ]

            # Each '}' at depth-0 ends a step. We take the first k matches.
            # For nested JSON this is approximate, but our step dicts are flat.
            step_positions = brace_positions[:k]

            if len(step_positions) < k:
                continue  # Not enough step boundaries found

            for s_idx in range(k):
                all_batch_idx.append(b_idx)
                all_step_pos.append(step_positions[s_idx])
                all_vjepa.append(latents[s_idx])

        if all_vjepa:
            result["latent_batch_idx"] = torch.tensor(all_batch_idx, dtype=torch.long)
            result["latent_step_pos"] = torch.tensor(all_step_pos, dtype=torch.long)
            result["latent_vjepa"] = torch.stack(all_vjepa, dim=0)

    return result


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

def compute_metrics(model, dataloader, tokenizer, device, max_target_len,
                    latent_head=None, latent_weight=0.1, temperature=0.07):
    """
    Compute validation loss and structural accuracy:
      - valid_json: fraction of outputs that are valid JSON with next_steps key
      - exact_match: fraction where predicted step list == gold step list
    Also computes latent grounding loss if latent_head is provided.
    """
    model.eval()
    if latent_head is not None:
        latent_head.eval()
    total_loss = 0.0
    total_latent_loss = 0.0
    n_batches = 0
    n_latent_batches = 0
    n_valid_json = 0
    n_exact = 0
    n_total = 0

    with torch.no_grad():
        for batch in dataloader:
            # Pop latent-specific keys
            latent_batch_idx = batch.pop("latent_batch_idx", None)
            latent_step_pos = batch.pop("latent_step_pos", None)
            latent_vjepa = batch.pop("latent_vjepa", None)

            # Model forward keys
            model_batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }

            need_hidden = latent_head is not None and latent_batch_idx is not None
            if need_hidden:
                model_batch["decoder_input_ids"] = batch["decoder_input_ids"].to(device)

            outputs = model(
                **model_batch,
                output_hidden_states=need_hidden,
            )
            total_loss += outputs.loss.item()
            n_batches += 1

            # Latent loss
            if need_hidden and latent_batch_idx is not None and len(latent_batch_idx) > 1:
                dec_hidden = outputs.decoder_hidden_states[-1]  # (B, T_dec, H)
                h_steps = dec_hidden[
                    latent_batch_idx.to(device),
                    latent_step_pos.to(device),
                ]
                z_text = latent_head(h_steps)
                z_video = nn.functional.normalize(
                    latent_vjepa.to(device).float(), dim=-1
                )
                l_latent = info_nce_loss(z_text, z_video, temperature)
                total_latent_loss += l_latent.item()
                n_latent_batches += 1

    # Generate a subset for structural metrics (slow, so cap at 200)
    model.eval()
    for sample in dataloader.dataset.samples[:200]:
        batch_inputs, batch_targets = sample[0], sample[1]
        input_enc = tokenizer(
            batch_inputs,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **input_enc,
                max_new_tokens=max_target_len,
                num_beams=1,
                do_sample=False,
            )
        pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        try:
            pred = json.loads(pred_text)
            if "next_steps" in pred:
                n_valid_json += 1
                # Check exact match
                gold = json.loads(batch_targets)
                if pred["next_steps"] == gold["next_steps"]:
                    n_exact += 1
        except (json.JSONDecodeError, KeyError):
            pass

        n_total += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_latent = total_latent_loss / max(n_latent_batches, 1) if n_latent_batches > 0 else 0.0
    valid_json_rate = n_valid_json / max(n_total, 1)
    exact_match_rate = n_exact / max(n_total, 1)

    return {
        "val_text_loss": avg_loss,
        "val_latent_loss": avg_latent,
        "val_loss": avg_loss + latent_weight * avg_latent,
        "valid_json_rate": valid_json_rate,
        "exact_match_rate": exact_match_rate,
        "n_eval_samples": n_total,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    _lazy_hf()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # --- Latent Projection Head (VLWM §3.1.1, Eq. 3-4) ---
    latent_head = None
    use_latent_grounding = (
        args.latent_dir is not None
        and os.path.isdir(args.latent_dir or "")
    )
    decoder_hidden_dim = model.config.d_model  # T5-small: 512

    if use_latent_grounding:
        latent_head = LatentProjectionHead(
            decoder_hidden_dim=decoder_hidden_dim,
            vjepa_dim=args.vjepa_dim,
        ).to(device)
        head_params = sum(p.numel() for p in latent_head.parameters())
        print(f"\n  Latent Projection Head:")
        print(f"    T5 decoder dim:  {decoder_hidden_dim}")
        print(f"    V-JEPA dim:      {args.vjepa_dim}")
        print(f"    Head params:     {head_params:,}")
        print(f"    InfoNCE τ:       {args.temperature}")
        print(f"    Latent weight α: {args.latent_weight}")
    else:
        if args.latent_dir:
            print(f"\n  ⚠️  --latent_dir={args.latent_dir} not found, text-only")
        else:
            print(f"\n  Training text-only (no latent grounding)")

    # Data
    print(f"Loading training data: {args.train_data}")
    train_ds = System1Dataset(
        args.train_data,
        latent_dir=args.latent_dir if use_latent_grounding else None,
        vjepa_dim=args.vjepa_dim,
    )
    print(f"  {len(train_ds)} training samples")

    val_ds = None
    if args.val_data and os.path.exists(args.val_data):
        val_ds = System1Dataset(
            args.val_data,
            latent_dir=args.latent_dir if use_latent_grounding else None,
            vjepa_dim=args.vjepa_dim,
        )
        print(f"  {len(val_ds)} validation samples")

    from functools import partial
    _collate = partial(
        collate_fn,
        tokenizer=tokenizer,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
    )

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

    # Optimizer — model params + latent head (if present)
    all_params = list(model.parameters())
    if latent_head is not None:
        all_params += list(latent_head.parameters())

    optimizer = torch.optim.AdamW(
        all_params,
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.06 * total_steps)  # 6% warmup

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
    print(f"Training System-1 model")
    print(f"  Grounding:   {'InfoNCE (VLWM Eq. 3-4)' if use_latent_grounding else 'None (text-only)'}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup:      {warmup_steps}")
    print(f"  Output:      {out_dir}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        if latent_head is not None:
            latent_head.train()

        epoch_text_loss = 0.0
        epoch_latent_loss = 0.0
        n_steps = 0
        n_latent_steps = 0
        t0 = time.time()

        for batch in train_loader:
            # Pop latent-specific keys
            latent_batch_idx = batch.pop("latent_batch_idx", None)
            latent_step_pos = batch.pop("latent_step_pos", None)
            latent_vjepa = batch.pop("latent_vjepa", None)

            # Determine if we need hidden states
            need_hidden = (
                latent_head is not None
                and latent_batch_idx is not None
            )

            # Model forward
            model_batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            if need_hidden:
                model_batch["decoder_input_ids"] = batch["decoder_input_ids"].to(device)

            outputs = model(
                **model_batch,
                output_hidden_states=need_hidden,
            )
            text_loss = outputs.loss

            # Latent grounding loss (InfoNCE)
            latent_loss = torch.tensor(0.0, device=device)
            if need_hidden and latent_batch_idx is not None and len(latent_batch_idx) > 1:
                dec_hidden = outputs.decoder_hidden_states[-1]  # (B, T_dec, H)
                h_steps = dec_hidden[
                    latent_batch_idx.to(device),
                    latent_step_pos.to(device),
                ]  # (N_steps, H)
                z_text = latent_head(h_steps)
                z_video = nn.functional.normalize(
                    latent_vjepa.to(device).float(), dim=-1
                )
                latent_loss = info_nce_loss(z_text, z_video, args.temperature)
                epoch_latent_loss += latent_loss.item()
                n_latent_steps += 1

            # Combined loss
            loss = text_loss + args.latent_weight * latent_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            epoch_text_loss += text_loss.item()
            n_steps += 1

        avg_text_loss = epoch_text_loss / max(n_steps, 1)
        avg_latent_loss = epoch_latent_loss / max(n_latent_steps, 1) if n_latent_steps > 0 else 0.0
        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_text_loss": round(avg_text_loss, 5),
            "train_latent_loss": round(avg_latent_loss, 5),
            "train_loss": round(avg_text_loss + args.latent_weight * avg_latent_loss, 5),
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(elapsed, 1),
        }

        # Validation
        if val_loader is not None:
            val_metrics = compute_metrics(
                model, val_loader, tokenizer, device, args.max_target_len,
                latent_head=latent_head,
                latent_weight=args.latent_weight,
                temperature=args.temperature,
            )
            entry.update(val_metrics)

            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                entry["is_best"] = True
                best_dir = out_dir / "best_model"
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                if latent_head is not None:
                    torch.save(
                        latent_head.state_dict(),
                        best_dir / "latent_head.pt",
                    )
                print(f"  ★ New best model saved (val_loss={best_val_loss:.4f})")
        else:
            # No val set: save every epoch
            best_dir = out_dir / "best_model"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            if latent_head is not None:
                torch.save(latent_head.state_dict(), best_dir / "latent_head.pt")

        log_entries.append(entry)

        # Write log incrementally
        with open(log_path, "w") as f:
            for e in log_entries:
                f.write(json.dumps(e) + "\n")

        # Print progress
        val_str = ""
        if "val_loss" in entry:
            val_str = (
                f"  val={entry['val_loss']:.4f}"
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
            f"{val_str}"
            f"  lr={optimizer.param_groups[0]['lr']:.2e}"
            f"  [{elapsed:.0f}s]"
        )

    # Save final model too
    final_dir = out_dir / "final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    if latent_head is not None:
        torch.save(latent_head.state_dict(), final_dir / "latent_head.pt")

    print(f"\nTraining complete.")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Log:           {log_path}")
    print(f"  Best model:    {out_dir / 'best_model'}")
    print(f"  Final model:   {final_dir}")
    if latent_head is not None:
        print(f"  Latent head:   {out_dir / 'best_model' / 'latent_head.pt'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train System-1 rollout generator (T5 seq2seq)"
    )
    parser.add_argument("--train_data", type=str,
                        default="data/crosstask/system1_train.jsonl")
    parser.add_argument("--val_data", type=str,
                        default="data/crosstask/system1_val.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/system1")
    parser.add_argument("--model_name", type=str,
                        default="google/flan-t5-small",
                        help="HuggingFace model ID (T5 variant)")

    # Latent Grounding (VLWM §3.1.1, Eq. 3-4)
    parser.add_argument("--latent_dir", type=str, default=None,
                        help="Path to V-JEPA latents. "
                             "If provided, enables InfoNCE grounding loss.")
    parser.add_argument("--vjepa_dim", type=int, default=1024,
                        help="V-JEPA feature dimension "
                             "(1024 for ViT-L, 1280 for ViT-H)")
    parser.add_argument("--latent_weight", type=float, default=0.1,
                        help="Weight α for InfoNCE: L = L_text + α·L_latent")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE temperature τ (default: 0.07)")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--max_target_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
