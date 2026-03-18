#!/usr/bin/env python3
"""
Train System-1 (rollout generator) using full-parameter fine-tuning with
GaLore for larger PLM backbones such as Perception-LM-8B.

This script reuses the existing causal-LM data format and optional V-JEPA
grounding path from `train_system1_plm_lora.py`, but replaces LoRA adapters
with full-parameter optimization using GaLore's low-rank gradient projection.

GaLore reference:
  - GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection
  - Official repo / package: github.com/jiaweizzhao/GaLore, `galore-torch`

What this script is for:
  - Training larger causal System-1 backbones (3B / 8B) when LoRA is too
    restrictive or you want full-parameter updates.
  - Reducing optimizer-memory overhead relative to standard AdamW while still
    updating the full model.

What this script is not:
  - It does not implement the single-GPU per-layer GaLore hooks from the
    official benchmark code. It uses the standard GaLore optimizer API, which
    is simpler and better aligned with this repository's existing training loop.

Saves:
  checkpoints/system1_plm_galore/best_model/   (full HF checkpoint)
  checkpoints/system1_plm_galore/final_model/  (full HF checkpoint)
  checkpoints/system1_plm_galore/latent_head.pt
  checkpoints/system1_plm_galore/training_log.jsonl

Example:
  python3 scripts/train_system1_plm_galore.py \
      --train_data data/coin/coin_system1_train.jsonl \
      --val_data data/coin/coin_system1_val.jsonl \
      --output_dir checkpoints/system1_plm_8b_galore \
      --model_name facebook/Perception-LM-8B \
      --optimizer galore_adamw8bit \
      --activation_checkpointing \
      --galore_rank 256 \
      --galore_update_proj_gap 200 \
      --galore_scale 0.25 \
      --epochs 5 \
      --batch_size 1 \
      --gradient_accumulation_steps 16 \
      --lr 1e-4
"""

import argparse
import importlib.util
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

_LOADED = False


def _lazy_imports():
    global _LOADED
    global AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
    global get_linear_schedule_with_warmup
    global GaLoreAdamW, GaLoreAdamW8bit
    if _LOADED:
        return

    from transformers import (
        AutoModelForCausalLM as _ACLM,
        AutoModelForImageTextToText as _AITT,
        AutoTokenizer as _AT,
    )
    AutoTokenizer = _AT
    AutoModelForCausalLM = _ACLM
    AutoModelForImageTextToText = _AITT

    try:
        from transformers import get_linear_schedule_with_warmup as _sched
    except ImportError:
        _sched = None
    get_linear_schedule_with_warmup = _sched

    try:
        from galore_torch import GaLoreAdamW as _GaLoreAdamW
        from galore_torch import GaLoreAdamW8bit as _GaLoreAdamW8bit
    except ImportError as exc:
        raise ImportError(
            "galore-torch is required for this script. "
            "Install it with: pip install galore-torch"
        ) from exc

    GaLoreAdamW = _GaLoreAdamW
    GaLoreAdamW8bit = _GaLoreAdamW8bit
    _LOADED = True


def _load_plm_training_helpers():
    helper_path = os.path.join(os.path.dirname(__file__), "train_system1_plm_lora.py")
    spec = importlib.util.spec_from_file_location("train_system1_plm_lora", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_plm_training_helpers()
CausalLMDataset = helpers.CausalLMDataset
LatentProjectionHead = helpers.LatentProjectionHead
collate_fn = helpers.collate_fn
evaluate = helpers.evaluate
generate_samples = helpers.generate_samples
info_nce_loss = helpers.info_nce_loss
_infer_hidden_dim = helpers._infer_hidden_dim


def load_model_and_tokenizer(args):
    _lazy_imports()

    print(f"Loading model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            torch_dtype=model_dtype,
            trust_remote_code=True,
        )
        print("  Loaded as AutoModelForImageTextToText")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=model_dtype,
            trust_remote_code=True,
        )
        print("  Loaded as AutoModelForCausalLM (fallback)")

    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
            print("  Tied input/output embeddings")
        except Exception as exc:
            print(f"  Warning: could not tie weights ({exc})")

    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        for param in model.model.vision_tower.parameters():
            param.requires_grad = False
        print("  Froze vision_tower parameters")

    if args.activation_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        print("  Enabled activation checkpointing")

    if hasattr(model, "config"):
        model.config.use_cache = False

    hidden_dim = _infer_hidden_dim(model)
    print(f"  Hidden dimension:     {hidden_dim}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:>12,}")
    print(f"  Trainable params:     {trainable_params:>12,}  (100% of trainable model)")

    return model, tokenizer, hidden_dim


def _matches_any(name: str, patterns):
    return any(p and p in name for p in patterns)


def split_galore_params(model, latent_head, args):
    galore_params = []
    regular_params = []

    exclude_patterns = [p.strip() for p in args.galore_exclude.split(",") if p.strip()]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        use_galore = (
            param.ndim >= 2
            and not _matches_any(name.lower(), exclude_patterns)
        )

        if use_galore:
            galore_params.append(param)
        else:
            regular_params.append(param)

    if latent_head is not None:
        regular_params.extend(latent_head.parameters())

    return galore_params, regular_params


def build_optimizer(galore_params, regular_params, args):
    _lazy_imports()

    if args.optimizer == "galore_adamw":
        optimizer_cls = GaLoreAdamW
    elif args.optimizer == "galore_adamw8bit":
        optimizer_cls = GaLoreAdamW8bit
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    param_groups = []
    if regular_params:
        param_groups.append({"params": regular_params})
    if galore_params:
        param_groups.append(
            {
                "params": galore_params,
                "rank": args.galore_rank,
                "update_proj_gap": args.galore_update_proj_gap,
                "scale": args.galore_scale,
                "proj_type": args.galore_proj_type,
            }
        )

    if not param_groups:
        raise ValueError("No trainable parameters found")

    return optimizer_cls(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )


def train(args):
    _lazy_imports()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, hidden_dim = load_model_and_tokenizer(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    latent_head = None
    use_latent_grounding = args.latent_dir is not None and os.path.isdir(args.latent_dir or "")
    if use_latent_grounding:
        latent_head = LatentProjectionHead(
            llm_hidden_dim=hidden_dim,
            vjepa_dim=args.vjepa_dim,
        ).to(device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        head_params = sum(p.numel() for p in latent_head.parameters())
        print("\n  Latent Projection Head:")
        print(f"    LLM hidden dim:  {hidden_dim}")
        print(f"    V-JEPA dim:      {args.vjepa_dim}")
        print(f"    Head params:     {head_params:,}")
        print(f"    InfoNCE tau:     {args.temperature}")
        print(f"    Latent weight:   {args.latent_weight}")
    elif args.latent_dir:
        print(f"\n  Warning: latent dir not found at {args.latent_dir}; training text-only")
    else:
        print("\n  Training text-only (no latent grounding)")

    train_ds = CausalLMDataset(
        args.train_data,
        tokenizer,
        args.max_seq_len,
        latent_dir=args.latent_dir if use_latent_grounding else None,
        vjepa_dim=args.vjepa_dim,
    )

    val_ds = None
    if args.val_data and os.path.exists(args.val_data):
        val_ds = CausalLMDataset(
            args.val_data,
            tokenizer,
            args.max_seq_len,
            latent_dir=args.latent_dir if use_latent_grounding else None,
            vjepa_dim=args.vjepa_dim,
        )

    from functools import partial
    from torch.utils.data import DataLoader

    _collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id or 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_collate,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    galore_params, regular_params = split_galore_params(model, latent_head, args)
    optimizer = build_optimizer(galore_params, regular_params, args)

    galore_count = sum(p.numel() for p in galore_params)
    regular_count = sum(p.numel() for p in regular_params)

    effective_batch = args.batch_size * args.gradient_accumulation_steps
    steps_per_epoch = max(len(train_loader) // args.gradient_accumulation_steps, 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = None
    if get_linear_schedule_with_warmup is not None:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    best_val_loss = float("inf")
    log_entries = []
    global_step = 0

    print(f"\n{'='*60}")
    print("Training System-1 (PLM + GaLore)")
    print(f"  Base model:     {args.model_name}")
    print(f"  Optimizer:      {args.optimizer}")
    print(f"  GaLore rank:    {args.galore_rank}")
    print(f"  Proj gap:       {args.galore_update_proj_gap}")
    print(f"  Proj scale:     {args.galore_scale}")
    print(f"  Proj type:      {args.galore_proj_type}")
    print(f"  GaLore params:  {galore_count:,}")
    print(f"  Regular params: {regular_count:,}")
    print(f"  Grounding:      {'InfoNCE' if use_latent_grounding else 'None'}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch (eff):    {effective_batch} ({args.batch_size} x {args.gradient_accumulation_steps})")
    print(f"  LR:             {args.lr}")
    print(f"  Warmup steps:   {warmup_steps}")
    print(f"  Max seq len:    {args.max_seq_len}")
    print(f"  Output:         {out_dir}")
    print(f"{'='*60}\n")

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
            latent_batch_idx = batch.pop("latent_batch_idx", None)
            latent_step_pos = batch.pop("latent_step_pos", None)
            latent_vjepa = batch.pop("latent_vjepa", None)
            batch = {k: v.to(device) for k, v in batch.items()}

            need_hidden = latent_head is not None and latent_batch_idx is not None
            outputs = model(
                **batch,
                output_hidden_states=need_hidden,
            )

            text_loss = outputs.loss
            latent_loss = torch.tensor(0.0, device=device)

            if need_hidden and latent_batch_idx is not None and len(latent_batch_idx) > 1:
                hidden_states = outputs.hidden_states[-1]
                h_steps = hidden_states[
                    latent_batch_idx.to(device),
                    latent_step_pos.to(device),
                ]
                z_text = latent_head(h_steps)
                z_video = torch.nn.functional.normalize(
                    latent_vjepa.to(device, dtype=z_text.dtype), dim=-1
                )
                latent_loss = info_nce_loss(z_text, z_video, args.temperature)
                epoch_latent_loss += latent_loss.item()
                n_latent_steps += 1

            total_loss = text_loss + args.latent_weight * latent_loss
            (total_loss / args.gradient_accumulation_steps).backward()

            epoch_text_loss += text_loss.item()
            n_micro_steps += 1

            if step_idx % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
                if latent_head is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(latent_head.parameters()),
                        args.max_grad_norm,
                    )
                optimizer.step()
                if scheduler is not None:
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

        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                latent_head=latent_head,
                latent_weight=args.latent_weight,
                temperature=args.temperature,
            )
            entry.update({
                "val_text_loss": round(val_metrics["val_text_loss"], 5),
                "val_latent_loss": round(val_metrics["val_latent_loss"], 5),
                "val_loss": round(val_metrics["val_combined_loss"], 5),
            })

            if epoch % args.eval_gen_every == 0 or epoch == args.epochs:
                gen_metrics = generate_samples(
                    model,
                    tokenizer,
                    val_ds,
                    device,
                    n_samples=min(args.eval_gen_samples, len(val_ds)),
                )
                entry.update(gen_metrics)

            val_loss = val_metrics["val_combined_loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                entry["is_best"] = True
                best_dir = out_dir / "best_model"
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                if latent_head is not None:
                    torch.save(latent_head.state_dict(), best_dir / "latent_head.pt")
                print(f"  New best model saved (val_loss={best_val_loss:.4f})")
        else:
            best_dir = out_dir / "best_model"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            if latent_head is not None:
                torch.save(latent_head.state_dict(), best_dir / "latent_head.pt")

        log_entries.append(entry)
        with open(log_path, "w") as handle:
            for log_entry in log_entries:
                handle.write(json.dumps(log_entry) + "\n")

        val_str = f"  val={entry['val_loss']:.4f}" if "val_loss" in entry else ""
        gen_str = ""
        if "valid_json_rate" in entry:
            gen_str = (
                f"  json={entry['valid_json_rate']:.1%}"
                f"  exact={entry['exact_match_rate']:.1%}"
            )
        latent_str = f"  L_lat={avg_latent_loss:.3f}" if avg_latent_loss > 0 else ""

        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  L_text={avg_text_loss:.4f}"
            f"{latent_str}"
            f"{val_str}{gen_str}"
            f"  lr={optimizer.param_groups[0]['lr']:.2e}"
            f"  [{elapsed:.0f}s]"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_dir = out_dir / "final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    if latent_head is not None:
        torch.save(latent_head.state_dict(), final_dir / "latent_head.pt")

    print("\nTraining complete.")
    print(f"  Best val loss: {best_val_loss:.4f}" if best_val_loss < float("inf") else "  Best val loss: n/a")
    print(f"  Log:           {log_path}")
    print(f"  Best model:    {out_dir / 'best_model'}")
    print(f"  Final model:   {final_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train System-1 rollout generator with full-parameter GaLore"
    )

    parser.add_argument("--train_data", type=str, default="data/crosstask/system1_train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/crosstask/system1_val.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/system1_plm_galore")

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/Perception-LM-8B",
        help="HF model ID for the full PLM backbone",
    )
    parser.add_argument(
        "--optimizer",
        choices=["galore_adamw", "galore_adamw8bit"],
        default="galore_adamw8bit",
    )
    parser.add_argument("--activation_checkpointing", action="store_true")

    parser.add_argument("--galore_rank", type=int, default=256)
    parser.add_argument("--galore_update_proj_gap", type=int, default=200)
    parser.add_argument("--galore_scale", type=float, default=0.25)
    parser.add_argument("--galore_proj_type", type=str, default="std")
    parser.add_argument(
        "--galore_exclude",
        type=str,
        default="embed,norm,ln_f,layernorm,lm_head",
        help="Comma-separated substrings of parameter names to exclude from GaLore projection",
    )

    parser.add_argument("--latent_dir", type=str, default=None)
    parser.add_argument("--vjepa_dim", type=int, default=1024)
    parser.add_argument("--latent_weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--eval_gen_every", type=int, default=2)
    parser.add_argument("--eval_gen_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
