#!/usr/bin/env python3
"""
Train a goal/trajectory latent alignment model against V-JEPA latents.

Purpose:
  - Map goal text -> latent goal state (V-JEPA-like space)
  - Map action/state-change trajectory text -> latent current state
  - Use energy E = ||z_traj - z_goal||^2 during planning

Input:
  - transition CSV (e.g. data/coin/coin_state_change_transitions_train.csv)
  - latent directory from extract_vjepa_latents.py

Output:
  - checkpoints/goal_latent/best_model.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def format_trajectory_for_energy(goal: str, steps: List[str]) -> str:
    """Build trajectory text used by the latent energy model."""
    lines = [f"Goal: {goal}", "Trajectory:"]
    if steps:
        for idx, step in enumerate(steps, start=1):
            lines.append(f"  {idx}) {step}")
    else:
        lines.append("  (No steps observed yet.)")
    return "\n".join(lines)


def _pool_latent(path: str) -> torch.Tensor:
    z = torch.load(path, map_location="cpu", weights_only=True)
    return z.mean(dim=0).float() if z.dim() == 2 else z.float()


def _load_video_latents(latent_dir: str, task_id: str, video_id: str) -> Dict[int, torch.Tensor]:
    vid_dir = os.path.join(latent_dir, str(task_id), str(video_id))
    if not os.path.isdir(vid_dir):
        return {}
    latents = {}
    for name in os.listdir(vid_dir):
        if not (name.startswith("segment_") and name.endswith(".pt")):
            continue
        seg = int(name.split("_")[1].split(".")[0])
        latents[seg] = _pool_latent(os.path.join(vid_dir, name))
    return latents


def _load_grouped_trajectories(csv_path: str) -> Dict[Tuple[str, str], dict]:
    grouped_rows: Dict[Tuple[str, str], list] = defaultdict(list)
    with open(csv_path) as handle:
        for row in csv.DictReader(handle):
            key = (row["task_id"], row["video_id"])
            grouped_rows[key].append(row)

    trajectories = {}
    for key, rows in grouped_rows.items():
        rows.sort(key=lambda r: int(r["seg_pos"]))
        steps = [f"{r['action']} | {r['state_change']}" for r in rows]
        last = rows[-1]
        steps.append(f"{last['next_action']} | {last['next_state_change']}")
        trajectories[key] = {
            "goal": rows[0]["task_goal"],
            "steps": steps,
        }
    return trajectories


class GoalLatentDataset(Dataset):
    """Samples (goal_text, trajectory_text) with target V-JEPA latents."""

    def __init__(self, csv_path: str, latent_dir: str):
        trajectories = _load_grouped_trajectories(csv_path)
        self.samples = []

        for (task_id, video_id), info in trajectories.items():
            steps = info["steps"]
            latents = _load_video_latents(latent_dir, task_id, video_id)
            if not latents:
                continue
            goal_seg = max(latents.keys())
            goal_target = latents[goal_seg]

            # Train trajectory encoder at each observed segment where a latent exists.
            for t in range(len(steps)):
                if t not in latents:
                    continue
                traj_text = format_trajectory_for_energy(info["goal"], steps[: t + 1])
                self.samples.append(
                    (
                        info["goal"],
                        traj_text,
                        latents[t],
                        goal_target,
                    )
                )

        if not self.samples:
            raise ValueError(
                "No training samples built. Check transition CSV and --latent_dir."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class GoalLatentModel(nn.Module):
    """Shared text encoder with separate heads for goal and trajectory."""

    def __init__(self, encoder_name: str, latent_dim: int, hidden_dim: int = 384):
        super().__init__()
        from transformers import AutoModel

        self.encoder_name = encoder_name
        self.latent_dim = latent_dim
        self.text_encoder = AutoModel.from_pretrained(encoder_name)
        enc_dim = self.text_encoder.config.hidden_size

        self.goal_head = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.traj_head = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._mean_pool(out.last_hidden_state, attention_mask)

    def encode_goal(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self._encode(input_ids, attention_mask)
        return F.normalize(self.goal_head(h), dim=-1)

    def encode_traj(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self._encode(input_ids, attention_mask)
        return F.normalize(self.traj_head(h), dim=-1)

    def forward(
        self,
        goal_input_ids: torch.Tensor,
        goal_attention_mask: torch.Tensor,
        traj_input_ids: torch.Tensor,
        traj_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_goal = self.encode_goal(goal_input_ids, goal_attention_mask)
        z_traj = self.encode_traj(traj_input_ids, traj_attention_mask)
        return z_goal, z_traj


def collate_fn(batch, tokenizer, max_goal_len: int, max_traj_len: int):
    goal_texts = [b[0] for b in batch]
    traj_texts = [b[1] for b in batch]
    traj_targets = torch.stack([b[2] for b in batch], dim=0)
    goal_targets = torch.stack([b[3] for b in batch], dim=0)

    goal_enc = tokenizer(
        goal_texts,
        padding="max_length",
        truncation=True,
        max_length=max_goal_len,
        return_tensors="pt",
    )
    traj_enc = tokenizer(
        traj_texts,
        padding="max_length",
        truncation=True,
        max_length=max_traj_len,
        return_tensors="pt",
    )

    return {
        "goal_input_ids": goal_enc.input_ids,
        "goal_attention_mask": goal_enc.attention_mask,
        "traj_input_ids": traj_enc.input_ids,
        "traj_attention_mask": traj_enc.attention_mask,
        "traj_targets": F.normalize(traj_targets, dim=-1),
        "goal_targets": F.normalize(goal_targets, dim=-1),
    }


def compute_val_loss(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            z_goal, z_traj = model(
                goal_input_ids=batch["goal_input_ids"].to(device),
                goal_attention_mask=batch["goal_attention_mask"].to(device),
                traj_input_ids=batch["traj_input_ids"].to(device),
                traj_attention_mask=batch["traj_attention_mask"].to(device),
            )
            goal_targets = batch["goal_targets"].to(device)
            traj_targets = batch["traj_targets"].to(device)
            loss_goal = F.mse_loss(z_goal, goal_targets)
            loss_traj = F.mse_loss(z_traj, traj_targets)
            loss = loss_goal + loss_traj
            total += loss.item()
            n += 1
    return total / max(n, 1)


def train(args):
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_ds = GoalLatentDataset(args.train_csv, args.latent_dir)
    print(f"Train samples: {len(train_ds)}")
    val_ds = None
    if args.val_csv and os.path.exists(args.val_csv):
        val_ds = GoalLatentDataset(args.val_csv, args.latent_dir)
        print(f"Val samples: {len(val_ds)}")

    # Infer latent dim from first sample target.
    latent_dim = int(train_ds[0][2].shape[-1])
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    model = GoalLatentModel(
        encoder_name=args.encoder_name,
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    from functools import partial

    collate = partial(
        collate_fn,
        tokenizer=tokenizer,
        max_goal_len=args.max_goal_len,
        max_traj_len=args.max_traj_len,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"

    best_val = float("inf")
    logs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps = 0
        t0 = time.time()
        for batch in train_loader:
            z_goal, z_traj = model(
                goal_input_ids=batch["goal_input_ids"].to(device),
                goal_attention_mask=batch["goal_attention_mask"].to(device),
                traj_input_ids=batch["traj_input_ids"].to(device),
                traj_attention_mask=batch["traj_attention_mask"].to(device),
            )
            goal_targets = batch["goal_targets"].to(device)
            traj_targets = batch["traj_targets"].to(device)

            loss_goal = F.mse_loss(z_goal, goal_targets)
            loss_traj = F.mse_loss(z_traj, traj_targets)
            loss = loss_goal + loss_traj

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

        entry = {
            "epoch": epoch,
            "train_loss": round(epoch_loss / max(n_steps, 1), 6),
            "time_sec": round(time.time() - t0, 1),
        }
        if val_loader is not None:
            val_loss = compute_val_loss(model, val_loader, device)
            entry["val_loss"] = round(val_loss, 6)
            if val_loss < best_val:
                best_val = val_loss
                entry["is_best"] = True
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "encoder_name": args.encoder_name,
                    "latent_dim": latent_dim,
                    "hidden_dim": args.hidden_dim,
                }
                torch.save(ckpt, out_dir / "best_model.pt")
                tokenizer.save_pretrained(out_dir / "tokenizer")
        logs.append(entry)
        with open(log_path, "w") as handle:
            for row in logs:
                handle.write(json.dumps(row) + "\n")

        val_msg = f" val={entry['val_loss']:.4f}" if "val_loss" in entry else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} "
            f"train={entry['train_loss']:.4f}{val_msg} "
            f"[{entry['time_sec']:.0f}s]"
        )

    final_ckpt = {
        "model_state_dict": model.state_dict(),
        "encoder_name": args.encoder_name,
        "latent_dim": latent_dim,
        "hidden_dim": args.hidden_dim,
    }
    torch.save(final_ckpt, out_dir / "final_model.pt")
    tokenizer.save_pretrained(out_dir / "tokenizer")
    print(f"Saved: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train goal/trajectory latent model")
    parser.add_argument("--train_csv", default="data/coin/coin_state_change_transitions_train.csv")
    parser.add_argument("--val_csv", default="data/coin/coin_state_change_transitions_val.csv")
    parser.add_argument("--latent_dir", default="data/coin/vjepa_latents")
    parser.add_argument("--output_dir", default="checkpoints/goal_latent")
    parser.add_argument("--encoder_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_goal_len", type=int, default=64)
    parser.add_argument("--max_traj_len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
