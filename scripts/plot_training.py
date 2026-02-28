#!/usr/bin/env python3
"""
Plot training curves and per-task breakdowns from training logs.

Reads training_log.jsonl files produced by train_system1.py, train_goal_model.py,
and train_critic.py, then generates publication-quality plots.

Outputs to plots/ directory:
  - system1_loss_curve.png
  - system1_metrics.png
  - goal_model_loss_curve.png
  - goal_model_metrics.png
  - critic_loss_curve.png
  - critic_accuracy.png
  - combined_overview.png

Usage:
    python3 scripts/plot_training.py \
        --system1_log   checkpoints/system1/training_log.jsonl \
        --goal_log      checkpoints/goal_model/training_log.jsonl \
        --critic_log    checkpoints/critic/training_log.jsonl \
        --output_dir    plots
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import numpy as np


def load_log(path: str) -> list:
    """Load a JSONL training log."""
    entries = []
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"  Loaded {len(entries)} epochs from {path}")
    return entries


def plot_system1(log: list, out_dir: Path):
    """Plot System-1 training curves."""
    if not log:
        return

    epochs = [e["epoch"] for e in log]
    train_loss = [e["train_loss"] for e in log]
    val_loss = [e.get("val_loss") for e in log]
    json_rate = [e.get("valid_json_rate") for e in log]
    exact_rate = [e.get("exact_match_rate") for e in log]

    # --- Loss curves ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_loss, "b-o", markersize=3, label="Train Loss")
    if any(v is not None for v in val_loss):
        ax1.plot(epochs, [v for v in val_loss], "r-s", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("System-1: Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark best epoch
    best_entries = [e for e in log if e.get("is_best")]
    if best_entries:
        best = best_entries[-1]
        ax1.axvline(x=best["epoch"], color="green", linestyle="--", alpha=0.5,
                     label=f"Best (epoch {best['epoch']})")
        ax1.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "system1_loss_curve.png", dpi=150)
    plt.close()

    # --- Metrics ---
    if any(v is not None for v in json_rate):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(epochs, [v or 0 for v in json_rate], "g-o", markersize=3)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Valid JSON Rate")
        ax1.set_title("System-1: Output Structure Quality")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, [v or 0 for v in exact_rate], "m-o", markersize=3)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Exact Match Rate")
        ax2.set_title("System-1: Exact Step-Sequence Match")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "system1_metrics.png", dpi=150)
        plt.close()


def plot_goal_model(log: list, out_dir: Path):
    """Plot Goal Model training curves."""
    if not log:
        return

    epochs = [e["epoch"] for e in log]
    train_loss = [e["train_loss"] for e in log]
    val_loss = [e.get("val_loss") for e in log]
    goal_acc = [e.get("goal_accuracy") for e in log]
    plan_json = [e.get("plan_valid_json_rate") for e in log]

    # --- Loss ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_loss, "b-o", markersize=3, label="Train Loss")
    if any(v is not None for v in val_loss):
        ax1.plot(epochs, [v for v in val_loss], "r-s", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Goal Model: Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    best_entries = [e for e in log if e.get("is_best")]
    if best_entries:
        best = best_entries[-1]
        ax1.axvline(x=best["epoch"], color="green", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "goal_model_loss_curve.png", dpi=150)
    plt.close()

    # --- Metrics ---
    if any(v is not None for v in goal_acc):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(epochs, [v or 0 for v in goal_acc], "g-o", markersize=3)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Goal Accuracy")
        ax1.set_title("Goal Model: Goal Prediction Accuracy")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, [v or 0 for v in plan_json], "m-o", markersize=3)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Valid JSON Rate")
        ax2.set_title("Goal Model: Plan Output Quality")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "goal_model_metrics.png", dpi=150)
        plt.close()


def plot_critic(log: list, out_dir: Path):
    """Plot Critic training curves."""
    if not log:
        return

    epochs = [e["epoch"] for e in log]
    train_loss = [e["train_loss"] for e in log]
    train_acc = [e.get("train_ranking_acc") for e in log]
    val_loss = [e.get("val_loss") for e in log]
    val_acc = [e.get("ranking_accuracy") for e in log]

    # --- Loss ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_loss, "b-o", markersize=3, label="Train Loss")
    if any(v is not None for v in val_loss):
        ax1.plot(epochs, [v for v in val_loss], "r-s", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Ranking Loss")
    ax1.set_title("Critic: Ranking Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "critic_loss_curve.png", dpi=150)
    plt.close()

    # --- Accuracy ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    if any(v is not None for v in train_acc):
        ax1.plot(epochs, [v or 0 for v in train_acc], "b-o", markersize=3, label="Train Ranking Acc")
    if any(v is not None for v in val_acc):
        ax1.plot(epochs, [v or 0 for v in val_acc], "r-s", markersize=3, label="Val Ranking Acc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Ranking Accuracy")
    ax1.set_title("Critic: Ranking Accuracy (C_pos < C_neg)")
    ax1.set_ylim(0.4, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if any(e.get("is_best") for e in log):
        best = [e for e in log if e.get("is_best")][-1]
        ax1.axvline(x=best["epoch"], color="green", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "critic_accuracy.png", dpi=150)
    plt.close()


def plot_combined(system1_log, goal_log, critic_log, out_dir: Path):
    """Combined overview: one figure with all models."""
    available = []
    if system1_log:
        available.append(("System-1", system1_log, "train_loss", "val_loss"))
    if goal_log:
        available.append(("Goal Model", goal_log, "train_loss", "val_loss"))
    if critic_log:
        available.append(("Critic", critic_log, "train_loss", "val_loss"))

    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, log, train_key, val_key) in zip(axes, available):
        epochs = [e["epoch"] for e in log]
        ax.plot(epochs, [e[train_key] for e in log], "b-o", markersize=2, label="Train")
        val_vals = [e.get(val_key) for e in log]
        if any(v is not None for v in val_vals):
            ax.plot(epochs, val_vals, "r-s", markersize=2, label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Overview — All Models", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "combined_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved combined_overview.png")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--system1_log", default="checkpoints/system1/training_log.jsonl")
    parser.add_argument("--goal_log", default="checkpoints/goal_model/training_log.jsonl")
    parser.add_argument("--critic_log", default="checkpoints/critic/training_log.jsonl")
    parser.add_argument("--output_dir", default="plots")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training logs...")
    s1_log = load_log(args.system1_log)
    goal_log = load_log(args.goal_log)
    critic_log = load_log(args.critic_log)

    print("\nGenerating plots...")
    plot_system1(s1_log, out_dir)
    plot_goal_model(goal_log, out_dir)
    plot_critic(critic_log, out_dir)
    plot_combined(s1_log, goal_log, critic_log, out_dir)

    print(f"\nAll plots saved to: {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        if f.suffix == ".png":
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
