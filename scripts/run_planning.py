#!/usr/bin/env python3
"""
System-2 Planning: generate K plans with System-1, rerank with Critic + Energy.

Pipeline:
  1. Load a test sample (goal + prefix)
  2. System-1 generates K candidate plans via temperature sampling
  3. Score each plan:
       score = α · C_critic(goal, prefix + plan)
             + β · Energy(z_predicted, z_goal)      [optional, needs V-JEPA latents]
  4. Select the plan with the LOWEST score

The Energy term is optional — when V-JEPA latents are not available,
the script runs critic-only reranking (α=1, β=0).

Saves:
  outputs/planning/{timestamp}_plans.jsonl

Usage (critic-only):
    python3 scripts/run_planning.py \
        --test_data       data/crosstask/system1_test.jsonl \
        --system1_model   checkpoints/system1/best_model \
        --critic_model    checkpoints/critic/best_model.pt \
        --K 5 \
        --temperature 0.8 \
        --output_dir      outputs/planning

Usage (with latent energy — when V-JEPA latents are available):
    python3 scripts/run_planning.py \
        --test_data       data/crosstask/system1_test.jsonl \
        --system1_model   checkpoints/system1/best_model \
        --critic_model    checkpoints/critic/best_model.pt \
        --latent_dir      data/crosstask/vjepa_latents \
        --alpha 0.7 --beta 0.3 \
        --K 5 \
        --output_dir      outputs/planning
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
_HF_LOADED = False
def _lazy_hf():
    global _HF_LOADED, AutoTokenizer, T5ForConditionalGeneration, AutoModel
    if _HF_LOADED:
        return
    from transformers import (
        AutoTokenizer as _AT,
        T5ForConditionalGeneration as _T5,
        AutoModel as _AM,
    )
    AutoTokenizer = _AT
    T5ForConditionalGeneration = _T5
    AutoModel = _AM
    _HF_LOADED = True


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_system1(model_path: str, device):
    """Load fine-tuned T5 System-1 model."""
    _lazy_hf()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_critic(model_path: str, device):
    """Load trained Critic model."""
    _lazy_hf()
    # Import CriticModel from train_critic
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_critic",
        os.path.join(os.path.dirname(__file__), "train_critic.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    encoder_name = ckpt.get("encoder_name", "sentence-transformers/all-MiniLM-L6-v2")

    model = mod.CriticModel(encoder_name)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    return model, tokenizer, mod.format_trajectory_text


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

def generate_k_plans(
    model,
    tokenizer,
    input_text: str,
    K: int,
    temperature: float,
    top_p: float,
    max_target_len: int,
    device,
) -> list:
    """Generate K diverse plans using temperature sampling."""
    enc = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    plans = []
    for _ in range(K):
        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_target_len,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=1,
            )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # Parse JSON output
        try:
            parsed = json.loads(text)
            steps = parsed.get("next_steps", [])
            step_strs = [
                f"{s['action']} | {s['state_change']}" for s in steps
            ]
            plans.append({
                "raw_text": text,
                "steps": step_strs,
                "valid": True,
            })
        except (json.JSONDecodeError, KeyError, TypeError):
            plans.append({
                "raw_text": text,
                "steps": [],
                "valid": False,
            })

    return plans


# ---------------------------------------------------------------------------
# Plan scoring
# ---------------------------------------------------------------------------

def score_plans_critic(
    critic_model,
    critic_tokenizer,
    format_traj_fn,
    goal: str,
    prefix_steps: list,
    plans: list,
    device,
    max_len: int = 256,
) -> list:
    """Score each plan using the Critic model. Lower cost = better."""
    scores = []
    for plan in plans:
        if not plan["valid"] or not plan["steps"]:
            scores.append(float("inf"))
            continue

        # Construct trajectory = prefix + plan steps
        full_steps = prefix_steps + plan["steps"]
        text = format_traj_fn(goal, full_steps)

        enc = critic_tokenizer(
            text, max_length=max_len, truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            cost = critic_model(enc.input_ids, enc.attention_mask).item()
        scores.append(cost)

    return scores


def score_plans_energy(
    latent_dir: str,
    video_id: str,
    task_id: str,
    plans: list,
) -> list:
    """
    Score plans using Energy(z_predicted_final, z_goal).

    Energy = ||z_predicted - z_goal||₂²

    z_goal is pre-extracted from V-JEPA for the final segment.
    z_predicted approximates where the trajectory would end.

    Returns list of energy scores (lower = better).

    NOTE: This is a placeholder for when V-JEPA latents become available.
    The latent directory should contain files like:
      {latent_dir}/{task_id}/{video_id}/segment_{seg_pos}.pt
    """
    # Check if latents exist
    vid_dir = os.path.join(latent_dir, task_id, video_id)
    if not os.path.exists(vid_dir):
        return [0.0] * len(plans)  # No energy penalty

    # Load goal latent (last segment)
    latent_files = sorted(
        [f for f in os.listdir(vid_dir) if f.endswith(".pt")],
        key=lambda f: int(f.split("_")[1].split(".")[0]),
    )
    if not latent_files:
        return [0.0] * len(plans)

    z_goal = torch.load(os.path.join(vid_dir, latent_files[-1]), weights_only=True)

    # For each plan, use the number of steps to estimate which latent
    # the trajectory would reach
    energies = []
    for plan in plans:
        if not plan["valid"]:
            energies.append(float("inf"))
            continue
        # Simple heuristic: energy based on plan length relative to total
        # (proper implementation would use a learned latent predictor)
        n_planned = len(plan["steps"])
        target_idx = min(len(latent_files) - 1, n_planned)
        z_pred_path = os.path.join(vid_dir, latent_files[target_idx])
        if os.path.exists(z_pred_path):
            z_pred = torch.load(z_pred_path, weights_only=True)
            energy = torch.norm(z_pred - z_goal, p=2).item() ** 2
        else:
            energy = 0.0
        energies.append(energy)

    return energies


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_planning(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print(f"Loading System-1 model: {args.system1_model}")
    s1_model, s1_tokenizer = load_system1(args.system1_model, device)

    print(f"Loading Critic model: {args.critic_model}")
    critic_model, critic_tokenizer, format_traj_fn = load_critic(
        args.critic_model, device
    )

    use_energy = args.latent_dir and os.path.exists(args.latent_dir)
    if use_energy:
        print(f"Using latent energy from: {args.latent_dir}")
        print(f"  α (critic weight): {args.alpha}")
        print(f"  β (energy weight): {args.beta}")
    else:
        print("No latent directory — running critic-only reranking")
        args.alpha = 1.0
        args.beta = 0.0

    # Load test data
    print(f"Loading test data: {args.test_data}")
    test_samples = []
    with open(args.test_data) as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))
    print(f"  {len(test_samples)} test samples")

    if args.max_samples:
        test_samples = test_samples[:args.max_samples]
        print(f"  Capped to {len(test_samples)} samples")

    # Run planning
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}_plans.jsonl"

    results = []
    t0 = time.time()

    for i, sample in enumerate(test_samples):
        input_text = sample["input_text"]
        gold_output = sample["output_text"]
        meta = sample.get("meta", {})

        # Extract goal and prefix from input_text
        goal = ""
        prefix_steps = []
        for line in input_text.split("\n"):
            if line.startswith("Goal:"):
                goal = line[len("Goal:"):].strip()
            elif line.strip() and line.strip()[0].isdigit() and ")" in line:
                # Parse "  1) action | state_change"
                step_text = line.strip().split(")", 1)[1].strip()
                prefix_steps.append(step_text)

        # Generate K plans
        plans = generate_k_plans(
            s1_model, s1_tokenizer, input_text,
            K=args.K, temperature=args.temperature, top_p=args.top_p,
            max_target_len=256, device=device,
        )

        # Score with critic
        critic_scores = score_plans_critic(
            critic_model, critic_tokenizer, format_traj_fn,
            goal, prefix_steps, plans, device,
        )

        # Score with energy (if available)
        energy_scores = [0.0] * len(plans)
        if use_energy:
            energy_scores = score_plans_energy(
                args.latent_dir,
                meta.get("video_id", ""),
                meta.get("task_id", ""),
                plans,
            )

        # Combined score
        combined_scores = [
            args.alpha * c + args.beta * e
            for c, e in zip(critic_scores, energy_scores)
        ]

        # Select best plan (lowest score)
        best_idx = int(np.argmin(combined_scores))

        result = {
            "sample_idx": i,
            "meta": meta,
            "goal": goal,
            "gold_output": gold_output,
            "K": args.K,
            "best_plan_idx": best_idx,
            "best_plan": plans[best_idx],
            "best_score": combined_scores[best_idx],
            "all_plans": [
                {
                    "plan": p,
                    "critic_score": cs,
                    "energy_score": es,
                    "combined_score": comb,
                }
                for p, cs, es, comb in zip(plans, critic_scores, energy_scores, combined_scores)
            ],
        }
        results.append(result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(test_samples)}] {elapsed:.0f}s")

    # Write results
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    n_valid = sum(1 for r in results if r["best_plan"]["valid"])

    print(f"\nPlanning complete.")
    print(f"  Samples:       {len(results)}")
    print(f"  Valid plans:   {n_valid}/{len(results)}")
    print(f"  Time:          {elapsed:.0f}s ({elapsed/len(results):.1f}s per sample)")
    print(f"  Output:        {out_path}")

    # Quick stats
    critic_scores_best = [r["best_score"] for r in results if r["best_plan"]["valid"]]
    if critic_scores_best:
        print(f"  Best scores:   mean={np.mean(critic_scores_best):.3f}, "
              f"std={np.std(critic_scores_best):.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="System-2 Planning: K-plan generation + reranking"
    )
    parser.add_argument("--test_data", default="data/crosstask/system1_test.jsonl")
    parser.add_argument("--system1_model", default="checkpoints/system1/best_model")
    parser.add_argument("--critic_model", default="checkpoints/critic/best_model.pt")
    parser.add_argument("--latent_dir", default=None,
                        help="Directory with V-JEPA latents (optional)")
    parser.add_argument("--K", type=int, default=5,
                        help="Number of candidate plans to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight for critic score")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Weight for latent energy score")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit test samples (for quick testing)")
    parser.add_argument("--output_dir", default="outputs/planning")
    args = parser.parse_args()
    run_planning(args)


if __name__ == "__main__":
    main()
