#!/usr/bin/env python3
"""
Unified evaluation suite for VLWM components.

Covers:
  1) System1 (without JEPA-trained grounding) and System1 (with JEPA-trained grounding)
  2) System1 + Critic (System2 reranking with critic only)
  3) System1 + Critic + Goal (critic + latent energy reranking)
  4) Goal model
  5) Critic model

Metrics for planning (System1 and combinations):
  - SR_exact: exact sequence match
  - SR_task: task success using order-threshold criterion
  - ordered_ratio: LCS-based order alignment
  - step_accuracy: positional match ratio
  - step_iou: set IoU overlap

Robustness:
  - Goal paraphrase robustness:
      sr_original, sr_paraphrased, sr_drop
  - Optional frame-noise robustness (if --frames_root is provided):
      re-generate interpretation from clean/noisy initial frames and compare SR

Cross-dataset:
  - Run same system metrics on a second dataset split (e.g., CrossTask)

Usage example:
  python3 scripts/run_full_evaluation.py \
      --system1_plain_model checkpoints/system1_plm_coin_lora_no_grounding/best_adapter \
      --system1_jepa_model checkpoints/system1_plm_coin_lora_grounded/best_adapter \
      --critic_model checkpoints/critic_coin/best_model.pt \
      --goal_model checkpoints/goal_model_coin/best_model \
      --goal_latent_model checkpoints/goal_latent_coin/best_model.pt \
      --plm_base_model facebook/Perception-LM-1B \
      --latent_dir data/coin/vjepa_latents \
      --output_dir outputs/evaluation_suite
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import re
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch


def _load_run_inference_module():
    spec = importlib.util.spec_from_file_location(
        "run_inference",
        os.path.join(os.path.dirname(__file__), "run_inference.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ri = _load_run_inference_module()


def read_jsonl(path: str, max_samples: Optional[int] = None):
    samples = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def parse_system1_input_text(input_text: str):
    goal = ""
    interpretation = ""
    prefix = []
    k = None

    for line in input_text.split("\n"):
        s = line.strip()
        if s.startswith("Goal:"):
            goal = s[len("Goal:"):].strip()
        elif s.startswith("Interpretation:"):
            interpretation = s[len("Interpretation:"):].strip()
        elif s and s[0].isdigit() and ")" in s:
            prefix.append(s.split(")", 1)[1].strip())
        elif s.startswith("Predict the next"):
            m = re.search(r"Predict the next\s+(\d+)\s+step", s)
            if m:
                k = int(m.group(1))

    return goal, interpretation, prefix, k


def compute_accuracy(pred_steps: List[str], gold_steps: List[str]) -> float:
    if not gold_steps:
        return 1.0 if not pred_steps else 0.0
    max_len = max(len(pred_steps), len(gold_steps))
    if max_len == 0:
        return 1.0
    correct = 0
    for i in range(max_len):
        p = pred_steps[i] if i < len(pred_steps) else None
        g = gold_steps[i] if i < len(gold_steps) else None
        if p == g:
            correct += 1
    return correct / max_len


def compute_iou(pred_steps: List[str], gold_steps: List[str]) -> float:
    pred_set = set(pred_steps)
    gold_set = set(gold_steps)
    if not pred_set and not gold_set:
        return 1.0
    union = pred_set | gold_set
    if not union:
        return 0.0
    return len(pred_set & gold_set) / len(union)


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def compute_ordered_ratio(pred_steps: List[str], gold_steps: List[str]) -> float:
    if not gold_steps:
        return 1.0 if not pred_steps else 0.0
    if not pred_steps:
        return 0.0
    lcs = lcs_length(pred_steps, gold_steps)
    return lcs / max(len(gold_steps), 1)


def build_planning_metrics(pred_steps: List[str], gold_steps: List[str], order_threshold: float):
    sr_exact = 1.0 if pred_steps == gold_steps else 0.0
    step_acc = compute_accuracy(pred_steps, gold_steps)
    step_iou = compute_iou(pred_steps, gold_steps)
    ordered = compute_ordered_ratio(pred_steps, gold_steps)
    sr_task = 1.0 if ordered >= order_threshold else 0.0
    return {
        "sr_exact": sr_exact,
        "sr_task": sr_task,
        "ordered_ratio": ordered,
        "step_accuracy": step_acc,
        "step_iou": step_iou,
    }


def aggregate_planning_metrics(rows: List[Dict]):
    if not rows:
        return {
            "n_samples": 0,
            "SR_exact": 0.0,
            "SR_task": 0.0,
            "ordered_ratio": 0.0,
            "step_accuracy": 0.0,
            "step_iou": 0.0,
        }
    return {
        "n_samples": len(rows),
        "SR_exact": round(float(np.mean([r["sr_exact"] for r in rows])), 4),
        "SR_task": round(float(np.mean([r["sr_task"] for r in rows])), 4),
        "ordered_ratio": round(float(np.mean([r["ordered_ratio"] for r in rows])), 4),
        "step_accuracy": round(float(np.mean([r["step_accuracy"] for r in rows])), 4),
        "step_iou": round(float(np.mean([r["step_iou"] for r in rows])), 4),
    }


def paraphrase_goal(goal: str, count: int = 3):
    core = (goal or "").strip()
    core = re.sub(r"^complete the task:\s*", "", core, flags=re.IGNORECASE)
    if not core:
        return []

    variants = [
        f"Complete the task: {core}.",
        f"Please carry out this task: {core}.",
        f"Perform the following objective: {core}.",
        f"Finish this procedure successfully: {core}.",
    ]
    if core.lower().startswith("make "):
        thing = core[5:]
        variants.append(f"Prepare {thing}.")
        variants.append(f"Cook {thing}.")

    uniq = []
    seen = set()
    for v in variants:
        n = normalize_text(v)
        if n and n not in seen:
            seen.add(n)
            uniq.append(v)
    return uniq[: max(count, 1)]


def _critic_cost(critic_model, critic_tokenizer, format_traj_fn, goal: str, steps: List[str], device):
    text = format_traj_fn(goal, steps)
    enc = critic_tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        return float(critic_model(enc.input_ids, enc.attention_mask).item())


def _generate_k_plans(system1, tokenizer, gen_mode, prompt: str, args, device):
    plans = []
    sampling_temp = args.temperature
    if args.K > 1 and sampling_temp <= 0:
        sampling_temp = 0.8

    for _ in range(args.K):
        raw = ri.generate_plan(
            system1,
            tokenizer,
            prompt,
            device,
            gen_mode=gen_mode,
            temperature=sampling_temp,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        steps = ri.parse_plan_json(raw)
        plans.append({"raw": raw, "steps": steps, "valid": len(steps) > 0})
    return plans


def _best_plan_with_critic(
    plans,
    goal: str,
    prefix_steps: List[str],
    critic_model,
    critic_tokenizer,
    format_traj_fn,
    device,
):
    scores = []
    for p in plans:
        if not p["valid"]:
            scores.append(math.inf)
            continue
        full_steps = prefix_steps + p["steps"]
        scores.append(_critic_cost(critic_model, critic_tokenizer, format_traj_fn, goal, full_steps, device))
    idx = min(range(len(scores)), key=lambda i: scores[i])
    return plans[idx], scores


def _best_plan_with_critic_goal(
    plans,
    goal: str,
    prefix_steps: List[str],
    critic_model,
    critic_tokenizer,
    format_traj_fn,
    goal_latent_model,
    goal_latent_tokenizer,
    goal_latent_format,
    latent_dir,
    task_id,
    video_id,
    alpha,
    beta,
    device,
):
    critic_scores = []
    energy_scores = []

    use_learned_energy = goal_latent_model is not None
    use_lookup_energy = (
        (not use_learned_energy)
        and latent_dir
        and os.path.isdir(latent_dir)
        and task_id
        and video_id
    )

    for p in plans:
        if not p["valid"]:
            critic_scores.append(math.inf)
            energy_scores.append(math.inf)
            continue

        full_steps = prefix_steps + p["steps"]
        c = _critic_cost(critic_model, critic_tokenizer, format_traj_fn, goal, full_steps, device)
        critic_scores.append(c)

        if use_learned_energy:
            e = ri.compute_learned_energy(
                goal_latent_model,
                goal_latent_tokenizer,
                goal_latent_format,
                goal,
                full_steps,
                device,
            )
        elif use_lookup_energy:
            e = ri.compute_energy(latent_dir, task_id, video_id, p["steps"], len(prefix_steps))
        else:
            e = 0.0
        energy_scores.append(float(e))

    if not use_learned_energy and not use_lookup_energy:
        alpha, beta = 1.0, 0.0

    combined = [alpha * c + beta * e for c, e in zip(critic_scores, energy_scores)]
    idx = min(range(len(combined)), key=lambda i: combined[i])
    return plans[idx], critic_scores, energy_scores, combined


def evaluate_system_config(samples, cfg_name: str, cfg, args, device):
    rows = []

    if cfg_name == "system1_no_jepa":
        s1_model, s1_tok, gen_mode = ri.load_system1(
            cfg.system1_plain_model, device, args.system1_type, args.plm_base_model
        )
        critic_model = critic_tok = critic_fmt = None
        goal_latent_model = goal_latent_tok = goal_latent_fmt = None
    elif cfg_name == "system1_with_jepa":
        s1_model, s1_tok, gen_mode = ri.load_system1(
            cfg.system1_jepa_model, device, args.system1_type, args.plm_base_model
        )
        critic_model = critic_tok = critic_fmt = None
        goal_latent_model = goal_latent_tok = goal_latent_fmt = None
    else:
        s1_model, s1_tok, gen_mode = ri.load_system1(
            cfg.system1_jepa_model, device, args.system1_type, args.plm_base_model
        )
        critic_model, critic_tok, critic_fmt = ri.load_critic(
            args.critic_model, device, args.critic_type, args.llm_base_critic
        )
        goal_latent_model = goal_latent_tok = goal_latent_fmt = None
        if cfg_name == "system1_critic_goal" and args.goal_latent_model:
            goal_latent_model, goal_latent_tok, goal_latent_fmt = ri.load_goal_latent_model(
                args.goal_latent_model, device
            )

    if args.K > 1 and args.temperature <= 0 and cfg_name in ("system1_critic", "system1_critic_goal"):
        print(
            f"  [{cfg_name}] requested K={args.K} with temperature={args.temperature}; "
            f"using fallback sampling temperature 0.8 for candidate diversity"
        )

    for idx, sample in enumerate(samples):
        goal, interp, prefix_steps, k_from_prompt = parse_system1_input_text(sample["input_text"])
        meta = sample.get("meta", {})
        k = int(meta.get("k", k_from_prompt or args.default_k))
        gold_steps = ri.parse_plan_json(sample.get("output_text", ""))

        prompt = ri.build_system1_prompt(
            goal=goal,
            prefix_steps=prefix_steps,
            k=k,
            interpretation=interp,
        )

        if cfg_name in ("system1_no_jepa", "system1_with_jepa"):
            raw = ri.generate_plan(
                s1_model,
                s1_tok,
                prompt,
                device,
                gen_mode=gen_mode,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
            pred_steps = ri.parse_plan_json(raw)
            extra = {"raw": raw}
        else:
            plans = _generate_k_plans(s1_model, s1_tok, gen_mode, prompt, args, device)
            if cfg_name == "system1_critic":
                best_plan, critic_scores = _best_plan_with_critic(
                    plans,
                    goal,
                    prefix_steps,
                    critic_model,
                    critic_tok,
                    critic_fmt,
                    device,
                )
                pred_steps = best_plan["steps"]
                extra = {
                    "best_plan": best_plan,
                    "critic_scores": critic_scores,
                }
            else:
                best_plan, critic_scores, energy_scores, combined = _best_plan_with_critic_goal(
                    plans,
                    goal,
                    prefix_steps,
                    critic_model,
                    critic_tok,
                    critic_fmt,
                    goal_latent_model,
                    goal_latent_tok,
                    goal_latent_fmt,
                    args.latent_dir,
                    str(meta.get("task_id", "")),
                    str(meta.get("video_id", "")),
                    args.alpha,
                    args.beta,
                    device,
                )
                pred_steps = best_plan["steps"]
                extra = {
                    "best_plan": best_plan,
                    "critic_scores": critic_scores,
                    "energy_scores": energy_scores,
                    "combined_scores": combined,
                }

        m = build_planning_metrics(pred_steps, gold_steps, args.order_threshold)
        rows.append(
            {
                "sample_idx": idx,
                "task": goal,
                "task_name": meta.get("task_name", ""),
                "video_id": meta.get("video_id", ""),
                "task_id": meta.get("task_id", ""),
                "prefix_steps": prefix_steps,
                "gold_steps": gold_steps,
                "pred_steps": pred_steps,
                **m,
                **extra,
            }
        )

        if (idx + 1) % 25 == 0:
            print(f"  [{cfg_name}] processed {idx+1}/{len(samples)} samples")

    del s1_model, s1_tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if cfg_name in ("system1_critic", "system1_critic_goal"):
        del critic_model, critic_tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if cfg_name == "system1_critic_goal" and args.goal_latent_model:
        del goal_latent_model, goal_latent_tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


def evaluate_goal_model(samples, goal_model_path: str, device):
    model, tok = ri.load_goal_model(goal_model_path, device)
    rows = []
    for idx, sample in enumerate(samples):
        input_text = sample.get("input_text", "")
        prefix = []
        for line in input_text.split("\n"):
            s = line.strip()
            if s and s[0].isdigit() and ")" in s:
                prefix.append(s.split(")", 1)[1].strip())

        prompt = ri.build_goal_prompt(prefix)
        enc = tok(prompt, max_length=384, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(**enc, max_new_tokens=128, num_beams=1, do_sample=False)
        pred = tok.decode(gen_ids[0], skip_special_tokens=True).strip()

        gold_raw = sample.get("output_text", "")
        gold_goal = gold_raw
        if gold_raw.strip().startswith("{"):
            try:
                parsed = json.loads(gold_raw)
                gold_goal = parsed.get("goal", "")
            except Exception:
                pass

        pred_n = normalize_text(pred)
        gold_n = normalize_text(gold_goal)
        exact = 1.0 if pred_n == gold_n else 0.0
        contain = 1.0 if (pred_n in gold_n or gold_n in pred_n) and pred_n and gold_n else 0.0

        rows.append(
            {
                "sample_idx": idx,
                "task": sample.get("task", ""),
                "task_name": sample.get("meta", {}).get("task_name", ""),
                "video_id": sample.get("meta", {}).get("video_id", ""),
                "pred_goal": pred,
                "gold_goal": gold_goal,
                "goal_exact": exact,
                "goal_contains": contain,
            }
        )

    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not rows:
        return rows, {"n_samples": 0, "goal_exact": 0.0, "goal_contains": 0.0}

    agg = {
        "n_samples": len(rows),
        "goal_exact": round(float(np.mean([r["goal_exact"] for r in rows])), 4),
        "goal_contains": round(float(np.mean([r["goal_contains"] for r in rows])), 4),
    }
    return rows, agg


def evaluate_critic_model(samples, args, device):
    model, tok, fmt = ri.load_critic(args.critic_model, device, args.critic_type, args.llm_base_critic)

    rows = []
    for idx, rec in enumerate(samples):
        goal = rec["goal"]
        c_good = _critic_cost(model, tok, fmt, goal, rec["good"], device)
        c_base = _critic_cost(model, tok, fmt, goal, rec["base"], device)
        c_bad = _critic_cost(model, tok, fmt, goal, rec["bad"], device)
        c_shuffled = _critic_cost(model, tok, fmt, goal, rec["shuffled"], device)

        good_base = 1.0 if c_good < c_base else 0.0
        base_bad = 1.0 if c_base < c_bad else 0.0
        base_shuf = 1.0 if c_base < c_shuffled else 0.0
        rank_acc = (good_base + base_bad + base_shuf) / 3.0

        rows.append(
            {
                "sample_idx": idx,
                "task_name": rec.get("metadata", {}).get("task_name", ""),
                "video_id": rec.get("metadata", {}).get("video_id", ""),
                "c_good": c_good,
                "c_base": c_base,
                "c_bad": c_bad,
                "c_shuffled": c_shuffled,
                "good_lt_base": good_base,
                "base_lt_bad": base_bad,
                "base_lt_shuffled": base_shuf,
                "ranking_accuracy": rank_acc,
            }
        )

    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not rows:
        return rows, {
            "n_samples": 0,
            "ranking_accuracy": 0.0,
            "good_lt_base": 0.0,
            "base_lt_bad": 0.0,
            "base_lt_shuffled": 0.0,
        }

    agg = {
        "n_samples": len(rows),
        "ranking_accuracy": round(float(np.mean([r["ranking_accuracy"] for r in rows])), 4),
        "good_lt_base": round(float(np.mean([r["good_lt_base"] for r in rows])), 4),
        "base_lt_bad": round(float(np.mean([r["base_lt_bad"] for r in rows])), 4),
        "base_lt_shuffled": round(float(np.mean([r["base_lt_shuffled"] for r in rows])), 4),
    }
    return rows, agg


def evaluate_paraphrase_robustness(samples, cfg_name: str, cfg, args, device, original_rows=None):
    if args.skip_paraphrase_eval:
        return {"skipped": True, "reason": "skip_paraphrase_eval"}
    if args.paraphrase_count <= 0:
        return {"skipped": True, "reason": "paraphrase_count<=0"}

    # Reuse full config evaluation machinery but only with altered goals.
    altered = []
    for sample in samples:
        goal, _, _, _ = parse_system1_input_text(sample["input_text"])
        paras = paraphrase_goal(goal, args.paraphrase_count)
        if not paras:
            continue
        chosen = paras[0]

        replaced = dict(sample)
        replaced_text = sample["input_text"].replace(f"Goal: {goal}", f"Goal: {chosen}", 1)
        replaced["input_text"] = replaced_text
        altered.append(replaced)

    if not altered:
        return {"skipped": True, "reason": "no-paraphraseable-goals"}

    if original_rows is None:
        original_rows = evaluate_system_config(samples, cfg_name, cfg, args, device)
    para_rows = evaluate_system_config(altered, cfg_name, cfg, args, device)

    sr_original = float(np.mean([r["sr_task"] for r in original_rows])) if original_rows else 0.0
    sr_para = float(np.mean([r["sr_task"] for r in para_rows])) if para_rows else 0.0
    return {
        "n_samples": len(para_rows),
        "sr_original": round(sr_original, 4),
        "sr_paraphrased": round(sr_para, 4),
        "sr_drop": round(sr_original - sr_para, 4),
    }


def find_frames_dir(frames_root: str, task_id: str, video_id: str):
    candidates = [
        Path(frames_root) / str(video_id),
        Path(frames_root) / str(task_id) / str(video_id),
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return str(c)
    return None


def add_noise_to_frame_dir(src_dir: str, dst_dir: str, std: float, max_images: int = 8):
    from PIL import Image

    files = sorted([p for p in Path(src_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    files = files[:max_images]
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    for p in files:
        arr = np.array(Image.open(p).convert("RGB")).astype(np.float32)
        noise = np.random.normal(0.0, std, size=arr.shape)
        out = np.clip(arr + noise, 0, 255).astype(np.uint8)
        Image.fromarray(out).save(Path(dst_dir) / p.name)


def evaluate_frame_noise_robustness(samples, cfg_name: str, cfg, args, device):
    if not args.frames_root:
        return {"skipped": True, "reason": "frames_root-not-provided"}

    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_name = args.interp_model or args.plm_base_model or "facebook/Perception-LM-1B"
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    interp_model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    clean_samples = []
    noisy_samples = []
    used = 0
    for sample in samples:
        meta = sample.get("meta", {})
        task_id = str(meta.get("task_id", ""))
        video_id = str(meta.get("video_id", ""))
        fdir = find_frames_dir(args.frames_root, task_id, video_id)
        if not fdir:
            continue

        clean_interp = ri.generate_interpretation_from_frames(
            frames_path=fdir,
            device=device,
            model_name=model_name,
            num_frames=args.num_frames,
            preloaded_model=interp_model,
            preloaded_processor=processor,
        )

        with tempfile.TemporaryDirectory() as tmp:
            add_noise_to_frame_dir(fdir, tmp, std=args.frame_noise_std, max_images=args.num_frames)
            noisy_interp = ri.generate_interpretation_from_frames(
                frames_path=tmp,
                device=device,
                model_name=model_name,
                num_frames=args.num_frames,
                preloaded_model=interp_model,
                preloaded_processor=processor,
            )

        clean_s = dict(sample)
        noisy_s = dict(sample)

        goal, interp, _, _ = parse_system1_input_text(sample["input_text"])
        clean_s["input_text"] = sample["input_text"].replace(
            f"Interpretation: {interp}", f"Interpretation: {clean_interp}", 1
        )
        noisy_s["input_text"] = sample["input_text"].replace(
            f"Interpretation: {interp}", f"Interpretation: {noisy_interp}", 1
        )

        clean_samples.append(clean_s)
        noisy_samples.append(noisy_s)
        used += 1
        if args.frame_noise_max_samples and used >= args.frame_noise_max_samples:
            break

    del interp_model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not clean_samples:
        return {"skipped": True, "reason": "no-frame-dirs-found"}

    clean_rows = evaluate_system_config(clean_samples, cfg_name, cfg, args, device)
    noisy_rows = evaluate_system_config(noisy_samples, cfg_name, cfg, args, device)

    sr_clean = float(np.mean([r["sr_task"] for r in clean_rows])) if clean_rows else 0.0
    sr_noisy = float(np.mean([r["sr_task"] for r in noisy_rows])) if noisy_rows else 0.0
    return {
        "n_samples": len(noisy_rows),
        "sr_clean": round(sr_clean, 4),
        "sr_noisy": round(sr_noisy, 4),
        "sr_drop": round(sr_clean - sr_noisy, 4),
        "noise_std": args.frame_noise_std,
    }


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def save_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")


def _to_csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def save_csv(path: Path, rows: List[Dict], prefix: Optional[Dict] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["empty"])
        return

    prefix = prefix or {}

    all_keys = []
    seen = set()
    for row in rows:
        keys = list(prefix.keys()) + list(row.keys())
        for k in keys:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            merged = {}
            merged.update(prefix)
            merged.update(row)
            writer.writerow({k: _to_csv_value(merged.get(k, "")) for k in all_keys})


def save_planning_payload(out_dir: Path, dataset_name: str, cfg_name: str, payload: Dict, run_meta: Dict):
    save_json(out_dir / f"{dataset_name}_{cfg_name}_aggregate.json", payload["aggregate"])
    save_jsonl(out_dir / f"{dataset_name}_{cfg_name}_per_sample.jsonl", payload["rows"])
    save_csv(
        out_dir / f"{dataset_name}_{cfg_name}_aggregate.csv",
        [payload["aggregate"]],
        prefix={
            **run_meta,
            "dataset": dataset_name,
            "config": cfg_name,
        },
    )
    save_csv(
        out_dir / f"{dataset_name}_{cfg_name}_per_sample.csv",
        payload["rows"],
        prefix={
            **run_meta,
            "dataset": dataset_name,
            "config": cfg_name,
        },
    )
    if dataset_name == "main":
        save_json(
            out_dir / f"{dataset_name}_{cfg_name}_robustness.json",
            {
                "paraphrase": payload.get("paraphrase_robustness", {}),
                "frame_noise": payload.get("frame_noise_robustness", {}),
            },
        )
        save_csv(
            out_dir / f"{dataset_name}_{cfg_name}_robustness.csv",
            [{
                "paraphrase": payload.get("paraphrase_robustness", {}),
                "frame_noise": payload.get("frame_noise_robustness", {}),
            }],
            prefix={
                **run_meta,
                "dataset": dataset_name,
                "config": cfg_name,
            },
        )


def evaluate_planning_bundle(dataset_name: str, dataset_path: str, cfg, args, device, out_dir: Path, run_meta: Dict):
    samples = read_jsonl(dataset_path, max_samples=args.max_samples)
    out = {
        "dataset": dataset_name,
        "path": dataset_path,
        "n_samples": len(samples),
        "configs": {},
    }

    for cfg_name in args.planning_configs:
        print(f"\nEvaluating [{dataset_name}] {cfg_name} ...")
        rows = evaluate_system_config(samples, cfg_name, cfg, args, device)
        payload = {
            "aggregate": aggregate_planning_metrics(rows),
            "rows": rows,
        }
        out["configs"][cfg_name] = payload

        if dataset_name == "main":
            para = evaluate_paraphrase_robustness(
                samples, cfg_name, cfg, args, device, original_rows=rows
            )
            out["configs"][cfg_name]["paraphrase_robustness"] = para

            frame_noise = evaluate_frame_noise_robustness(samples, cfg_name, cfg, args, device)
            out["configs"][cfg_name]["frame_noise_robustness"] = frame_noise

        save_planning_payload(out_dir, dataset_name, cfg_name, out["configs"][cfg_name], run_meta)

    return out


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation for System1/System2/Goal/Critic")

    # Datasets
    parser.add_argument("--system1_test", default="data/coin/coin_system1_test.jsonl")
    parser.add_argument("--system1_cross_test", default="data/crosstask/system1_test.jsonl")
    parser.add_argument("--goal_test", default="data/coin/coin_goal_test.jsonl")
    parser.add_argument("--goal_cross_test", default="data/crosstask/goal_test.jsonl")
    parser.add_argument("--critic_test", default="data/coin/coin_critic_test.jsonl")
    parser.add_argument("--critic_cross_test", default="data/crosstask/critic_test.jsonl")

    # System1 models
    parser.add_argument("--system1_plain_model", required=True,
                        help="System1 checkpoint without JEPA grounding")
    parser.add_argument("--system1_jepa_model", default=None,
                        help="System1 checkpoint trained with JEPA grounding")
    parser.add_argument("--system1_type", choices=["t5", "plm"], default="plm")
    parser.add_argument("--plm_base_model", default="facebook/Perception-LM-1B")

    # Critic/Goal models
    parser.add_argument("--critic_model", required=True)
    parser.add_argument("--critic_type", choices=["mlp", "llm"], default="mlp")
    parser.add_argument("--llm_base_critic", default=None)
    parser.add_argument("--goal_model", default=None)
    parser.add_argument("--goal_latent_model", default=None,
                        help="Optional learned goal-latent checkpoint for System1+Critic+Goal")

    # Latent settings for lookup energy fallback
    parser.add_argument("--latent_dir", default=None)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)

    # Generation
    parser.add_argument("--default_k", type=int, default=3)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    # Robustness
    parser.add_argument("--order_threshold", type=float, default=0.6)
    parser.add_argument("--paraphrase_count", type=int, default=3)

    parser.add_argument("--frames_root", default=None,
                        help="Optional root dir for frame folders for frame-noise robustness")
    parser.add_argument("--interp_model", default="facebook/Perception-LM-1B")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--frame_noise_std", type=float, default=20.0)
    parser.add_argument("--frame_noise_max_samples", type=int, default=100)

    # Runtime/output
    parser.add_argument("--run_id", default=None,
                        help="Optional run identifier. If omitted, auto-generated timestamp ID is used")
    parser.add_argument("--run_description", default="",
                        help="Free-text description for this evaluation run")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/evaluation_suite")
    parser.add_argument(
        "--planning_configs",
        default="system1_no_jepa,system1_with_jepa,system1_critic,system1_critic_goal",
        help=(
            "Comma-separated planning configs to run. "
            "Options: system1_no_jepa,system1_with_jepa,system1_critic,system1_critic_goal"
        ),
    )
    parser.add_argument(
        "--skip_goal_eval",
        action="store_true",
        help="Skip standalone Goal model evaluation block",
    )
    parser.add_argument(
        "--skip_cross_eval",
        action="store_true",
        help="Skip CrossTask planning/goal/critic evaluation blocks",
    )
    parser.add_argument(
        "--skip_critic_eval",
        action="store_true",
        help="Skip standalone Critic model evaluation block",
    )
    parser.add_argument(
        "--skip_paraphrase_eval",
        action="store_true",
        help="Skip goal-paraphrase robustness evaluation for main planning configs",
    )

    args = parser.parse_args()

    valid_configs = {
        "system1_no_jepa",
        "system1_with_jepa",
        "system1_critic",
        "system1_critic_goal",
    }
    args.planning_configs = [c.strip() for c in args.planning_configs.split(",") if c.strip()]
    bad = [c for c in args.planning_configs if c not in valid_configs]
    if bad:
        raise ValueError(f"Invalid planning config(s): {bad}. Valid: {sorted(valid_configs)}")

    if not args.system1_jepa_model:
        args.system1_jepa_model = args.system1_plain_model

    needs_critic = any(c in args.planning_configs for c in ("system1_critic", "system1_critic_goal"))
    if needs_critic and not args.critic_model:
        raise ValueError("--critic_model is required for selected planning configs")

    needs_goal_latent_combo = "system1_critic_goal" in args.planning_configs
    if needs_goal_latent_combo and (not args.goal_latent_model and not args.latent_dir):
        print("WARNING: system1_critic_goal selected without --goal_latent_model/--latent_dir; it will fallback to critic-only scoring")

    if not args.skip_goal_eval and not args.goal_model:
        raise ValueError("--goal_model is required unless --skip_goal_eval is set")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_description = args.run_description or ""
    run_meta = {
        "run_id": run_id,
        "run_description": run_description,
    }

    cfg = SimpleNamespace(
        system1_plain_model=args.system1_plain_model,
        system1_jepa_model=args.system1_jepa_model,
    )

    # A) Planning evaluations (main + cross dataset)
    planning_main = evaluate_planning_bundle("main", args.system1_test, cfg, args, device, out_dir, run_meta)
    if args.skip_cross_eval:
        planning_cross = {"configs": {}}
    else:
        planning_cross = evaluate_planning_bundle("cross", args.system1_cross_test, cfg, args, device, out_dir, run_meta)

    # B) Goal model evaluations (main + cross)
    if args.skip_goal_eval:
        goal_rows_main, goal_agg_main = [], {"skipped": True}
        goal_rows_cross, goal_agg_cross = [], {"skipped": True}
    else:
        goal_rows_main, goal_agg_main = evaluate_goal_model(
            read_jsonl(args.goal_test, max_samples=args.max_samples),
            args.goal_model,
            device,
        )
        if args.skip_cross_eval:
            goal_rows_cross, goal_agg_cross = [], {"skipped": True}
        else:
            goal_rows_cross, goal_agg_cross = evaluate_goal_model(
                read_jsonl(args.goal_cross_test, max_samples=args.max_samples),
                args.goal_model,
                device,
            )

    # C) Critic model evaluations (main + cross)
    if args.skip_critic_eval:
        critic_rows_main, critic_agg_main = [], {"skipped": True}
        critic_rows_cross, critic_agg_cross = [], {"skipped": True}
    else:
        critic_rows_main, critic_agg_main = evaluate_critic_model(
            read_jsonl(args.critic_test, max_samples=args.max_samples),
            args,
            device,
        )
        if args.skip_cross_eval:
            critic_rows_cross, critic_agg_cross = [], {"skipped": True}
        else:
            critic_rows_cross, critic_agg_cross = evaluate_critic_model(
                read_jsonl(args.critic_cross_test, max_samples=args.max_samples),
                args,
                device,
            )

    summary = {
        "run_id": run_id,
        "run_description": run_description,
        "device": str(device),
        "planning_main": {
            k: v["aggregate"] for k, v in planning_main["configs"].items()
        },
        "planning_cross": {
            k: v["aggregate"] for k, v in planning_cross["configs"].items()
        },
        "goal_main": goal_agg_main,
        "goal_cross": goal_agg_cross,
        "critic_main": critic_agg_main,
        "critic_cross": critic_agg_cross,
    }

    # Save artifacts
    save_json(out_dir / "summary.json", summary)

    summary_rows = []
    for section, metrics in summary.items():
        if not isinstance(metrics, dict):
            continue
        if section in ("planning_main", "planning_cross"):
            for cfg_name, cfg_metrics in metrics.items():
                row = {"section": section, "config": cfg_name}
                row.update(cfg_metrics)
                summary_rows.append(row)
        elif section in ("goal_main", "goal_cross", "critic_main", "critic_cross"):
            row = {"section": section, "config": "-"}
            row.update(metrics)
            summary_rows.append(row)

    save_csv(out_dir / "summary.csv", summary_rows, prefix=run_meta)

    save_json(out_dir / "goal_main_aggregate.json", goal_agg_main)
    save_jsonl(out_dir / "goal_main_per_sample.jsonl", goal_rows_main)
    save_json(out_dir / "goal_cross_aggregate.json", goal_agg_cross)
    save_jsonl(out_dir / "goal_cross_per_sample.jsonl", goal_rows_cross)
    save_csv(
        out_dir / "goal_main_aggregate.csv",
        [goal_agg_main],
        prefix={
            "run_id": run_id,
            "run_description": run_description,
            "dataset": "main",
            "component": "goal",
        },
    )
    save_csv(
        out_dir / "goal_main_per_sample.csv",
        goal_rows_main,
        prefix={
            "run_id": run_id,
            "run_description": run_description,
            "dataset": "main",
            "component": "goal",
        },
    )
    save_csv(
        out_dir / "goal_cross_aggregate.csv",
        [goal_agg_cross],
        prefix={
            "run_id": run_id,
            "run_description": run_description,
            "dataset": "cross",
            "component": "goal",
        },
    )
    save_csv(
        out_dir / "goal_cross_per_sample.csv",
        goal_rows_cross,
        prefix={
            "run_id": run_id,
            "run_description": run_description,
            "dataset": "cross",
            "component": "goal",
        },
    )

    save_json(out_dir / "critic_main_aggregate.json", critic_agg_main)
    save_jsonl(out_dir / "critic_main_per_sample.jsonl", critic_rows_main)
    save_json(out_dir / "critic_cross_aggregate.json", critic_agg_cross)
    save_jsonl(out_dir / "critic_cross_per_sample.jsonl", critic_rows_cross)
    save_csv(
        out_dir / "critic_main_aggregate.csv",
        [critic_agg_main],
        prefix={
            "run_id": run_id,
            "run_description": run_description,
            "dataset": "main",
            "component": "critic",
        },
    )
    save_csv(
        out_dir / "critic_main_per_sample.csv",
        critic_rows_main,
        prefix={
            "run_id": run_id,
            "run_description": run_description,
            "dataset": "main",
            "component": "critic",
        },
    )
    save_csv(
        out_dir / "critic_cross_aggregate.csv",
        [critic_agg_cross],
        prefix={
            "run_id": run_id,
            "run_description": run_description,
            "dataset": "cross",
            "component": "critic",
        },
    )
    save_csv(
        out_dir / "critic_cross_per_sample.csv",
        critic_rows_cross,
        prefix={
            "run_id": run_id,
            "run_description": run_description,
            "dataset": "cross",
            "component": "critic",
        },
    )

    print("\n============================================================")
    print("Evaluation complete")
    print("============================================================")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()
