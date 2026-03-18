#!/usr/bin/env python3
"""
Evaluate planning quality using VPA benchmark metrics from the proposal:

  1. Success Rate (SR)  — fraction of samples where the predicted step
     sequence exactly matches the gold sequence (plan-level accuracy)

  2. Mean Accuracy (mAcc) — step-level accuracy, averaged over samples.
     For each sample, what fraction of predicted steps match the gold at
     each position?

  3. Mean IoU (mIoU) — intersection over union between predicted and gold
     step sets, averaged over samples (action-proposal accuracy).

These are the standard VPA metrics from Patel et al., 2023, used by VLWM.

Two evaluation modes:
  A. System-1 only (greedy decode, no reranking)
  B. System-2 (reranked plans from run_planning.py output)

Saves:
  outputs/evaluation/{mode}_results.json     (aggregate metrics)
  outputs/evaluation/{mode}_per_task.json    (per-task breakdown)
  outputs/evaluation/{mode}_per_sample.jsonl (per-sample details)

Usage (System-1 greedy):
    python3 scripts/run_evaluation.py \
        --mode system1 \
        --test_data       data/crosstask/system1_test.jsonl \
        --system1_model   checkpoints/system1/best_model \
        --output_dir      outputs/evaluation

Usage (System-2 from planning output):
    python3 scripts/run_evaluation.py \
        --mode system2 \
        --planning_output outputs/planning/YYYYMMDD_HHMMSS_plans.jsonl \
        --output_dir      outputs/evaluation

Usage (both, for comparison):
    python3 scripts/run_evaluation.py \
        --mode both \
        --test_data       data/crosstask/system1_test.jsonl \
        --system1_model   checkpoints/system1/best_model \
        --planning_output outputs/planning/YYYYMMDD_HHMMSS_plans.jsonl \
        --output_dir      outputs/evaluation
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_sr(pred_steps: list, gold_steps: list) -> float:
    """Success Rate: 1 if exact sequence match, 0 otherwise."""
    return 1.0 if pred_steps == gold_steps else 0.0


def compute_accuracy(pred_steps: list, gold_steps: list) -> float:
    """Step-level accuracy: fraction of positions where pred == gold."""
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


def compute_iou(pred_steps: list, gold_steps: list) -> float:
    """IoU: |intersection| / |union| of step sets."""
    pred_set = set(pred_steps)
    gold_set = set(gold_steps)
    if not pred_set and not gold_set:
        return 1.0
    intersection = pred_set & gold_set
    union = pred_set | gold_set
    return len(intersection) / len(union) if union else 0.0


def extract_step_strings(output_text: str) -> list:
    """Parse gold output JSON to list of 'action | state_change' strings."""
    try:
        parsed = json.loads(output_text)
        steps = parsed.get("next_steps", [])
        return [f"{s['action']} | {s['state_change']}" for s in steps]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


# ---------------------------------------------------------------------------
# System-1 evaluation (greedy decode)
# ---------------------------------------------------------------------------

def _load_system1_model(model_path, device, model_type="t5", base_model_name=None):
    """Load System-1 model. Returns (model, tokenizer, gen_mode)."""
    from transformers import AutoTokenizer

    if model_type == "t5":
        from transformers import T5ForConditionalGeneration
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return model, tokenizer, "seq2seq"

    elif model_type == "plm":
        from transformers import AutoModelForCausalLM
        from peft import PeftModel

        if base_model_name is None:
            base_model_name = "facebook/Perception-LM-1B"

        is_adapter = os.path.isdir(model_path) and os.path.exists(
            os.path.join(model_path, "adapter_config.json")
        )
        tokenizer_source = model_path if os.path.isdir(model_path) else base_model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Try VLM class, fallback to causal LM
        if is_adapter:
            try:
                from transformers import AutoModelForImageTextToText
                base = AutoModelForImageTextToText.from_pretrained(
                    base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
                )
            except Exception:
                base = AutoModelForCausalLM.from_pretrained(
                    base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
                )

            model = PeftModel.from_pretrained(base, model_path)
        else:
            try:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
                )

        model.to(device)
        model.eval()
        return model, tokenizer, "causal"

    else:
        raise ValueError(f"Unknown system1_type: {model_type}")


def evaluate_system1(test_data_path: str, model_path: str, device,
                     max_samples=None, model_type="t5", base_model_name=None):
    """Run System-1 greedy decoding on test set and compute metrics."""
    print(f"  Loading model ({model_type}): {model_path}")
    model, tokenizer, gen_mode = _load_system1_model(
        model_path, device, model_type, base_model_name
    )

    print(f"  Loading test data: {test_data_path}")
    samples = []
    with open(test_data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]

    results = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        input_text = sample["input_text"]
        gold_output = sample["output_text"]
        meta = sample.get("meta", {})

        gold_steps = extract_step_strings(gold_output)

        # Greedy decode
        enc = tokenizer(
            input_text, max_length=512, truncation=True, return_tensors="pt"
        ).to(device)
        prompt_len = enc.input_ids.shape[1]

        with torch.no_grad():
            gen_ids = model.generate(
                **enc, max_new_tokens=256, num_beams=1, do_sample=False,
            )

        if gen_mode == "causal":
            # Strip prompt tokens for causal LM
            completion_ids = gen_ids[0][prompt_len:]
            pred_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        else:
            pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        pred_steps = extract_step_strings(pred_text)

        sr = compute_sr(pred_steps, gold_steps)
        acc = compute_accuracy(pred_steps, gold_steps)
        iou = compute_iou(pred_steps, gold_steps)

        results.append({
            "sample_idx": i,
            "meta": meta,
            "gold_steps": gold_steps,
            "pred_steps": pred_steps,
            "sr": sr,
            "accuracy": acc,
            "iou": iou,
            "valid_json": len(pred_steps) > 0 or pred_text.strip() == '{"next_steps": []}',
        })

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(samples)}] {time.time()-t0:.0f}s")

    return results


# ---------------------------------------------------------------------------
# System-2 evaluation (from planning output)
# ---------------------------------------------------------------------------

def evaluate_system2(planning_output_path: str):
    """Evaluate reranked plans from run_planning.py output."""
    print(f"  Loading planning output: {planning_output_path}")
    results = []

    with open(planning_output_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            gold_steps = extract_step_strings(rec["gold_output"])
            best_plan = rec.get("best_plan", {})
            pred_steps = best_plan.get("steps", [])

            sr = compute_sr(pred_steps, gold_steps)
            acc = compute_accuracy(pred_steps, gold_steps)
            iou = compute_iou(pred_steps, gold_steps)

            results.append({
                "sample_idx": rec.get("sample_idx", 0),
                "meta": rec.get("meta", {}),
                "gold_steps": gold_steps,
                "pred_steps": pred_steps,
                "sr": sr,
                "accuracy": acc,
                "iou": iou,
                "valid_json": best_plan.get("valid", False),
                "best_plan_idx": rec.get("best_plan_idx", 0),
                "best_score": rec.get("best_score", 0.0),
            })

    return results


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_metrics(results: list) -> dict:
    """Compute aggregate SR, mAcc, mIoU."""
    if not results:
        return {"SR": 0.0, "mAcc": 0.0, "mIoU": 0.0, "n_samples": 0}

    sr_list = [r["sr"] for r in results]
    acc_list = [r["accuracy"] for r in results]
    iou_list = [r["iou"] for r in results]

    return {
        "SR": round(np.mean(sr_list), 4),
        "mAcc": round(np.mean(acc_list), 4),
        "mIoU": round(np.mean(iou_list), 4),
        "n_samples": len(results),
        "valid_json_rate": round(
            sum(1 for r in results if r["valid_json"]) / len(results), 4
        ),
    }


def per_task_metrics(results: list) -> dict:
    """Compute metrics per task."""
    by_task = defaultdict(list)
    for r in results:
        tid = r["meta"].get("task_id", "unknown")
        tname = r["meta"].get("task_name", "unknown")
        by_task[(tid, tname)].append(r)

    task_metrics = {}
    for (tid, tname), task_results in sorted(by_task.items()):
        m = aggregate_metrics(task_results)
        m["task_id"] = tid
        m["task_name"] = tname
        task_metrics[tid] = m

    return task_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate VPA metrics")
    parser.add_argument("--mode", choices=["system1", "system2", "both"],
                        default="system1")
    parser.add_argument("--test_data", default="data/crosstask/system1_test.jsonl")
    parser.add_argument("--system1_model", default="checkpoints/system1/best_model")
    parser.add_argument("--system1_type", choices=["t5", "plm"], default="t5",
                        help="System-1 model type: t5 (seq2seq) or plm (causal LM + LoRA)")
    parser.add_argument("--plm_base_model", default=None,
                        help="Base model name for PLM (e.g. facebook/Perception-LM-1B)")
    parser.add_argument("--planning_output", default=None,
                        help="JSONL from run_planning.py (for system2 mode)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="outputs/evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes_to_run = []
    if args.mode in ("system1", "both"):
        modes_to_run.append("system1")
    if args.mode in ("system2", "both"):
        if not args.planning_output or not os.path.exists(args.planning_output):
            print("ERROR: --planning_output required for system2 mode")
            return
        modes_to_run.append("system2")

    for mode in modes_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating: {mode.upper()}")
        print(f"{'='*60}")

        if mode == "system1":
            results = evaluate_system1(
                args.test_data, args.system1_model, device, args.max_samples,
                model_type=args.system1_type,
                base_model_name=args.plm_base_model,
            )
        else:
            results = evaluate_system2(args.planning_output)

        # Aggregate
        agg = aggregate_metrics(results)
        print(f"\n  Aggregate metrics:")
        print(f"    SR   = {agg['SR']:.4f}  ({agg['SR']*100:.1f}%)")
        print(f"    mAcc = {agg['mAcc']:.4f}  ({agg['mAcc']*100:.1f}%)")
        print(f"    mIoU = {agg['mIoU']:.4f}  ({agg['mIoU']*100:.1f}%)")
        print(f"    Valid JSON: {agg['valid_json_rate']*100:.1f}%")

        # Per-task
        task_m = per_task_metrics(results)
        print(f"\n  Per-task breakdown:")
        print(f"    {'Task':>30s}  {'SR':>6s}  {'mAcc':>6s}  {'mIoU':>6s}  {'N':>5s}")
        print(f"    {'-'*30}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}")
        for tid, m in sorted(task_m.items()):
            print(
                f"    {m['task_name']:>30s}  "
                f"{m['SR']:6.1%}  {m['mAcc']:6.1%}  {m['mIoU']:6.1%}  "
                f"{m['n_samples']:5d}"
            )

        # Save
        with open(out_dir / f"{mode}_results.json", "w") as f:
            json.dump(agg, f, indent=2)
        with open(out_dir / f"{mode}_per_task.json", "w") as f:
            json.dump(task_m, f, indent=2)
        with open(out_dir / f"{mode}_per_sample.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"\n  Saved to {out_dir}/{mode}_*.json")

    # If both modes ran, print comparison
    if len(modes_to_run) == 2:
        s1_agg_path = out_dir / "system1_results.json"
        s2_agg_path = out_dir / "system2_results.json"
        if s1_agg_path.exists() and s2_agg_path.exists():
            s1 = json.load(open(s1_agg_path))
            s2 = json.load(open(s2_agg_path))
            print(f"\n{'='*60}")
            print(f"COMPARISON: System-1 vs System-2")
            print(f"{'='*60}")
            print(f"  {'Metric':>8s}  {'System-1':>10s}  {'System-2':>10s}  {'Δ':>8s}")
            for metric in ("SR", "mAcc", "mIoU"):
                v1 = s1[metric]
                v2 = s2[metric]
                delta = v2 - v1
                sign = "+" if delta >= 0 else ""
                print(f"  {metric:>8s}  {v1:10.4f}  {v2:10.4f}  {sign}{delta:7.4f}")


if __name__ == "__main__":
    main()
