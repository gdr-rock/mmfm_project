#!/usr/bin/env python3
"""Research-oriented evaluation for one System-1 model on coin_2.

Design choices:
  - Evaluate one System-1 checkpoint per run; compare checkpoints later with
    scripts/compare_coin2_evaluations.py.
  - Generate a shared candidate pool once per sample and rerank that same pool
    with critic / goal energy. This avoids the confound in the older suite
    where each planning config regenerated different candidates.
  - Use prompt formats that exactly match coin_2 training prompts:
      * goal_only
      * goal_plus_interpretation
      * goal_plus_interpretation_prefix (derived from held-out task rows)
  - Report stricter and more informative metrics than the old summary:
      * exact_match
      * task_success (gold steps appear in order as a subsequence)
      * ordered_ratio (LCS / gold length)
      * step_precision / recall / f1 / IoU
      * step_accuracy
      * edit_similarity
      * length diagnostics
      * reranking gain vs greedy
      * oracle@K gap to expose candidate-pool headroom
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _load_local_module(module_name: str, filename: str):
    module_path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


efp = _load_local_module("evaluate_full_plans_clean", "evaluate_full_plans_clean.py")
_coin2_utils = _load_local_module("coin2_utils", "coin2_utils.py")
build_full_plan_prompt = _coin2_utils.build_full_plan_prompt
build_remaining_plan_prompt = _coin2_utils.build_remaining_plan_prompt


MAIN_METRICS = [
    "exact_match",
    "task_success",
    "ordered_ratio",
    "step_precision",
    "step_recall",
    "step_f1",
    "step_iou",
    "step_accuracy",
    "edit_similarity",
]
GAIN_METRICS = [
    "gain_vs_greedy_exact_match",
    "gain_vs_greedy_task_success",
    "gain_vs_greedy_ordered_ratio",
    "gain_vs_greedy_step_f1",
    "gain_vs_greedy_step_accuracy",
    "oracle_gap_exact_match",
    "oracle_gap_task_success",
    "oracle_gap_ordered_ratio",
    "oracle_gap_step_f1",
]


@dataclass
class EvalCase:
    case_id: int
    base_sample_id: int
    condition_family: str
    condition_label: str
    prompt_variant: str
    prefix_len: int
    goal: str
    interpretation: str
    prefix_steps: list[str]
    gold_steps: list[str]
    meta: dict
    source_dataset: str

    def prompt_text(self) -> str:
        if self.prefix_len > 0:
            return build_remaining_plan_prompt(
                self.goal,
                [split_step_text(step) for step in self.prefix_steps],
                interpretation=self.interpretation,
            )
        return build_full_plan_prompt(
            self.goal,
            interpretation=self.interpretation,
        )


def split_step_text(step: str) -> tuple[str, str]:
    if " | " not in step:
        return step.strip(), ""
    action, state_change = step.split(" | ", 1)
    return action.strip(), state_change.strip()


def normalize_step(step: str) -> str:
    return efp.normalize_step(step)


def lcs_length(a: list[str], b: list[str]) -> int:
    return efp.lcs_length(a, b)


def levenshtein_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        curr = [i]
        for j, y in enumerate(b, start=1):
            cost = 0 if x == y else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def compute_sequence_metrics(pred_steps: list[str], gold_steps: list[str]) -> dict:
    pred_norm = [normalize_step(step) for step in pred_steps]
    gold_norm = [normalize_step(step) for step in gold_steps]

    exact_match = float(pred_norm == gold_norm)
    max_len = max(len(pred_norm), len(gold_norm), 1)
    pos_matches = sum(1 for pred, gold in zip(pred_norm, gold_norm) if pred == gold)
    step_accuracy = pos_matches / max_len

    pred_set = set(pred_norm)
    gold_set = set(gold_norm)
    overlap = len(pred_set & gold_set)
    union = pred_set | gold_set
    step_iou = overlap / len(union) if union else 1.0
    step_precision = overlap / len(pred_set) if pred_set else (1.0 if not gold_set else 0.0)
    step_recall = overlap / len(gold_set) if gold_set else (1.0 if not pred_set else 0.0)
    if step_precision + step_recall > 0:
        step_f1 = 2.0 * step_precision * step_recall / (step_precision + step_recall)
    else:
        step_f1 = 0.0

    lcs = lcs_length(pred_norm, gold_norm)
    ordered_ratio = lcs / max(len(gold_norm), 1)
    task_success = float(lcs == len(gold_norm)) if gold_norm else float(not pred_norm)

    edit_distance = levenshtein_distance(pred_norm, gold_norm)
    edit_similarity = 1.0 - (edit_distance / max_len)

    return {
        "exact_match": exact_match,
        "task_success": task_success,
        "ordered_ratio": ordered_ratio,
        "step_precision": step_precision,
        "step_recall": step_recall,
        "step_f1": step_f1,
        "step_iou": step_iou,
        "step_accuracy": step_accuracy,
        "edit_similarity": edit_similarity,
        "pred_len": len(pred_steps),
        "gold_len": len(gold_steps),
        "length_delta": len(pred_steps) - len(gold_steps),
    }


def metric_quality_key(metrics: dict) -> tuple:
    return (
        metrics["exact_match"],
        metrics["task_success"],
        metrics["ordered_ratio"],
        metrics["step_f1"],
        metrics["step_accuracy"],
        metrics["edit_similarity"],
        -abs(metrics["length_delta"]),
    )


def bootstrap_mean_ci(values: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) == 1 or n_boot <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(sample.mean()))
    low, high = np.quantile(np.asarray(boots), [0.025, 0.975])
    return mean, float(low), float(high)


def stable_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def normalize_score_list(scores: list[float], mode: str) -> list[float]:
    finite = [score for score in scores if math.isfinite(score)]
    if not finite:
        return [math.inf] * len(scores)

    if mode == "none":
        return list(scores)

    if mode == "zscore":
        mean = stable_mean(finite)
        std = float(np.std(np.asarray(finite, dtype=np.float64)))
        if std < 1e-8:
            return [0.0 if math.isfinite(score) else math.inf for score in scores]
        return [((score - mean) / std) if math.isfinite(score) else math.inf for score in scores]

    lo = min(finite)
    hi = max(finite)
    if hi - lo < 1e-8:
        return [0.0 if math.isfinite(score) else math.inf for score in scores]
    return [((score - lo) / (hi - lo)) if math.isfinite(score) else math.inf for score in scores]


def parse_prefix_lengths(raw: str) -> list[int]:
    if not raw:
        return []
    vals = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals.append(int(chunk))
    return sorted(set(v for v in vals if v > 0))


def load_jsonl(path: str, max_samples: Optional[int] = None) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def build_eval_cases(args) -> list[EvalCase]:
    goal_only_rows = load_jsonl(args.goal_only_data, max_samples=args.max_samples)
    goal_interp_rows = load_jsonl(args.goal_interp_data, max_samples=args.max_samples)

    cases: list[EvalCase] = []
    case_id = 0

    for idx, row in enumerate(goal_only_rows):
        meta = row.get("meta", {})
        cases.append(
            EvalCase(
                case_id=case_id,
                base_sample_id=idx,
                condition_family="goal_only",
                condition_label="goal_only",
                prompt_variant="goal_only",
                prefix_len=0,
                goal=row.get("goal", ""),
                interpretation="",
                prefix_steps=[],
                gold_steps=list(row.get("gold_steps", [])),
                meta=meta,
                source_dataset=str(args.goal_only_data),
            )
        )
        case_id += 1

    for idx, row in enumerate(goal_interp_rows):
        meta = row.get("meta", {})
        goal = row.get("goal", "")
        interpretation = row.get("interpretation", "")
        gold_steps = list(row.get("gold_steps", []))
        cases.append(
            EvalCase(
                case_id=case_id,
                base_sample_id=idx,
                condition_family="goal_plus_interpretation",
                condition_label="goal_plus_interpretation",
                prompt_variant="goal_plus_interpretation",
                prefix_len=0,
                goal=goal,
                interpretation=interpretation,
                prefix_steps=[],
                gold_steps=gold_steps,
                meta=meta,
                source_dataset=str(args.goal_interp_data),
            )
        )
        case_id += 1

        for prefix_len in args.prefix_lengths:
            if prefix_len >= len(gold_steps):
                continue
            cases.append(
                EvalCase(
                    case_id=case_id,
                    base_sample_id=idx,
                    condition_family="goal_plus_interpretation_prefix",
                    condition_label=f"goal_plus_interpretation_prefix@{prefix_len}",
                    prompt_variant="goal_plus_interpretation_prefix",
                    prefix_len=prefix_len,
                    goal=goal,
                    interpretation=interpretation,
                    prefix_steps=gold_steps[:prefix_len],
                    gold_steps=gold_steps[prefix_len:],
                    meta=meta,
                    source_dataset=str(args.goal_interp_data),
                )
            )
            case_id += 1

    return cases


def candidate_temperatures(args) -> list[tuple[str, float]]:
    temps: list[tuple[str, float]] = [("greedy", 0.0)]
    if args.K <= 1:
        return temps
    sample_temp = args.sampling_temperature if args.sampling_temperature > 0 else 0.8
    for idx in range(args.K - 1):
        temps.append((f"sample_{idx + 1}", sample_temp))
    return temps


def generate_candidates_for_case(
    case: EvalCase,
    *,
    system1_model,
    system1_tokenizer,
    gen_mode: str,
    args,
    device,
    critic_model,
    critic_tokenizer,
    goal_model,
    goal_tokenizer,
) -> list[dict]:
    prompt = case.prompt_text()
    candidates = []
    temps = candidate_temperatures(args)

    for cand_idx, (strategy, temperature) in enumerate(temps):
        raw = efp.generate_plan(
            system1_model,
            system1_tokenizer,
            prompt,
            device,
            gen_mode=gen_mode,
            temperature=temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        steps = efp.parse_plan_json(raw)
        full_steps = case.prefix_steps + steps
        metrics = compute_sequence_metrics(steps, case.gold_steps)
        critic_score = math.inf
        goal_score = math.inf

        if steps and critic_model is not None:
            enc = critic_tokenizer(
                efp.format_trajectory_text(case.goal, full_steps),
                max_length=args.critic_max_len,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                critic_score = float(critic_model(enc.input_ids, enc.attention_mask).item())

        if steps:
            if goal_model is not None:
                goal_score = float(
                    efp.compute_learned_energy(
                        goal_model,
                        goal_tokenizer,
                        case.goal,
                        full_steps,
                        device,
                    )
                )
            elif args.latent_dir:
                goal_score = float(
                    efp.compute_lookup_energy(
                        args.latent_dir,
                        case.meta,
                        full_steps,
                    )
                )

        candidates.append(
            {
                "candidate_idx": cand_idx,
                "strategy": strategy,
                "temperature": temperature,
                "prompt": prompt,
                "raw": raw,
                "steps": steps,
                "full_steps": full_steps,
                "valid": bool(steps),
                "metrics": metrics,
                "critic_score": critic_score,
                "goal_score": goal_score,
            }
        )

    critic_norm = normalize_score_list(
        [cand["critic_score"] for cand in candidates],
        args.score_normalization,
    )
    goal_norm = normalize_score_list(
        [cand["goal_score"] for cand in candidates],
        args.score_normalization,
    )

    for cand, c_norm, g_norm in zip(candidates, critic_norm, goal_norm):
        cand["critic_score_norm"] = c_norm
        cand["goal_score_norm"] = g_norm
        if math.isfinite(c_norm) and math.isfinite(g_norm):
            cand["combined_score"] = args.alpha * c_norm + args.beta * g_norm
        elif math.isfinite(c_norm):
            cand["combined_score"] = c_norm
        elif math.isfinite(g_norm):
            cand["combined_score"] = g_norm
        else:
            cand["combined_score"] = math.inf

    return candidates


def best_index_by_metric(candidates: list[dict], key_name: str) -> int:
    valid = [cand for cand in candidates if cand["valid"]]
    if not valid:
        return 0
    return min(valid, key=lambda cand: cand[key_name])["candidate_idx"]


def oracle_index(candidates: list[dict]) -> int:
    valid = [cand for cand in candidates if cand["valid"]]
    if not valid:
        return 0
    return max(valid, key=lambda cand: metric_quality_key(cand["metrics"]))["candidate_idx"]


def available_configs(args, critic_model, goal_model) -> list[str]:
    configs = ["system1_greedy"]
    if critic_model is not None:
        configs.append("system1_critic")
    if goal_model is not None or args.latent_dir:
        configs.append("system1_goal")
    if critic_model is not None and (goal_model is not None or args.latent_dir):
        configs.append("system1_critic_goal")
    return configs


def select_candidate_index(config_name: str, candidates: list[dict]) -> int:
    if config_name == "system1_greedy":
        return 0
    if config_name == "system1_critic":
        return best_index_by_metric(candidates, "critic_score")
    if config_name == "system1_goal":
        return best_index_by_metric(candidates, "goal_score")
    if config_name == "system1_critic_goal":
        return best_index_by_metric(candidates, "combined_score")
    raise ValueError(f"Unknown config: {config_name}")


def case_json_record(case: EvalCase, candidates: list[dict], selected_indices: dict[str, int]) -> dict:
    return {
        "case": asdict(case),
        "selected_indices": selected_indices,
        "candidates": candidates,
    }


def flatten_case_rows(
    case: EvalCase,
    candidates: list[dict],
    selected_indices: dict[str, int],
    *,
    run_id: str,
    model_tag: str,
    system1_model_path: str,
) -> list[dict]:
    greedy = candidates[0]
    oracle = candidates[oracle_index(candidates)]
    n_valid = sum(1 for cand in candidates if cand["valid"])
    unique_valid = len({tuple(cand["steps"]) for cand in candidates if cand["valid"]})

    rows = []
    for config_name, cand_idx in selected_indices.items():
        chosen = candidates[cand_idx]
        row = {
            "run_id": run_id,
            "model_tag": model_tag,
            "system1_model": system1_model_path,
            "config": config_name,
            "case_id": case.case_id,
            "base_sample_id": case.base_sample_id,
            "condition_family": case.condition_family,
            "condition_label": case.condition_label,
            "prompt_variant": case.prompt_variant,
            "prefix_len": case.prefix_len,
            "source_dataset": case.source_dataset,
            "task_id": str(case.meta.get("task_id", "")),
            "task_name": str(case.meta.get("task_name", "")),
            "video_id": str(case.meta.get("video_id", "")),
            "goal": case.goal,
            "n_candidates": len(candidates),
            "n_valid_candidates": n_valid,
            "n_unique_valid_candidates": unique_valid,
            "selected_idx": cand_idx,
            "selected_strategy": chosen["strategy"],
            "greedy_idx": 0,
            "oracle_idx": oracle["candidate_idx"],
            "selected_is_oracle": float(cand_idx == oracle["candidate_idx"]),
            "selected_is_greedy": float(cand_idx == 0),
            "critic_score": chosen["critic_score"],
            "goal_score": chosen["goal_score"],
            "combined_score": chosen["combined_score"],
            "greedy_critic_score": greedy["critic_score"],
            "greedy_goal_score": greedy["goal_score"],
            "greedy_combined_score": greedy["combined_score"],
            "oracle_critic_score": oracle["critic_score"],
            "oracle_goal_score": oracle["goal_score"],
            "oracle_combined_score": oracle["combined_score"],
            "gold_len": len(case.gold_steps),
        }

        for metric_name in MAIN_METRICS + ["pred_len", "gold_len", "length_delta"]:
            row[metric_name] = chosen["metrics"][metric_name]
            row[f"greedy_{metric_name}"] = greedy["metrics"][metric_name]
            row[f"oracle_{metric_name}"] = oracle["metrics"][metric_name]

        for metric_name in ("exact_match", "task_success", "ordered_ratio", "step_f1", "step_accuracy"):
            row[f"gain_vs_greedy_{metric_name}"] = (
                chosen["metrics"][metric_name] - greedy["metrics"][metric_name]
            )
            row[f"oracle_gap_{metric_name}"] = (
                oracle["metrics"][metric_name] - chosen["metrics"][metric_name]
            )

        rows.append(row)

    return rows


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["empty"])
        return

    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows: list[dict], group_keys: list[str], *, bootstrap_iters: int, seed: int) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    out_rows = []
    for group_key, group in sorted(grouped.items()):
        record = {key: value for key, value in zip(group_keys, group_key)}
        record["n_samples"] = len(group)
        record["n_tasks"] = len({row["task_id"] for row in group})
        record["n_videos"] = len({row["video_id"] for row in group})

        for metric_name in MAIN_METRICS + [
            "pred_len",
            "gold_len",
            "length_delta",
            "n_valid_candidates",
            "n_unique_valid_candidates",
            "selected_is_oracle",
            "selected_is_greedy",
        ] + GAIN_METRICS:
            mean, low, high = bootstrap_mean_ci(
                [float(row[metric_name]) for row in group],
                bootstrap_iters,
                seed + abs(hash((metric_name,) + group_key)) % 100000,
            )
            record[metric_name] = round(mean, 4)
            record[f"{metric_name}_ci_low"] = round(low, 4)
            record[f"{metric_name}_ci_high"] = round(high, 4)

        out_rows.append(record)
    return out_rows


def build_markdown_table(rows: list[dict], metrics: list[str], group_keys: list[str]) -> str:
    if not rows:
        return "| empty |\n|---|\n| no rows |\n"

    headers = list(group_keys) + metrics
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals = []
        for key in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def save_run_artifacts(
    out_dir: Path,
    *,
    args,
    cases: list[EvalCase],
    case_records: list[dict],
    flat_rows: list[dict],
    summary_label_rows: list[dict],
    summary_family_rows: list[dict],
    per_task_rows: list[dict],
) -> None:
    run_config = {
        "run_id": args.run_id,
        "model_tag": args.model_tag,
        "system1_model": args.system1_model,
        "system1_type": args.system1_type,
        "plm_base_model": args.plm_base_model,
        "critic_model": args.critic_model,
        "goal_latent_model": args.goal_latent_model,
        "latent_dir": args.latent_dir,
        "K": args.K,
        "sampling_temperature": args.sampling_temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "alpha": args.alpha,
        "beta": args.beta,
        "score_normalization": args.score_normalization,
        "prefix_lengths": args.prefix_lengths,
        "task_success_definition": "gold step sequence appears as an ordered subsequence of the prediction",
        "note": "Critic/goal ablations reuse the same candidate pool as greedy for each sample.",
        "n_cases": len(cases),
    }

    write_json(out_dir / "run_config.json", run_config)
    write_jsonl(out_dir / "per_case.jsonl", case_records)
    write_csv(out_dir / "per_sample.csv", flat_rows)
    write_jsonl(out_dir / "per_sample.jsonl", flat_rows)
    write_csv(out_dir / "summary_overall.csv", summary_label_rows)
    write_json(out_dir / "summary_overall.json", summary_label_rows)
    write_csv(out_dir / "summary_family.csv", summary_family_rows)
    write_json(out_dir / "summary_family.json", summary_family_rows)
    write_csv(out_dir / "per_task_summary.csv", per_task_rows)
    write_json(out_dir / "per_task_summary.json", per_task_rows)
    (out_dir / "summary_overall.md").write_text(
        build_markdown_table(
            summary_label_rows,
            ["exact_match", "task_success", "ordered_ratio", "step_f1", "step_accuracy"],
            ["config", "condition_label", "n_samples"],
        )
    )


def plot_metric_bars(summary_rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: matplotlib unavailable ({exc})")
        return

    metric_names = ["exact_match", "task_success", "ordered_ratio", "step_f1", "step_accuracy"]
    plot_rows = [row for row in summary_rows if row["condition_label"] != "goal_plus_interpretation_prefix"]
    if not plot_rows:
        plot_rows = summary_rows

    conditions = sorted({row["condition_label"] for row in plot_rows})
    configs = sorted({row["config"] for row in plot_rows})
    x = np.arange(len(conditions))
    width = 0.8 / max(len(configs), 1)

    fig, axes = plt.subplots(len(metric_names), 1, figsize=(11, 3.4 * len(metric_names)), sharex=True)
    if len(metric_names) == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metric_names):
        for cfg_idx, config in enumerate(configs):
            vals = []
            errs_low = []
            errs_high = []
            for condition in conditions:
                row = next(
                    (
                        item
                        for item in plot_rows
                        if item["config"] == config and item["condition_label"] == condition
                    ),
                    None,
                )
                if row is None:
                    vals.append(np.nan)
                    errs_low.append(0.0)
                    errs_high.append(0.0)
                else:
                    vals.append(row[metric_name])
                    errs_low.append(row[metric_name] - row[f"{metric_name}_ci_low"])
                    errs_high.append(row[f"{metric_name}_ci_high"] - row[metric_name])
            pos = x - 0.4 + width / 2.0 + cfg_idx * width
            ax.bar(pos, vals, width=width, label=config)
            ax.errorbar(pos, vals, yerr=[errs_low, errs_high], fmt="none", ecolor="black", capsize=3, linewidth=1)

        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(conditions, rotation=20, ha="right")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("coin_2 Metrics by Prompt Condition")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "metrics_by_condition.png", dpi=180)
    plt.close(fig)


def plot_prefix_curves(summary_rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_rows = [
        row
        for row in summary_rows
        if row["condition_family"] == "goal_plus_interpretation_prefix"
    ]
    if not plot_rows:
        return

    configs = sorted({row["config"] for row in plot_rows})
    prefix_vals = sorted({int(row["prefix_len"]) for row in plot_rows})
    metric_names = ["exact_match", "task_success", "ordered_ratio", "step_accuracy"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    axes = axes.flatten()
    for ax, metric_name in zip(axes, metric_names):
        for config in configs:
            xs = []
            ys = []
            for prefix_len in prefix_vals:
                row = next(
                    (
                        item
                        for item in plot_rows
                        if item["config"] == config and int(item["prefix_len"]) == prefix_len
                    ),
                    None,
                )
                if row is None:
                    continue
                xs.append(prefix_len)
                ys.append(row[metric_name])
            if xs:
                ax.plot(xs, ys, marker="o", label=config)
        ax.set_title(metric_name)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlabel("Observed Prefix Steps")
    axes[0].legend(fontsize=8)
    fig.suptitle("Prefix-Diagnostic Performance vs Observed Steps")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "prefix_curves.png", dpi=180)
    plt.close(fig)


def plot_accuracy_by_gold_len(flat_rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    filtered = [row for row in flat_rows if row["condition_family"] in {"goal_only", "goal_plus_interpretation"}]
    if not filtered:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    families = ["goal_only", "goal_plus_interpretation"]
    configs = sorted({row["config"] for row in filtered})
    for ax, family in zip(axes, families):
        fam_rows = [row for row in filtered if row["condition_family"] == family]
        gold_lens = sorted({int(row["gold_len"]) for row in fam_rows})
        for config in configs:
            xs = []
            ys = []
            for gold_len in gold_lens:
                group = [
                    row["step_accuracy"]
                    for row in fam_rows
                    if row["config"] == config and int(row["gold_len"]) == gold_len
                ]
                if not group:
                    continue
                xs.append(gold_len)
                ys.append(stable_mean(group))
            if xs:
                ax.plot(xs, ys, marker="o", label=config)
        ax.set_title(family)
        ax.set_xlabel("Gold Plan Length")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylabel("Step Accuracy")
    axes[0].legend(fontsize=8)
    fig.suptitle("Accuracy vs Gold Plan Length")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "accuracy_by_gold_len.png", dpi=180)
    plt.close(fig)


def plot_reranking_gains(summary_family_rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_rows = [
        row
        for row in summary_family_rows
        if row["config"] != "system1_greedy"
    ]
    if not plot_rows:
        return

    families = sorted({row["condition_family"] for row in plot_rows})
    configs = sorted({row["config"] for row in plot_rows})
    metrics = ["gain_vs_greedy_ordered_ratio", "gain_vs_greedy_step_f1"]
    x = np.arange(len(families))
    width = 0.8 / max(len(configs), 1)

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10.5, 4.2 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metrics):
        for cfg_idx, config in enumerate(configs):
            vals = []
            for family in families:
                row = next(
                    (
                        item
                        for item in plot_rows
                        if item["config"] == config and item["condition_family"] == family
                    ),
                    None,
                )
                vals.append(0.0 if row is None else row[metric_name])
            pos = x - 0.4 + width / 2.0 + cfg_idx * width
            ax.bar(pos, vals, width=width, label=config)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(families, rotation=20, ha="right")
    axes[0].legend(fontsize=8)
    fig.suptitle("Reranking Gain vs Greedy")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "reranking_gains.png", dpi=180)
    plt.close(fig)


def plot_task_heatmaps(per_task_rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    target_families = ["goal_only", "goal_plus_interpretation"]
    for family in target_families:
        family_rows = [row for row in per_task_rows if row["condition_family"] == family]
        if not family_rows:
            continue

        task_names = sorted({row["task_name"] for row in family_rows})
        configs = sorted({row["config"] for row in family_rows})
        matrix = np.full((len(task_names), len(configs)), np.nan)
        for i, task_name in enumerate(task_names):
            for j, config in enumerate(configs):
                row = next(
                    (
                        item
                        for item in family_rows
                        if item["task_name"] == task_name and item["config"] == config
                    ),
                    None,
                )
                if row is not None:
                    matrix[i, j] = row["ordered_ratio"]

        fig_h = max(4.5, 0.28 * len(task_names))
        fig, ax = plt.subplots(figsize=(10, fig_h))
        im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(len(configs)))
        ax.set_xticklabels(configs, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(task_names)))
        ax.set_yticklabels(task_names)
        ax.set_title(f"Per-Task Ordered Ratio Heatmap: {family}")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"task_heatmap_{family}.png", dpi=180)
        plt.close(fig)


def make_plots(out_dir: Path, flat_rows: list[dict], summary_label_rows: list[dict], summary_family_rows: list[dict], per_task_rows: list[dict]) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_bars(summary_label_rows, out_dir)
    plot_prefix_curves(summary_label_rows, out_dir)
    plot_accuracy_by_gold_len(flat_rows, out_dir)
    plot_reranking_gains(summary_family_rows, out_dir)
    plot_task_heatmaps(per_task_rows, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one System-1 model on coin_2 with clean reranking ablations")
    parser.add_argument("--system1_model", required=True)
    parser.add_argument("--model_tag", default=None, help="Short label used in saved summaries")
    parser.add_argument("--system1_type", choices=["t5", "plm"], default="plm")
    parser.add_argument("--plm_base_model", default="facebook/Perception-LM-1B")
    parser.add_argument("--critic_model", default=None)
    parser.add_argument("--goal_latent_model", default=None)
    parser.add_argument("--latent_dir", default=None,
                        help="Optional lookup-latent fallback. Kept for compatibility, but learned goal-latent scoring is preferred.")
    parser.add_argument("--goal_only_data", default="data/coin_2/coin2_system1_test_goal_only.jsonl")
    parser.add_argument("--goal_interp_data", default="data/coin_2/coin2_system1_test_goal_plus_interpretation.jsonl")
    parser.add_argument("--prefix_lengths", default="1,2,3",
                        help="Comma-separated prefix lengths for the held-out prefix diagnostic derived from goal+interpretation test rows")
    parser.add_argument("--K", type=int, default=5,
                        help="Total candidate pool size per sample. Candidate 0 is always greedy; the rest are sampled.")
    parser.add_argument("--sampling_temperature", type=float, default=0.8,
                        help="Sampling temperature for non-greedy candidates")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--critic_max_len", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight for normalized critic score in critic+goal reranking")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight for normalized goal score in critic+goal reranking")
    parser.add_argument("--score_normalization", choices=["minmax", "zscore", "none"], default="minmax")
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--output_dir", default="outputs/coin_2")
    args = parser.parse_args()

    if args.K < 1:
        raise ValueError("--K must be >= 1")

    args.system1_model = efp.resolve_existing_local_path(args.system1_model)
    if args.critic_model:
        args.critic_model = efp.resolve_existing_local_path(args.critic_model)
    if args.goal_latent_model:
        args.goal_latent_model = efp.resolve_existing_local_path(args.goal_latent_model)
    if args.latent_dir:
        args.latent_dir = efp.resolve_existing_local_path(args.latent_dir)
    args.goal_only_data = efp.resolve_existing_local_path(args.goal_only_data)
    args.goal_interp_data = efp.resolve_existing_local_path(args.goal_interp_data)
    args.prefix_lengths = parse_prefix_lengths(args.prefix_lengths)
    args.model_tag = args.model_tag or Path(args.system1_model).name
    args.run_id = args.run_id or f"{args.model_tag}_coin2_eval"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir).resolve() / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading System-1 model: {args.system1_model}")
    system1_model, system1_tokenizer, gen_mode = efp.load_system1(
        args.system1_model,
        device,
        args.system1_type,
        args.plm_base_model,
    )
    critic_model, critic_tokenizer = efp.load_critic(args.critic_model, device)
    goal_model, goal_tokenizer = efp.load_goal_latent_model(args.goal_latent_model, device)
    configs = available_configs(args, critic_model, goal_model)
    print(f"Planning configs: {', '.join(configs)}")

    cases = build_eval_cases(args)
    print(f"Loaded {len(cases)} evaluation cases")

    case_records = []
    flat_rows = []

    # Check for existing checkpoint
    checkpoint_per_case = out_dir / "per_case.jsonl"
    checkpoint_per_sample = out_dir / "per_sample.jsonl"
    
    evaluated_case_ids = set()
    if checkpoint_per_case.exists() and checkpoint_per_sample.exists():
        print(f"Loading existing checkpoints from {out_dir}...")
        case_records = load_jsonl(str(checkpoint_per_case))
        flat_rows = load_jsonl(str(checkpoint_per_sample))
        evaluated_case_ids = {r["case_id"] for r in case_records}
        print(f"Resuming evaluation, found {len(evaluated_case_ids)} completed cases.")

    for idx, case in enumerate(cases, start=1):
        if case.case_id in evaluated_case_ids:
            continue

        candidates = generate_candidates_for_case(
            case,
            system1_model=system1_model,
            system1_tokenizer=system1_tokenizer,
            gen_mode=gen_mode,
            args=args,
            device=device,
            critic_model=critic_model,
            critic_tokenizer=critic_tokenizer,
            goal_model=goal_model,
            goal_tokenizer=goal_tokenizer,
        )
        selected_indices = {
            config_name: select_candidate_index(config_name, candidates)
            for config_name in configs
        }
        case_records.append(case_json_record(case, candidates, selected_indices))
        flat_rows.extend(
            flatten_case_rows(
                case,
                candidates,
                selected_indices,
                run_id=args.run_id,
                model_tag=args.model_tag,
                system1_model_path=args.system1_model,
            )
        )

        if idx == 1 or idx % 10 == 0 or idx == len(cases):
            print(f"Processed {idx}/{len(cases)} cases", flush=True)
            # Save intermediate progress
            write_jsonl(checkpoint_per_case, case_records)
            write_jsonl(checkpoint_per_sample, flat_rows)

    summary_label_rows = aggregate_rows(
        flat_rows,
        ["config", "condition_family", "condition_label", "prefix_len"],
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )
    summary_family_rows = aggregate_rows(
        flat_rows,
        ["config", "condition_family"],
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )
    per_task_rows = aggregate_rows(
        flat_rows,
        ["config", "condition_family", "condition_label", "task_id", "task_name"],
        bootstrap_iters=max(args.bootstrap_iters // 2, 200),
        seed=args.seed,
    )

    save_run_artifacts(
        out_dir,
        args=args,
        cases=cases,
        case_records=case_records,
        flat_rows=flat_rows,
        summary_label_rows=summary_label_rows,
        summary_family_rows=summary_family_rows,
        per_task_rows=per_task_rows,
    )
    make_plots(out_dir, flat_rows, summary_label_rows, summary_family_rows, per_task_rows)

    print(f"Saved evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
