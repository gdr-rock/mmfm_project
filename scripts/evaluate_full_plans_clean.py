#!/usr/bin/env python3
"""Standalone full-plan evaluation for COIN-style planning.

This script intentionally does not depend on the existing evaluation pipeline.
It evaluates a simpler task:
  - Input: goal + interpretation, or goal + initial frames
  - Model output: a full plan as action/state-change JSON
  - Reranking: critic cost + optional goal-energy score
  - Metrics: full-plan exact/ordering/overlap metrics only
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_INTERP_PROMPT = (
    "Given these initial frames of the video, describe in one sentence "
    "what the person is about to do and what the finished result should be."
)


def resolve_existing_local_path(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return path_str

    raw = Path(os.path.expandvars(os.path.expanduser(path_str)))
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, repo_root / raw]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
        if candidate.name == "best_model":
            alt = candidate.with_name("best_adapter")
            if alt.exists():
                print(f"Using adapter directory instead of missing checkpoint: {alt}")
                return str(alt.resolve())

    return path_str


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def build_full_plan_prompt(goal: str, interpretation: str = "") -> str:
    lines = [f"Goal: {goal}"]
    if interpretation:
        lines.append(f"Interpretation: {interpretation}")
    lines.append("")
    lines.append(
        'Generate the full plan to complete the task. Output JSON only: '
        '{"next_steps": [{"action": "...", "state_change": "..."}, ...]}'
    )
    return "\n".join(lines)


def parse_plan_json(text: str) -> list[str]:
    def _decode_json_string(s: str) -> str:
        try:
            return json.loads(f"\"{s}\"")
        except Exception:
            return s

    def _steps_from_list(step_list):
        if not isinstance(step_list, list):
            return []
        out = []
        for item in step_list:
            if not isinstance(item, dict):
                continue
            action = item.get("action")
            state_change = item.get("state_change")
            if isinstance(action, str) and isinstance(state_change, str):
                out.append(f"{action.strip()} | {state_change.strip()}")
        return out

    text = (text or "").strip()
    if not text:
        return []

    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()

    candidates = [text]
    if '"next_steps"' in text and not text.lstrip().startswith("{"):
        candidates.append("{" + text + "}")
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace:last_brace + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            steps = _steps_from_list(parsed.get("next_steps", []))
            if steps:
                return steps
        elif isinstance(parsed, list):
            steps = _steps_from_list(parsed)
            if steps:
                return steps

    action_vals = [
        _decode_json_string(value)
        for value in re.findall(r'"action"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    ]
    state_vals = [
        _decode_json_string(value)
        for value in re.findall(r'"state_change"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    ]
    n = min(len(action_vals), len(state_vals))
    if n > 0:
        return [
            f"{action_vals[i].strip()} | {state_vals[i].strip()}"
            for i in range(n)
        ]
    return []


def normalize_step(step: str) -> str:
    text = (step or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9| ]", "", text)
    return text


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            old = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = old
    return dp[-1]


def compute_full_plan_metrics(pred_steps: list[str], gold_steps: list[str]) -> dict:
    pred_norm = [normalize_step(step) for step in pred_steps]
    gold_norm = [normalize_step(step) for step in gold_steps]

    exact_match = float(pred_norm == gold_norm)
    max_len = max(len(pred_norm), len(gold_norm), 1)
    pos_matches = sum(1 for p, g in zip(pred_norm, gold_norm) if p == g)
    step_accuracy = pos_matches / max_len

    pred_set = set(pred_norm)
    gold_set = set(gold_norm)
    union = pred_set | gold_set
    step_iou = len(pred_set & gold_set) / len(union) if union else 1.0

    lcs = lcs_length(pred_norm, gold_norm)
    ordered_ratio = lcs / max(len(gold_norm), 1)

    return {
        "exact_match": exact_match,
        "step_accuracy": step_accuracy,
        "step_iou": step_iou,
        "ordered_ratio": ordered_ratio,
        "pred_len": len(pred_steps),
        "gold_len": len(gold_steps),
        "length_delta": len(pred_steps) - len(gold_steps),
    }


class CriticModel(nn.Module):
    def __init__(self, encoder_name: str, freeze_encoder: bool = False):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_dim = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _mean_pool(self, output, attention_mask):
        tokens = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(tokens.size()).float()
        return torch.sum(tokens * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(output, attention_mask)
        return self.head(pooled).squeeze(-1)


def format_trajectory_text(goal: str, steps: list[str]) -> str:
    parts = [f"[GOAL] {goal}"]
    parts.extend(steps)
    return " [SEP] ".join(parts)


class GoalLatentModel(nn.Module):
    def __init__(self, encoder_name: str, latent_dim: int, hidden_dim: int = 384):
        super().__init__()
        from transformers import AutoModel

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
        return F.normalize(self.goal_head(self._encode(input_ids, attention_mask)), dim=-1)

    def encode_traj(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.traj_head(self._encode(input_ids, attention_mask)), dim=-1)


def format_trajectory_for_energy(goal: str, steps: list[str]) -> str:
    lines = [f"Goal: {goal}", "Trajectory:"]
    if steps:
        for idx, step in enumerate(steps, start=1):
            lines.append(f"  {idx}) {step}")
    else:
        lines.append("  (No steps observed yet.)")
    return "\n".join(lines)


def has_tokenizer_files(path_str: str) -> bool:
    if not path_str or not os.path.isdir(path_str):
        return False
    files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "sentencepiece.bpe.model",
        "vocab.json",
        "merges.txt",
    )
    return any(os.path.exists(os.path.join(path_str, name)) for name in files)


def load_system1(model_path: str, device, model_type: str, base_model_name: Optional[str]):
    from transformers import AutoTokenizer

    model_path = resolve_existing_local_path(model_path)

    if model_type == "t5":
        from transformers import T5ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        return model.to(device).eval(), tokenizer, "seq2seq"

    if model_type != "plm":
        raise ValueError(f"Unsupported system1_type: {model_type}")

    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    base_model_name = base_model_name or "facebook/Perception-LM-1B"
    is_adapter = os.path.isdir(model_path) and os.path.exists(
        os.path.join(model_path, "adapter_config.json")
    )

    tokenizer_source = model_path if has_tokenizer_files(model_path) else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_adapter:
        try:
            from transformers import AutoModelForImageTextToText

            base = AutoModelForImageTextToText.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        except Exception:
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

    return model.to(device).eval(), tokenizer, "causal"


def load_critic(model_path: Optional[str], device):
    if not model_path:
        return None, None

    from transformers import AutoTokenizer

    model_path = resolve_existing_local_path(model_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    encoder_name = ckpt.get("encoder_name", "sentence-transformers/all-MiniLM-L6-v2")
    model = CriticModel(encoder_name)
    model.load_state_dict(ckpt["model_state_dict"])
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    return model.to(device).eval(), tokenizer


def load_goal_latent_model(model_path: Optional[str], device):
    if not model_path:
        return None, None

    from transformers import AutoTokenizer

    model_path = resolve_existing_local_path(model_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = GoalLatentModel(
        encoder_name=ckpt["encoder_name"],
        latent_dim=ckpt["latent_dim"],
        hidden_dim=ckpt.get("hidden_dim", 384),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    tokenizer = AutoTokenizer.from_pretrained(ckpt["encoder_name"])
    return model.to(device).eval(), tokenizer


def generate_plan(
    model,
    tokenizer,
    prompt: str,
    device,
    gen_mode: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    if gen_mode == "causal" and not prompt.rstrip().endswith("Assistant:"):
        prompt = prompt + "\n\nAssistant:"

    enc = tokenizer(prompt, max_length=768, truncation=True, return_tensors="pt").to(device)
    prompt_len = enc.input_ids.shape[1]
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    do_sample = temperature > 0

    with torch.no_grad():
        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            num_beams=1,
            pad_token_id=pad_token_id,
        )

    if gen_mode == "causal":
        completion_ids = gen_ids[0][prompt_len:]
        return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


def generate_interpretation_from_frames(
    frames_path: str,
    device,
    model_name: str,
    num_frames: int,
    prompt: str,
    max_new_tokens: int = 96,
) -> str:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    frames_p = Path(frames_path)
    if not frames_p.exists():
        raise FileNotFoundError(f"Frames path not found: {frames_path}")

    content = []
    if frames_p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        img_files = sorted([p for p in frames_p.iterdir() if p.suffix.lower() in exts])[:num_frames]
        if not img_files:
            raise FileNotFoundError(f"No frame images found in {frames_path}")
        for img_f in img_files:
            content.append({"type": "image", "url": str(img_f)})
    else:
        raise ValueError("Frames mode expects a directory of initial frame images")
    content.append({"type": "text", "text": prompt})

    conversation = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        [conversation],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    input_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)[0].strip()


def resolve_frames_path(frames_root: str, meta: dict) -> str:
    root = Path(resolve_existing_local_path(frames_root))
    video_id = str(meta.get("video_id", ""))
    task_id = str(meta.get("task_id", ""))
    candidates = [
        root / video_id,
        root / task_id / video_id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve initial frames for video {video_id}. Checked: {checked}")


def compute_lookup_energy(latent_dir: str, meta: dict, plan_steps: list[str]) -> float:
    task_id = str(meta.get("task_id", ""))
    video_id = str(meta.get("video_id", ""))
    vid_dir = Path(resolve_existing_local_path(latent_dir)) / task_id / video_id
    if not vid_dir.is_dir():
        return 0.0

    latent_files = sorted(
        [path for path in vid_dir.iterdir() if path.name.startswith("segment_") and path.suffix == ".pt"],
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if not latent_files:
        return 0.0

    def _pool(path: Path) -> torch.Tensor:
        z = torch.load(path, map_location="cpu", weights_only=True)
        return z.mean(dim=0).float() if z.dim() == 2 else z.float()

    z_goal = _pool(latent_files[-1])
    target_idx = min(len(latent_files) - 1, max(len(plan_steps) - 1, 0))
    z_pred = _pool(latent_files[target_idx])
    return torch.norm(z_pred - z_goal, p=2).item() ** 2


def compute_learned_energy(
    goal_model,
    goal_tokenizer,
    goal: str,
    plan_steps: list[str],
    device,
) -> float:
    goal_enc = goal_tokenizer(
        [goal],
        max_length=64,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    traj_text = format_trajectory_for_energy(goal, plan_steps)
    traj_enc = goal_tokenizer(
        [traj_text],
        max_length=384,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        z_goal = goal_model.encode_goal(goal_enc.input_ids, goal_enc.attention_mask)[0]
        z_traj = goal_model.encode_traj(traj_enc.input_ids, traj_enc.attention_mask)[0]
    return torch.norm(z_traj - z_goal, p=2).item() ** 2


@dataclass
class SampleOutcome:
    sample_id: int
    goal: str
    interpretation: str
    best_idx: int
    valid_candidates: int
    best_metrics: dict
    combined_score: float
    critic_score: float
    goal_score: float
    gold_steps: list[str]
    best_steps: list[str]
    plans: list[dict]


def plot_summary(out_dir: Path, summary: dict, outcomes: list[SampleOutcome]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: matplotlib unavailable ({exc})")
        return

    metric_names = ["exact_match", "ordered_ratio", "step_iou", "step_accuracy", "valid_json_rate"]
    metric_values = [summary.get(name, 0.0) for name in metric_names]

    plt.figure(figsize=(8, 4.5))
    plt.bar(metric_names, metric_values)
    plt.ylim(0.0, 1.0)
    plt.title("Mean Full-Plan Metrics")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "summary_metrics.png", dpi=150)
    plt.close()

    gold_lens = [len(outcome.gold_steps) for outcome in outcomes]
    pred_lens = [len(outcome.best_steps) for outcome in outcomes]
    plt.figure(figsize=(5, 5))
    plt.scatter(gold_lens, pred_lens, alpha=0.6)
    lim = max(gold_lens + pred_lens + [1])
    plt.plot([0, lim], [0, lim], linestyle="--", color="gray")
    plt.xlabel("Gold Length")
    plt.ylabel("Predicted Length")
    plt.title("Plan Lengths")
    plt.tight_layout()
    plt.savefig(out_dir / "length_scatter.png", dpi=150)
    plt.close()

    combined_scores = [outcome.combined_score for outcome in outcomes if math.isfinite(outcome.combined_score)]
    if combined_scores:
        plt.figure(figsize=(7, 4.5))
        plt.hist(combined_scores, bins=min(20, max(5, len(combined_scores) // 2)))
        plt.xlabel("Best Combined Score")
        plt.ylabel("Count")
        plt.title("Selected Plan Scores")
        plt.tight_layout()
        plt.savefig(out_dir / "combined_score_hist.png", dpi=150)
        plt.close()


def summarize_outcomes(outcomes: list[SampleOutcome]) -> dict:
    if not outcomes:
        return {}

    def mean_metric(name: str) -> float:
        return statistics.fmean(outcome.best_metrics[name] for outcome in outcomes)

    return {
        "samples": len(outcomes),
        "valid_json_rate": statistics.fmean(1.0 if outcome.valid_candidates > 0 else 0.0 for outcome in outcomes),
        "exact_match": mean_metric("exact_match"),
        "ordered_ratio": mean_metric("ordered_ratio"),
        "step_iou": mean_metric("step_iou"),
        "step_accuracy": mean_metric("step_accuracy"),
        "avg_length_delta": mean_metric("length_delta"),
        "avg_pred_len": mean_metric("pred_len"),
        "avg_gold_len": mean_metric("gold_len"),
        "avg_combined_score": statistics.fmean(outcome.combined_score for outcome in outcomes),
        "avg_critic_score": statistics.fmean(outcome.critic_score for outcome in outcomes),
        "avg_goal_score": statistics.fmean(outcome.goal_score for outcome in outcomes),
    }


def save_tables(out_dir: Path, outcomes: list[SampleOutcome], summary: dict) -> None:
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with (out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])

    with (out_dir / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "goal",
                "best_idx",
                "valid_candidates",
                "combined_score",
                "critic_score",
                "goal_score",
                "exact_match",
                "ordered_ratio",
                "step_iou",
                "step_accuracy",
                "pred_len",
                "gold_len",
                "length_delta",
            ]
        )
        for outcome in outcomes:
            metrics = outcome.best_metrics
            writer.writerow(
                [
                    outcome.sample_id,
                    outcome.goal,
                    outcome.best_idx,
                    outcome.valid_candidates,
                    outcome.combined_score,
                    outcome.critic_score,
                    outcome.goal_score,
                    metrics["exact_match"],
                    metrics["ordered_ratio"],
                    metrics["step_iou"],
                    metrics["step_accuracy"],
                    metrics["pred_len"],
                    metrics["gold_len"],
                    metrics["length_delta"],
                ]
            )

    jsonl_rows = []
    for outcome in outcomes:
        jsonl_rows.append(
            {
                "sample_id": outcome.sample_id,
                "goal": outcome.goal,
                "interpretation": outcome.interpretation,
                "gold_steps": outcome.gold_steps,
                "best_steps": outcome.best_steps,
                "best_idx": outcome.best_idx,
                "valid_candidates": outcome.valid_candidates,
                "critic_score": outcome.critic_score,
                "goal_score": outcome.goal_score,
                "combined_score": outcome.combined_score,
                "metrics": outcome.best_metrics,
                "plans": outcome.plans,
            }
        )
    write_jsonl(out_dir / "per_sample.jsonl", jsonl_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone full-plan evaluation")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--input_mode", choices=["text", "frames"], default="text")
    parser.add_argument("--frames_root", default=None)
    parser.add_argument("--system1_model", required=True)
    parser.add_argument("--system1_type", choices=["t5", "plm"], default="plm")
    parser.add_argument("--plm_base_model", default="facebook/Perception-LM-1B")
    parser.add_argument("--critic_model", default=None)
    parser.add_argument("--goal_latent_model", default=None)
    parser.add_argument("--latent_dir", default=None)
    parser.add_argument("--interp_model", default="facebook/Perception-LM-1B")
    parser.add_argument("--interp_prompt", default=DEFAULT_INTERP_PROMPT)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0, help="Critic weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Goal score weight")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/full_plan_eval_clean")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.test_data = resolve_existing_local_path(args.test_data)
    args.system1_model = resolve_existing_local_path(args.system1_model)
    if args.critic_model:
        args.critic_model = resolve_existing_local_path(args.critic_model)
    if args.goal_latent_model:
        args.goal_latent_model = resolve_existing_local_path(args.goal_latent_model)
    if args.latent_dir:
        args.latent_dir = resolve_existing_local_path(args.latent_dir)
    if args.frames_root:
        args.frames_root = resolve_existing_local_path(args.frames_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(args.test_data, max_samples=args.max_samples)
    print(f"Loaded {len(data)} evaluation samples from {args.test_data}")
    print(f"Device: {device}")

    system1_model, system1_tokenizer, gen_mode = load_system1(
        args.system1_model, device, args.system1_type, args.plm_base_model
    )
    critic_model, critic_tokenizer = load_critic(args.critic_model, device)
    goal_model, goal_tokenizer = load_goal_latent_model(args.goal_latent_model, device)

    outcomes: list[SampleOutcome] = []

    for sample_idx, sample in enumerate(data):
        goal = sample["goal"]
        interpretation = sample.get("interpretation", "") or ""
        if args.input_mode == "frames":
            if not args.frames_root:
                raise ValueError("--frames_root is required when --input_mode frames")
            frames_path = resolve_frames_path(args.frames_root, sample.get("meta", {}))
            interpretation = generate_interpretation_from_frames(
                frames_path=frames_path,
                device=device,
                model_name=args.interp_model,
                num_frames=args.num_frames,
                prompt=args.interp_prompt,
            )

        prompt = build_full_plan_prompt(goal, interpretation)
        gold_steps = sample["gold_steps"]

        plans = []
        for _ in range(args.K):
            raw = generate_plan(
                system1_model,
                system1_tokenizer,
                prompt,
                device,
                gen_mode,
                args.temperature,
                args.top_p,
                args.max_new_tokens,
            )
            pred_steps = parse_plan_json(raw)
            metrics = compute_full_plan_metrics(pred_steps, gold_steps)

            critic_score = 0.0
            if critic_model is not None:
                critic_text = format_trajectory_text(goal, pred_steps)
                enc = critic_tokenizer(
                    critic_text,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    critic_score = float(critic_model(enc.input_ids, enc.attention_mask).item())

            goal_score = 0.0
            if goal_model is not None:
                goal_score = compute_learned_energy(
                    goal_model,
                    goal_tokenizer,
                    goal,
                    pred_steps,
                    device,
                )
            elif args.latent_dir and sample.get("meta", {}).get("video_id"):
                goal_score = compute_lookup_energy(args.latent_dir, sample["meta"], pred_steps)

            combined = args.alpha * critic_score + args.beta * goal_score
            if not pred_steps:
                combined = float("inf")
                critic_score = float("inf") if critic_model is not None else critic_score
                goal_score = float("inf") if (goal_model is not None or args.latent_dir) else goal_score

            plans.append(
                {
                    "raw": raw,
                    "steps": pred_steps,
                    "metrics": metrics,
                    "critic_score": critic_score,
                    "goal_score": goal_score,
                    "combined_score": combined,
                }
            )

        best_idx = min(range(len(plans)), key=lambda idx: plans[idx]["combined_score"])
        best = plans[best_idx]
        valid_candidates = sum(1 for plan in plans if plan["steps"])

        outcomes.append(
            SampleOutcome(
                sample_id=sample_idx,
                goal=goal,
                interpretation=interpretation,
                best_idx=best_idx,
                valid_candidates=valid_candidates,
                best_metrics=best["metrics"],
                combined_score=best["combined_score"],
                critic_score=best["critic_score"],
                goal_score=best["goal_score"],
                gold_steps=gold_steps,
                best_steps=best["steps"],
                plans=plans,
            )
        )

        print(
            f"[{sample_idx + 1}/{len(data)}] valid={valid_candidates}/{args.K} "
            f"exact={best['metrics']['exact_match']:.3f} "
            f"ord={best['metrics']['ordered_ratio']:.3f} "
            f"iou={best['metrics']['step_iou']:.3f}"
        )

    summary = summarize_outcomes(outcomes)
    save_tables(out_dir, outcomes, summary)
    plot_summary(out_dir, summary, outcomes)

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
