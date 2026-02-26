from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.training.baseline_conditioning import build_labels_with_baseline_conditioning_mask
from src.training.baseline_schema import BaselineSample, build_baseline_training_text
from src.training.dataset_schema import System1Sample, build_system1_training_text
from src.training.losses import IGNORE_INDEX, autoregressive_ce_loss
from src.training.system1_conditioning import build_labels_with_conditioning_mask

try:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    AdamW = None
    DataLoader = None
    Dataset = object

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    AutoModelForCausalLM = None
    AutoTokenizer = None


Mode = Literal["system1", "baseline"]


@dataclass
class TextPair:
    prefix: str
    target: str


class JsonlTrajectoryDataset(Dataset):
    def __init__(self, jsonl_path: str, mode: Mode) -> None:
        if torch is None:
            raise ModuleNotFoundError("torch is required for dataset/training runtime")
        self.mode = mode
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        self.rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"No records found in {self.path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> TextPair:
        row = self.rows[idx]
        if self.mode == "system1":
            sample = System1Sample(
                config=_pick(row, "config", "system_prompt", default="predict goal-plan trajectory"),
                context=_pick(row, "visual_context", "context"),
                goal_description=_pick(row, "goal_description", "goal"),
                goal_interpretation=_pick(row, "goal_interpretation"),
                actions=_pick(row, "actions"),
                delta_states=_pick(row, "delta_states", "states"),
            )
            prefix, target = build_system1_training_text(sample)
            return TextPair(prefix=prefix, target=target)

        sample = BaselineSample(
            config=_pick(row, "config", "system_prompt", default="predict procedural actions"),
            visual_context=_pick(row, "visual_context", "context"),
            goal_description=_pick(row, "goal_description", "goal"),
            actions=_pick(row, "actions"),
            asr_text=row.get("asr_text"),
        )
        prefix, target = build_baseline_training_text(sample)
        return TextPair(prefix=prefix, target=target)


class TrainingBatchCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        if torch is None:
            raise ModuleNotFoundError("torch is required for collation")
        self.tokenizer = tokenizer
        self.max_length = max_length

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must provide pad_token_id or eos_token_id")
            tokenizer.pad_token = tokenizer.eos_token
            pad_id = tokenizer.pad_token_id
        self.pad_token_id = int(pad_id)

    def __call__(self, samples: List[TextPair]) -> Dict[str, torch.Tensor]:
        input_id_rows: List[List[int]] = []
        prefix_lens: List[int] = []

        for sample in samples:
            prefix_ids = self.tokenizer.encode(sample.prefix, add_special_tokens=False)
            target_ids = self.tokenizer.encode(sample.target, add_special_tokens=False)

            full_ids = prefix_ids + target_ids
            if self.tokenizer.eos_token_id is not None:
                full_ids = full_ids + [int(self.tokenizer.eos_token_id)]

            if len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]
            prefix_len = min(len(prefix_ids), len(full_ids))

            input_id_rows.append(full_ids)
            prefix_lens.append(prefix_len)

        max_len = max(len(x) for x in input_id_rows)
        padded_ids: List[List[int]] = []
        padded_mask: List[List[int]] = []

        for ids in input_id_rows:
            pad_n = max_len - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_n)
            padded_mask.append([1] * len(ids) + [0] * pad_n)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "prefix_lengths": torch.tensor(prefix_lens, dtype=torch.long),
        }


def run_training(mode: Mode, cfg: Dict[str, Any], dry_run: bool = False, max_steps_override: Optional[int] = None) -> Dict[str, float]:
    if torch is None:
        raise ModuleNotFoundError("torch is required to run training")
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ModuleNotFoundError("transformers is required to run training")

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    model_name = cfg["model"]["pretrained"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    io_cfg = cfg["io"]
    train_ds = JsonlTrajectoryDataset(io_cfg["train_jsonl"], mode=mode)
    collator = TrainingBatchCollator(tokenizer=tokenizer, max_length=_max_length_for_mode(cfg, mode))

    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)

    epochs = int(cfg["training"].get("epochs", 1))
    train_steps = len(train_loader) * epochs
    if max_steps_override is not None:
        train_steps = min(train_steps, max_steps_override)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 2e-5)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )
    scheduler = _build_scheduler(
        optimizer=optimizer,
        total_steps=max(1, train_steps),
        warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.03)),
    )

    grad_clip = float(cfg["training"].get("grad_clip_norm", 1.0))
    amp_mode = str(cfg["training"].get("mixed_precision", "bf16")).lower()

    global_step = 0
    total_loss = 0.0
    model.train()

    for _epoch in range(epochs):
        for batch in train_loader:
            if max_steps_override is not None and global_step >= max_steps_override:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prefix_lengths = batch["prefix_lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)

            use_amp = device.type == "cuda" and amp_mode in {"bf16", "fp16"}
            amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                if mode == "system1":
                    labels = build_labels_with_conditioning_mask(
                        input_ids=input_ids,
                        prefix_lengths=prefix_lengths,
                        ignore_index=IGNORE_INDEX,
                    )
                else:
                    labels = build_labels_with_baseline_conditioning_mask(
                        input_ids=input_ids,
                        prefix_lengths=prefix_lengths,
                        ignore_index=IGNORE_INDEX,
                    )
                loss = autoregressive_ce_loss(logits=logits, labels=labels, ignore_index=IGNORE_INDEX)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.detach().item())
            global_step += 1

            if dry_run:
                break

        if dry_run:
            break

    out_dir = Path(io_cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{mode}_last.pt"
    torch.save(
        {
            "mode": mode,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        ckpt_path,
    )

    avg_loss = total_loss / max(global_step, 1)
    return {
        "steps": float(global_step),
        "avg_loss": float(avg_loss),
        "checkpoint": str(ckpt_path),
    }


def _build_scheduler(optimizer: Any, total_steps: int, warmup_ratio: float):
    if torch is None:
        raise ModuleNotFoundError("torch is required for scheduler")

    warmup_steps = int(total_steps * max(0.0, min(1.0, warmup_ratio)))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _max_length_for_mode(cfg: Dict[str, Any], mode: Mode) -> int:
    model_cfg = cfg.get("model", {})
    if mode == "system1":
        return int(model_cfg.get("max_context_tokens", 11500))
    return int(model_cfg.get("decoder_context_tokens", 4096))


def _pick(row: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    if default is not None:
        return default
    raise KeyError(f"None of the keys found in sample: {keys}")
