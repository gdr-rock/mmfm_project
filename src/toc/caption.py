from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    BlipForConditionalGeneration,
)


class SegmentCaptioner:
    def __init__(self, model_name: str, fallback_model_name: Optional[str], max_new_tokens: int) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.processor = None
        self.model = None
        self.model_type = None
        self._load_model()

    def _load_model(self) -> None:
        candidates = [self.model_name]
        if self.fallback_model_name:
            candidates.append(self.fallback_model_name)

        last_err = None
        for name in candidates:
            try:
                self.processor = self._load_processor(name)
                self.model = self._load_model_for_name(name).to(self.device)
                self.model.eval()
                self.model_name = name
                return
            except Exception as exc:
                last_err = exc
                continue

        raise RuntimeError(f"Failed to load caption model(s): {candidates}. Last error: {last_err}")

    def _load_processor(self, name: str):
        # Prefer default behavior (usually fast). Some checkpoints do not support slow processors.
        try:
            return AutoProcessor.from_pretrained(name, trust_remote_code=True)
        except Exception:
            # Backward compatibility for older checkpoints that may require explicit slow mode.
            return AutoProcessor.from_pretrained(name, trust_remote_code=True, use_fast=False)

    def _load_model_for_name(self, name: str):
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        self.model_type = getattr(cfg, "model_type", None)

        # BLIP checkpoints are vision-to-text models, not causal LM-only.
        if self.model_type == "blip":
            return BlipForConditionalGeneration.from_pretrained(name, trust_remote_code=True)

        # Perception-LM uses its own conditional generation class.
        if self.model_type == "perception_lm":
            from transformers.models.perception_lm.modeling_perception_lm import (
                PerceptionLMForConditionalGeneration,
            )

            return PerceptionLMForConditionalGeneration.from_pretrained(name, trust_remote_code=True)

        loaders = (AutoModelForCausalLM.from_pretrained,)
        last_err = None
        for loader in loaders:
            try:
                return loader(name, trust_remote_code=True)
            except Exception as exc:
                last_err = exc
                continue

        raise RuntimeError(f"No compatible model class found for {name}. Last error: {last_err}")

    def caption_segment(self, frames_bgr: list[np.ndarray]) -> str:
        if self.processor is None or self.model is None:
            return ""
        if not frames_bgr:
            return ""

        images = [Image.fromarray(frame[:, :, ::-1]) for frame in frames_bgr]

        prompt = "Describe this video segment in one concise sentence."
        if self.model_type == "perception_lm":
            image_token = getattr(getattr(self.processor, "tokenizer", None), "image_token", "<image>")
            prompt = f"{image_token}\n{prompt}"

        inputs = self._prepare_caption_inputs(images, prompt)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        model_dtype = next(self.model.parameters()).dtype
        prepared_inputs = self._move_and_cast(inputs, model_dtype=model_dtype)
        prepared_inputs = self._normalize_generation_inputs(prepared_inputs, model_dtype=model_dtype)

        # PerceptionLM is sensitive to pixel dtype/device alignment with vision tower weights.
        if "pixel_values" in prepared_inputs and torch.is_tensor(prepared_inputs["pixel_values"]):
            prepared_inputs["pixel_values"] = prepared_inputs["pixel_values"].to(
                device=self.device,
                dtype=model_dtype,
            )

        with torch.no_grad():
            out = self.model.generate(**prepared_inputs, max_new_tokens=self.max_new_tokens)

        text = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        return self._clean_caption_text(text)

    def _clean_caption_text(self, text: str) -> str:
        cleaned = text.strip()
        prompt = "Describe this video segment in one concise sentence."
        if cleaned.startswith(prompt):
            cleaned = cleaned[len(prompt) :].strip(" :.-\n\t")
        if cleaned.startswith("<image>"):
            cleaned = cleaned[len("<image>") :].strip(" :.-\n\t")
        return cleaned

    def _move_and_cast(self, value, model_dtype: torch.dtype):
        if torch.is_tensor(value):
            t = value.to(self.device)
            if torch.is_floating_point(t):
                t = t.to(model_dtype)
            return t
        if isinstance(value, Mapping):
            return {k: self._move_and_cast(v, model_dtype) for k, v in value.items()}
        if isinstance(value, list):
            return [self._move_and_cast(v, model_dtype) for v in value]
        if isinstance(value, tuple):
            return tuple(self._move_and_cast(v, model_dtype) for v in value)
        return value

    def _normalize_generation_inputs(self, inputs: dict, model_dtype: torch.dtype) -> dict:
        out = dict(inputs)
        for key in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
            if key not in out:
                continue
            v = out[key]
            if torch.is_tensor(v):
                out[key] = v.to(self.device)
            elif isinstance(v, (list, tuple)):
                out[key] = torch.as_tensor(v, device=self.device)

        if "pixel_values" in out:
            pv = out["pixel_values"]
            if torch.is_tensor(pv):
                out["pixel_values"] = pv.to(device=self.device, dtype=model_dtype)
            elif isinstance(pv, (list, tuple)):
                out["pixel_values"] = torch.as_tensor(pv, device=self.device, dtype=model_dtype)

        return out

    def _prepare_caption_inputs(self, images: list[Image.Image], prompt: str):
        if self.model_type == "perception_lm":
            # PerceptionLM expects image placeholder count to match visual features.
            # Use a single representative frame to keep token/feature alignment stable.
            image = images[len(images) // 2]
            return self.processor(images=image, text=prompt, return_tensors="pt")

        attempts = [
            {"images": images, "text": prompt, "return_tensors": "pt"},
            {"images": images[0], "text": prompt, "return_tensors": "pt"},
            {"videos": [images], "text": prompt, "return_tensors": "pt"},
        ]

        last_err = None
        for kwargs in attempts:
            try:
                return self.processor(**kwargs)
            except Exception as exc:
                last_err = exc
                continue

        raise RuntimeError(f"Failed to prepare caption inputs. Last error: {last_err}")
