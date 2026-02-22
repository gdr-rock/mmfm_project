from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor


@dataclass
class TemporalFeatureStream:
    features: np.ndarray
    segments: list[tuple[float, float]]
    representative_frames: list[np.ndarray]
    duration_seconds: float


class PerceptionFeatureEncoder:
    def __init__(self, model_name: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backend = "transformers"
        self.processor = None
        self.model = None
        self.timm_model = None
        self.timm_transform = None
        self.open_clip_model = None
        self.open_clip_preprocess = None

        try:
            self.processor = self._load_processor(model_name)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.model.eval()
        except Exception:
            try:
                self._load_open_clip_model(model_name)
                self.backend = "open_clip"
            except Exception:
                self._load_timm_model(model_name)
                self.backend = "timm"

    def _load_processor(self, model_name: str):
        try:
            return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            try:
                return AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                # Some encoder checkpoints expose no processor on HF.
                return None

    def _load_timm_model(self, model_name: str) -> None:
        import timm
        from timm.data import create_transform, resolve_data_config

        repo_id = model_name if model_name.startswith("hf-hub:") else f"hf-hub:{model_name}"
        self.timm_model = timm.create_model(repo_id, pretrained=True).to(self.device)
        self.timm_model.eval()
        data_cfg = resolve_data_config({}, model=self.timm_model)
        self.timm_transform = create_transform(**data_cfg, is_training=False)

    def _load_open_clip_model(self, model_name: str) -> None:
        import open_clip

        repo_id = model_name if model_name.startswith("hf-hub:") else f"hf-hub:{model_name}"
        model, _, preprocess = open_clip.create_model_and_transforms(repo_id)
        self.open_clip_model = model.to(self.device)
        self.open_clip_model.eval()
        self.open_clip_preprocess = preprocess

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, 1), dtype=np.float32)

        feats: list[np.ndarray] = []
        for image in images:
            pil_image = Image.fromarray(image[:, :, ::-1])
            feat = self._encode_single_image(pil_image)
            feats.append(feat)
        return np.vstack(feats).astype(np.float32)

    def _encode_single_image(self, image: Image.Image) -> np.ndarray:
        if self.backend == "open_clip":
            return self._encode_single_image_open_clip(image)
        if self.backend == "timm":
            return self._encode_single_image_timm(image)

        with torch.no_grad():
            inputs = self._prepare_inputs(image)
            image_features = self._extract_image_features(inputs)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
        return image_features[0].detach().cpu().numpy()

    def _encode_single_image_timm(self, image: Image.Image) -> np.ndarray:
        if self.timm_model is None or self.timm_transform is None:
            raise RuntimeError("timm backend is not initialized.")

        x = self.timm_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.timm_model.forward_features(x) if hasattr(self.timm_model, "forward_features") else self.timm_model(x)

        feat = self._coerce_feature_tensor(out)
        feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat[0].detach().cpu().numpy()

    def _encode_single_image_open_clip(self, image: Image.Image) -> np.ndarray:
        if self.open_clip_model is None or self.open_clip_preprocess is None:
            raise RuntimeError("open_clip backend is not initialized.")

        x = self.open_clip_preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.open_clip_model.encode_image(x, normalize=True)
        return feat[0].detach().cpu().numpy()

    def _prepare_inputs(self, image: Image.Image) -> dict[str, torch.Tensor]:
        if self.processor is None:
            return self._manual_image_inputs(image)

        attempts: list[dict[str, Any]] = [
            {"images": image, "return_tensors": "pt"},
            {"images": [image], "return_tensors": "pt"},
            {"videos": [[image]], "return_tensors": "pt"},
        ]
        last_err = None
        for kwargs in attempts:
            try:
                data = self.processor(**kwargs)
                if isinstance(data, dict):
                    return {k: v.to(self.device) for k, v in data.items() if isinstance(v, torch.Tensor)}
            except Exception as exc:
                last_err = exc
                continue
        if last_err is not None:
            return self._manual_image_inputs(image)
        raise RuntimeError("Failed to prepare encoder inputs for image.")

    def _manual_image_inputs(self, image: Image.Image) -> dict[str, torch.Tensor]:
        # Fallback for models without published processors.
        size = 224
        rgb = image.convert("RGB").resize((size, size), Image.BICUBIC)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        pixel_values = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return {"pixel_values": pixel_values}

    def _extract_image_features(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        # Transformers APIs differ by model/version; prefer explicit helper when present.
        if hasattr(self.model, "get_image_features"):
            out = self.model.get_image_features(**inputs)
            if isinstance(out, torch.Tensor):
                return out
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                return out.last_hidden_state.mean(dim=1)

        out = self.model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state.mean(dim=1)

        return self._coerce_feature_tensor(out)

    def _coerce_feature_tensor(self, out: Any) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            if out.ndim > 2:
                return out.mean(dim=tuple(range(1, out.ndim - 1)))
            return out
        if isinstance(out, (list, tuple)) and out:
            return self._coerce_feature_tensor(out[0])
        if isinstance(out, dict):
            if "pooler_output" in out and out["pooler_output"] is not None:
                return self._coerce_feature_tensor(out["pooler_output"])
            if "last_hidden_state" in out and out["last_hidden_state"] is not None:
                return self._coerce_feature_tensor(out["last_hidden_state"])
            for value in out.values():
                try:
                    return self._coerce_feature_tensor(value)
                except Exception:
                    continue
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return self._coerce_feature_tensor(out.pooler_output)
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return self._coerce_feature_tensor(out.last_hidden_state)
        raise RuntimeError("Could not extract image features from encoder output.")


def extract_temporal_features(
    video_path: str,
    encoder: PerceptionFeatureEncoder,
    temporal_item_seconds: float,
    frames_per_item: int,
) -> TemporalFeatureStream:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0

    segments: list[tuple[float, float]] = []
    reps: list[np.ndarray] = []
    item_vectors: list[np.ndarray] = []

    start = 0.0
    while start < duration:
        end = min(start + temporal_item_seconds, duration)
        sample_times = np.linspace(start, end, num=max(1, frames_per_item), endpoint=False)
        frames: list[np.ndarray] = []
        for t in sample_times:
            idx = int(t * fps) if fps > 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)

        if frames:
            emb = encoder.encode_images(frames)
            item_vectors.append(emb.mean(axis=0))
            reps.append(frames[len(frames) // 2])
            segments.append((start, end))

        start = end

    cap.release()

    if not item_vectors:
        return TemporalFeatureStream(
            features=np.zeros((0, 1), dtype=np.float32),
            segments=[],
            representative_frames=[],
            duration_seconds=duration,
        )

    return TemporalFeatureStream(
        features=np.vstack(item_vectors).astype(np.float32),
        segments=segments,
        representative_frames=reps,
        duration_seconds=duration,
    )
