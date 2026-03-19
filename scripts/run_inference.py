#!/usr/bin/env python3
"""
Standalone inference for the VLWM-style planning pipeline.

╔══════════════════════════════════════════════════════════════════╗
║  WHAT THIS SCRIPT DOES                                         ║
║  1. System-1 (fast):   goal + prefix → next-step predictions   ║
║  2. System-2 (search): generate K plans → rerank with Critic   ║
║                        + optional V-JEPA energy → pick best    ║
╚══════════════════════════════════════════════════════════════════╝

Inputs at inference time
────────────────────────
  • Goal:   free-text description of the task
            e.g. "Complete the task: Make Pancakes."
  • Prefix: list of already-observed steps, each "action | state_change"
            e.g. ["pour egg | Egg is now poured in.",
                   "add flour | Flour is now added to the mixture."]
  • k:      how many future steps to predict (default: 3)

The script constructs the same prompt format used during training:

    Goal: Complete the task: Make Pancakes.
    Interpretation: <optional>

    Progress so far:
      1) pour egg | Egg is now poured in.
      2) add flour | Flour is now added to the mixture.

    Predict the next 3 step(s). Output JSON only: {"next_steps": [{"action": "...", "state_change": "..."}, ...]}

Model variants supported
────────────────────────
  System-1:  --system1_type t5   → T5-small seq2seq (default)
             --system1_type plm  → PerceptionLM-1B + LoRA (causal LM)

  Critic:    --critic_type mlp   → MiniLM + MLP head (default)
             --critic_type llm   → Llama-3.2-1B + LoRA + cost head

  Energy:    --latent_dir        → V-JEPA latent directory (optional)

Usage examples
──────────────
  # 1) System-1 greedy (T5, text-only) — quickest
  python3 scripts/run_inference.py \\
      --system1_model checkpoints/system1/best_model \\
      --goal "Complete the task: Make Pancakes." \\
      --prefix "pour egg | Egg is now poured in." \\
               "add flour | Flour is now added to the mixture." \\
      --k 3

  # 2) System-2 with T5 + MiniLM critic
  python3 scripts/run_inference.py \\
      --mode system2 \\
      --system1_model checkpoints/system1/best_model \\
      --critic_model  checkpoints/critic/best_model.pt \\
      --goal "Complete the task: Make Pancakes." \\
      --prefix "pour egg | Egg is now poured in." \\
      --K 5 --temperature 0.8

  # 3) System-2 with PLM+LoRA + LLM critic + V-JEPA energy
  python3 scripts/run_inference.py \\
      --mode system2 \\
      --system1_type plm \\
      --system1_model checkpoints/system1_plm_lora/best_adapter \\
      --plm_base_model facebook/Perception-LM-1B \\
      --critic_type llm \\
      --critic_model checkpoints/critic_llm_lora/best_adapter \\
      --llm_base_critic meta-llama/Llama-3.2-1B \\
      --latent_dir data/crosstask/vjepa_latents \\
      --task_id 91515 --video_id KUDnfzXsB3w \\
      --goal "Complete the task: Make Pancakes." \\
      --prefix "pour egg | Egg is now poured in." \\
      --K 5 --alpha 0.7 --beta 0.3

  # 4) From a test JSONL sample (auto-fills goal, prefix, meta)
  python3 scripts/run_inference.py \\
      --mode system2 \\
      --from_jsonl data/crosstask/system1_test.jsonl \\
      --sample_idx 0 \\
      --system1_model checkpoints/system1/best_model \\
      --critic_model  checkpoints/critic/best_model.pt \\
      --K 5

  # 5) Goal model inference (what is the task given a partial trajectory?)
  python3 scripts/run_inference.py \\
      --mode goal \\
      --goal_model checkpoints/goal_model/best_model \\
      --prefix "pour egg | Egg is now poured in." \\
               "add flour | Flour is now added to the mixture."

  # 6) Vision-grounded interpretation from initial video frames
  #    PLM's vision tower generates the Interpretation: line from frames.
  python3 scripts/run_inference.py \\
      --system1_model checkpoints/system1/best_model \\
      --goal "Complete the task: Make Pancakes." \\
      --frames path/to/video.mp4 \\
      --num_frames 8 \\
      --k 3

  # 7) Vision-grounded interpretation with PLM+LoRA (model reuse)
  #    If --interp_model matches --plm_base_model, we can share the model.
  python3 scripts/run_inference.py \\
      --system1_type plm \\
      --system1_model checkpoints/system1_plm_lora/best_adapter \\
      --plm_base_model facebook/Perception-LM-1B \\
      --goal "Complete the task: Make Pancakes." \\
      --frames initial_frames/ \\
      --interp_model facebook/Perception-LM-1B \\
      --k 3
"""

import argparse
import json
import os
import re
import sys
import importlib.util
from pathlib import Path

import torch

# Must match scripts/train_system1_plm_lora.py
CAUSAL_PROMPT_SUFFIX = "\n\nAssistant:"

# ─────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────

def build_system1_prompt(goal: str, prefix_steps: list, k: int,
                         interpretation: str = "") -> str:
    """
    Build the exact prompt format used in System-1 training data.

    This matches scripts/06_build_system1_dataset.py output.
    """
    lines = [f"Goal: {goal}"]
    if interpretation:
        lines.append(f"Interpretation: {interpretation}")
    lines.append("")
    if prefix_steps:
        lines.append("Progress so far:")
        for i, step in enumerate(prefix_steps, 1):
            lines.append(f"  {i}) {step}")
    else:
        lines.append("Progress so far:")
        lines.append("  (No steps observed yet.)")
    lines.append("")
    lines.append(
        f'Predict the next {k} step(s). Output JSON only: '
        '{"next_steps": [{"action": "...", "state_change": "..."}, ...]}'
    )
    return "\n".join(lines)


def build_goal_prompt(prefix_steps: list) -> str:
    """Build the prompt format used in Goal Model training data."""
    lines = ["Observed steps:"]
    for i, step in enumerate(prefix_steps, 1):
        lines.append(f"  {i}) {step}")
    lines.append("")
    lines.append("Identify the overall goal of this task.")
    return "\n".join(lines)


def parse_plan_json(text: str) -> list:
    """Parse model output into list of 'action | state_change' strings."""

    def _decode_json_string(s: str) -> str:
        try:
            return json.loads(f"\"{s}\"")
        except Exception:
            return s

    def _steps_from_list(step_list):
        if not isinstance(step_list, list):
            return []
        out = []
        for s in step_list:
            if not isinstance(s, dict):
                continue
            action = s.get("action")
            state_change = s.get("state_change")
            if isinstance(action, str) and isinstance(state_change, str):
                out.append(f"{action.strip()} | {state_change.strip()}")
        return out

    text = (text or "").strip()
    if not text:
        return []

    # Common cleanup: fenced code block payloads.
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1).strip()

    # Try several JSON candidates (raw, wrapped, extracted object).
    candidates = [text]
    if '"next_steps"' in text and not text.lstrip().startswith("{"):
        candidates.append("{" + text + "}")
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace:last_brace + 1])

    for cand in candidates:
        try:
            parsed = json.loads(cand)
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

    # Fallback: recover quoted key-value pairs from malformed JSON-like output.
    action_vals = [
        _decode_json_string(v)
        for v in re.findall(r'"action"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    ]
    state_vals = [
        _decode_json_string(v)
        for v in re.findall(r'"state_change"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    ]
    n = min(len(action_vals), len(state_vals))
    if n > 0:
        return [
            f"{action_vals[i].strip()} | {state_vals[i].strip()}"
            for i in range(n)
        ]

    return []


# ─────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────

def load_system1(model_path: str, device, model_type: str = "t5",
                 base_model_name: str = None):
    """Load System-1 model. Returns (model, tokenizer, gen_mode)."""
    from transformers import AutoTokenizer

    if model_type == "t5":
        from transformers import T5ForConditionalGeneration
        print(f"Loading System-1 (T5): {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model.to(device).eval()
        return model, tokenizer, "seq2seq"

    elif model_type == "plm":
        from transformers import AutoModelForCausalLM
        from peft import PeftModel

        if base_model_name is None:
            base_model_name = "facebook/Perception-LM-1B"

        is_adapter = os.path.isdir(model_path) and os.path.exists(
            os.path.join(model_path, "adapter_config.json")
        )
        if is_adapter:
            print(f"Loading System-1 (PLM+LoRA): base={base_model_name}, adapter={model_path}")
        else:
            print(f"Loading System-1 (PLM full checkpoint): {model_path}")

        # Prefer adapter tokenizer if saved during training.
        tok_source = model_path if os.path.isdir(model_path) else base_model_name
        tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if is_adapter:
            try:
                from transformers import AutoModelForImageTextToText
                base = AutoModelForImageTextToText.from_pretrained(
                    base_model_name, dtype=torch.bfloat16, trust_remote_code=True
                )
            except Exception:
                base = AutoModelForCausalLM.from_pretrained(
                    base_model_name, dtype=torch.bfloat16, trust_remote_code=True
                )

            if hasattr(base, "tie_weights"):
                try:
                    base.tie_weights()
                    print("  Tied input/output embeddings")
                except Exception as e:
                    print(f"  Warning: could not tie weights ({e})")

            model = PeftModel.from_pretrained(base, model_path)
        else:
            try:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(
                    model_path, dtype=torch.bfloat16, trust_remote_code=True
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, dtype=torch.bfloat16, trust_remote_code=True
                )

        model.to(device).eval()
        return model, tokenizer, "causal"

    else:
        raise ValueError(f"Unknown system1_type: {model_type}")


def load_critic(model_path: str, device, critic_type: str = "mlp",
                base_model_name: str = None):
    """Load Critic model. Returns (model, tokenizer, format_fn)."""
    from transformers import AutoTokenizer
    import importlib.util

    if critic_type == "mlp":
        from transformers import AutoModel
        print(f"Loading Critic (MiniLM+MLP): {model_path}")
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
        model.to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        return model, tokenizer, mod.format_trajectory_text

    elif critic_type == "llm":
        from transformers import AutoModelForCausalLM
        from peft import PeftModel

        if base_model_name is None:
            base_model_name = "meta-llama/Llama-3.2-1B"
        print(f"Loading Critic (LLM+LoRA): base={base_model_name}, adapter={model_path}")

        spec = importlib.util.spec_from_file_location(
            "train_critic_llm_lora",
            os.path.join(os.path.dirname(__file__), "train_critic_llm_lora.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        peft_model = PeftModel.from_pretrained(base, model_path)
        peft_model.eval()

        hidden_dim = base.config.hidden_size
        critic = mod.LLMCriticWithHead(peft_model, hidden_dim)

        head_path = os.path.join(model_path, "cost_head.pt")
        if os.path.exists(head_path):
            critic.load_head(head_path)
        else:
            print(f"  ⚠️  cost_head.pt not found at {head_path}")

        critic.to(device).eval()
        return critic, tokenizer, mod.format_trajectory

    else:
        raise ValueError(f"Unknown critic_type: {critic_type}")


def load_goal_model(model_path: str, device):
    """Load Goal Model (T5 seq2seq)."""
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    print(f"Loading Goal Model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device).eval()
    return model, tokenizer


def load_goal_latent_model(model_path: str, device):
    """Load learned goal-latent model used for energy scoring."""
    from transformers import AutoTokenizer

    spec = importlib.util.spec_from_file_location(
        "train_goal_latent_model",
        os.path.join(os.path.dirname(__file__), "train_goal_latent_model.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = mod.GoalLatentModel(
        encoder_name=ckpt["encoder_name"],
        latent_dim=ckpt["latent_dim"],
        hidden_dim=ckpt.get("hidden_dim", 384),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt["encoder_name"])
    return model, tokenizer, mod.format_trajectory_for_energy


# ─────────────────────────────────────────────────────────────────
# Vision-grounded interpretation  (Option 3: PLM frames → text)
# ─────────────────────────────────────────────────────────────────

DEFAULT_INTERP_PROMPT = (
    "Given these initial frames of the video, describe in one sentence "
    "what the person is about to do and what the end result should look like."
)


def generate_interpretation_from_frames(
    frames_path: str,
    device,
    model_name: str = "facebook/Perception-LM-1B",
    num_frames: int = 8,
    prompt: str = DEFAULT_INTERP_PROMPT,
    max_new_tokens: int = 128,
    preloaded_model=None,
    preloaded_processor=None,
) -> str:
    """
    Use PLM's vision tower to generate a scene interpretation from video
    frames or a short video clip.

    Parameters
    ----------
    frames_path : str
        Path to a video file (.mp4/.webm) **or** a directory of frame images
        (.jpg/.png). If a directory, the first ``num_frames`` images (sorted
        alphabetically) are used.  If a video file, the first ``num_frames``
        frames are extracted (NOT uniformly sampled — we deliberately take the
        opening frames so PLM sees the *initial* scene state).
    device : torch.device
    model_name : str
        HuggingFace model ID for PLM (1B or 8B).
    num_frames : int
        Number of frames to sample from the video (only for video files).
    prompt : str
        The text prompt sent alongside the visual input.
    max_new_tokens : int
        Max tokens for the generated interpretation.
    preloaded_model : optional
        An already-loaded ``AutoModelForImageTextToText`` (avoids double-loading
        when ``--system1_type plm`` is also in use).
    preloaded_processor : optional
        The corresponding ``AutoProcessor``.

    Returns
    -------
    str
        The generated interpretation text.
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    # ── Load model / processor (or reuse) ──
    if preloaded_model is not None and preloaded_processor is not None:
        model = preloaded_model
        processor = preloaded_processor
        print("  (reusing preloaded PLM for interpretation)")
    else:
        print(f"Loading PLM for interpretation: {model_name}")
        processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()

    # ── Build conversation payload ──
    frames_p = Path(frames_path)
    if frames_p.is_dir():
        # Directory of frame images — take first num_frames (sorted)
        from PIL import Image
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        img_files = sorted(
            [f for f in frames_p.iterdir() if f.suffix.lower() in exts]
        )[:num_frames]
        if not img_files:
            print(f"  ⚠️  No images found in {frames_path}")
            return ""
        content = []
        for img_f in img_files:
            content.append({"type": "image", "url": str(img_f)})
        content.append({"type": "text", "text": prompt})
        print(f"  Using {len(img_files)} frame images from directory")
    elif frames_p.is_file():
        # Video file — extract the FIRST num_frames frames (initial scene).
        # We deliberately avoid the processor's default uniform sampling,
        # which would spread frames across the whole video.  We only want
        # the opening scene so PLM describes the starting state.
        from PIL import Image
        try:
            import decord
            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(str(frames_p))
            total = len(vr)
            n = min(num_frames, total)
            # Take the very first n frames
            frame_indices = list(range(n))
            frames_np = vr.get_batch(frame_indices).asnumpy()  # (N, H, W, 3)
            pil_frames = [Image.fromarray(f) for f in frames_np]
        except ImportError:
            # Fallback: OpenCV
            import cv2
            cap = cv2.VideoCapture(str(frames_p))
            pil_frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                pil_frames.append(
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )
            cap.release()

        if not pil_frames:
            print(f"  ⚠️  Could not extract frames from {frames_path}")
            return ""

        # Save temporary frames and pass as images (not video)
        # so we bypass the processor's uniform video sampling.
        content = []
        for pil_f in pil_frames:
            content.append({"type": "image", "image": pil_f})
        content.append({"type": "text", "text": prompt})
        print(f"  Extracted first {len(pil_frames)} frames from video "
              f"(total: {len(vr) if 'vr' in dir() else '?'} frames)")
    else:
        print(f"  ⚠️  frames path not found: {frames_path}")
        return ""

    conversation = [{"role": "user", "content": content}]

    # ── Tokenize & generate ──
    apply_kwargs = dict(
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = processor.apply_chat_template([conversation], **apply_kwargs)
    inputs = inputs.to(device)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    input_length = inputs["input_ids"].shape[1]
    gen_text = processor.batch_decode(
        gen_ids[:, input_length:], skip_special_tokens=True
    )[0].strip()

    return gen_text


# ─────────────────────────────────────────────────────────────────
# V-JEPA energy scoring
# ─────────────────────────────────────────────────────────────────

def compute_energy(latent_dir: str, task_id: str, video_id: str,
                   plan_steps: list, n_prefix: int) -> float:
    """
    Compute Energy(z_predicted, z_goal) = ||z_pred − z_goal||₂².

    z_goal = last segment latent (V-JEPA).
    z_pred = segment latent at the position the plan would reach.
    Latents are mean-pooled from (T_tokens, D) → (D,).
    """
    vid_dir = os.path.join(latent_dir, str(task_id), str(video_id))
    if not os.path.exists(vid_dir):
        return 0.0

    latent_files = sorted(
        [f for f in os.listdir(vid_dir)
         if f.endswith(".pt") and f.startswith("segment_")],
        key=lambda f: int(f.split("_")[1].split(".")[0]),
    )
    if not latent_files:
        return 0.0

    def _load_pool(path):
        z = torch.load(path, weights_only=True)
        return z.mean(dim=0) if z.dim() == 2 else z

    z_goal = _load_pool(os.path.join(vid_dir, latent_files[-1]))

    target_idx = min(len(latent_files) - 1, n_prefix + len(plan_steps))
    z_pred = _load_pool(os.path.join(vid_dir, latent_files[target_idx]))

    return torch.norm(z_pred - z_goal, p=2).item() ** 2


def compute_learned_energy(
    goal_latent_model,
    goal_latent_tokenizer,
    format_traj_fn,
    goal: str,
    full_steps: list,
    device,
    max_goal_len: int = 64,
    max_traj_len: int = 384,
) -> float:
    """Compute learned energy E=||z_traj(goal,steps)-z_goal(goal)||^2."""
    goal_enc = goal_latent_tokenizer(
        [goal],
        max_length=max_goal_len,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    traj_text = format_traj_fn(goal, full_steps)
    traj_enc = goal_latent_tokenizer(
        [traj_text],
        max_length=max_traj_len,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        z_goal = goal_latent_model.encode_goal(
            goal_enc.input_ids, goal_enc.attention_mask
        )[0]
        z_traj = goal_latent_model.encode_traj(
            traj_enc.input_ids, traj_enc.attention_mask
        )[0]
    return torch.norm(z_traj - z_goal, p=2).item() ** 2


# ─────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────

def generate_plan(model, tokenizer, prompt: str, device,
                  gen_mode: str = "seq2seq",
                  temperature: float = 0.0,
                  top_p: float = 0.9,
                  max_new_tokens: int = 256) -> str:
    """Generate a single plan from a prompt."""
    if gen_mode == "causal" and not prompt.rstrip().endswith("Assistant:"):
        prompt = prompt + CAUSAL_PROMPT_SUFFIX

    enc = tokenizer(
        prompt, max_length=512, truncation=True, return_tensors="pt"
    ).to(device)
    prompt_len = enc.input_ids.shape[1]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

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
    else:
        return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────
# Main inference modes
# ─────────────────────────────────────────────────────────────────

def run_system1(args, device):
    """System-1 only: greedy decode (or temperature sampling) from one prompt."""
    model, tokenizer, gen_mode = load_system1(
        args.system1_model, device, args.system1_type, args.plm_base_model
    )

    prompt = build_system1_prompt(
        goal=args.goal,
        prefix_steps=args.prefix or [],
        k=args.k,
        interpretation=args.interpretation or "",
    )

    print(f"\n{'─'*60}")
    print("PROMPT:")
    print(prompt)
    print(f"{'─'*60}\n")

    raw = generate_plan(
        model, tokenizer, prompt, device,
        gen_mode=gen_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    steps = parse_plan_json(raw)

    print("RAW OUTPUT:")
    print(raw)
    print(f"\nPARSED STEPS ({len(steps)}):")
    for i, s in enumerate(steps, 1):
        print(f"  {i}) {s}")
    if not steps:
        print("  (no valid JSON parsed)")

    return {"raw": raw, "steps": steps}


def run_system2(args, device):
    """System-2: generate K plans, rerank with Critic + optional Energy."""
    # Load models
    s1_model, s1_tokenizer, gen_mode = load_system1(
        args.system1_model, device, args.system1_type, args.plm_base_model
    )
    critic_model, critic_tokenizer, format_traj_fn = load_critic(
        args.critic_model, device, args.critic_type, args.llm_base_critic
    )
    goal_latent_model = None
    goal_latent_tokenizer = None
    goal_latent_format = None
    if args.goal_latent_model:
        goal_latent_model, goal_latent_tokenizer, goal_latent_format = load_goal_latent_model(
            args.goal_latent_model, device
        )

    prompt = build_system1_prompt(
        goal=args.goal,
        prefix_steps=args.prefix or [],
        k=args.k,
        interpretation=args.interpretation or "",
    )

    print(f"\n{'─'*60}")
    print("PROMPT:")
    print(prompt)
    print(f"{'─'*60}\n")

    # Generate K plans
    print(f"Generating {args.K} candidate plans (T={args.temperature})...")
    plans = []
    for i in range(args.K):
        raw = generate_plan(
            s1_model, s1_tokenizer, prompt, device,
            gen_mode=gen_mode,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        steps = parse_plan_json(raw)
        plans.append({
            "raw": raw,
            "steps": steps,
            "valid": len(steps) > 0,
        })
        status = f"{len(steps)} steps" if steps else "invalid JSON"
        print(f"  Plan {i+1}: {status}")

    # Score with critic
    print(f"\nScoring with Critic ({args.critic_type})...")
    prefix_steps = args.prefix or []
    critic_scores = []
    for plan in plans:
        if not plan["valid"]:
            critic_scores.append(float("inf"))
            continue
        full_steps = prefix_steps + plan["steps"]
        text = format_traj_fn(args.goal, full_steps)
        enc = critic_tokenizer(
            text, max_length=512, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            cost = critic_model(enc.input_ids, enc.attention_mask).item()
        critic_scores.append(cost)

    # Score with energy (optional)
    energy_scores = [0.0] * len(plans)
    use_learned_energy = goal_latent_model is not None
    use_lookup_energy = (
        (not use_learned_energy)
        and args.latent_dir and os.path.isdir(args.latent_dir or "")
        and args.task_id and args.video_id
    )
    if use_learned_energy:
        print(f"Computing learned energy (α={args.alpha}, β={args.beta})...")
        for i, plan in enumerate(plans):
            if not plan["valid"]:
                energy_scores[i] = float("inf")
                continue
            full_steps = prefix_steps + plan["steps"]
            energy_scores[i] = compute_learned_energy(
                goal_latent_model,
                goal_latent_tokenizer,
                goal_latent_format,
                args.goal,
                full_steps,
                device,
            )
    elif use_lookup_energy:
        print(f"Computing V-JEPA energy (α={args.alpha}, β={args.beta})...")
        for i, plan in enumerate(plans):
            if not plan["valid"]:
                energy_scores[i] = float("inf")
                continue
            energy_scores[i] = compute_energy(
                args.latent_dir, args.task_id, args.video_id,
                plan["steps"], len(prefix_steps),
            )
    else:
        args.alpha = 1.0
        args.beta = 0.0

    # Combined scores
    combined = [
        args.alpha * c + args.beta * e
        for c, e in zip(critic_scores, energy_scores)
    ]
    best_idx = min(range(len(combined)), key=lambda i: combined[i])

    # Print results
    print(f"\n{'─'*60}")
    print("PLAN RANKING:")
    print(f"{'─'*60}")
    for i, (p, cs, es, comb) in enumerate(
        zip(plans, critic_scores, energy_scores, combined)
    ):
        marker = " ◀ BEST" if i == best_idx else ""
        status = "✓" if p["valid"] else "✗"
        print(f"  [{status}] Plan {i+1}: critic={cs:.4f}  "
              f"energy={es:.4f}  combined={comb:.4f}{marker}")
        if p["steps"]:
            for j, s in enumerate(p["steps"], 1):
                print(f"        {j}) {s}")

    print(f"\n{'─'*60}")
    print(f"BEST PLAN (#{best_idx+1}, score={combined[best_idx]:.4f}):")
    print(f"{'─'*60}")
    for j, s in enumerate(plans[best_idx]["steps"], 1):
        print(f"  {j}) {s}")

    return {
        "best_idx": best_idx,
        "best_plan": plans[best_idx],
        "best_score": combined[best_idx],
        "all_plans": plans,
        "critic_scores": critic_scores,
        "energy_scores": energy_scores,
    }


def run_goal(args, device):
    """Goal Model inference: predict the task goal from observed steps."""
    model, tokenizer = load_goal_model(args.goal_model, device)

    prompt = build_goal_prompt(args.prefix or [])

    print(f"\n{'─'*60}")
    print("PROMPT:")
    print(prompt)
    print(f"{'─'*60}\n")

    enc = tokenizer(
        prompt, max_length=384, truncation=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        gen_ids = model.generate(
            **enc, max_new_tokens=128, num_beams=1, do_sample=False
        )
    pred_goal = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    print(f"PREDICTED GOAL: {pred_goal}")
    return {"predicted_goal": pred_goal}


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VLWM-style planning inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # System-1 greedy
  python3 scripts/run_inference.py \\
      --system1_model checkpoints/system1/best_model \\
      --goal "Complete the task: Make Pancakes." \\
      --prefix "pour egg | Egg is now poured in."

  # System-2 with critic reranking
  python3 scripts/run_inference.py --mode system2 \\
      --system1_model checkpoints/system1/best_model \\
      --critic_model  checkpoints/critic/best_model.pt \\
      --goal "Complete the task: Make Pancakes." \\
      --prefix "pour egg | Egg is now poured in." \\
      --K 5 --temperature 0.8

  # Goal prediction
  python3 scripts/run_inference.py --mode goal \\
      --goal_model checkpoints/goal_model/best_model \\
      --prefix "pour egg | Egg is now poured in." \\
               "add flour | Flour is now added."
        """
    )

    # Mode
    parser.add_argument("--mode", choices=["system1", "system2", "goal"],
                        default="system1",
                        help="system1=greedy plan, system2=K plans+reranking, "
                             "goal=predict task goal from observations")

    # Input (free-text)
    parser.add_argument("--goal", type=str, default="",
                        help="Task goal text (e.g. 'Complete the task: Make Pancakes.')")
    parser.add_argument("--prefix", nargs="*", default=None,
                        help="Observed steps, each 'action | state_change'")
    parser.add_argument("--interpretation", type=str, default="",
                        help="Optional task interpretation line")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of future steps to predict")

    # Input (from JSONL)
    parser.add_argument("--from_jsonl", type=str, default=None,
                        help="Load a sample from a test JSONL file instead of --goal/--prefix")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Sample index within the JSONL file")

    # System-1 model
    parser.add_argument("--system1_type", choices=["t5", "plm"], default="t5")
    parser.add_argument("--system1_model", default="checkpoints/system1/best_model")
    parser.add_argument("--plm_base_model", default=None,
                        help="HuggingFace base model for PLM (e.g. facebook/Perception-LM-1B)")

    # Critic model (system2 only)
    parser.add_argument("--critic_type", choices=["mlp", "llm"], default="mlp")
    parser.add_argument("--critic_model", default="checkpoints/critic/best_model.pt")
    parser.add_argument("--llm_base_critic", default=None,
                        help="HuggingFace base model for LLM critic")

    # Goal model (goal mode only)
    parser.add_argument("--goal_model", default="checkpoints/goal_model/best_model")

    # V-JEPA energy (system2 only, optional)
    parser.add_argument("--latent_dir", default=None,
                        help="V-JEPA latent directory (enables energy scoring)")
    parser.add_argument("--goal_latent_model", default=None,
                        help="Path to learned goal-latent checkpoint (.pt). "
                             "If provided, uses learned energy from "
                             "goal + action/state-change trajectory.")
    parser.add_argument("--task_id", default=None,
                        help="CrossTask task ID (needed for energy scoring)")
    parser.add_argument("--video_id", default=None,
                        help="CrossTask video ID (needed for energy scoring)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Critic weight in combined score")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Energy weight in combined score")

    # Vision-grounded interpretation (Option 3 — PLM frames → interpretation)
    parser.add_argument("--frames", type=str, default=None,
                        help="Path to a video file (.mp4/.webm) or directory of "
                             "frame images (.jpg/.png). When provided, PLM's "
                             "vision tower generates the Interpretation: line "
                             "from visual input (overrides --interpretation).")
    parser.add_argument("--interp_model", type=str,
                        default="facebook/Perception-LM-1B",
                        help="HuggingFace model for frame interpretation "
                             "(default: Perception-LM-1B). If --system1_type plm "
                             "and the base model matches, the same model is reused.")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of frames to sample from video "
                             "(only for video files, default: 8)")
    parser.add_argument("--interp_prompt", type=str,
                        default=DEFAULT_INTERP_PROMPT,
                        help="Custom prompt for frame interpretation")

    # Generation params
    parser.add_argument("--K", type=int, default=5,
                        help="Number of candidate plans (system2 only)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=greedy, >0=stochastic)")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    # Output
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save results to a JSON file")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load from JSONL if specified ──
    if args.from_jsonl:
        print(f"Loading sample {args.sample_idx} from {args.from_jsonl}")
        samples = []
        with open(args.from_jsonl) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        if args.sample_idx >= len(samples):
            print(f"ERROR: sample_idx {args.sample_idx} out of range "
                  f"(file has {len(samples)} samples)")
            sys.exit(1)

        sample = samples[args.sample_idx]
        meta = sample.get("meta", {})

        # Parse the input_text to extract goal, interpretation, and prefix
        input_text = sample["input_text"]
        parsed_goal = ""
        parsed_interp = ""
        parsed_prefix = []
        for line in input_text.split("\n"):
            if line.startswith("Goal:"):
                parsed_goal = line[len("Goal:"):].strip()
            elif line.startswith("Interpretation:"):
                parsed_interp = line[len("Interpretation:"):].strip()
            elif line.strip() == "(No steps observed yet.)":
                # Zero-prefix sample — leave parsed_prefix empty
                pass
            elif line.strip() and line.strip()[0].isdigit() and ")" in line:
                step_text = line.strip().split(")", 1)[1].strip()
                parsed_prefix.append(step_text)

        if not args.goal:
            args.goal = parsed_goal
        if not args.interpretation:
            args.interpretation = parsed_interp
        if not args.prefix:
            args.prefix = parsed_prefix
        if not args.k:
            args.k = meta.get("k", 3)
        if not args.task_id:
            args.task_id = str(meta.get("task_id", ""))
        if not args.video_id:
            args.video_id = str(meta.get("video_id", ""))

        # Show gold for comparison
        gold_steps = parse_plan_json(sample.get("output_text", ""))
        if gold_steps:
            print(f"GOLD STEPS ({len(gold_steps)}):")
            for i, s in enumerate(gold_steps, 1):
                print(f"  {i}) {s}")
            print()

    # ── Validate inputs ──
    if args.mode in ("system1", "system2") and not args.goal:
        print("ERROR: --goal is required (or use --from_jsonl)")
        sys.exit(1)
    if args.mode == "goal" and not args.prefix:
        print("ERROR: --prefix is required for goal mode")
        sys.exit(1)

    # ── Vision-grounded interpretation (Option 3) ──
    if args.frames and args.mode in ("system1", "system2"):
        if args.interpretation:
            print("NOTE: --interpretation provided alongside --frames; "
                  "overriding with vision-grounded interpretation.")
        print(f"\nGenerating interpretation from frames: {args.frames}")
        args.interpretation = generate_interpretation_from_frames(
            frames_path=args.frames,
            device=device,
            model_name=args.interp_model,
            num_frames=args.num_frames,
            prompt=args.interp_prompt,
        )
        if args.interpretation:
            print(f"  → Interpretation: {args.interpretation}\n")
        else:
            print("  ⚠️  Interpretation generation returned empty — "
                  "continuing without.\n")

    # ── Dispatch ──
    if args.mode == "system1":
        result = run_system1(args, device)
    elif args.mode == "system2":
        result = run_system2(args, device)
    elif args.mode == "goal":
        result = run_goal(args, device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # ── Save ──
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
