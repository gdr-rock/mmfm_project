# Minimal CrossTask Automation (3 Things Only)

This repo is now focused on exactly your 3 steps:
1. Get perception encoder/caption models.
2. Run PerceptionLM detailed video captions and build hierarchical caption tree.
3. Repeat automatically for `N` videos (download one-by-one and delete after each run).

## 1) Install framework

```bash
sudo apt update
sudo apt install -y ffmpeg
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## 2) Get models

### 2.1 Encoder checkpoint
```bash
python3 scripts/01_download_model.py \
  --url "<ENCODER_CHECKPOINT_URL>" \
  --output checkpoints/perception_encoder.pt
```

### 2.2 PerceptionLM caption model
```bash
python3 scripts/01_download_model.py \
  --use_hf \
  --hf_repo_id "<PERCEPTIONLM_HF_REPO>" \
  --target_dir checkpoints/perceptionlm
```

If private HF repo:
```bash
huggingface-cli login
```

## 3) Prepare URL list

Edit `data/crosstask/video_urls.txt`:
```text
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
https://www.youtube.com/watch?v=VIDEO_ID_3
https://www.youtube.com/watch?v=VIDEO_ID_4
https://www.youtube.com/watch?v=VIDEO_ID_5
```

## 4) Run automatic cycle for N videos

### Captions + hierarchical tree first
```bash
python3 scripts/13_stream_crosstask_pipeline.py \
  --mode captions_only \
  --url_list data/crosstask/video_urls.txt \
  --max_videos 5 \
  --caption_model_path checkpoints/perceptionlm \
  --strict_caption_model \
  --num_caption_steps 12 \
  --output_dir outputs/caption_tree_runs
```

### Latents later
```bash
python3 scripts/13_stream_crosstask_pipeline.py \
  --mode latents_only \
  --url_list data/crosstask/video_urls.txt \
  --max_videos 5 \
  --encoder_checkpoint checkpoints/perception_encoder.pt \
  --output_dir outputs/latent_runs
```

### Both together (optional)
```bash
python3 scripts/13_stream_crosstask_pipeline.py \
  --mode both \
  --url_list data/crosstask/video_urls.txt \
  --max_videos 5 \
  --caption_model_path checkpoints/perceptionlm \
  --encoder_checkpoint checkpoints/perception_encoder.pt \
  --strict_caption_model \
  --output_dir outputs/full_runs
```

## Important behavior

- One video is downloaded, processed, then deleted by default.
- Use `--keep_downloaded` to keep local video files.
- Tree output per video: `outputs/.../<video_key>/plans/caption_tree_crosstask.json`
- Raw detailed caption text per video: `outputs/.../<video_key>/plans/detailed_captions.txt`
- Summary file: `outputs/.../streaming_summary.json`
