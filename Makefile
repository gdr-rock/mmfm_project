PYTHON ?= python3

.PHONY: install prepare_urls prepare_urls_official download_encoder download_caption_model run_captions run_latents run_both run_smoke

install:
	$(PYTHON) -m pip install -r requirements.txt

prepare_urls:
	$(PYTHON) scripts/02_prepare_video_urls.py --input data/crosstask/video_urls.raw.txt --output data/crosstask/video_urls.txt --max_videos 5

prepare_urls_official:
	$(PYTHON) scripts/03_prepare_official_crosstask_urls.py --max_videos 5 --output_url_list data/crosstask/video_urls.txt

download_encoder:
	$(PYTHON) scripts/01_download_model.py --url "<ENCODER_CHECKPOINT_URL>" --output checkpoints/perception_encoder.pt

download_caption_model:
	$(PYTHON) scripts/01_download_model.py --use_hf --hf_repo_id "<PERCEPTIONLM_HF_REPO>" --target_dir checkpoints/perceptionlm

run_captions:
	$(PYTHON) scripts/13_stream_crosstask_pipeline.py --mode captions_only --url_list data/crosstask/video_urls.txt --max_videos 5 --caption_model_path checkpoints/perceptionlm --strict_caption_model --num_caption_steps 12 --output_dir outputs/caption_tree_runs

run_latents:
	$(PYTHON) scripts/13_stream_crosstask_pipeline.py --mode latents_only --url_list data/crosstask/video_urls.txt --max_videos 5 --encoder_checkpoint checkpoints/perception_encoder.pt --output_dir outputs/latent_runs

run_both:
	$(PYTHON) scripts/13_stream_crosstask_pipeline.py --mode both --url_list data/crosstask/video_urls.txt --max_videos 5 --caption_model_path checkpoints/perceptionlm --encoder_checkpoint checkpoints/perception_encoder.pt --strict_caption_model --output_dir outputs/full_runs

run_smoke:
	$(PYTHON) scripts/13_stream_crosstask_pipeline.py --mode captions_only --url_list data/crosstask/video_urls.txt --max_videos 2 --output_dir outputs/smoke_runs --skip_download
