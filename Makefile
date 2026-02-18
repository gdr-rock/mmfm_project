PYTHON ?= python

.PHONY: format lint test smoke prepare_dataset cache_latents train_critic train_bridges evaluate

format:
	$(PYTHON) -m black src scripts tests

lint:
	$(PYTHON) -m ruff check src scripts tests

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) scripts/00_smoke_test.py --use_dummy_data --dry_run

prepare_dataset:
	$(PYTHON) scripts/01_prepare_dataset.py --dataset crosstask --config configs/dataset_crosstask.yaml --output_dir outputs --use_dummy_data

cache_latents:
	$(PYTHON) scripts/02_cache_jepa_latents.py --dataset crosstask --config configs/models.yaml --output_dir outputs --use_dummy_data

train_critic:
	$(PYTHON) scripts/05_train_text_critic.py --dataset crosstask --config configs/models.yaml --output_dir outputs --use_dummy_data

train_bridges:
	$(PYTHON) scripts/06_train_bridge_transition.py --dataset crosstask --config configs/models.yaml --output_dir outputs --use_dummy_data
	$(PYTHON) scripts/07_train_bridge_goal.py --dataset crosstask --config configs/models.yaml --output_dir outputs --use_dummy_data

evaluate:
	$(PYTHON) scripts/09_eval_vpa.py --dataset crosstask --config configs/eval.yaml --output_dir outputs --use_dummy_data
	$(PYTHON) scripts/10_eval_consistency_retrieval.py --dataset crosstask --config configs/eval.yaml --output_dir outputs --use_dummy_data
	$(PYTHON) scripts/11_eval_robustness.py --dataset crosstask --config configs/eval.yaml --output_dir outputs --use_dummy_data
