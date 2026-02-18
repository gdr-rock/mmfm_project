# Data Layout

Place datasets under this directory. No proprietary data is included.

Expected structure:

- `data/crosstask/`
  - `videos/`
  - `annotations/`
  - `splits/train.txt`
  - `splits/val.txt`
  - `splits/test.txt`
- `data/coin/`
  - `videos/`
  - `COIN.json`
  - `splits/train.txt`
  - `splits/val.txt`
  - `splits/test.txt`

Generated artifacts:

- Cached latents: `outputs/cache/`
- Processed manifests: `outputs/data/`
- Evaluation reports: `outputs/eval/`
