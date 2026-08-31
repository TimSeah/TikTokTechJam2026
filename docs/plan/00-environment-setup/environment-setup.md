# Phase 0 — Environment & Repo Setup

Goal: a working Python environment with all dependencies installed and verified, plus a pushed
repo skeleton, before any model code is written.

## Steps

1. Create a public GitHub repository and clone it locally.
2. Create a virtual environment (`venv` or `conda`).
3. Install core dependencies: `torch`, `torchvision`, `open_clip_torch` (or OpenAI `clip`),
   `scikit-learn`, `Pillow`, `numpy`, `pandas`, `tqdm`, `joblib`.
4. Write a one-off sanity script that loads the CLIP model and embeds a single test image, printing
   the embedding shape.
5. Add a `LICENSE` file (MIT or Apache) — the competition rules require custom code to be released
   under one of these.
6. Set up `.gitignore` (`data/`, `*.npy` embedding caches, `outputs/preds.json`, `venv/`,
   `__pycache__/`). **Do not** gitignore `outputs/probe.joblib`/model weights or
   `outputs/*.md`/`*.csv` reports — the competition rules require winning teams to open-source
   model weights, so trained artifacts must be committed.
7. Create the target folder skeleton: `src/detector/`, `data/`, `outputs/`.
8. Initial commit and push.

## Definition of done

- [ ] `python -c "import torch, open_clip; print('ok')"` (or equivalent for the chosen CLIP
      package) runs without error.
- [ ] The sanity script embeds a sample image and prints an embedding vector of the expected shape
      (e.g. 512-d for ViT-B/32).
- [ ] `LICENSE` file present (MIT or Apache).
- [ ] Repo skeleton exists and is pushed to GitHub with an initial commit.
- [ ] `.gitignore` excludes datasets and large regenerated caches only — trained model weights and
      report files are NOT excluded.

## Time budget

30 minutes.

## Risks

- Package install failures (network/pip resolution, GPU wheel mismatch) — install the CPU-only
  `torch` wheel explicitly if the default install fails or if no local GPU/CUDA-equivalent is
  available.
