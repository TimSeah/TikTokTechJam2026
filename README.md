# Real or Fake?

Detects AI-generated images that survive real-world editing, like compression, blurring, cropping,
and resizing, using a hybrid CLIP + frequency-domain model trained specifically for robustness, not
just clean-lab accuracy.

Built for **TikTok TechJam 2026, Track 5: Robust Detection of AI-Generated Images Under Real-World
Transformations**.

## Status

🚧 In Progress

## Approach

A hybrid detector: frozen CLIP (ViT-B/32) semantic embeddings fused with a lightweight, non-learned
frequency-domain (FFT/DCT) feature branch, trained on CIFAKE with both clean and randomly-augmented
copies of each image (JPEG compression, Gaussian blur, resize, Gaussian noise, color jitter, center
crop), so robustness is learned during training rather than only measured afterward. A held-out,
unseen-generator subset (WildFake-derived COCO val2017 + DALL·E Advanced) is used only to check
cross-generator generalization, never for training.

### Why this isn't a direct replication

A bare frozen-CLIP linear probe is essentially the published "Universal Fake Image Detectors"
method unmodified. Fusing in a frequency-domain branch and training on augmented data (rather than
only evaluating robustness after the fact) makes this pipeline my own design, while still using an
explicitly whitelisted pretrained backbone (CLIP).

## Results

| Metric | Score |
| --- | --- |
| AUC (clean) | TBD |
| AUC (robust, averaged across all 6 transform families) | TBD |
| **Final Score** (0.50 × AUC_clean + 0.50 × AUC_robust) | TBD |
| Cross-generator AUC (unseen-generator validation set) | TBD |
| Parameter count | TBD (< 2B) |

See [outputs/robustness_table.csv](outputs/robustness_table.csv) for the full per-transform,
per-severity breakdown.

## Setup and installation

```bash
git clone <this-repo-url>
cd TikTokTechJam2026
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Datasets (CIFAKE + the validation-only WildFake subset) are not committed to the repo; see
[docs/plan/01-data-acquisition/data-acquisition.md](docs/plan/01-data-acquisition/data-acquisition.md)
for download instructions and expected folder layout under `data/`.

## Steps to reproduce results

```bash
# 1. Build cached fused (CLIP + frequency) embeddings for clean + augmented training data
python src/detector/embed.py

# 2. Train the classifier head
python src/detector/train_probe.py

# 3. Evaluate clean/robust AUC, the composite Final Score, and cross-generator AUC
python src/detector/evaluate.py

# 4. Run inference on any folder of images
python src/predict.py <image_dir> --out preds.json
```

`predict.py` writes a JSON list of `{"image_path": ..., "pred": ...}` objects, where `pred` is a
continuous AI-generated confidence score (not a hard label).

## Repository layout

```
LICENSE
data/                        # gitignored, CIFAKE + validation-only downloads
src/
  detector/
    transforms.py
    freq_features.py
    embed.py
    train_probe.py
    evaluate.py
  predict.py                  # image dir -> JSON of {image_path, pred}
outputs/
  probe.joblib                # committed trained classifier head
  model_card.md
  robustness_table.csv
  error_analysis.md
  trade_offs.md
requirements.txt
```

## Limitations & what I'd improve with more time

TBD

## Team

Solo project: all components (data pipeline, model, evaluation, docs) built by TBD.

## Documentation

- [docs/problem_statement.md](docs/problem_statement.md): full hackathon problem statement (all
  tracks), plus supplementary workshop notes for Track 5 in §5.7.
- [docs/plan/README.md](docs/plan/README.md): chosen approach, timeline, target repo layout, and
  per-phase execution plans.

## License

TBD (MIT or Apache-2.0, per competition rules), see [LICENSE](LICENSE).
