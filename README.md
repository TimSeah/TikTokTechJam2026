# Real or Fake?

A lightweight detector for AI-generated images under JPEG compression, blur, resizing, noise,
color jitter, and cropping. It fuses frozen CLIP ViT-B/32 semantic embeddings with a fixed 32-bin
radial log-FFT branch, then fits a scaled logistic-regression classifier on clean and individually
augmented CIFAKE images.

Built for **TikTok TechJam 2026, Track 5: Robust Detection of AI-Generated Images Under Real-World
Transformations**.

## Results

The held-out CIFAKE test split contains 20,000 images. Robust AUC is the mean across all 14
individually applied transform/severity conditions. The organizer workshop did not define severity
aggregation, so a family-balanced sensitivity score is also reported.

| Metric | Score |
| --- | ---: |
| Clean AUC | 0.988409 |
| Clean accuracy / F1 at 0.5 | 0.946750 / 0.947112 |
| Condition-weighted robust AUC | 0.956211 |
| Family-balanced robust AUC | 0.957598 |
| **Final Score**: 0.5 clean + 0.5 condition-weighted robust | **0.972310** |
| Family-balanced sensitivity score | 0.973004 |
| Cross-generator AUC | Not run; exact 26 GB WildFake source exceeded the optional data gate |
| Backbone parameters | 151,277,313 frozen + 545 final linear coefficients/intercept |

| Ablation | Clean AUC | Robust AUC | Final Score |
| --- | ---: | ---: | ---: |
| Semantic, clean training | 0.988755 | 0.917770 | 0.953263 |
| Semantic + FFT, clean training | **0.991760** | 0.910941 | 0.951351 |
| Semantic + FFT, clean + augmented training | 0.988409 | **0.956211** | **0.972310** |

Frequency fusion improves clean AUC, but is fragile under degradation without augmented training.
Adding one seeded corruption per training image costs 0.00335 clean AUC versus the clean hybrid and
gains 0.04527 robust AUC. Heavy blur sigma 2.0 (AUC 0.886651) and 0.25x resizing (AUC 0.882709) are
the weakest conditions.

Full tables: [outputs/robustness_table.csv](outputs/robustness_table.csv) and
[outputs/ablation_table.csv](outputs/ablation_table.csv).

## Approach

Each image produces two features: a normalized 512-dimensional CLIP vector and a 32-dimensional
radially averaged log-FFT magnitude vector. Three `StandardScaler + LogisticRegression` pipelines
provide controlled ablations. The submitted `hybrid_augmented` model trains on all 100,000 clean
CIFAKE training images and one deterministic, individually transformed copy of each image.

### Why this isn't a direct replication

A bare frozen-CLIP linear probe is essentially the published "Universal Fake Image Detectors"
method unmodified. Fusing in a frequency-domain branch and training on augmented data (rather than
only evaluating robustness after the fact) changes both the signal space and the training protocol.
The ablation above reports each contribution rather than assuming it helps.

## Setup

### Windows 11 with AMD Radeon RX 7900 XTX

Requires Python 3.12 and AMD Software: Adrenalin Edition 26.2.2 or newer. The environment below has
been validated locally with Adrenalin 26.6.4, PyTorch 2.9.1, and ROCm 7.2.1.

```powershell
git clone https://github.com/TimSeah/TikTokTechJam2026.git
cd TikTokTechJam2026
uv python install 3.12
uv venv --python 3.12 .venv-amd
.\.venv-amd\Scripts\Activate.ps1
$env:ROCM_SDK_TARGET_FAMILY = "custom"
uv pip install -r requirements-amd.txt
python scripts/check_environment.py --require-gpu --benchmark-clip
```

### Colab, CUDA, or CPU

Use a Python 3.12 environment. Colab already supplies a compatible CUDA build of PyTorch.

```bash
pip install -r requirements.txt
python scripts/check_environment.py --benchmark-clip --batch-size 1 --device cpu
```

Datasets are not committed to the repo; see
[docs/plan/01-data-acquisition/data-acquisition.md](docs/plan/01-data-acquisition/data-acquisition.md)
and [data/README.md](data/README.md). Download CIFAKE with the authenticated Kaggle CLI:

```powershell
.\.venv-amd\Scripts\kaggle.exe datasets download `
  -d birdy654/cifake-real-and-ai-generated-synthetic-images `
  -p data/downloads --unzip
```

## Reproduce

Run from the repository root. On native AMD, set
`$env:ROCM_SDK_TARGET_FAMILY = "custom"` in each new terminal.

```powershell
# 1. Deterministic manifests
python -m src.detector.data --data-root data/downloads --output-dir data/manifests `
  --validate-sample 1000

# 2. Restartable training caches
python -m src.detector.embed --manifest data/manifests/train.csv `
  --data-root data/downloads --output-dir data/features/train-clean `
  --condition clean --device auto --batch-size 512 --workers 8
python -m src.detector.embed --manifest data/manifests/train.csv `
  --data-root data/downloads --output-dir data/features/train-augmented `
  --condition augmented --device auto --batch-size 512 --workers 8
python -m src.detector.embed --manifest data/manifests/test.csv `
  --data-root data/downloads --output-dir data/features/test-clean `
  --condition clean --device auto --batch-size 512 --workers 8

# 3. Train all ablations and save the CPU-loadable bundle
python -m src.detector.train_probe `
  --clean-train-cache data/features/train-clean `
  --augmented-train-cache data/features/train-augmented `
  --clean-eval-cache data/features/test-clean `
  --output outputs/model.joblib --metrics-out outputs/clean_metrics.json

# 4. Extract the 14 transformed test caches
$conditions = @(
  'jpeg_q90', 'jpeg_q70', 'jpeg_q50', 'jpeg_q30',
  'blur_sigma0.5', 'blur_sigma1.0', 'blur_sigma2.0',
  'resize_scale0.5', 'resize_scale0.25',
  'noise_sigma0.02', 'noise_sigma0.05', 'noise_sigma0.10',
  'jitter_amount0.20', 'crop_ratio0.80'
)
foreach ($condition in $conditions) {
  python -m src.detector.embed --manifest data/manifests/test.csv `
    --data-root data/downloads --output-dir "data/features/test-$condition" `
    --condition $condition --device auto --batch-size 512 --workers 8
}

# 5. Calculate clean, robustness, ablation, and composite-score tables
python -m src.detector.evaluate --model outputs/model.joblib `
  --features-root data/features --output-dir outputs
```

For a quick integration check, add `--limit 200` to feature extraction. Feature shards are written
atomically and existing shards are skipped on restart.

## Inference

```powershell
python src/predict.py --input_dir path/to/images --out preds.json --device auto
python src/predict.py --input_dir path/to/images --out preds.json --device cpu
```

The CLI recursively reads `.jpg`, `.jpeg`, `.png`, and `.webp`, skips corrupt files, and writes a
deterministically ordered JSON list. Its default judge-facing schema is exactly:

```json
[{"image_path": "example.jpg", "pred": 0.731}]
```

`pred` is continuous `P(FAKE)`. `--include_label` optionally adds a human-readable label at the
documented 0.5 threshold. The final bundle was validated in fresh GPU and CPU processes; the two
smoke-test probabilities agreed within 4.6e-9. End-to-end fresh-process inference over 100 held-out
images took 15.504 seconds on the RX 7900 XTX and 46.260 seconds on the Ryzen 7 7800X3D, including
model loading and feature extraction.

## Interactive Demo

The React + FastAPI challenge compares a player's label and response time with the final detector
over ten rounds. It defaults to `outputs/model.joblib` and the local CIFAKE test directory.

```powershell
Set-Location webapp/frontend
npm ci
npm run build
Set-Location ../..
python -m uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The production build, `/api/status`, image delivery, detector-backed
guess endpoint, and a complete game round were verified locally. Use another port if 8000 is
already occupied.

## Repository Layout

```text
LICENSE
data/                        # gitignored datasets, manifests, feature caches
src/
  detector/
    data.py
    transforms.py
    freq_features.py
    features.py
    embed.py
    model.py
    train_probe.py
    evaluate.py
    analyze_errors.py
  predict.py                 # image directory -> JSON confidence scores
webapp/                      # React + FastAPI human-versus-detector challenge
outputs/
  model.joblib               # committed CPU-loadable inference bundle
  model_card.md
  robustness_table.csv
  ablation_table.csv
  error_analysis.md
  trade_offs.md
requirements.txt
```

## Limitations

- CIFAKE is only 32x32 and uses one generated-image process, so strong in-distribution AUC does not
  establish broad modern-generator detection.
- Cross-generator WildFake evaluation was not run; no substitute result is presented.
- The fixed FFT branch helps clean AUC but amplifies dataset-specific low-level cues and degrades
  sharply under blur and resizing unless augmentation is used.
- Probabilities are useful for ranking but are not calibrated for every deployment distribution.
- With more time, evaluate the exact held-out WildFake subset, add generator-diverse training data,
  learn calibration on a separate validation set, and test higher-resolution inputs.

## Team

Solo project by Timothy Seah: data pipeline, detector, robustness evaluation, inference CLI,
analysis, and documentation.

## Documentation

- [docs/problem_statement.md](docs/problem_statement.md): full hackathon problem statement (all
  tracks), plus supplementary workshop notes for Track 5 in §5.7.
- [docs/plan/README.md](docs/plan/README.md): chosen approach, timeline, target repo layout, and
  per-phase execution plans.
- [outputs/model_card.md](outputs/model_card.md): model details and measured limitations.
- [outputs/error_analysis.md](outputs/error_analysis.md): four high-confidence heavy-blur errors.
- [outputs/trade_offs.md](outputs/trade_offs.md): clean/robustness and feasibility discussion.

## License

MIT, see [LICENSE](LICENSE). CIFAKE and CLIP retain their respective upstream terms.
