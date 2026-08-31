# Demo Script (2-4 Minutes)

## 0:00-0:30 - Problem and Design

Show the README title and architecture. Explain that ordinary edits break detector shortcuts, so
the promoted model uses frozen CLIP semantics, balanced three-domain fitting, and one individual
corruption per training image. Note that the earlier FFT branch is retained only as an ablation.

## 0:30-1:15 - Live CPU Inference

Open a terminal at the repository root and run:

```powershell
python src/predict.py --input_dir data/smoke-input --out outputs/preds.json --device cpu
Get-Content outputs/preds.json
```

Point out the exact `image_path` + continuous `pred` schema and that the same 47 KB classifier
bundle is device-independent; OpenCLIP weights load separately.

## 1:15-2:10 - Results and Model Selection

Show `outputs/robustness_table.csv` and the promoted model's CIFAKE Final Score 0.927228 across all
14 conditions. Then show `outputs/native_metrics.json`: the semantic model passed SID validation
(0.991900), COCO/DALL-E (0.902925), and LAION/DALL-E matched (0.912700) gates. Use
`outputs/ablation_table.csv` to explain why the higher-scoring historical hybrid was not promoted.

## 2:10-2:55 - Honest Failure Analysis

Open `outputs/error_analysis.md` and show one false positive and one false negative under blur sigma
2.0. Then open the cross-domain section of `docs/technical_report.pdf`: the frozen submitted hybrid
falls to chance AUC on three 400-image SID/WildFake samples and predicts every image as fake, while
its semantic-only component retains useful ranking. Explain that known-transform robustness and
cross-generator transfer are separate objectives.

## 2:55-3:25 - Close

Mention deterministic manifests, restartable caches, GPU/CPU parity, and the main limitation:
CIFAKE is 32x32 and generator-narrow. State that the full optional WildFake set was not evaluated
and the smaller promotion gates do not replace a hidden benchmark. The defensible claim is measured
transform robustness plus gate-tested native transfer, not universal detection.

After recording, upload the video publicly to YouTube and add its URL to the README and Devpost.
