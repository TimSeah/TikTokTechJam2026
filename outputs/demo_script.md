# Demo Script (2-4 Minutes)

## 0:00-0:30 - Problem and Design

Show the README title and architecture. Explain that ordinary edits break detector shortcuts, so
the model combines frozen CLIP semantics, radial FFT evidence, and one individual corruption per
training image.

## 0:30-1:15 - Live CPU Inference

Open a terminal at the repository root and run:

```powershell
python src/predict.py --input_dir data/smoke-input --out outputs/preds.json --device cpu
Get-Content outputs/preds.json
```

Point out the exact `image_path` + continuous `pred` schema and that the same 47 KB classifier
bundle is device-independent; OpenCLIP weights load separately.

## 1:15-2:10 - Results and Ablation

Show `outputs/ablation_table.csv`. The clean hybrid has the best clean AUC, but augmentation raises
robust AUC from 0.910941 to 0.956211. Show `outputs/robustness_table.csv` and the headline Final
Score 0.972310 across all 14 conditions.

## 2:10-2:50 - Honest Failure Analysis

Open `outputs/error_analysis.md` and show one false positive and one false negative under blur sigma
2.0. Explain that heavy blur destroys texture/frequency evidence and leaves ambiguous coarse shapes.

## 2:50-3:15 - Close

Mention deterministic manifests, restartable caches, GPU/CPU parity, and the main limitation:
CIFAKE is 32x32 and generator-narrow. State that the exact optional WildFake set was not evaluated,
so the project claims transform robustness, not broad cross-generator generalization.

After recording, upload the video publicly to YouTube and add its URL to the README and Devpost.