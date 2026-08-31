# Devpost Written Project Description

## Inspiration

AI-image detectors often look strong on untouched benchmark files and fail after ordinary social
media edits. This project asks a stricter question: can a small, reproducible detector keep ranking
AI-generated images correctly after compression, blur, resizing, noise, color changes, or cropping?

## What It Does

Real or Fake? accepts a directory of images and returns a JSON list containing each path and a
continuous AI-generated confidence score. The same detector runs on AMD ROCm, CUDA, or CPU.

## How It Works

The model fuses two complementary signals. Frozen OpenCLIP ViT-B/32 provides a 512-dimensional
semantic embedding, while a fixed 32-bin radial log-FFT profile captures frequency structure. A
scaled logistic-regression head is trained on all 100,000 CIFAKE training images plus one seeded,
individually transformed copy of every image. Transform families are sampled approximately
uniformly and never stacked.

Three controlled ablations separate semantic features, frequency fusion, and robustness-aware
training. Evaluation uses 20,000 held-out CIFAKE images and all 14 required transform/severity
conditions. Each condition is applied separately and all models reuse the same cached features.

## Results

The final model reaches clean AUC 0.988409, condition-weighted robust AUC 0.956211, and the workshop
composite Final Score 0.972310. A family-balanced sensitivity calculation gives 0.973004. Augmented
training sacrifices 0.003351 clean AUC versus the clean hybrid but gains 0.045270 robust AUC. Heavy
blur and 0.25x resizing remain the clearest failure modes.

The exact WildFake COCO val2017 + DALL-E Advanced benchmark was not run because its required 26 GB
source archive exceeded the optional acquisition time gate. No substitute result is presented.

## How It Is Different

A frozen-CLIP linear probe is only the baseline. The submitted approach adds an explicit
frequency-domain signal and a deterministic one-corruption-per-image training protocol. The
ablation table shows that frequency fusion helps clean discrimination, while augmentation is what
delivers the large robustness gain.

## Built With

Python 3.12, PyTorch 2.9.1 with ROCm 7.2.1, OpenCLIP 3.3.0, scikit-learn, NumPy, Pillow, joblib,
Kaggle CIFAKE, VS Code, Git, and GitHub. Training and evaluation ran locally on an AMD Radeon RX
7900 XTX and Ryzen 7 7800X3D.

## Challenges and Lessons

Native Windows ROCm required a custom target-family workaround for a workspace path containing
spaces. Restartable feature shards made the expensive extraction safe to resume. The ablations also
showed a useful negative result: adding frequency features without transform-aware training made
robustness worse, despite improving clean AUC.

## What's Next

Evaluate the exact WildFake reference set, add generator-diverse and higher-resolution training
data, calibrate probabilities on a separate deployment-like validation set, and test learned
frequency representations without losing the current CPU-portable inference path.