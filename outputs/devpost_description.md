# Devpost Written Project Description

## Inspiration

AI-image detectors often look strong on untouched benchmark files and fail after ordinary social
media edits. This project asks a stricter question: can a small, reproducible detector keep ranking
AI-generated images correctly after compression, blur, resizing, noise, color changes, or cropping?

## What It Does

Real or Fake? accepts a directory of images and returns a JSON list containing each path and a
continuous AI-generated confidence score. The same detector runs on AMD ROCm, CUDA, or CPU.

## How It Works

The promoted `semantic_native_mixed` model standardizes a normalized 512-dimensional frozen
OpenCLIP ViT-B/32 embedding and fits a logistic-regression head. Training uses 4,000 REAL and 4,000
FAKE images from each of CIFAKE, SID-Set, and allowed WildFake groups. Each of the 24,000 source
images supplies one clean view and one seeded, individually transformed view, for 48,000 fitting
rows. Transform families are sampled approximately uniformly and never stacked.

The earlier hybrid with a 32-bin radial log-FFT profile remains as an ablation. Its blind-test
failure showed that a frequency feature that works on CIFAKE can collapse on native-resolution
images. Evaluation uses 20,000 held-out CIFAKE images and all 14 required transform/severity
conditions. Each condition is applied separately and all models reuse the same cached features.

## Results

The final model reaches clean AUC 0.957820, condition-weighted robust AUC 0.896636, and the workshop
composite Final Score 0.927228. A family-balanced sensitivity calculation gives a robust AUC of
0.900314. Heavy blur, 0.25x resizing, and high noise remain the clearest CIFAKE failure modes.

The full WildFake COCO val2017 + DALL-E Advanced benchmark was not run because its required 26 GB
source archive exceeded the optional acquisition time gate. After freezing the artifact, smaller
balanced SID and WildFake diagnostics revealed the central negative result: the submitted hybrid
ranked at chance and predicted every native-resolution sample as fake, while its semantic-only
component retained useful ranking. This motivated the final three-domain semantic model, which
passed held-out SID validation AUC 0.991900, COCO/DALL-E AUC 0.902925, and LAION/DALL-E matched AUC
0.912700 promotion gates.

## How It Is Different

A frozen-CLIP linear probe is only the baseline. The final approach adds balanced multi-domain
fitting, deterministic one-corruption-per-image training, disjoint threshold calibration, and
promotion gates that detect score collapse. The ablation documents why frequency fusion was removed
from the deployed detector.

## Built With

Python 3.12, PyTorch 2.9.1 with ROCm 7.2.1, OpenCLIP 3.3.0, scikit-learn, NumPy, Pillow, joblib,
Kaggle CIFAKE, VS Code, Git, and GitHub. Training and evaluation ran locally on an AMD Radeon RX
7900 XTX and Ryzen 7 7800X3D.

## Challenges and Lessons

Native Windows ROCm required a custom target-family workaround for a workspace path containing
spaces. Restartable feature shards made the expensive extraction safe to resume. The ablations and
native diagnostics also showed a useful negative result: frequency features can improve the CIFAKE
score while making the detector unsafe on native-resolution inputs.

## What's Next

Add a larger source-disjoint hidden evaluation, more generator-diverse native-resolution training
data, deployment-prevalence calibration, compound-transform tests, and fairness analysis. Test any
future frequency feature behind the same anti-collapse promotion gates.
