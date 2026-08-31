# Devpost Written Project Description

## Inspiration

AI-image detectors often look strong on untouched benchmark files and fail after ordinary social
media edits. This project asks a stricter question: can a small, reproducible detector keep ranking
AI-generated images correctly after compression, blur, resizing, noise, color changes, or cropping?

## What It Does

Real or Fake? has two ways to use the detector:

- A reproducible CLI accepts a directory of images and returns a JSON list containing each path and
  a continuous AI-generated confidence score.
- A live ten-round Human vs Machine game lets a player classify the same transformed images as the
  detector, then compares accuracy and response time.

The game shows the applied transformation before every decision. After each guess, it reveals the
ground truth, the player's result, the detector's label, fake confidence, and inference time. The
React frontend is deployed on Cloudflare Pages, and a FastAPI backend runs the current OpenCLIP
detector on a Modal T4 GPU. The deployed game uses an 800-image balanced challenge pool: 400 SID-Set
and 400 WildFake evaluation samples, with 400 REAL and 400 FAKE images overall.

## How It Works

The promoted `semantic_native_mixed` model standardizes a normalized 512-dimensional frozen
OpenCLIP ViT-B/32 embedding and fits a logistic-regression head. Training uses 4,000 REAL and 4,000
FAKE images from each of CIFAKE, SID-Set, and allowed WildFake groups. Each of the 24,000 source
images supplies one clean view and one seeded, individually transformed view, for 48,000 fitting
rows. Transform families are sampled approximately uniformly and never stacked.

For the live challenge, the backend chooses one of six representative transformations per round:
JPEG compression, Gaussian blur, downscaling, Gaussian noise, color jitter, or center cropping. A
stored seed makes the transformation deterministic, so the detector scores exactly the same pixels
served to the player. The browser receives only the transformed PNG and augmentation label; ground
truth and model output remain server-side until the player submits a guess.

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
from the deployed detector. The interactive game also turns the robustness protocol into something
visible: players see which real-world edit was applied and compete against the detector on the same
pixels instead of viewing a precomputed benchmark score.

## Built With

Python 3.12, PyTorch 2.9.1 with ROCm 7.2.1, OpenCLIP 3.3.0, scikit-learn, NumPy, Pillow, joblib,
FastAPI, React 19, TypeScript, Vite, Modal, Cloudflare Pages, Kaggle CIFAKE, SID-Set, WildFake, VS
Code, Git, and GitHub. Training and evaluation ran locally on an AMD Radeon RX 7900 XTX and Ryzen 7
7800X3D; the production detector runs with CUDA on a Modal T4.

## Challenges and Lessons

Native Windows ROCm required a custom target-family workaround for a workspace path containing
spaces. Restartable feature shards made the expensive extraction safe to resume. The ablations and
native diagnostics also showed a useful negative result: frequency features can improve the CIFAKE
score while making the detector unsafe on native-resolution inputs. Deploying the interactive game
added another reproducibility constraint: each transformed image had to be regenerated from the
same seed for inference and delivery, while round labels and predictions remained private until the
guess was submitted.

## What's Next

Add a larger source-disjoint hidden evaluation, more generator-diverse native-resolution training
data, deployment-prevalence calibration, compound-transform tests, and fairness analysis. Expand
the game with opt-in aggregate player statistics and test any future frequency feature behind the
same anti-collapse promotion gates.

## Live Demo

Play the [deployed game](https://real-vs-ai.pages.dev/).
