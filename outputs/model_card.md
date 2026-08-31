# Model Card: Real or Fake?

## Summary

`hybrid_augmented` is a binary AI-generated image detector. It concatenates a normalized frozen
OpenCLIP `ViT-B-32-quickgelu` (`openai`) image embedding with a fixed 32-bin radial log-FFT feature,
standardizes the 544 values, and predicts `P(FAKE)` with logistic regression.

## Training Data and Protocol

- CIFAKE published train split: 50,000 REAL + 50,000 FAKE, all 32x32 RGB JPEG.
- Seed: 2026. Manifest SHA-256 is stored in `outputs/model.joblib`.
- Training rows: every clean image plus one deterministic transformed copy per image.
- One transform is applied individually, never stacked. Families are sampled approximately
  uniformly: JPEG, Gaussian blur, down/up-resize, Gaussian noise, color jitter, and center crop.
- CLIP is frozen. The final logistic-regression head has 544 coefficients plus one intercept.
- OpenCLIP backbone parameter count: 151,277,313, below the 2B competition limit.
- Verified warm GPU benchmark: approximately 500 images/second at batch 256 on an AMD RX 7900 XTX.
- Measured fresh-process inference for 100 images: 15.504 seconds on GPU and 46.260 seconds on CPU,
  including model loading and feature extraction.

## Held-Out CIFAKE Results

| Model | Clean AUC | Robust AUC | Final Score |
| --- | ---: | ---: | ---: |
| Semantic, clean training | 0.988755 | 0.917770 | 0.953263 |
| Semantic + FFT, clean training | 0.991760 | 0.910941 | 0.951351 |
| **Semantic + FFT, clean + augmented training** | **0.988409** | **0.956211** | **0.972310** |

Robust AUC is the arithmetic mean of all 14 transform/severity AUC rows. A family-balanced
sensitivity calculation gives robust AUC 0.957598 and Final Score 0.973004. Accuracy and F1 use a
fixed 0.5 threshold only for interpretation; AUC and submitted predictions are continuous.

## Intended Use

This is a hackathon prototype for ranking likely AI-generated images and studying transform
robustness. It is not suitable as sole evidence for moderation, attribution, copyright, fraud, or
disciplinary decisions.

## Limitations

CIFAKE is low-resolution and generator-narrow. Heavy blur sigma 2.0 and resize 0.25 reduce AUC to
0.886651 and 0.882709 respectively. The exact WildFake COCO val2017 + DALL-E Advanced reference
set was not evaluated because its required 26 GB source archive exceeded the optional acquisition
gate. Scores therefore establish in-distribution transform robustness, not cross-generator
generalization or probability calibration on real deployments.

## Non-Replication Note

A plain frozen-CLIP linear probe is included only as an ablation. The submitted detector changes the
feature space through explicit frequency fusion and changes training through a deterministic,
family-balanced, one-corruption-per-image augmentation protocol. The results report where each
choice helps and hurts.