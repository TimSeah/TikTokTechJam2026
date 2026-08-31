# Model Card: Real or Fake?

## Summary

`semantic_native_mixed` is a binary AI-generated image detector. It standardizes a normalized
512-dimensional frozen OpenCLIP `ViT-B-32-quickgelu` (`openai`) image embedding and predicts
`P(FAKE)` with logistic regression. The semantic-only artifact bypasses the FFT branch that caused
the previous hybrid model to collapse on native-resolution images.

## Training Data and Protocol

- Balanced fitting sample per domain: 4,000 REAL + 4,000 FAKE from CIFAKE, SID-Set, and WildFake.
- SID-Set revision: `dc03ead57929879319ce30a82bfcfb8d317b10bd`.
- WildFake source revision: `3c4a1d3824e593167f9b6f682c079c3c17516214`.
- WildFake groups: ADM, DDPM, GALIP, GigaGAN, VQGAN, VQVAE, AFHQ, CelebA-HQ, and LSUN-Church.
- COCO val2017, DALL-E Advanced, and all frozen evaluation IDs/hashes were excluded from fitting.
- Training rows: 48,000, comprising one clean and one transformed view of 24,000 source images.
- One transform is applied individually, never stacked. Families are sampled approximately
  uniformly: JPEG, Gaussian blur, down/up-resize, Gaussian noise, color jitter, and center crop.
- The decision threshold `0.7819586396` was calibrated on a disjoint 500 REAL + 500 FAKE SID split.
- CLIP is frozen. The final logistic-regression head has 512 coefficients plus one intercept.
- OpenCLIP backbone parameter count: 151,277,313, below the 2B competition limit.
- Feature extraction, fitting, gates, and promotion took 547.063 seconds on an AMD RX 7900 XTX,
  under a hard 3,600-second wall-clock budget.

## Held-Out CIFAKE Results

| Model | Clean AUC | Robust AUC | Final Score |
| --- | ---: | ---: | ---: |
| **Semantic, three domains, clean + augmented** | **0.957820** | **0.896636** | **0.927228** |

Robust AUC is the arithmetic mean of all 14 transform/severity AUC rows. A family-balanced
sensitivity calculation gives robust AUC 0.900314 and Final Score 0.929067. AUC and submitted
predictions are continuous.

## Native Promotion Gates

| Gate | ROC AUC | Predicted FAKE rate | REAL / FAKE median |
| --- | ---: | ---: | ---: |
| SID calibration | 0.993972 | 0.5130 | 0.0073 / 0.9990 |
| SID validation | 0.991900 | 0.5075 | 0.0178 / 0.9993 |
| WildFake COCO/DALL-E | 0.902925 | 0.5275 | 0.1650 / 0.9891 |
| WildFake LAION/DALL-E matched | 0.912700 | 0.4050 | 0.0075 / 0.9333 |

Every gate passed its minimum AUC, minimum score spread, 5%-95% predicted-class-rate bounds, and
FAKE-median-above-REAL check before promotion.

## Intended Use

This is a hackathon prototype for ranking likely AI-generated images and studying transform
robustness. It is not suitable as sole evidence for moderation, attribution, copyright, fraud, or
disciplinary decisions.

## Limitations

CIFAKE is low-resolution and generator-narrow. Heavy blur sigma 2.0, resize 0.25, and noise sigma
0.10 reduce AUC to 0.785540, 0.774426, and 0.797451 respectively. The native gates contain only 400
images each and cannot establish universal cross-generator generalization. WildFake fitting covers
nine allowed groups but unseen generators, edits, and capture pipelines can still shift the score
distribution. Scores are not calibrated to deployment prevalence. See
[cross_domain_summary.json](cross_domain_summary.json) for the original hybrid failure diagnostic.

## Non-Replication Note

A plain frozen-CLIP linear probe is included only as an earlier ablation. This detector adds
balanced multi-domain fitting, deterministic family-balanced augmentation, disjoint threshold
calibration, frozen-data exclusions, and anti-collapse promotion gates. The iteration history
reports why the earlier frequency-fusion approach was removed from production inference.
