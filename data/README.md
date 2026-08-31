# Data

Datasets, manifests, and feature caches are generated locally and are not committed to Git. The
completed run used this layout:

```text
data/
  downloads/
    train/{REAL,FAKE}/
    test/{REAL,FAKE}/
  manifests/{train,test,development}.csv
  features/
    train-{clean,augmented}/
    test-{clean,<transform_condition>}/
    sid-final-{clean,augmented,calibration}/
    wildfake-final-{clean,augmented}/
  native-train/
    sid-final/{REAL,FAKE}/
    wildfake-final/{REAL,FAKE}/
```

## Completed Dataset

- Source: [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- License shown by Kaggle: MIT
- Published split preserved without re-splitting
- Train: 100,000 images, 50,000 REAL and 50,000 FAKE
- Test: 20,000 images, 10,000 REAL and 10,000 FAKE
- Format observed: 32x32 RGB JPEG
- Label mapping: REAL = 0, FAKE = 1
- Manifest/development-sample seed: 2026
- Development manifest: 200 train images, balanced 100/100

`python -m src.detector.data --data-root data/downloads --output-dir data/manifests
--validate-sample 1000` generated the manifests and successfully decoded 1,000 seeded rows from
each one. The training and evaluation programs consume those manifests instead of independently
enumerating images.

## Cross-Domain Evaluation and Follow-Up Data

The exact organizer reference is the WildFake COCO val2017 (4,998 real) plus DALL-E Advanced
(8,843 fake) subset. The official ModelScope manifests were checked and match those counts, but
the full evaluation was not run: obtaining DALL-E Advanced required downloading a 26 GB source
archive and exceeded the optional 30-minute acquisition gate.

After the original artifact was frozen, three balanced 400-image diagnostic samples were obtained
from versioned Hugging Face sources: SID-Set validation, WildFake COCO/DALL-E Advanced, and a
LAION/DALL-E matched configuration. These samples were not used to fit the reported model and are
not presented as the full organizer benchmark. Their protocol and results are recorded in
[outputs/cross_domain_summary.json](../outputs/cross_domain_summary.json) and the
[technical report](../docs/technical_report.pdf).

The deployed semantic model uses a deterministic SID-Set sample with 4,000 images per class for
training and 500 per class for threshold calibration. A separate WildFake sample contributes 4,000
images per class from `ADM`, `DDPM`, `GALIP`, `GigaGAN`, `VQGAN`, `VQVAE`, `AFHQ`, `CelebA-HQ`,
and `LSUN-Church`. WildFake COCO val2017 and DALL-E Advanced are hard-forbidden by the acquisition
code. All 17,000 selected SID/WildFake train and calibration images are hash-unique and have zero
hash overlap with the frozen evaluation manifests.

Each training image has one clean view and one deterministic transformed view. Exactly one of JPEG,
blur, resize, noise, color jitter, or center crop is applied; transforms are never stacked. Dataset
revisions, source paths, split roles, dimensions, byte sizes, and SHA-256 values are retained in the
local provenance files.
