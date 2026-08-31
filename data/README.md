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

## Optional Cross-Generator Set

The exact organizer reference is the WildFake COCO val2017 (4,998 real) plus DALL-E Advanced
(8,843 fake) subset. The official ModelScope manifests were checked and match those counts, but
cross-generator evaluation was not run: obtaining DALL-E Advanced required downloading a 26 GB
source archive and exceeded the optional 30-minute acquisition gate. No substitute is reported as
the organizer benchmark.

Streaming fallback, not used in this run: [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set),
labels 0 and 1 only.
