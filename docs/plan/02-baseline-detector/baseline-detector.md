# Phase 2 — Baseline Detector

Goal: train a **hybrid** real-vs-fake classifier — frozen CLIP ViT-B/32 semantic embeddings fused
with a lightweight frequency-domain feature branch — on **augmented** training data (clean +
transformed copies of each image), then freeze it for use by Phases 3 and 4.

Reference: [../../problem_statement.md §5.7](../../problem_statement.md#57-technical-workshop-notes-supplementary)
(Slides 6–8: baseline pipeline, signal sources, augmentation-during-training methodology).

## Why hybrid + augmented, not a plain frozen-CLIP linear probe

The competition rules say: *"Do not directly replicate existing models or approaches."* A bare
frozen-CLIP-embedding + logistic-regression classifier is essentially the published "Universal Fake
Image Detectors" method unmodified. Fusing in a frequency-domain branch (Slide 7's "Go hybrid"
insight — CLIP semantics catch what frequency artifacts miss and vice versa) and training on
augmented data (Slide 8's stated "key idea") makes this my own pipeline while still using an
explicitly whitelisted pretrained backbone (CLIP) and staying cheap enough for the time budget.

## Steps

1. `src/detector/transforms.py` — implement the 6 required transform families (JPEG compression,
   Gaussian blur, resize, Gaussian noise, color jitter, center crop) at their specified parameters.
   Built here (not in Phase 3) because training needs them first; Phase 3 will import and reuse
   this module for evaluation.
2. `src/detector/freq_features.py` — a small, non-learned frequency-domain feature extractor (e.g.
   radially-averaged FFT magnitude spectrum binned into ~16–32 bands, or a DCT high-frequency energy
   ratio). No training required for this branch — it's a fixed transform, not a model.
3. `src/detector/embed.py` — for each CIFAKE train image, generate the clean version plus a small
   number of randomly-parameterized augmented copies (e.g. 2–3 per image, each with one randomly
   chosen transform from Phase 2's `transforms.py`, applied individually per the competition's
   "not combined" rule). Compute CLIP embeddings + frequency features for every copy, and
   concatenate the two feature vectors. Cache to disk. Do the same (clean only) for the CIFAKE test
   split, used purely for held-out evaluation.
4. `src/detector/train_probe.py` — train a `sklearn.linear_model.LogisticRegression` (or a small
   1–2 layer `torch.nn.Linear`/MLP) on the fused, augmented train features; evaluate on the clean
   CIFAKE test split (accuracy, AUC, F1).
5. Save the trained head (`outputs/probe.joblib` or `.pt`) plus `outputs/model_card.md` documenting:
   backbone(s) used, training data (including the augmentation recipe), test accuracy/AUC, total
   parameter count, and a short "novelty note" explaining how this differs from a direct
   reproduction of an existing published method.
6. Sanity-check inference end-to-end on a handful of individual images (embed → fuse → predict).

## Definition of done

- [ ] `transforms.py` and `freq_features.py` exist and are unit-sanity-checked independently of the
      classifier.
- [ ] Cached fused (CLIP + frequency) embeddings exist for the augmented train set and the clean
      test split.
- [ ] The trained head achieves clearly-above-chance accuracy on the clean CIFAKE test split
      (record the exact number — a sanity bar, not a hard requirement).
- [ ] Probe weights, the exact CLIP checkpoint, and the frequency-feature recipe are saved and
      reproducible from a single script invocation.
- [ ] `outputs/model_card.md` records: backbone(s), training data + augmentation recipe, test
      accuracy/AUC, confirmed total parameter count (must be < 2B), and the novelty/non-replication
      note.

## Time budget

1 hour 15 minutes (grew from 1h to cover the frequency branch and augmented-copy embedding; Phase 3
shrinks by the same amount since it reuses `transforms.py` built here).

## Risks

- CLIP embedding extraction (now multiplied by augmented copies) too slow on CPU → subsample the
  CIFAKE train set (e.g. 5–10k images) and/or reduce to 1–2 augmented copies per image rather than
  a full transform sweep.
- If the frequency branch adds negligible measurable accuracy, keep it anyway and report that
  honestly — its role is as much about avoiding direct replication as about raw accuracy gain.
