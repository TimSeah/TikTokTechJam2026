# Phase 2 — Baseline Detector

Goal: first prove a 200-image end-to-end vertical slice, then train a **hybrid** real-vs-fake
classifier — frozen CLIP ViT-B/32 semantic embeddings fused with a lightweight frequency-domain
feature branch — on full, augmented CIFAKE data using the selected GPU path. Produce restartable
feature caches, a CPU-loadable inference artifact, and ablations that measure the contribution of
frequency fusion and augmentation.

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

1. Add a shared run configuration containing the data manifest, seed, backbone/checkpoint, input
      size, batch size, device, frequency bins, and augmentation recipe. Every training and evaluation
      command must accept `--device auto` and a small `--limit` for smoke tests.
2. `src/detector/transforms.py` — implement the 6 required transform families (JPEG compression,
   Gaussian blur, resize, Gaussian noise, color jitter, center crop) at their specified parameters.
   Built here (not in Phase 3) because training needs them first; Phase 3 will import and reuse
      this module for evaluation. Unit-check shape, value range, and determinism under a fixed seed.
3. `src/detector/freq_features.py` — compute a fixed 32-bin radially-averaged log-FFT magnitude
      spectrum. Unit-check output length and finite values for RGB, grayscale, and constant images.
4. `src/detector/embed.py` — load the manifest once, batch image preprocessing, and run CLIP under
      `torch.inference_mode()` with FP16 autocast on GPU. Start at batch size 256 and increase only
      after measuring memory and throughput. Keep CLIP embeddings and frequency features as separate
      arrays so ablations can reuse the same extraction.
5. Run a balanced 200-image vertical slice through transforms → feature extraction → training →
      saved artifact → prediction before starting the full extraction.
6. For the full train split, cache every clean image plus one seeded, randomly selected transform
      applied individually. Write restartable shards containing semantic features, frequency features,
      labels, image IDs, and transform metadata. Add a second augmented copy only after all required
      deliverables exist.
7. Cache the clean held-out test features separately. Phase 3 will add one cache per transformed
      evaluation condition so all model variants use exactly the same images.
8. `src/detector/train_probe.py` — train three `StandardScaler` +
      `sklearn.linear_model.LogisticRegression` pipelines using the same held-out split:
      - semantic-only features trained on clean images;
      - semantic + frequency features trained on clean images;
      - semantic + frequency features trained on clean and augmented images (the final model).
9. Save one artifact bundle containing the final scaler/classifier, class mapping, CLIP model and
      checkpoint names, preprocessing, FFT recipe, feature dimensions, manifest hash, and random seed.
      Do not save only a bare classifier whose expected feature preprocessing is implicit.
10. Write `outputs/model_card.md` with the three clean-result rows, training data and augmentation
       recipe, measured parameter count, extraction throughput, and a concise non-replication note.
11. Sanity-check the saved artifact on individual images using both the selected GPU and forced CPU.

## Definition of done

- [ ] The 200-image vertical slice completes before full extraction begins.
- [ ] Transform and frequency-feature tests pass for shape, range, determinism, and finite values.
- [ ] Restartable clean/augmented train caches and clean test caches contain features, labels, IDs,
      and metadata with matching row counts.
- [ ] All three ablation pipelines train on identical manifests and record clean AUC/accuracy/F1.
- [ ] The final artifact records all preprocessing/configuration needed for standalone inference and
      loads on CPU.
- [ ] `outputs/model_card.md` records the exact checkpoint, data manifest, augmentation recipe,
      throughput, parameter count (<2B), ablations, and novelty/non-replication note.

## Time budget

3 hours 15 minutes after the compute/data gate: 1 hour 45 minutes for implementation and the
vertical slice, followed by 1 hour 30 minutes for full extraction and training.

## Risks

- CLIP operation fails on native ROCm → switch to the already-verified Colab path; if it fails there,
      replace CLIP with pretrained ResNet18 while preserving the frequency branch and ablations.
- Full extraction exceeds the 4:00 hard cut → stop after the current shard and use a deterministic,
      balanced subset of up to 20k train and 4k test images. Keep one augmented copy per train image.
- GPU is underutilized → increase batch size and data-loader workers after measuring; do not redesign
      the model during the timed run.
- Colab terminates → resume from completed shards and never keep the only model/cache copy in the
      ephemeral runtime.
- If the frequency branch adds negligible measurable accuracy, keep it anyway and report that
      honestly in the ablation rather than claiming an unsupported improvement.
