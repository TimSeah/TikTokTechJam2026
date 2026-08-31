# Track 5 Execution Plan — Robust Detection of AI-Generated Images

References:
- [../problem_statement.md](../problem_statement.md#5-robust-detection-of-aigenerated-images-under-realworld-transformations)
- [../problem_statement.md §5.7](../problem_statement.md#57-technical-workshop-notes-supplementary)
  — organizer workshop slides with competition rules, scoring formula, and recommended methodology
  not present elsewhere in the base problem statement.

## Chosen approach

**A hybrid detector: CLIP (ViT-B/32) semantic embeddings fused with a lightweight frequency-domain
feature branch (FFT/DCT-based), trained on augmented CIFAKE data (clean + transformed copies of
each training image) and a shallow classifier head (logistic regression / small MLP).** Evaluated
for robustness against the 6 required transform families, plus a cross-generator generalization
check on the WildFake-derived COCO val2017 / DALL·E Advanced validation subset — used **only** for
evaluation, never for training, per the problem statement's rules.

## Why this approach

- **Avoids disqualification risk.** The workshop slides state explicitly: *"Do not directly
  replicate existing models or approaches"* and *"Just using a pre-trained AI image detection model
  would result in disqualification."* A bare frozen-CLIP-linear-probe is the well-known published
  "Universal Fake Image Detectors" method essentially unmodified — fusing in a frequency-domain
  branch and training on augmented data makes this my own pipeline, not a literal reproduction,
  while still using an explicitly whitelisted pretrained backbone (CLIP is named as approved).
- **Matches the graded methodology, not just the graded metric.** The slides frame "training for
  robustness with augmentation" as the key idea (not merely evaluating robustness after the fact),
  and separately recommend the CLIP-semantics + frequency-patch hybrid as the strongest simple
  design (Slide 7: "Go hybrid"). Building both in directly targets what reviewers were told to look
  for.
- The `<2B` parameter constraint is trivially satisfied (frozen CLIP ViT-B/32 ≈ 151M params + cheap
  frequency features + a linear/shallow head).
- No GPU is strictly required — frozen backbones plus a shallow head train in minutes on CPU, even
  with augmented training data.
- It minimizes time spent on model training/tuning and maximizes time spent on the parts that are
  actually judged: the robustness score formula, generalization check, error analysis, and
  reproducible deliverables.

## Timeline (~5h coding budget within a 12h deadline)

| Phase | Folder | Time budget |
| --- | --- | --- |
| 0. Environment & Repo Setup | [00-environment-setup](00-environment-setup/environment-setup.md) | 0:00–0:30 |
| 1. Data Acquisition | [01-data-acquisition](01-data-acquisition/data-acquisition.md) | 0:30–1:00 |
| 2. Baseline Detector (hybrid + augmented training) | [02-baseline-detector](02-baseline-detector/baseline-detector.md) | 1:00–2:15 |
| 3. Robustness Pipeline (eval only — reuses Phase 2's transforms) | [03-robustness-pipeline](03-robustness-pipeline/robustness-pipeline.md) | 2:15–3:00 |
| 4. Inference CLI Script | [04-inference-script](04-inference-script/inference-script.md) | 3:00–3:45 |
| 5. Error Analysis | [05-error-analysis](05-error-analysis/error-analysis.md) | 3:45–4:20 |
| 6. Deliverables Packaging | [06-deliverables-packaging](06-deliverables-packaging/deliverables-packaging.md) | 4:20–5:00 |

Total: ~5h. Phase 2 grew by 15 min to cover the frequency-feature branch and augmented-training
data; Phase 3 shrank by the same amount since it now reuses the transform functions Phase 2 already
built, rather than authoring them from scratch. Remaining ~7h of the 12h deadline is slack for setup
friction, re-recording the demo video, sleep, and submission-portal issues.

## Deliverable mapping (per problem statement §5.5)

| # | Required deliverable | Produced in |
| --- | --- | --- |
| 1 | Written Project Description (Devpost) | Phase 6 |
| 2 | Public Code/GitHub Repository + inference script (image dir → JSON) | Phases 2–4 |
| 3 | Demo Video | Phase 6 |
| 4 | Robustness Evaluation Summary | Phase 3 |
| 5 | Error Analysis Note | Phase 5 |

## Global definition of done (project-level)

- [ ] `predict.py <image_dir> --out preds.json` runs end-to-end and produces valid JSON with
      `image_path` + `pred` for every image in the directory.
- [ ] A robustness table exists comparing clean vs. each of the 6 transform families (at the listed
      severities, or a documented reduced subset) on a held-out test split.
- [ ] The composite score `Final Score = 0.50 × AUC_clean + 0.50 × AUC_robust` is computed and
      reported, plus a cross-generator/unseen-generator AUC from the validation-only subset.
- [ ] An error-analysis note cites concrete false positive / false negative examples with a stated
      hypothesis for each.
- [ ] A trade-offs write-up covers robustness vs. clean accuracy, generalization vs. specialization,
      and complexity vs. feasibility.
- [ ] README documents setup, reproduction steps, limitations, team contributions, and an explicit
      "why this isn't a direct replication of an existing method" note.
- [ ] No training occurred on the WildFake/COCO–DALL·E validation-only subset (verify before
      submission).
- [ ] Model uses fewer than 2B parameters (confirm and record the count); only publicly available
      pretrained backbones are used (CLIP is explicitly whitelisted).
- [ ] A LICENSE file (MIT or Apache) is present in the repo.
- [ ] Trained model weights (`outputs/probe.joblib` or equivalent) are committed/published in the
      repo, not gitignored.
- [ ] Repo is pushed, public, and linked in the Devpost description; demo video is uploaded to
      YouTube (public) and linked.

## Target repository layout (for the implementation phase)

```
<project-root>/
  LICENSE                      # MIT or Apache — required by competition rules
  data/                        # gitignored — CIFAKE + validation-only downloads
    cifake/{train,test}/{REAL,FAKE}/
    validation_only/           # COCO val2017 subset + DALL·E Advanced (demo-only, never trained on)
  src/
    detector/
      transforms.py             # the 6 required augmentation families (built in Phase 2, reused by Phase 3)
      freq_features.py           # lightweight FFT/DCT frequency-domain feature extraction
      embed.py                    # CLIP embedding extraction (clean + augmented copies)
      train_probe.py                # trains the classifier head on fused CLIP + frequency features
      evaluate.py                    # robustness table + Final Score computation
    predict.py                        # required CLI: image dir -> JSON
  outputs/
    probe.joblib                      # or .pt — trained classifier head (committed, not gitignored)
    model_card.md                     # backbone, novelty note, parameter count, accuracy
    robustness_table.csv
    error_analysis.md
    trade_offs.md
    error_examples/
    preds.json                         # gitignored — regenerated output, not part of the submission artifact set
  README.md
  requirements.txt
```

## Risks & fallbacks

- CLIP (`open_clip`/OpenAI `clip`) download blocked or slow → fall back to a smaller `open_clip`
  checkpoint, or torchvision's pretrained ResNet18 features.
- CIFAKE download friction (Kaggle account/API token needed) → fall back to SID_Set
  (Hugging Face, no auth required) as the primary training set.
- WildFake ModelScope access/translation friction for the validation-only subset → substitute a
  smaller public real/fake sample set of comparable structure and clearly document it in the README
  as a stand-in, not the official validation set.
- Time overrun in Phase 2 (baseline detector) from embedding augmented copies + frequency features
  → reduce the number of augmented copies per training image (e.g. 2 random transforms instead of
  a full sweep) and/or subsample the training set further; the frequency branch itself is cheap
  (no learned weights), so it should not be the bottleneck.
- If the frequency-domain branch turns out to add negligible accuracy → keep it anyway and report
  that finding honestly in `model_card.md`/trade-offs; its purpose is as much about demonstrating a
  non-replicated, hybrid design as about raw accuracy gain.
- Time overrun in Phase 3 (robustness sweep) → cover only the mildest + harshest severity per
  transform instead of all listed values, and document the reduced sweep as a scoping decision in
  the README's limitations section.

## Next step

Review this plan and each phase folder. Once approved, implementation starts at
[00-environment-setup](00-environment-setup/environment-setup.md) and proceeds phase by phase.
